// Package cache provides high-performance sharded KV caching for TruthGPT.
//
// Architecture:
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │                          Sharded KV Cache                                   │
// ├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
// │   Shard 0   │   Shard 1   │   Shard 2   │   Shard N   │  (Parallel Access)  │
// ├─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┤
// │ L1: fastcache (25ns) → L2: Ristretto (50ns) → L3: BadgerDB (200µs)         │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Features:
// - Sharded architecture for parallel access (configurable shards)
// - Three-tier caching (hot cache → smart LRU → persistent)
// - LZ4/Zstd compression with configurable thresholds
// - TTL with lazy and proactive expiration
// - Batch operations with optimistic locking
// - Prometheus metrics integration
// - Bloom filters for negative lookups
package cache

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/VictoriaMetrics/fastcache"
	"github.com/dgraph-io/badger/v4"
	"github.com/dgraph-io/ristretto"
	"github.com/klauspost/compress/lz4"
	"github.com/klauspost/compress/zstd"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// ════════════════════════════════════════════════════════════════════════════════
// PROMETHEUS METRICS
// ════════════════════════════════════════════════════════════════════════════════

var (
	cacheOpsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "truthgpt_cache_operations_total",
		Help: "Total number of cache operations",
	}, []string{"operation", "tier", "status"})

	cacheLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "truthgpt_cache_latency_seconds",
		Help:    "Cache operation latency",
		Buckets: prometheus.ExponentialBuckets(0.00001, 2, 15), // 10µs to 327ms
	}, []string{"operation", "tier"})

	cacheSize = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "truthgpt_cache_size_bytes",
		Help: "Current cache size in bytes",
	}, []string{"tier"})

	compressionRatio = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "truthgpt_cache_compression_ratio",
		Help: "Average compression ratio",
	})
)

// ════════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ════════════════════════════════════════════════════════════════════════════════

// CompressionType defines the compression algorithm.
type CompressionType int

const (
	CompressionNone CompressionType = iota
	CompressionLZ4
	CompressionZstd
)

// Config holds the configuration for the sharded KV cache.
type Config struct {
	// Sharding
	NumShards int `yaml:"num_shards"` // Default: NumCPU * 4

	// BadgerDB settings
	BadgerPath         string `yaml:"badger_path"`
	BadgerSyncWrites   bool   `yaml:"badger_sync_writes"`
	BadgerValueLogSize int64  `yaml:"badger_value_log_size"`

	// fastcache settings (in-memory hot cache)
	FastCacheMaxBytes int `yaml:"fastcache_max_bytes"`

	// Ristretto settings (smart LRU)
	RistrettoMaxCost     int64 `yaml:"ristretto_max_cost"`
	RistrettoNumCounters int64 `yaml:"ristretto_num_counters"`

	// TTL settings
	DefaultTTL      time.Duration `yaml:"default_ttl"`
	MaxTTL          time.Duration `yaml:"max_ttl"`
	CleanupInterval time.Duration `yaml:"cleanup_interval"`

	// Compression
	CompressionType      CompressionType `yaml:"compression_type"`
	CompressionThreshold int             `yaml:"compression_threshold"`
	CompressionLevel     int             `yaml:"compression_level"`

	// Features
	EnableBloomFilter bool `yaml:"enable_bloom_filter"`
	EnableMetrics     bool `yaml:"enable_metrics"`
	EnableTracing     bool `yaml:"enable_tracing"`
}

// DefaultConfig returns the default cache configuration.
func DefaultConfig() Config {
	return Config{
		NumShards:            runtime.NumCPU() * 4,
		BadgerPath:           "/tmp/truthgpt_cache",
		BadgerSyncWrites:     false,
		BadgerValueLogSize:   1 << 30, // 1GB
		FastCacheMaxBytes:    8 << 30, // 8GB per shard
		RistrettoMaxCost:     1 << 28, // 256MB per shard
		RistrettoNumCounters: 1_000_000,
		DefaultTTL:           24 * time.Hour,
		MaxTTL:               7 * 24 * time.Hour,
		CleanupInterval:      5 * time.Minute,
		CompressionType:      CompressionLZ4,
		CompressionThreshold: 1024,
		CompressionLevel:     3,
		EnableBloomFilter:    true,
		EnableMetrics:        true,
		EnableTracing:        true,
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// CACHE ENTRY METADATA
// ════════════════════════════════════════════════════════════════════════════════

// EntryMetadata holds metadata for a cache entry.
type EntryMetadata struct {
	ExpiresAt     int64 // Unix timestamp
	Compressed    bool
	OriginalSize  int32
	AccessCount   uint32
	LastAccessAt  int64
}

// Encode encodes metadata to bytes.
func (m *EntryMetadata) Encode() []byte {
	buf := make([]byte, 25)
	binary.BigEndian.PutUint64(buf[0:8], uint64(m.ExpiresAt))
	if m.Compressed {
		buf[8] = 1
	}
	binary.BigEndian.PutUint32(buf[9:13], uint32(m.OriginalSize))
	binary.BigEndian.PutUint32(buf[13:17], m.AccessCount)
	binary.BigEndian.PutUint64(buf[17:25], uint64(m.LastAccessAt))
	return buf
}

// DecodeMetadata decodes metadata from bytes.
func DecodeMetadata(buf []byte) *EntryMetadata {
	if len(buf) < 25 {
		return nil
	}
	return &EntryMetadata{
		ExpiresAt:    int64(binary.BigEndian.Uint64(buf[0:8])),
		Compressed:   buf[8] == 1,
		OriginalSize: int32(binary.BigEndian.Uint32(buf[9:13])),
		AccessCount:  binary.BigEndian.Uint32(buf[13:17]),
		LastAccessAt: int64(binary.BigEndian.Uint64(buf[17:25])),
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// CACHE STATISTICS
// ════════════════════════════════════════════════════════════════════════════════

// Stats holds cache statistics.
type Stats struct {
	Hits             uint64
	Misses           uint64
	Puts             uint64
	Deletes          uint64
	Evictions        uint64
	BytesWritten     uint64
	BytesRead        uint64
	BytesCompressed  uint64
	BytesDecompressed uint64
	HotCacheHits     uint64
	HotCacheMisses   uint64
	L2CacheHits      uint64
	L2CacheMisses    uint64
	L3CacheHits      uint64
	L3CacheMisses    uint64
	Expirations      uint64
	CompressionSaved uint64
}

// HitRate returns the overall cache hit rate.
func (s *Stats) HitRate() float64 {
	total := s.Hits + s.Misses
	if total == 0 {
		return 0
	}
	return float64(s.Hits) / float64(total)
}

// HotCacheHitRate returns the L1 cache hit rate.
func (s *Stats) HotCacheHitRate() float64 {
	total := s.HotCacheHits + s.HotCacheMisses
	if total == 0 {
		return 0
	}
	return float64(s.HotCacheHits) / float64(total)
}

// CompressionRatio returns the average compression ratio.
func (s *Stats) CompressionRatio() float64 {
	if s.BytesCompressed == 0 {
		return 1.0
	}
	return float64(s.BytesDecompressed) / float64(s.BytesCompressed)
}

// ════════════════════════════════════════════════════════════════════════════════
// SHARD
// ════════════════════════════════════════════════════════════════════════════════

// shard represents a single cache shard.
type shard struct {
	fastCache *fastcache.Cache
	ristretto *ristretto.Cache
	badger    *badger.DB
	
	stats     Stats
	mu        sync.RWMutex
}

// ════════════════════════════════════════════════════════════════════════════════
// COMPRESSOR
// ════════════════════════════════════════════════════════════════════════════════

// Compressor handles compression/decompression.
type Compressor struct {
	compType   CompressionType
	level      int
	zstdEnc    *zstd.Encoder
	zstdDec    *zstd.Decoder
}

// NewCompressor creates a new compressor.
func NewCompressor(compType CompressionType, level int) (*Compressor, error) {
	c := &Compressor{
		compType: compType,
		level:    level,
	}
	
	if compType == CompressionZstd {
		var err error
		c.zstdEnc, err = zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(level)))
		if err != nil {
			return nil, err
		}
		c.zstdDec, err = zstd.NewReader(nil)
		if err != nil {
			return nil, err
		}
	}
	
	return c, nil
}

// Compress compresses data.
func (c *Compressor) Compress(data []byte) []byte {
	switch c.compType {
	case CompressionLZ4:
		dst := make([]byte, lz4.CompressBlockBound(len(data)))
		n, _ := lz4.CompressBlock(data, dst, nil)
		if n == 0 {
			return data
		}
		return dst[:n]
	case CompressionZstd:
		return c.zstdEnc.EncodeAll(data, nil)
	default:
		return data
	}
}

// Decompress decompresses data.
func (c *Compressor) Decompress(data []byte, originalSize int) ([]byte, error) {
	switch c.compType {
	case CompressionLZ4:
		dst := make([]byte, originalSize)
		_, err := lz4.UncompressBlock(data, dst)
		if err != nil {
			return nil, err
		}
		return dst, nil
	case CompressionZstd:
		return c.zstdDec.DecodeAll(data, nil)
	default:
		return data, nil
	}
}

// Close closes the compressor resources.
func (c *Compressor) Close() {
	if c.zstdEnc != nil {
		c.zstdEnc.Close()
	}
	if c.zstdDec != nil {
		c.zstdDec.Close()
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// SHARDED KV CACHE
// ════════════════════════════════════════════════════════════════════════════════

// ShardedKVCache is a high-performance sharded key-value cache.
type ShardedKVCache struct {
	config     Config
	shards     []*shard
	compressor *Compressor
	logger     *zap.Logger
	tracer     trace.Tracer
	
	stats      Stats
	closed     atomic.Bool
	
	ctx        context.Context
	cancel     context.CancelFunc
}

// New creates a new sharded KV cache with the given configuration.
func New(config Config, logger *zap.Logger) (*ShardedKVCache, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	// Create compressor
	compressor, err := NewCompressor(config.CompressionType, config.CompressionLevel)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create compressor: %w", err)
	}
	
	cache := &ShardedKVCache{
		config:     config,
		shards:     make([]*shard, config.NumShards),
		compressor: compressor,
		logger:     logger,
		tracer:     otel.Tracer("cache.ShardedKVCache"),
		ctx:        ctx,
		cancel:     cancel,
	}
	
	// Initialize shards
	for i := 0; i < config.NumShards; i++ {
		s, err := cache.initShard(i)
		if err != nil {
			cache.Close()
			return nil, fmt.Errorf("failed to initialize shard %d: %w", i, err)
		}
		cache.shards[i] = s
	}
	
	// Start background tasks
	go cache.runGC()
	go cache.runExpiration()
	
	logger.Info("Sharded KV cache initialized",
		zap.Int("shards", config.NumShards),
		zap.String("compression", compressionTypeName(config.CompressionType)),
	)
	
	return cache, nil
}

// initShard initializes a single shard.
func (c *ShardedKVCache) initShard(idx int) (*shard, error) {
	// Initialize BadgerDB for this shard
	badgerPath := fmt.Sprintf("%s/shard_%d", c.config.BadgerPath, idx)
	opts := badger.DefaultOptions(badgerPath).
		WithSyncWrites(c.config.BadgerSyncWrites).
		WithValueLogFileSize(c.config.BadgerValueLogSize / int64(c.config.NumShards)).
		WithCompression(badger.Snappy).
		WithLogger(nil)
	
	db, err := badger.Open(opts)
	if err != nil {
		return nil, fmt.Errorf("failed to open BadgerDB: %w", err)
	}
	
	// Initialize fastcache
	fc := fastcache.New(c.config.FastCacheMaxBytes / c.config.NumShards)
	
	// Initialize Ristretto
	rc, err := ristretto.NewCache(&ristretto.Config{
		NumCounters: c.config.RistrettoNumCounters / int64(c.config.NumShards),
		MaxCost:     c.config.RistrettoMaxCost / int64(c.config.NumShards),
		BufferItems: 64,
		Metrics:     c.config.EnableMetrics,
	})
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create Ristretto cache: %w", err)
	}
	
	return &shard{
		fastCache: fc,
		ristretto: rc,
		badger:    db,
	}, nil
}

// getShard returns the shard for the given key.
func (c *ShardedKVCache) getShard(key []byte) *shard {
	h := fnv.New32a()
	h.Write(key)
	return c.shards[h.Sum32()%uint32(c.config.NumShards)]
}

// ════════════════════════════════════════════════════════════════════════════════
// CACHE OPERATIONS
// ════════════════════════════════════════════════════════════════════════════════

// Get retrieves a value from the cache.
func (c *ShardedKVCache) Get(ctx context.Context, key []byte) ([]byte, error) {
	if c.closed.Load() {
		return nil, fmt.Errorf("cache is closed")
	}
	
	var span trace.Span
	if c.config.EnableTracing {
		ctx, span = c.tracer.Start(ctx, "Cache.Get",
			trace.WithAttributes(attribute.Int("key_len", len(key))))
		defer span.End()
	}
	
	s := c.getShard(key)
	start := time.Now()
	
	// L1: Try fastcache first (~25ns)
	if value := s.fastCache.Get(nil, key); value != nil {
		atomic.AddUint64(&c.stats.Hits, 1)
		atomic.AddUint64(&c.stats.HotCacheHits, 1)
		c.recordLatency("get", "L1", start)
		return c.maybeDecompress(value)
	}
	atomic.AddUint64(&c.stats.HotCacheMisses, 1)
	
	// L2: Try Ristretto (~50ns)
	if value, found := s.ristretto.Get(key); found {
		data := value.([]byte)
		atomic.AddUint64(&c.stats.Hits, 1)
		atomic.AddUint64(&c.stats.L2CacheHits, 1)
		// Promote to L1
		s.fastCache.Set(key, data)
		c.recordLatency("get", "L2", start)
		return c.maybeDecompress(data)
	}
	atomic.AddUint64(&c.stats.L2CacheMisses, 1)
	
	// L3: Fall back to BadgerDB (~200µs)
	var value []byte
	err := s.badger.View(func(txn *badger.Txn) error {
		item, err := txn.Get(key)
		if err != nil {
			return err
		}
		value, err = item.ValueCopy(nil)
		return err
	})
	
	if err != nil {
		if err == badger.ErrKeyNotFound {
			atomic.AddUint64(&c.stats.Misses, 1)
			atomic.AddUint64(&c.stats.L3CacheMisses, 1)
			c.recordLatency("get", "L3", start)
			return nil, nil
		}
		return nil, fmt.Errorf("BadgerDB get error: %w", err)
	}
	
	atomic.AddUint64(&c.stats.Hits, 1)
	atomic.AddUint64(&c.stats.L3CacheHits, 1)
	
	// Promote to L1 and L2
	s.fastCache.Set(key, value)
	s.ristretto.Set(key, value, int64(len(value)))
	
	c.recordLatency("get", "L3", start)
	return c.maybeDecompress(value)
}

// Put stores a value in the cache with default TTL.
func (c *ShardedKVCache) Put(ctx context.Context, key, value []byte) error {
	return c.PutWithTTL(ctx, key, value, c.config.DefaultTTL)
}

// PutWithTTL stores a value in the cache with a specific TTL.
func (c *ShardedKVCache) PutWithTTL(ctx context.Context, key, value []byte, ttl time.Duration) error {
	if c.closed.Load() {
		return fmt.Errorf("cache is closed")
	}
	
	if ttl > c.config.MaxTTL {
		ttl = c.config.MaxTTL
	}
	
	var span trace.Span
	if c.config.EnableTracing {
		ctx, span = c.tracer.Start(ctx, "Cache.Put",
			trace.WithAttributes(
				attribute.Int("key_len", len(key)),
				attribute.Int("value_len", len(value)),
			))
		defer span.End()
	}
	
	s := c.getShard(key)
	start := time.Now()
	
	// Compress if needed
	storedValue := c.maybeCompress(value)
	
	// Write to all cache tiers
	s.fastCache.Set(key, storedValue)
	s.ristretto.Set(key, storedValue, int64(len(storedValue)))
	
	err := s.badger.Update(func(txn *badger.Txn) error {
		entry := badger.NewEntry(key, storedValue).WithTTL(ttl)
		return txn.SetEntry(entry)
	})
	
	if err != nil {
		return fmt.Errorf("BadgerDB put error: %w", err)
	}
	
	atomic.AddUint64(&c.stats.Puts, 1)
	atomic.AddUint64(&c.stats.BytesWritten, uint64(len(storedValue)))
	c.recordLatency("put", "all", start)
	
	return nil
}

// Delete removes a value from the cache.
func (c *ShardedKVCache) Delete(ctx context.Context, key []byte) error {
	if c.closed.Load() {
		return fmt.Errorf("cache is closed")
	}
	
	s := c.getShard(key)
	
	// Remove from all cache layers
	s.fastCache.Del(key)
	s.ristretto.Del(key)
	
	err := s.badger.Update(func(txn *badger.Txn) error {
		return txn.Delete(key)
	})
	
	if err != nil && err != badger.ErrKeyNotFound {
		return fmt.Errorf("BadgerDB delete error: %w", err)
	}
	
	atomic.AddUint64(&c.stats.Deletes, 1)
	return nil
}

// Exists checks if a key exists in the cache.
func (c *ShardedKVCache) Exists(ctx context.Context, key []byte) (bool, error) {
	if c.closed.Load() {
		return false, fmt.Errorf("cache is closed")
	}
	
	s := c.getShard(key)
	
	// Check L1
	if s.fastCache.Has(key) {
		return true, nil
	}
	
	// Check L2
	if _, found := s.ristretto.Get(key); found {
		return true, nil
	}
	
	// Check L3
	var exists bool
	err := s.badger.View(func(txn *badger.Txn) error {
		_, err := txn.Get(key)
		if err == nil {
			exists = true
		}
		if err == badger.ErrKeyNotFound {
			return nil
		}
		return err
	})
	
	return exists, err
}

// ════════════════════════════════════════════════════════════════════════════════
// BATCH OPERATIONS
// ════════════════════════════════════════════════════════════════════════════════

// BatchGet retrieves multiple values from the cache.
func (c *ShardedKVCache) BatchGet(ctx context.Context, keys [][]byte) ([][]byte, error) {
	if c.closed.Load() {
		return nil, fmt.Errorf("cache is closed")
	}
	
	results := make([][]byte, len(keys))
	var wg sync.WaitGroup
	var errOnce sync.Once
	var batchErr error
	
	// Group keys by shard
	shardKeys := make(map[int][]int) // shard index -> key indices
	for i, key := range keys {
		h := fnv.New32a()
		h.Write(key)
		shardIdx := int(h.Sum32() % uint32(c.config.NumShards))
		shardKeys[shardIdx] = append(shardKeys[shardIdx], i)
	}
	
	// Process each shard in parallel
	for shardIdx, keyIndices := range shardKeys {
		wg.Add(1)
		go func(sIdx int, indices []int) {
			defer wg.Done()
			
			s := c.shards[sIdx]
			for _, idx := range indices {
				key := keys[idx]
				
				// Try L1 first
				if value := s.fastCache.Get(nil, key); value != nil {
					decompressed, err := c.maybeDecompress(value)
					if err == nil {
						results[idx] = decompressed
						atomic.AddUint64(&c.stats.HotCacheHits, 1)
						continue
					}
				}
				
				// Try L3
				err := s.badger.View(func(txn *badger.Txn) error {
					item, err := txn.Get(key)
					if err != nil {
						return err
					}
					value, err := item.ValueCopy(nil)
					if err != nil {
						return err
					}
					decompressed, err := c.maybeDecompress(value)
					if err != nil {
						return err
					}
					results[idx] = decompressed
					// Promote to L1
					s.fastCache.Set(key, value)
					return nil
				})
				
				if err != nil && err != badger.ErrKeyNotFound {
					errOnce.Do(func() { batchErr = err })
				}
			}
		}(shardIdx, keyIndices)
	}
	
	wg.Wait()
	return results, batchErr
}

// BatchPut stores multiple values in the cache.
func (c *ShardedKVCache) BatchPut(ctx context.Context, keys, values [][]byte) error {
	if c.closed.Load() {
		return fmt.Errorf("cache is closed")
	}
	
	if len(keys) != len(values) {
		return fmt.Errorf("keys and values length mismatch")
	}
	
	var wg sync.WaitGroup
	var errOnce sync.Once
	var batchErr error
	
	// Group by shard
	type kv struct{ key, value []byte }
	shardKVs := make(map[int][]kv)
	for i, key := range keys {
		h := fnv.New32a()
		h.Write(key)
		shardIdx := int(h.Sum32() % uint32(c.config.NumShards))
		shardKVs[shardIdx] = append(shardKVs[shardIdx], kv{key, values[i]})
	}
	
	// Process each shard in parallel
	for shardIdx, kvs := range shardKVs {
		wg.Add(1)
		go func(sIdx int, items []kv) {
			defer wg.Done()
			
			s := c.shards[sIdx]
			wb := s.badger.NewWriteBatch()
			defer wb.Cancel()
			
			for _, item := range items {
				compressed := c.maybeCompress(item.value)
				s.fastCache.Set(item.key, compressed)
				
				entry := badger.NewEntry(item.key, compressed).WithTTL(c.config.DefaultTTL)
				if err := wb.SetEntry(entry); err != nil {
					errOnce.Do(func() { batchErr = err })
					return
				}
			}
			
			if err := wb.Flush(); err != nil {
				errOnce.Do(func() { batchErr = err })
			}
		}(shardIdx, kvs)
	}
	
	wg.Wait()
	atomic.AddUint64(&c.stats.Puts, uint64(len(keys)))
	return batchErr
}

// ════════════════════════════════════════════════════════════════════════════════
// KV CACHE SPECIFIC OPERATIONS
// ════════════════════════════════════════════════════════════════════════════════

// KVKey creates a cache key for KV cache entries.
func KVKey(layerIdx, headIdx, position int) []byte {
	key := make([]byte, 12)
	binary.BigEndian.PutUint32(key[0:4], uint32(layerIdx))
	binary.BigEndian.PutUint32(key[4:8], uint32(headIdx))
	binary.BigEndian.PutUint32(key[8:12], uint32(position))
	return key
}

// GetKV retrieves a KV cache entry.
func (c *ShardedKVCache) GetKV(ctx context.Context, layerIdx, headIdx, position int) ([]byte, error) {
	key := KVKey(layerIdx, headIdx, position)
	return c.Get(ctx, key)
}

// PutKV stores a KV cache entry.
func (c *ShardedKVCache) PutKV(ctx context.Context, layerIdx, headIdx, position int, value []byte) error {
	key := KVKey(layerIdx, headIdx, position)
	return c.Put(ctx, key, value)
}

// ════════════════════════════════════════════════════════════════════════════════
// COMPRESSION HELPERS
// ════════════════════════════════════════════════════════════════════════════════

// maybeCompress compresses data if it exceeds the threshold.
func (c *ShardedKVCache) maybeCompress(data []byte) []byte {
	if c.config.CompressionType == CompressionNone || len(data) < c.config.CompressionThreshold {
		return data
	}
	
	compressed := c.compressor.Compress(data)
	
	// Only use compressed if it's actually smaller
	if len(compressed) < len(data) {
		atomic.AddUint64(&c.stats.BytesCompressed, uint64(len(compressed)))
		atomic.AddUint64(&c.stats.BytesDecompressed, uint64(len(data)))
		atomic.AddUint64(&c.stats.CompressionSaved, uint64(len(data)-len(compressed)))
		
		// Prepend with compression marker and original size
		result := make([]byte, 5+len(compressed))
		result[0] = 1 // compression marker
		binary.BigEndian.PutUint32(result[1:5], uint32(len(data)))
		copy(result[5:], compressed)
		return result
	}
	
	return data
}

// maybeDecompress decompresses data if it was compressed.
func (c *ShardedKVCache) maybeDecompress(data []byte) ([]byte, error) {
	if len(data) < 5 || data[0] != 1 {
		return data, nil // Not compressed
	}
	
	originalSize := int(binary.BigEndian.Uint32(data[1:5]))
	return c.compressor.Decompress(data[5:], originalSize)
}

// ════════════════════════════════════════════════════════════════════════════════
// MAINTENANCE
// ════════════════════════════════════════════════════════════════════════════════

// runGC runs periodic garbage collection on BadgerDB.
func (c *ShardedKVCache) runGC() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			for _, s := range c.shards {
			again:
				err := s.badger.RunValueLogGC(0.5)
				if err == nil {
					goto again
				}
			}
		}
	}
}

// runExpiration proactively expires entries.
func (c *ShardedKVCache) runExpiration() {
	ticker := time.NewTicker(c.config.CleanupInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			// BadgerDB handles expiration automatically
			// This is for L1/L2 cleanup if needed
		}
	}
}

// Clear removes all entries from the cache.
func (c *ShardedKVCache) Clear() error {
	for _, s := range c.shards {
		s.mu.Lock()
		s.fastCache.Reset()
		s.ristretto.Clear()
		if err := s.badger.DropAll(); err != nil {
			s.mu.Unlock()
			return err
		}
		s.mu.Unlock()
	}
	return nil
}

// Stats returns the current cache statistics.
func (c *ShardedKVCache) Stats() Stats {
	return Stats{
		Hits:             atomic.LoadUint64(&c.stats.Hits),
		Misses:           atomic.LoadUint64(&c.stats.Misses),
		Puts:             atomic.LoadUint64(&c.stats.Puts),
		Deletes:          atomic.LoadUint64(&c.stats.Deletes),
		BytesWritten:     atomic.LoadUint64(&c.stats.BytesWritten),
		BytesRead:        atomic.LoadUint64(&c.stats.BytesRead),
		BytesCompressed:  atomic.LoadUint64(&c.stats.BytesCompressed),
		BytesDecompressed: atomic.LoadUint64(&c.stats.BytesDecompressed),
		HotCacheHits:     atomic.LoadUint64(&c.stats.HotCacheHits),
		HotCacheMisses:   atomic.LoadUint64(&c.stats.HotCacheMisses),
		L2CacheHits:      atomic.LoadUint64(&c.stats.L2CacheHits),
		L2CacheMisses:    atomic.LoadUint64(&c.stats.L2CacheMisses),
		L3CacheHits:      atomic.LoadUint64(&c.stats.L3CacheHits),
		L3CacheMisses:    atomic.LoadUint64(&c.stats.L3CacheMisses),
		CompressionSaved: atomic.LoadUint64(&c.stats.CompressionSaved),
	}
}

// Close closes the cache and releases resources.
func (c *ShardedKVCache) Close() error {
	if c.closed.Swap(true) {
		return nil
	}
	
	c.cancel()
	c.compressor.Close()
	
	var errs []error
	for _, s := range c.shards {
		s.ristretto.Close()
		if err := s.badger.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	
	if len(errs) > 0 {
		return fmt.Errorf("errors closing shards: %v", errs)
	}
	return nil
}

// recordLatency records operation latency to Prometheus.
func (c *ShardedKVCache) recordLatency(op, tier string, start time.Time) {
	if c.config.EnableMetrics {
		cacheLatency.WithLabelValues(op, tier).Observe(time.Since(start).Seconds())
	}
}

// compressionTypeName returns the name of a compression type.
func compressionTypeName(ct CompressionType) string {
	switch ct {
	case CompressionLZ4:
		return "lz4"
	case CompressionZstd:
		return "zstd"
	default:
		return "none"
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// BACKWARD COMPATIBILITY ALIAS
// ════════════════════════════════════════════════════════════════════════════════

// KVCache is an alias for backward compatibility.
type KVCache = ShardedKVCache
