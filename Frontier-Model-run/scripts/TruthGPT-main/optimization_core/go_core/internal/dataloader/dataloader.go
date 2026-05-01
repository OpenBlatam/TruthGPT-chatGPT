// Package dataloader provides efficient parallel data loading for training.
package dataloader

import (
	"bufio"
	"encoding/json"
	"io"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// ════════════════════════════════════════════════════════════════════════════════
// TYPES
// ════════════════════════════════════════════════════════════════════════════════

// Sample represents a single training sample.
type Sample struct {
	ID       string         `json:"id,omitempty"`
	Text     string         `json:"text"`
	Label    int            `json:"label,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Batch represents a batch of samples.
type Batch struct {
	Samples []Sample
	Index   int
}

// Config holds dataloader configuration.
type Config struct {
	BatchSize   int
	NumWorkers  int
	Shuffle     bool
	DropLast    bool
	PrefetchN   int
	BufferSize  int
	MaxSamples  int // 0 = unlimited
	Seed        int64
}

// DefaultConfig returns default configuration.
func DefaultConfig() Config {
	return Config{
		BatchSize:  32,
		NumWorkers: 4,
		Shuffle:    true,
		DropLast:   false,
		PrefetchN:  2,
		BufferSize: 1000,
		MaxSamples: 0,
		Seed:       time.Now().UnixNano(),
	}
}

// Stats holds dataloader statistics.
type Stats struct {
	SamplesLoaded   int64
	BatchesProduced int64
	LoadTimeMs      int64
	Errors          int64
}

// ════════════════════════════════════════════════════════════════════════════════
// DATALOADER
// ════════════════════════════════════════════════════════════════════════════════

// DataLoader provides efficient parallel data loading.
type DataLoader struct {
	config  Config
	sources []string
	stats   Stats
	rng     *rand.Rand
	mu      sync.Mutex
	
	batchCh chan Batch
	done    chan struct{}
	wg      sync.WaitGroup
}

// New creates a new DataLoader.
func New(config Config) *DataLoader {
	return &DataLoader{
		config:  config,
		sources: make([]string, 0),
		rng:     rand.New(rand.NewSource(config.Seed)),
		done:    make(chan struct{}),
	}
}

// AddSource adds a data source (file path).
func (d *DataLoader) AddSource(path string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.sources = append(d.sources, path)
}

// Start begins data loading.
func (d *DataLoader) Start() <-chan Batch {
	d.batchCh = make(chan Batch, d.config.PrefetchN)
	
	d.wg.Add(1)
	go d.loadLoop()
	
	return d.batchCh
}

// Stop stops the dataloader.
func (d *DataLoader) Stop() {
	close(d.done)
	d.wg.Wait()
}

// Stats returns current statistics.
func (d *DataLoader) Stats() Stats {
	return Stats{
		SamplesLoaded:   atomic.LoadInt64(&d.stats.SamplesLoaded),
		BatchesProduced: atomic.LoadInt64(&d.stats.BatchesProduced),
		LoadTimeMs:      atomic.LoadInt64(&d.stats.LoadTimeMs),
		Errors:          atomic.LoadInt64(&d.stats.Errors),
	}
}

func (d *DataLoader) loadLoop() {
	defer d.wg.Done()
	defer close(d.batchCh)
	
	// Load all samples
	samples := d.loadAllSamples()
	if len(samples) == 0 {
		return
	}
	
	// Shuffle if needed
	if d.config.Shuffle {
		d.rng.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})
	}
	
	// Create batches
	batchIdx := 0
	for i := 0; i < len(samples); i += d.config.BatchSize {
		select {
		case <-d.done:
			return
		default:
		}
		
		end := i + d.config.BatchSize
		if end > len(samples) {
			if d.config.DropLast {
				break
			}
			end = len(samples)
		}
		
		batch := Batch{
			Samples: samples[i:end],
			Index:   batchIdx,
		}
		
		select {
		case d.batchCh <- batch:
			atomic.AddInt64(&d.stats.BatchesProduced, 1)
		case <-d.done:
			return
		}
		
		batchIdx++
	}
}

func (d *DataLoader) loadAllSamples() []Sample {
	var allSamples []Sample
	var mu sync.Mutex
	var wg sync.WaitGroup
	
	sampleCh := make(chan []Sample, d.config.NumWorkers)
	
	// Start workers
	for w := 0; w < d.config.NumWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := workerID; i < len(d.sources); i += d.config.NumWorkers {
				start := time.Now()
				samples := d.loadFile(d.sources[i])
				atomic.AddInt64(&d.stats.LoadTimeMs, time.Since(start).Milliseconds())
				
				if len(samples) > 0 {
					sampleCh <- samples
				}
			}
		}(w)
	}
	
	// Collector
	go func() {
		wg.Wait()
		close(sampleCh)
	}()
	
	// Collect results
	for samples := range sampleCh {
		mu.Lock()
		allSamples = append(allSamples, samples...)
		atomic.AddInt64(&d.stats.SamplesLoaded, int64(len(samples)))
		
		// Check max samples
		if d.config.MaxSamples > 0 && len(allSamples) >= d.config.MaxSamples {
			allSamples = allSamples[:d.config.MaxSamples]
			mu.Unlock()
			break
		}
		mu.Unlock()
	}
	
	return allSamples
}

func (d *DataLoader) loadFile(path string) []Sample {
	file, err := os.Open(path)
	if err != nil {
		atomic.AddInt64(&d.stats.Errors, 1)
		return nil
	}
	defer file.Close()
	
	var samples []Sample
	decoder := json.NewDecoder(bufio.NewReaderSize(file, 64*1024))
	
	for {
		var sample Sample
		if err := decoder.Decode(&sample); err != nil {
			if err == io.EOF {
				break
			}
			atomic.AddInt64(&d.stats.Errors, 1)
			continue
		}
		samples = append(samples, sample)
	}
	
	return samples
}

// ════════════════════════════════════════════════════════════════════════════════
// STREAMING DATALOADER
// ════════════════════════════════════════════════════════════════════════════════

// StreamingDataLoader loads data in a streaming fashion without loading all into memory.
type StreamingDataLoader struct {
	config   Config
	sources  []string
	stats    Stats
	
	currentFile *os.File
	currentIdx  int
	reader      *bufio.Reader
	mu          sync.Mutex
}

// NewStreaming creates a new streaming dataloader.
func NewStreaming(config Config) *StreamingDataLoader {
	return &StreamingDataLoader{
		config:  config,
		sources: make([]string, 0),
	}
}

// AddSource adds a data source.
func (s *StreamingDataLoader) AddSource(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sources = append(s.sources, path)
}

// NextBatch returns the next batch of samples.
func (s *StreamingDataLoader) NextBatch() (*Batch, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	var samples []Sample
	
	for len(samples) < s.config.BatchSize {
		sample, err := s.nextSample()
		if err != nil {
			if len(samples) > 0 && !s.config.DropLast {
				break
			}
			return nil, err
		}
		samples = append(samples, *sample)
		atomic.AddInt64(&s.stats.SamplesLoaded, 1)
	}
	
	if len(samples) == 0 {
		return nil, io.EOF
	}
	
	atomic.AddInt64(&s.stats.BatchesProduced, 1)
	
	return &Batch{
		Samples: samples,
		Index:   int(s.stats.BatchesProduced),
	}, nil
}

func (s *StreamingDataLoader) nextSample() (*Sample, error) {
	for {
		// Open file if needed
		if s.currentFile == nil {
			if s.currentIdx >= len(s.sources) {
				return nil, io.EOF
			}
			
			file, err := os.Open(s.sources[s.currentIdx])
			if err != nil {
				s.currentIdx++
				atomic.AddInt64(&s.stats.Errors, 1)
				continue
			}
			
			s.currentFile = file
			s.reader = bufio.NewReaderSize(file, 64*1024)
		}
		
		// Read line
		line, err := s.reader.ReadBytes('\n')
		if err != nil {
			s.currentFile.Close()
			s.currentFile = nil
			s.currentIdx++
			
			if err == io.EOF {
				continue
			}
			return nil, err
		}
		
		// Parse JSON
		var sample Sample
		if err := json.Unmarshal(line, &sample); err != nil {
			atomic.AddInt64(&s.stats.Errors, 1)
			continue
		}
		
		return &sample, nil
	}
}

// Reset resets the dataloader to the beginning.
func (s *StreamingDataLoader) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.currentFile != nil {
		s.currentFile.Close()
		s.currentFile = nil
	}
	s.currentIdx = 0
}

// Close closes the dataloader.
func (s *StreamingDataLoader) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.currentFile != nil {
		return s.currentFile.Close()
	}
	return nil
}

// Stats returns current statistics.
func (s *StreamingDataLoader) Stats() Stats {
	return Stats{
		SamplesLoaded:   atomic.LoadInt64(&s.stats.SamplesLoaded),
		BatchesProduced: atomic.LoadInt64(&s.stats.BatchesProduced),
		LoadTimeMs:      atomic.LoadInt64(&s.stats.LoadTimeMs),
		Errors:          atomic.LoadInt64(&s.stats.Errors),
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// SAMPLER
// ════════════════════════════════════════════════════════════════════════════════

// Sampler defines the interface for sampling strategies.
type Sampler interface {
	Sample(n int) []int
	Reset()
}

// RandomSampler samples randomly with replacement.
type RandomSampler struct {
	size int
	rng  *rand.Rand
}

// NewRandomSampler creates a new random sampler.
func NewRandomSampler(size int, seed int64) *RandomSampler {
	return &RandomSampler{
		size: size,
		rng:  rand.New(rand.NewSource(seed)),
	}
}

// Sample returns n random indices.
func (s *RandomSampler) Sample(n int) []int {
	indices := make([]int, n)
	for i := range indices {
		indices[i] = s.rng.Intn(s.size)
	}
	return indices
}

// Reset resets the sampler (no-op for random).
func (s *RandomSampler) Reset() {}

// SequentialSampler samples sequentially.
type SequentialSampler struct {
	size    int
	current int
	mu      sync.Mutex
}

// NewSequentialSampler creates a new sequential sampler.
func NewSequentialSampler(size int) *SequentialSampler {
	return &SequentialSampler{size: size}
}

// Sample returns n sequential indices.
func (s *SequentialSampler) Sample(n int) []int {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	indices := make([]int, 0, n)
	for i := 0; i < n && s.current < s.size; i++ {
		indices = append(indices, s.current)
		s.current++
	}
	return indices
}

// Reset resets the sampler to the beginning.
func (s *SequentialSampler) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.current = 0
}

// WeightedSampler samples with weights.
type WeightedSampler struct {
	weights []float64
	cumsum  []float64
	rng     *rand.Rand
}

// NewWeightedSampler creates a new weighted sampler.
func NewWeightedSampler(weights []float64, seed int64) *WeightedSampler {
	// Compute cumulative sum
	cumsum := make([]float64, len(weights))
	sum := 0.0
	for i, w := range weights {
		sum += w
		cumsum[i] = sum
	}
	
	// Normalize
	for i := range cumsum {
		cumsum[i] /= sum
	}
	
	return &WeightedSampler{
		weights: weights,
		cumsum:  cumsum,
		rng:     rand.New(rand.NewSource(seed)),
	}
}

// Sample returns n weighted random indices.
func (s *WeightedSampler) Sample(n int) []int {
	indices := make([]int, n)
	for i := range indices {
		r := s.rng.Float64()
		for j, c := range s.cumsum {
			if r <= c {
				indices[i] = j
				break
			}
		}
	}
	return indices
}

// Reset resets the sampler (no-op for weighted).
func (s *WeightedSampler) Reset() {}












