// Package compression provides high-performance compression for TruthGPT.
//
// Supported algorithms:
// - LZ4: Ultra-fast compression (~5GB/s)
// - Zstd: High ratio compression with adjustable levels
package compression

import (
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/pierrec/lz4/v4"
)

// ════════════════════════════════════════════════════════════════════════════════
// ALGORITHM TYPES
// ════════════════════════════════════════════════════════════════════════════════

// Algorithm represents a compression algorithm.
type Algorithm int

const (
	// None - no compression
	None Algorithm = iota
	// LZ4 - ultra-fast compression (~5GB/s)
	LZ4
	// Zstd - high ratio compression
	Zstd
)

func (a Algorithm) String() string {
	switch a {
	case LZ4:
		return "lz4"
	case Zstd:
		return "zstd"
	default:
		return "none"
	}
}

// ParseAlgorithm parses a string into an Algorithm.
func ParseAlgorithm(s string) Algorithm {
	switch s {
	case "lz4":
		return LZ4
	case "zstd":
		return Zstd
	default:
		return None
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// COMPRESSION STATS
// ════════════════════════════════════════════════════════════════════════════════

// Stats holds compression statistics.
type Stats struct {
	OriginalSize   int
	CompressedSize int
	Algorithm      Algorithm
	Duration       time.Duration
}

// Ratio returns the compression ratio.
func (s Stats) Ratio() float64 {
	if s.OriginalSize == 0 {
		return 1.0
	}
	return float64(s.CompressedSize) / float64(s.OriginalSize)
}

// Savings returns the space savings percentage.
func (s Stats) Savings() float64 {
	return (1.0 - s.Ratio()) * 100
}

// Throughput returns the compression throughput in bytes/second.
func (s Stats) Throughput() float64 {
	if s.Duration == 0 {
		return 0
	}
	return float64(s.OriginalSize) / s.Duration.Seconds()
}

// ════════════════════════════════════════════════════════════════════════════════
// COMPRESSOR
// ════════════════════════════════════════════════════════════════════════════════

// Compressor provides compression/decompression operations.
type Compressor struct {
	algorithm   Algorithm
	level       int
	
	// Encoder/decoder pools for thread safety
	zstdEncoderPool sync.Pool
	zstdDecoderPool sync.Pool
}

// NewCompressor creates a new Compressor.
func NewCompressor(algorithm Algorithm, level int) *Compressor {
	c := &Compressor{
		algorithm: algorithm,
		level:     level,
	}
	
	// Initialize Zstd encoder pool
	c.zstdEncoderPool = sync.Pool{
		New: func() interface{} {
			enc, _ := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(level)))
			return enc
		},
	}
	
	// Initialize Zstd decoder pool
	c.zstdDecoderPool = sync.Pool{
		New: func() interface{} {
			dec, _ := zstd.NewReader(nil)
			return dec
		},
	}
	
	return c
}

// DefaultCompressor returns a default LZ4 compressor.
func DefaultCompressor() *Compressor {
	return NewCompressor(LZ4, 1)
}

// ════════════════════════════════════════════════════════════════════════════════
// COMPRESS OPERATIONS
// ════════════════════════════════════════════════════════════════════════════════

// Compress compresses the input data.
func (c *Compressor) Compress(data []byte) ([]byte, error) {
	switch c.algorithm {
	case LZ4:
		return c.compressLZ4(data)
	case Zstd:
		return c.compressZstd(data)
	default:
		return data, nil
	}
}

// CompressWithStats compresses the input data and returns statistics.
func (c *Compressor) CompressWithStats(data []byte) ([]byte, Stats, error) {
	start := time.Now()
	
	compressed, err := c.Compress(data)
	if err != nil {
		return nil, Stats{}, err
	}
	
	stats := Stats{
		OriginalSize:   len(data),
		CompressedSize: len(compressed),
		Algorithm:      c.algorithm,
		Duration:       time.Since(start),
	}
	
	return compressed, stats, nil
}

// compressLZ4 compresses data using LZ4.
func (c *Compressor) compressLZ4(data []byte) ([]byte, error) {
	// Pre-allocate buffer for worst case
	buf := make([]byte, lz4.CompressBlockBound(len(data)))
	
	n, err := lz4.CompressBlock(data, buf, nil)
	if err != nil {
		return nil, fmt.Errorf("LZ4 compress error: %w", err)
	}
	
	// If incompressible, return original with marker
	if n == 0 {
		result := make([]byte, len(data)+1)
		result[0] = 0 // Marker: uncompressed
		copy(result[1:], data)
		return result, nil
	}
	
	// Return compressed with marker
	result := make([]byte, n+1)
	result[0] = 1 // Marker: compressed
	copy(result[1:], buf[:n])
	return result, nil
}

// compressZstd compresses data using Zstd.
func (c *Compressor) compressZstd(data []byte) ([]byte, error) {
	enc := c.zstdEncoderPool.Get().(*zstd.Encoder)
	defer c.zstdEncoderPool.Put(enc)
	
	return enc.EncodeAll(data, nil), nil
}

// ════════════════════════════════════════════════════════════════════════════════
// DECOMPRESS OPERATIONS
// ════════════════════════════════════════════════════════════════════════════════

// Decompress decompresses the input data.
func (c *Compressor) Decompress(data []byte) ([]byte, error) {
	switch c.algorithm {
	case LZ4:
		return c.decompressLZ4(data)
	case Zstd:
		return c.decompressZstd(data)
	default:
		return data, nil
	}
}

// decompressLZ4 decompresses LZ4 data.
func (c *Compressor) decompressLZ4(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data")
	}
	
	// Check marker
	if data[0] == 0 {
		// Uncompressed
		return data[1:], nil
	}
	
	// Compressed - try progressively larger buffers
	compressed := data[1:]
	for size := len(compressed) * 2; size <= len(compressed)*100; size *= 2 {
		buf := make([]byte, size)
		n, err := lz4.UncompressBlock(compressed, buf)
		if err == nil {
			return buf[:n], nil
		}
	}
	
	return nil, fmt.Errorf("LZ4 decompress failed: buffer too small")
}

// decompressZstd decompresses Zstd data.
func (c *Compressor) decompressZstd(data []byte) ([]byte, error) {
	dec := c.zstdDecoderPool.Get().(*zstd.Decoder)
	defer c.zstdDecoderPool.Put(dec)
	
	return dec.DecodeAll(data, nil)
}

// ════════════════════════════════════════════════════════════════════════════════
// STREAMING COMPRESSION
// ════════════════════════════════════════════════════════════════════════════════

// StreamingCompressor provides streaming compression.
type StreamingCompressor struct {
	algorithm Algorithm
	level     int
}

// NewStreamingCompressor creates a new streaming compressor.
func NewStreamingCompressor(algorithm Algorithm, level int) *StreamingCompressor {
	return &StreamingCompressor{
		algorithm: algorithm,
		level:     level,
	}
}

// NewCompressWriter creates a new compression writer.
func (s *StreamingCompressor) NewCompressWriter(w io.Writer) (io.WriteCloser, error) {
	switch s.algorithm {
	case LZ4:
		return lz4.NewWriter(w), nil
	case Zstd:
		return zstd.NewWriter(w, zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(s.level)))
	default:
		return &nopWriteCloser{w}, nil
	}
}

// NewDecompressReader creates a new decompression reader.
func (s *StreamingCompressor) NewDecompressReader(r io.Reader) (io.Reader, error) {
	switch s.algorithm {
	case LZ4:
		return lz4.NewReader(r), nil
	case Zstd:
		return zstd.NewReader(r)
	default:
		return r, nil
	}
}

// nopWriteCloser wraps a writer with a no-op Close.
type nopWriteCloser struct {
	io.Writer
}

func (n *nopWriteCloser) Close() error {
	return nil
}

// ════════════════════════════════════════════════════════════════════════════════
// CONVENIENCE FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════════

// CompressLZ4 compresses data using LZ4.
func CompressLZ4(data []byte) ([]byte, error) {
	return DefaultCompressor().compressLZ4(data)
}

// DecompressLZ4 decompresses LZ4 data.
func DecompressLZ4(data []byte) ([]byte, error) {
	return DefaultCompressor().decompressLZ4(data)
}

// CompressZstd compresses data using Zstd with default level.
func CompressZstd(data []byte, level int) ([]byte, error) {
	c := NewCompressor(Zstd, level)
	return c.compressZstd(data)
}

// DecompressZstd decompresses Zstd data.
func DecompressZstd(data []byte) ([]byte, error) {
	c := NewCompressor(Zstd, 3)
	return c.decompressZstd(data)
}












