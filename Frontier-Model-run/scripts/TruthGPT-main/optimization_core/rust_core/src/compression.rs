//! High-Performance Compression Module
//!
//! Provides ultra-fast compression/decompression using LZ4 and Zstd.
//! Optimized for tensor data and KV cache storage.
//!
//! ## Performance
//!
//! | Algorithm | Compress | Decompress | Ratio |
//! |-----------|----------|------------|-------|
//! | LZ4 | 5 GB/s | 8 GB/s | 2-3x |
//! | Zstd (level 3) | 400 MB/s | 1 GB/s | 3-5x |
//! | Zstd (level 9) | 100 MB/s | 1 GB/s | 4-6x |

use anyhow::{anyhow, Result};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Compression algorithm options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 - fastest compression (~5GB/s)
    LZ4,
    /// Zstd - balanced speed/ratio
    Zstd,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::LZ4
    }
}

impl CompressionAlgorithm {
    /// Get algorithm name
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LZ4 => "lz4",
            Self::Zstd => "zstd",
        }
    }

    /// Parse from string
    /// Note: This method name avoids conflict with std::str::FromStr::from_str
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "lz4" => Self::LZ4,
            "zstd" | "zstandard" => Self::Zstd,
            "none" | "" => Self::None,
            _ => Self::LZ4,
        }
    }
}

/// Compressor with configurable settings
pub struct Compressor {
    algorithm: CompressionAlgorithm,
    compression_level: i32,
}

impl Compressor {
    /// Create new compressor with specified algorithm
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self {
            algorithm,
            compression_level: 3, // Default level
        }
    }

    /// Create LZ4 compressor (fastest)
    pub fn lz4() -> Self {
        Self::new(CompressionAlgorithm::LZ4)
    }

    /// Create Zstd compressor (better ratio)
    pub fn zstd() -> Self {
        Self::new(CompressionAlgorithm::Zstd)
    }

    /// Set compression level (1-22 for zstd, ignored for lz4)
    pub fn with_level(mut self, level: i32) -> Self {
        self.compression_level = level.clamp(1, 22);
        self
    }

    /// Get the algorithm
    pub fn algorithm(&self) -> CompressionAlgorithm {
        self.algorithm
    }

    /// Get the compression level
    pub fn level(&self) -> i32 {
        self.compression_level
    }

    /// Compress data
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::Zstd => compress_zstd_level(data, self.compression_level),
            _ => compress(data, &self.algorithm),
        }
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        decompress(data, &self.algorithm)
    }

    /// Compress with statistics
    pub fn compress_with_stats(&self, data: &[u8]) -> Result<(Vec<u8>, CompressionStats)> {
        compress_with_stats(data, &self.algorithm)
    }
}

impl Default for Compressor {
    fn default() -> Self {
        Self::lz4()
    }
}

/// Compress data with specified algorithm
pub fn compress(data: &[u8], algorithm: &CompressionAlgorithm) -> Result<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::None => Ok(data.to_vec()),
        CompressionAlgorithm::LZ4 => compress_lz4(data),
        CompressionAlgorithm::Zstd => compress_zstd(data),
    }
}

/// Decompress data with specified algorithm
pub fn decompress(data: &[u8], algorithm: &CompressionAlgorithm) -> Result<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::None => Ok(data.to_vec()),
        CompressionAlgorithm::LZ4 => decompress_lz4(data),
        CompressionAlgorithm::Zstd => decompress_zstd(data),
    }
}

/// LZ4 compression - ~5GB/s throughput
fn compress_lz4(data: &[u8]) -> Result<Vec<u8>> {
    Ok(compress_prepend_size(data))
}

/// LZ4 decompression
fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>> {
    decompress_size_prepended(data).map_err(|e| anyhow!("LZ4 decompression failed: {}", e))
}

/// Zstd compression with default level (3)
fn compress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    zstd::encode_all(data, 3).map_err(|e| anyhow!("Zstd compression failed: {}", e))
}

/// Zstd compression with custom level
pub fn compress_zstd_level(data: &[u8], level: i32) -> Result<Vec<u8>> {
    let level = level.clamp(1, 22);
    zstd::encode_all(data, level).map_err(|e| anyhow!("Zstd compression failed: {}", e))
}

/// Zstd decompression
fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    zstd::decode_all(data).map_err(|e| anyhow!("Zstd decompression failed: {}", e))
}

/// Compression statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_time_us: u64,
    pub decompression_time_us: u64,
    pub algorithm: String,
}

impl CompressionStats {
    /// Calculate compression ratio (compressed/original)
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            self.compressed_size as f64 / self.original_size as f64
        }
    }

    /// Calculate space savings (1 - ratio)
    pub fn space_savings(&self) -> f64 {
        1.0 - self.compression_ratio()
    }

    /// Calculate space savings in bytes
    pub fn bytes_saved(&self) -> usize {
        self.original_size.saturating_sub(self.compressed_size)
    }

    /// Calculate compression throughput (MB/s)
    pub fn compression_throughput_mbps(&self) -> f64 {
        if self.compression_time_us == 0 {
            0.0
        } else {
            (self.original_size as f64 / 1_000_000.0) / (self.compression_time_us as f64 / 1_000_000.0)
        }
    }

    /// Calculate decompression throughput (MB/s)
    pub fn decompression_throughput_mbps(&self) -> f64 {
        if self.decompression_time_us == 0 {
            0.0
        } else {
            (self.original_size as f64 / 1_000_000.0) / (self.decompression_time_us as f64 / 1_000_000.0)
        }
    }
}

/// Compress with statistics
pub fn compress_with_stats(
    data: &[u8],
    algorithm: &CompressionAlgorithm,
) -> Result<(Vec<u8>, CompressionStats)> {
    let start = Instant::now();
    let compressed = compress(data, algorithm)?;
    let compression_time = start.elapsed();

    let stats = CompressionStats {
        original_size: data.len(),
        compressed_size: compressed.len(),
        compression_time_us: compression_time.as_micros() as u64,
        decompression_time_us: 0,
        algorithm: algorithm.name().to_string(),
    };

    Ok((compressed, stats))
}

/// Decompress with statistics
pub fn decompress_with_stats(
    data: &[u8],
    algorithm: &CompressionAlgorithm,
    original_size: usize,
) -> Result<(Vec<u8>, CompressionStats)> {
    let start = Instant::now();
    let decompressed = decompress(data, algorithm)?;
    let decompression_time = start.elapsed();

    let stats = CompressionStats {
        original_size,
        compressed_size: data.len(),
        compression_time_us: 0,
        decompression_time_us: decompression_time.as_micros() as u64,
        algorithm: algorithm.name().to_string(),
    };

    Ok((decompressed, stats))
}

/// Streaming compressor for large data
pub struct StreamingCompressor {
    algorithm: CompressionAlgorithm,
    buffer: Vec<u8>,
    chunk_size: usize,
    total_input: usize,
    total_output: usize,
}

impl StreamingCompressor {
    /// Create new streaming compressor
    pub fn new(algorithm: CompressionAlgorithm, chunk_size: usize) -> Self {
        Self {
            algorithm,
            buffer: Vec::with_capacity(chunk_size),
            chunk_size,
            total_input: 0,
            total_output: 0,
        }
    }

    /// Add data to buffer and compress full chunks
    pub fn write(&mut self, data: &[u8]) -> Result<Vec<Vec<u8>>> {
        self.buffer.extend_from_slice(data);
        self.total_input += data.len();
        
        let mut chunks = Vec::new();

        while self.buffer.len() >= self.chunk_size {
            let chunk: Vec<u8> = self.buffer.drain(..self.chunk_size).collect();
            let compressed = compress(&chunk, &self.algorithm)?;
            self.total_output += compressed.len();
            chunks.push(compressed);
        }

        Ok(chunks)
    }

    /// Flush remaining data
    pub fn flush(&mut self) -> Result<Option<Vec<u8>>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let data = std::mem::take(&mut self.buffer);
        let compressed = compress(&data, &self.algorithm)?;
        self.total_output += compressed.len();
        Ok(Some(compressed))
    }

    /// Get total input bytes processed
    pub fn total_input(&self) -> usize {
        self.total_input
    }

    /// Get total output bytes produced
    pub fn total_output(&self) -> usize {
        self.total_output
    }

    /// Get overall compression ratio
    pub fn overall_ratio(&self) -> f64 {
        if self.total_input == 0 {
            0.0
        } else {
            self.total_output as f64 / self.total_input as f64
        }
    }
}

/// Batch compression utilities
pub struct BatchCompressor {
    algorithm: CompressionAlgorithm,
    level: i32,
}

impl BatchCompressor {
    pub fn new(algorithm: CompressionAlgorithm, level: i32) -> Self {
        Self { algorithm, level }
    }

    /// Compress multiple items in parallel
    pub fn compress_batch(&self, items: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;

        items
            .par_iter()
            .map(|data| match self.algorithm {
                CompressionAlgorithm::Zstd => compress_zstd_level(data, self.level),
                _ => compress(data, &self.algorithm),
            })
            .collect()
    }

    /// Decompress multiple items in parallel
    pub fn decompress_batch(&self, items: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;

        items
            .par_iter()
            .map(|data| decompress(data, &self.algorithm))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz4_roundtrip() {
        let original = b"Hello, World! This is a test of the LZ4 compression.".to_vec();
        let compressed = compress(&original, &CompressionAlgorithm::LZ4).unwrap();
        let decompressed = decompress(&compressed, &CompressionAlgorithm::LZ4).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_zstd_roundtrip() {
        let original = b"Hello, World! This is a test of the Zstd compression.".to_vec();
        let compressed = compress(&original, &CompressionAlgorithm::Zstd).unwrap();
        let decompressed = decompress(&compressed, &CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_no_compression() {
        let original = b"Hello, World!".to_vec();
        let result = compress(&original, &CompressionAlgorithm::None).unwrap();
        assert_eq!(original, result);
    }

    #[test]
    fn test_compression_ratio() {
        // Create repetitive data (compresses well)
        let original: Vec<u8> = (0..1000).map(|i| (i % 10) as u8).collect();
        let (compressed, stats) = compress_with_stats(&original, &CompressionAlgorithm::LZ4).unwrap();

        assert!(compressed.len() < original.len());
        assert!(stats.compression_ratio() < 1.0);
        assert!(stats.space_savings() > 0.0);
    }

    #[test]
    fn test_streaming_compressor() {
        let mut compressor = StreamingCompressor::new(CompressionAlgorithm::LZ4, 100);

        // Write data in chunks
        let _ = compressor.write(&[1u8; 50]).unwrap();
        let chunks = compressor.write(&[2u8; 60]).unwrap();
        assert_eq!(chunks.len(), 1); // Should produce one chunk

        let final_chunk = compressor.flush().unwrap();
        assert!(final_chunk.is_some());
    }

    #[test]
    fn test_compressor_builder() {
        let compressor = Compressor::zstd().with_level(9);
        assert_eq!(compressor.algorithm(), CompressionAlgorithm::Zstd);
        assert_eq!(compressor.level(), 9);
    }

    #[test]
    fn test_zstd_levels() {
        let data = b"Hello, World! This is a test.".repeat(100).to_vec();
        
        let compressed_low = compress_zstd_level(&data, 1).unwrap();
        let compressed_high = compress_zstd_level(&data, 19).unwrap();
        
        // Higher level should give better compression
        assert!(compressed_high.len() <= compressed_low.len());
        
        // Both should decompress correctly
        let decompressed = decompress(&compressed_high, &CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(data, decompressed);
    }
}
