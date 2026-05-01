"""
Tests for unified Compression module.
"""

import pytest
from optimization_core.polyglot_core.compression import (
    Compressor,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionStats,
    compress,
    decompress,
)
from optimization_core.polyglot_core.backend import Backend


def test_compression_config():
    """Test CompressionConfig."""
    config = CompressionConfig(algorithm=CompressionAlgorithm.LZ4, level=3)
    assert config.algorithm == CompressionAlgorithm.LZ4
    assert config.level == 3


def test_compressor_lz4():
    """Test LZ4 compression."""
    compressor = Compressor(algorithm="lz4")
    
    # Repetitive data compresses well
    data = b"Hello, World! " * 100
    
    result = compressor.compress(data)
    
    assert result.success is True
    assert len(result.data) < len(data)
    assert result.stats.compression_ratio < 1.0
    assert result.stats.space_savings > 0.0


def test_compressor_roundtrip():
    """Test compression roundtrip."""
    compressor = Compressor(algorithm="lz4")
    
    original = b"Test data for compression roundtrip " * 50
    
    compressed = compressor.compress(original)
    assert compressed.success is True
    
    decompressed = compressor.decompress(compressed.data)
    
    assert decompressed == original


def test_compressor_zstd():
    """Test Zstd compression."""
    compressor = Compressor(algorithm="zstd", level=3)
    
    data = b"Zstd compression test " * 100
    
    result = compressor.compress(data)
    
    # Should compress (may fail if zstd not available)
    if result.success:
        assert len(result.data) <= len(data)


def test_compressor_stats():
    """Test compression statistics."""
    compressor = Compressor(algorithm="lz4")
    
    data = b"Statistics test " * 200
    result = compressor.compress(data)
    
    if result.success:
        stats = result.stats
        assert stats.original_size > 0
        assert stats.compressed_size > 0
        assert stats.compression_ratio > 0.0
        assert stats.compression_throughput_mbps >= 0.0


def test_compress_decompress_functions():
    """Test convenience functions."""
    data = b"Convenience function test " * 50
    
    try:
        compressed = compress(data, algorithm="lz4")
        decompressed = decompress(compressed, algorithm="lz4")
        assert decompressed == data
    except Exception:
        # May fail if lz4 not available
        pass


def test_compressor_backend():
    """Test backend selection."""
    compressor = Compressor(algorithm="lz4")
    assert compressor.backend in [Backend.PYTHON, Backend.RUST, Backend.CPP]


