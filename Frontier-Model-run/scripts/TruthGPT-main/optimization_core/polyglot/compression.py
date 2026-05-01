"""
Unified Compression Interface

Provides Python interface to Rust and Go compression backends.
"""
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from truthgpt_rust import PyCompressor
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class CompressionAlgorithm:
    LZ4 = "lz4"
    ZSTD = "zstd"
    NONE = "none"

class Compressor:
    """
    Unified compression interface.
    
    Automatically selects best available backend:
    1. Rust (fastest, most features)
    2. Python fallback (zlib/gzip)
    """
    
    def __init__(
        self,
        algorithm: str = CompressionAlgorithm.LZ4,
        level: int = 3,
        backend: Optional[str] = None,
    ):
        self.algorithm = algorithm
        self.level = level
        
        if backend is None:
            backend = "rust" if RUST_AVAILABLE else "python"
        
        self.backend = backend
        self._compressor = None
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup backend implementation."""
        if self.backend == "rust" and RUST_AVAILABLE:
            try:
                self._compressor = PyCompressor(
                    algorithm=self.algorithm,
                    level=self.level,
                )
                logger.info(f"Using Rust compression backend ({self.algorithm})")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Rust compressor: {e}")
        
        self._compressor = None
        logger.info(f"Using Python compression backend ({self.algorithm})")
    
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.backend == "rust" and self._compressor:
            try:
                return bytes(self._compressor.compress(list(data)))
            except Exception as e:
                logger.warning(f"Rust compression failed: {e}, falling back")
        
        return self._compress_python(data)
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        if self.backend == "rust" and self._compressor:
            try:
                return bytes(self._compressor.decompress(list(data)))
            except Exception as e:
                logger.warning(f"Rust decompression failed: {e}, falling back")
        
        return self._decompress_python(data)
    
    def compress_with_stats(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Compress with statistics."""
        if self.backend == "rust" and self._compressor:
            try:
                compressed, stats = self._compressor.compress_with_stats(list(data))
                return bytes(compressed), stats
            except Exception as e:
                logger.warning(f"Rust compression failed: {e}, falling back")
        
        import time
        start = time.perf_counter()
        compressed = self._compress_python(data)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        
        stats = {
            "original_size": len(data),
            "compressed_size": len(compressed),
            "ratio": len(compressed) / len(data) if len(data) > 0 else 0.0,
            "savings": 1.0 - (len(compressed) / len(data)) if len(data) > 0 else 0.0,
            "time_us": elapsed_us,
        }
        
        return compressed, stats
    
    def _compress_python(self, data: bytes) -> bytes:
        """Python fallback compression."""
        if self.algorithm == CompressionAlgorithm.LZ4:
            try:
                import lz4.frame
                return lz4.frame.compress(data, compression_level=self.level)
            except ImportError:
                logger.warning("lz4 not available, using zlib")
                import zlib
                return zlib.compress(data, level=self.level)
        
        elif self.algorithm == CompressionAlgorithm.ZSTD:
            try:
                import zstandard as zstd
                cctx = zstd.ZstdCompressor(level=self.level)
                return cctx.compress(data)
            except ImportError:
                logger.warning("zstandard not available, using zlib")
                import zlib
                return zlib.compress(data, level=self.level)
        
        else:
            return data
    
    def _decompress_python(self, data: bytes) -> bytes:
        """Python fallback decompression."""
        if self.algorithm == CompressionAlgorithm.LZ4:
            try:
                import lz4.frame
                return lz4.frame.decompress(data)
            except ImportError:
                import zlib
                return zlib.decompress(data)
        
        elif self.algorithm == CompressionAlgorithm.ZSTD:
            try:
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(data)
            except ImportError:
                import zlib
                return zlib.decompress(data)
        
        else:
            return data

def create_compressor(
    algorithm: str = CompressionAlgorithm.LZ4,
    level: int = 3
) -> Compressor:
    """Factory function to create compressor."""
    return Compressor(algorithm=algorithm, level=level)

__all__ = ["Compressor", "CompressionAlgorithm", "create_compressor"]













