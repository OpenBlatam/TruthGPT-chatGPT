"""
Unified KV Cache Interface

Provides Python interface to Rust, Go, and C++ KV cache implementations.
"""
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from truthgpt_rust import PyKVCache
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

try:
    import _cpp_core as cpp_core
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

class KVCache:
    """
    Unified KV cache interface.
    
    Automatically selects best available backend:
    1. Rust (fastest, most features)
    2. C++ (GPU-optimized)
    3. Python fallback (dict-based)
    """
    
    def __init__(
        self,
        max_size: int = 8192,
        eviction_strategy: str = "adaptive",
        enable_compression: bool = True,
        backend: Optional[str] = None,
    ):
        self.max_size = max_size
        self.eviction_strategy = eviction_strategy
        self.enable_compression = enable_compression
        
        if backend is None:
            if RUST_AVAILABLE:
                backend = "rust"
            elif CPP_AVAILABLE:
                backend = "cpp"
            else:
                backend = "python"
        
        self.backend = backend
        self._cache = None
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup backend implementation."""
        if self.backend == "rust" and RUST_AVAILABLE:
            try:
                self._cache = PyKVCache(
                    max_size=self.max_size,
                    eviction_strategy=self.eviction_strategy,
                    enable_compression=self.enable_compression,
                )
                logger.info("Using Rust KV cache backend")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Rust cache: {e}")
        
        if self.backend == "cpp" and CPP_AVAILABLE:
            try:
                config = cpp_core.memory.KVCacheConfig()
                config.max_cache_size = self.max_size * 1024 * 1024
                config.eviction_strategy = getattr(
                    cpp_core.memory.EvictionStrategy,
                    self.eviction_strategy.upper(),
                    cpp_core.memory.EvictionStrategy.S3FIFO
                )
                self._cache = cpp_core.memory.UltraKVCache(config)
                logger.info("Using C++ KV cache backend")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize C++ cache: {e}")
        
        self._cache = {}
        self._access_order = []
        logger.info("Using Python fallback KV cache")
    
    def get(self, layer_idx: int, position: int, key: str = "") -> Optional[bytes]:
        """Get cached value."""
        if self.backend == "rust":
            return self._cache.get(layer_idx, position)
        elif self.backend == "cpp":
            result = self._cache.get(layer_idx, position, key)
            if result:
                import numpy as np
                k = result.get("key")
                v = result.get("value")
                return (k.tobytes(), v.tobytes())
            return None
        else:
            cache_key = (layer_idx, position, key)
            if cache_key in self._cache:
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                return self._cache[cache_key]
            return None
    
    def put(
        self,
        layer_idx: int,
        position: int,
        data: bytes,
        key: str = "",
        priority: float = 1.0,
    ):
        """Store value in cache."""
        if self.backend == "rust":
            self._cache.put(layer_idx, position, list(data))
        elif self.backend == "cpp":
            import numpy as np
            if isinstance(data, tuple):
                k_data, v_data = data
                k = np.frombuffer(k_data, dtype=np.float32)
                v = np.frombuffer(v_data, dtype=np.float32)
                self._cache.put(layer_idx, position, k, v, key, priority)
            else:
                arr = np.frombuffer(data, dtype=np.float32)
                self._cache.put(layer_idx, position, arr, arr, key, priority)
        else:
            cache_key = (layer_idx, position, key)
            if cache_key in self._cache:
                self._access_order.remove(cache_key)
            elif len(self._cache) >= self.max_size:
                self._evict()
            self._cache[cache_key] = data
            self._access_order.append(cache_key)
    
    def _evict(self):
        """Evict entry using LRU."""
        if self._access_order:
            key_to_remove = self._access_order.pop(0)
            del self._cache[key_to_remove]
    
    def clear(self):
        """Clear all cached data."""
        if self.backend == "rust":
            self._cache.clear()
        elif self.backend == "cpp":
            self._cache.clear()
        else:
            self._cache.clear()
            self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.backend == "rust":
            return self._cache.stats()
        elif self.backend == "cpp":
            stats = self._cache.stats()
            return {
                "hit_count": stats.hit_count,
                "miss_count": stats.miss_count,
                "hit_rate": stats.hit_rate(),
                "current_size": stats.current_memory_bytes,
            }
        else:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": 0.0,
            }
    
    def size(self) -> int:
        """Get current cache size."""
        if self.backend == "rust":
            return self._cache.size()
        elif self.backend == "cpp":
            return self._cache.size()
        else:
            return len(self._cache)













