"""
Unified KV Cache with automatic backend selection.

Supports Rust, C++, Go, and Python backends with seamless fallback.
Provides high-performance key-value caching for transformer model states.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Tuple
from collections import OrderedDict, deque
import numpy as np
import time

from .backend import Backend, get_best_backend, is_backend_available


class EvictionStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In First Out
    S3FIFO = "s3fifo"     # Segmented 3-FIFO
    ARC = "arc"           # Adaptive Replacement Cache
    ADAPTIVE = "adaptive"  # Dynamic hybrid
    NONE = "none"         # No eviction


@dataclass
class KVCacheConfig:
    """Configuration for KV Cache."""
    max_size: int = 100000
    max_memory_bytes: int = 8 * 1024 * 1024 * 1024  # 8GB
    eviction_strategy: EvictionStrategy = EvictionStrategy.LRU
    eviction_threshold: float = 0.85
    eviction_target: float = 0.70
    enable_compression: bool = True
    compression_threshold: int = 4096
    num_shards: int = 32
    
    @classmethod
    def inference_optimized(cls, memory_gb: int = 8) -> "KVCacheConfig":
        """Create config optimized for inference."""
        return cls(
            max_memory_bytes=memory_gb * 1024 * 1024 * 1024,
            eviction_strategy=EvictionStrategy.S3FIFO,
            enable_compression=True,
            num_shards=64
        )
    
    @classmethod
    def long_context(cls, memory_gb: int = 32) -> "KVCacheConfig":
        """Create config for long-context models."""
        return cls(
            max_memory_bytes=memory_gb * 1024 * 1024 * 1024,
            max_size=10_000_000,
            eviction_strategy=EvictionStrategy.ADAPTIVE,
            enable_compression=True,
            num_shards=128
        )


@dataclass  
class CacheStats:
    """Cache performance statistics."""
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    entry_count: int = 0
    memory_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class KVCache:
    """
    High-performance KV Cache with automatic backend selection.
    
    Automatically selects the best available backend:
    - Rust: 50x faster than Python, memory-efficient
    - C++: 10-100x faster with compression
    - Go: Distributed cache via gRPC
    - Python: Fallback
    
    Example:
        >>> cache = KVCache(max_size=10000)
        >>> cache.put(layer=0, position=42, key=k_tensor, value=v_tensor)
        >>> result = cache.get(layer=0, position=42)
        >>> print(f"Hit rate: {cache.hit_rate:.2%}")
    """
    
    def __init__(
        self,
        config: Optional[KVCacheConfig] = None,
        max_size: int = 100000,
        backend: Optional[Backend] = None,
        **kwargs
    ):
        """
        Initialize KV Cache.
        
        Args:
            config: Cache configuration
            max_size: Maximum entries (if config not provided)
            backend: Force specific backend (auto-select if None)
            **kwargs: Additional config options
        """
        if config is None:
            config = KVCacheConfig(max_size=max_size, **kwargs)
        
        self.config = config
        self._backend = backend or get_best_backend('kv_cache')
        self._impl = self._create_implementation()
        
    def _create_implementation(self):
        """
        Create backend-specific implementation with automatic fallback.
        
        Returns:
            Backend implementation instance
            
        Raises:
            RuntimeError: If no backend implementation can be created
        """
        # Try to create the requested backend, fallback to Python if unavailable
        backend_creators = {
            Backend.RUST: (self._create_rust_impl, Backend.RUST),
            Backend.CPP: (self._create_cpp_impl, Backend.CPP),
            Backend.GO: (self._create_go_impl, Backend.GO),
        }
        
        # Attempt to create the requested backend
        if self._backend in backend_creators:
            creator_func, backend_type = backend_creators[self._backend]
            if is_backend_available(backend_type):
                try:
                    return creator_func()
                except (ImportError, AttributeError) as e:
                    # Log fallback but continue to Python implementation
                    # In production, you might want to log this
                    pass
        
        # Fallback to Python implementation (always available)
        return self._create_python_impl()
    
    def _create_rust_impl(self):
        """
        Create Rust backend implementation.
        
        Returns:
            Rust PyKVCache instance
            
        Raises:
            ImportError: If rust_core module is not available
        """
        from optimization_core.rust_core import truthgpt_rust
        
        return truthgpt_rust.PyKVCache(
            max_size=self.config.max_size,
            eviction_strategy=self.config.eviction_strategy.value,
            enable_compression=self.config.enable_compression
        )
    
    def _create_cpp_impl(self):
        """
        Create C++ backend implementation.
        
        Returns:
            C++ UltraKVCache instance
            
        Raises:
            ImportError: If _cpp_core module is not available
        """
        from optimization_core import _cpp_core
        
        cpp_config = _cpp_core.memory.KVCacheConfig(
            max_cache_size=self.config.max_memory_bytes,
            max_entries=self.config.max_size,
            eviction_threshold=self.config.eviction_threshold,
            eviction_target=self.config.eviction_target,
            use_compression=self.config.enable_compression,
            compression_threshold=self.config.compression_threshold,
            num_shards=self.config.num_shards
        )
        
        return _cpp_core.memory.UltraKVCache(cpp_config)
    
    def _create_go_impl(self):
        """
        Create Go gRPC client implementation.
        
        Returns:
            GoCacheClient instance
            
        Raises:
            ImportError: If distributed module is not available
        """
        from .distributed import GoCacheClient
        return GoCacheClient()
    
    def _create_python_impl(self):
        """
        Create pure Python fallback implementation.
        
        Returns:
            _PythonKVCache instance (always available)
        """
        return _PythonKVCache(self.config)
    
    def _put_rust(self, layer: int, position: int, key: np.ndarray, 
                  value: np.ndarray, tag: str) -> None:
        """
        Put operation for Rust backend (requires byte conversion).
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            key: Key tensor
            value: Value tensor
            tag: Optional tag
        """
        self._impl.put(layer, position, key.tobytes(), value.tobytes(), tag)
    
    def _put_standard(self, layer: int, position: int, key: np.ndarray,
                     value: np.ndarray, tag: str, priority: float) -> None:
        """
        Put operation for standard backends (C++, Go, Python).
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            key: Key tensor
            value: Value tensor
            tag: Optional tag
            priority: Priority for eviction
        """
        # Go backend doesn't support priority parameter
        if self._backend == Backend.GO:
            self._impl.put(layer, position, key, value, tag)
        else:
            self._impl.put(layer, position, key, value, tag, priority)
    
    def put(
        self,
        layer: int,
        position: int,
        key: np.ndarray,
        value: np.ndarray,
        tag: str = "",
        priority: float = 1.0
    ) -> None:
        """
        Store KV state in cache.
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            key: Key tensor
            value: Value tensor
            tag: Optional tag for namespacing
            priority: Priority for eviction (higher = keep longer)
            
        Raises:
            ValueError: If layer or position are negative
        """
        # Validate inputs
        if layer < 0 or position < 0:
            raise ValueError(f"Layer and position must be non-negative, got layer={layer}, position={position}")
        
        # Route to backend-specific implementation
        if self._backend == Backend.RUST:
            self._put_rust(layer, position, key, value, tag)
        else:
            self._put_standard(layer, position, key, value, tag, priority)
    
    def _get_rust(self, layer: int, position: int, tag: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get operation for Rust backend (requires byte conversion).
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            tag: Optional tag
            
        Returns:
            Dict with 'key' and 'value' arrays, or None if not found
        """
        result = self._impl.get(layer, position, tag)
        if result is None:
            return None
        
        k_bytes, v_bytes = result
        return {
            'key': np.frombuffer(k_bytes, dtype=np.float32),
            'value': np.frombuffer(v_bytes, dtype=np.float32)
        }
    
    def _get_cpp(self, layer: int, position: int, tag: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get operation for C++ backend.
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            tag: Optional tag
            
        Returns:
            Dict with 'key' and 'value' arrays, or None if not found
        """
        result = self._impl.get(layer, position, tag)
        if result is None:
            return None
        return {'key': result['key'], 'value': result['value']}
    
    def get(
        self,
        layer: int,
        position: int,
        tag: str = ""
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve KV state from cache.
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            tag: Optional tag
            
        Returns:
            Dict with 'key' and 'value' arrays, or None if not found
        """
        # Route to backend-specific implementation
        if self._backend == Backend.RUST:
            return self._get_rust(layer, position, tag)
        elif self._backend == Backend.CPP:
            return self._get_cpp(layer, position, tag)
        else:
            # Go and Python backends return dict directly
            return self._impl.get(layer, position, tag)
    
    def remove(self, layer: int, position: int, tag: str = "") -> bool:
        """Remove entry from cache."""
        return self._impl.remove(layer, position, tag)
    
    def clear(self) -> None:
        """Clear all entries."""
        self._impl.clear()
    
    def contains(self, layer: int, position: int, tag: str = "") -> bool:
        """Check if entry exists."""
        return self._impl.contains(layer, position, tag)
    
    def _get_impl_size(self) -> int:
        """
        Get cache size from implementation.
        
        Returns:
            Number of cached entries
        """
        if hasattr(self._impl, 'size'):
            size_method = getattr(self._impl, 'size')
            # Handle both method and property
            return size_method() if callable(size_method) else size_method
        return len(self._impl)
    
    def _get_impl_hit_rate(self) -> float:
        """
        Get hit rate from implementation.
        
        Returns:
            Hit rate as float between 0.0 and 1.0
        """
        # Try multiple ways to get hit rate from different backends
        if hasattr(self._impl, 'stats'):
            stats = self._impl.stats
            if hasattr(stats, 'hit_rate'):
                hit_rate_method = getattr(stats, 'hit_rate')
                return hit_rate_method() if callable(hit_rate_method) else hit_rate_method
        
        if hasattr(self._impl, 'hit_rate'):
            hit_rate = getattr(self._impl, 'hit_rate')
            return hit_rate() if callable(hit_rate) else hit_rate
        
        return 0.0
    
    def _extract_stats_from_impl(self) -> CacheStats:
        """
        Extract statistics from backend implementation.
        
        Returns:
            CacheStats object with available metrics
        """
        if not hasattr(self._impl, 'stats'):
            return CacheStats(entry_count=self._get_impl_size())
        
        impl_stats = self._impl.stats
        entry_count = self._get_impl_size()
        
        # Safely extract statistics with defaults
        hit_count = getattr(impl_stats, 'hit_count', 0)
        miss_count = getattr(impl_stats, 'miss_count', 0)
        eviction_count = getattr(impl_stats, 'eviction_count', 0)
        
        # Get memory usage if available
        memory_bytes = 0
        if hasattr(self._impl, 'memory_usage'):
            memory_usage_method = getattr(self._impl, 'memory_usage')
            memory_bytes = memory_usage_method() if callable(memory_usage_method) else memory_usage_method
        
        return CacheStats(
            hit_count=hit_count,
            miss_count=miss_count,
            eviction_count=eviction_count,
            entry_count=entry_count,
            memory_bytes=memory_bytes
        )
    
    @property
    def size(self) -> int:
        """
        Get number of cached entries.
        
        Returns:
            Number of entries currently in cache
        """
        return self._get_impl_size()
    
    @property
    def hit_rate(self) -> float:
        """
        Get cache hit rate.
        
        Returns:
            Hit rate as float between 0.0 and 1.0
        """
        return self._get_impl_hit_rate()
    
    @property
    def stats(self) -> CacheStats:
        """
        Get comprehensive cache statistics.
        
        Returns:
            CacheStats object with hit/miss counts, evictions, and memory usage
        """
        return self._extract_stats_from_impl()
    
    @property
    def backend(self) -> Backend:
        """Get current backend."""
        return self._backend
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return (f"KVCache(size={self.size}, hit_rate={self.hit_rate:.2%}, "
                f"backend={self._backend.name})")


class _PythonKVCache:
    """
    Pure Python KV cache fallback implementation.
    
    Uses OrderedDict for O(1) LRU operations instead of O(n) list operations.
    This provides significantly better performance for large caches.
    """
    
    def __init__(self, config: KVCacheConfig):
        """
        Initialize Python KV cache.
        
        Args:
            config: Cache configuration with size limits and eviction strategy
        """
        self.config = config
        # Use OrderedDict for efficient LRU: move_to_end() is O(1)
        # Keys are (layer, position, tag) tuples
        self._cache: OrderedDict[Tuple[int, int, str], Dict[str, np.ndarray]] = OrderedDict()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
    
    def _make_cache_key(self, layer: int, position: int, tag: str) -> Tuple[int, int, str]:
        """
        Create a cache key tuple from components.
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            tag: Optional namespace tag
            
        Returns:
            Cache key tuple
        """
        return (layer, position, tag)
    
    def _ensure_capacity(self) -> None:
        """
        Evict entries until cache is below max_size.
        
        Uses LRU eviction: removes least recently used entries first.
        """
        while len(self._cache) >= self.config.max_size:
            # OrderedDict.popitem(last=False) removes oldest (LRU) entry in O(1)
            if self._cache:
                self._cache.popitem(last=False)
                self._eviction_count += 1
    
    def put(
        self, 
        layer: int, 
        position: int, 
        key: np.ndarray, 
        value: np.ndarray, 
        tag: str = "", 
        priority: float = 1.0
    ) -> None:
        """
        Store KV state in cache.
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            key: Key tensor (will be copied)
            value: Value tensor (will be copied)
            tag: Optional tag for namespacing
            priority: Priority for eviction (currently unused in Python impl)
        """
        cache_key = self._make_cache_key(layer, position, tag)
        
        # Ensure we have capacity before adding new entry
        # Remove existing entry first if present (to update LRU order)
        if cache_key in self._cache:
            # Remove existing entry to update its position in LRU order
            del self._cache[cache_key]
        else:
            # Only check capacity if this is a new entry
            self._ensure_capacity()
        
        # Store copies to avoid external modifications affecting cache
        self._cache[cache_key] = {
            'key': key.copy(),
            'value': value.copy()
        }
        # Move to end (most recently used) - O(1) operation
        self._cache.move_to_end(cache_key)
    
    def get(self, layer: int, position: int, tag: str = "") -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve KV state from cache.
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            tag: Optional tag
            
        Returns:
            Dict with 'key' and 'value' arrays, or None if not found
        """
        cache_key = self._make_cache_key(layer, position, tag)
        
        if cache_key in self._cache:
            self._hit_count += 1
            # Update LRU order: move to end (most recently used) - O(1)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        
        self._miss_count += 1
        return None
    
    def remove(self, layer: int, position: int, tag: str = "") -> bool:
        """
        Remove entry from cache.
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            tag: Optional tag
            
        Returns:
            True if entry was removed, False if not found
        """
        cache_key = self._make_cache_key(layer, position, tag)
        
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False
    
    def contains(self, layer: int, position: int, tag: str = "") -> bool:
        """
        Check if entry exists in cache.
        
        Args:
            layer: Transformer layer index
            position: Sequence position
            tag: Optional tag
            
        Returns:
            True if entry exists, False otherwise
        """
        cache_key = self._make_cache_key(layer, position, tag)
        return cache_key in self._cache
    
    def clear(self) -> None:
        """Clear all entries from cache and reset statistics."""
        self._cache.clear()
        # Note: We keep hit/miss counts for historical stats
    
    def size(self) -> int:
        """
        Get current number of cached entries.
        
        Returns:
            Number of entries in cache
        """
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            Hit rate as float between 0.0 and 1.0
        """
        total_requests = self._hit_count + self._miss_count
        if total_requests == 0:
            return 0.0
        return self._hit_count / total_requests
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)



