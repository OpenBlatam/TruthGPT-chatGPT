"""
Cache utilities for optimization_core.

Provides comprehensive caching utilities with support for:
- In-memory caching with LRU eviction
- Disk-based caching with TTL
- Thread-safe operations
- Function result caching decorators

Consolidates functionality from utils/cache_utils.py and related modules.
"""

from __future__ import annotations

import logging
import time
import hashlib
import pickle
import threading
from typing import Any, Dict, Optional, Callable, Tuple, TypeVar, Union
from functools import wraps
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# ════════════════════════════════════════════════════════════════════════════════
# CACHE STATISTICS
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Cache miss rate (0.0 to 1.0)."""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'size': self.size,
            'max_size': self.max_size,
            'total_requests': self.total_requests,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
        }
    
    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.size = 0


# ════════════════════════════════════════════════════════════════════════════════
# MEMORY CACHE
# ════════════════════════════════════════════════════════════════════════════════

class MemoryCache:
    """
    Thread-safe in-memory cache with LRU eviction and TTL support.
    
    Uses OrderedDict for efficient LRU operations.
    """
    
    def __init__(
        self,
        max_size: int = 128,
        default_ttl: Optional[float] = None
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Tuple[Any, Optional[float]]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)
    
    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Cache key (MD5 hash)
        """
        try:
            key_data = pickle.dumps((args, sorted(kwargs.items())))
            return hashlib.md5(key_data).hexdigest()
        except (pickle.PickleError, TypeError):
            # Fallback for non-picklable objects
            key_str = str((args, sorted(kwargs.items())))
            return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return default
            
            value, expiry = self._cache[key]
            
            # Check expiration
            if expiry is not None and time.time() > expiry:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return default
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            # Evict if needed
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()
            
            ttl = ttl if ttl is not None else self.default_ttl
            expiry = time.time() + ttl if ttl is not None else None
            
            # Update or add
            if key in self._cache:
                self._cache.move_to_end(key)
            
            self._cache[key] = (value, expiry)
            self._stats.size = len(self._cache)
    
    def _evict_oldest(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            self._cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                size=self._stats.size,
                max_size=self._stats.max_size
            )
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._stats.reset()


# ════════════════════════════════════════════════════════════════════════════════
# DISK CACHE
# ════════════════════════════════════════════════════════════════════════════════

class DiskCache:
    """
    Disk-based cache with TTL support.
    
    Stores cached values as pickle files in a directory.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        default_ttl: Optional[float] = None
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get cache file path for key.
        
        Args:
            key: Cache key
        
        Returns:
            Path to cache file
        """
        # Sanitize key for filesystem
        safe_key = hashlib.md5(key.encode() if isinstance(key, str) else key).hexdigest()
        return self.cache_dir / f"{safe_key}.pkl"
    
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            with self._lock:
                self._stats.misses += 1
            return default
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                value, expiry = data
            
            # Check expiration
            if expiry is not None and time.time() > expiry:
                cache_path.unlink()
                with self._lock:
                    self._stats.misses += 1
                return default
            
            with self._lock:
                self._stats.hits += 1
            return value
        except Exception as e:
            logger.warning(f"Failed to load cache for {key}: {e}")
            with self._lock:
                self._stats.misses += 1
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        cache_path = self._get_cache_path(key)
        ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl if ttl is not None else None
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((value, expiry), f, protocol=pickle.HIGHEST_PROTOCOL)
            with self._lock:
                self._stats.size = len(list(self.cache_dir.glob("*.pkl")))
        except Exception as e:
            logger.warning(f"Failed to save cache for {key}: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)
        try:
            if cache_path.exists():
                cache_path.unlink()
                with self._lock:
                    self._stats.size = len(list(self.cache_dir.glob("*.pkl")))
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to delete cache for {key}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            self._stats.size = 0
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(list(self.cache_dir.glob("*.pkl")))
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = self.size()
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                size=self._stats.size,
                max_size=0  # Disk cache has no max size
            )
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._stats.reset()


# ════════════════════════════════════════════════════════════════════════════════
# CACHE DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def cached(
    cache: Optional[MemoryCache] = None,
    ttl: Optional[float] = None,
    max_size: int = 128,
    key_func: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """
    Decorator for caching function results.
    
    Args:
        cache: Cache instance (creates new MemoryCache if None)
        ttl: Time-to-live in seconds (overrides cache default if provided)
        max_size: Maximum cache size (only used if cache is None)
        key_func: Optional function to generate cache keys
    
    Example:
        >>> @cached(ttl=3600, max_size=64)
        >>> def expensive_function(x, y):
        ...     return x + y
    """
    if cache is None:
        cache = MemoryCache(max_size=max_size, default_ttl=ttl)
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                try:
                    key_data = pickle.dumps((args, sorted(kwargs.items())))
                    cache_key = hashlib.md5(key_data).hexdigest()
                except (pickle.PickleError, TypeError):
                    # Fallback for non-picklable objects
                    key_str = str((args, sorted(kwargs.items())))
                    cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Compute result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        # Add cache management methods
        wrapper.clear_cache = cache.clear  # type: ignore
        wrapper.cache_size = cache.size  # type: ignore
        wrapper.cache_stats = cache.get_stats  # type: ignore
        
        return wrapper  # type: ignore
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Statistics
    'CacheStats',
    # Cache classes
    'MemoryCache',
    'DiskCache',
    # Decorators
    'cached',
]












