"""
💾 Distributed Cache Manager
Redis-based caching with LRU eviction and TTL support
"""

import hashlib
import json
import pickle
import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
from threading import Lock

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class InMemoryCache:
    """Thread-safe in-memory LRU cache with TTL"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            if key in self._timestamps:
                ttl = self._timestamps.get(key, {}).get("ttl", self.default_ttl)
                elapsed = time.time() - self._timestamps[key]["created_at"]
                if elapsed > ttl:
                    self._evict(key)
                    return None
            
            # Move to end (LRU)
            value = self._cache.pop(key)
            self._cache[key] = value
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        with self._lock:
            # Remove if exists
            if key in self._cache:
                self._cache.pop(key)
            
            # Check size limit
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
            
            # Set value
            self._cache[key] = value
            self._timestamps[key] = {
                "created_at": time.time(),
                "ttl": ttl or self.default_ttl
            }
    
    def delete(self, key: str):
        """Delete key from cache"""
        with self._lock:
            self._evict(key)
    
    def _evict(self, key: str):
        """Evict key from cache"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self):
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": 0.0,  # Would need to track hits/misses
            }


class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        key_prefix: str = "inference:"
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Install with: pip install redis")
        
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._client = redis.from_url(redis_url, decode_responses=False)
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, key: str) -> str:
        """Make prefixed key"""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            full_key = self._make_key(key)
            data = self._client.get(full_key)
            
            if data is None:
                self._misses += 1
                return None
            
            self._hits += 1
            return pickle.loads(data)
        
        except Exception as e:
            # Log error but don't fail
            print(f"Cache get error: {e}")
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis"""
        try:
            full_key = self._make_key(key)
            data = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            self._client.setex(full_key, ttl, data)
        
        except Exception as e:
            # Log error but don't fail
            print(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete key from Redis"""
        try:
            full_key = self._make_key(key)
            self._client.delete(full_key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    def clear(self):
        """Clear all keys with prefix"""
        try:
            keys = self._client.keys(f"{self.key_prefix}*")
            if keys:
                self._client.delete(*keys)
        except Exception as e:
            print(f"Cache clear error: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "backend": "redis",
            "url": self.redis_url
        }


class CacheManager:
    """Unified cache manager with automatic backend selection"""
    
    def __init__(
        self,
        backend: str = "memory",
        redis_url: Optional[str] = None,
        max_size: int = 10000,
        default_ttl: int = 3600
    ):
        self.backend = backend.lower()
        
        if self.backend == "redis" and redis_url:
            if not REDIS_AVAILABLE:
                print("Warning: Redis not available, falling back to in-memory cache")
                self.backend = "memory"
                self._cache = InMemoryCache(max_size=max_size, default_ttl=default_ttl)
            else:
                self._cache = RedisCache(redis_url=redis_url, default_ttl=default_ttl)
        else:
            self._cache = InMemoryCache(max_size=max_size, default_ttl=default_ttl)
            if self.backend == "redis":
                print("Warning: No Redis URL provided, using in-memory cache")
                self.backend = "memory"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        self._cache.set(key, value, ttl)
    
    def delete(self, key: str):
        """Delete key from cache"""
        self._cache.delete(key)
    
    def clear(self):
        """Clear entire cache"""
        self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self._cache.stats()
        stats["backend"] = self.backend
        return stats


# Global cache manager instance
import os
cache_manager = CacheManager(
    backend=os.getenv("CACHE_BACKEND", "memory"),
    redis_url=os.getenv("REDIS_URL"),
    max_size=int(os.getenv("CACHE_MAX_SIZE", "10000")),
    default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "3600"))
)



