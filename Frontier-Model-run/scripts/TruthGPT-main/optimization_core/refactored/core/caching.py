"""
Intelligent Caching System
==========================

Advanced caching system with:
- Multiple cache backends (memory, Redis, file)
- TTL and LRU eviction
- Cache warming and preloading
- Cache statistics and monitoring
- Distributed caching support
"""

import time
import hashlib
import pickle
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict
from enum import Enum
import asyncio
from pathlib import Path


class CacheBackend(Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Cache entry data structure"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0


class CacheManager:
    """
    Intelligent caching system with multiple backends.
    
    Features:
    - Multiple cache backends
    - TTL and LRU eviction
    - Cache warming
    - Statistics and monitoring
    - Distributed caching
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.backend = CacheBackend(self.config.get('backend', 'memory'))
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Cache size management
        self.max_size = self.config.get('max_size', 1000)
        self.max_memory_mb = self.config.get('max_memory_mb', 100)
        
        # TTL management
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5 minutes
        self.cleanup_thread = None
        self.running = False
        
        # Initialize backend
        self._initialize_backend()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _initialize_backend(self):
        """Initialize cache backend"""
        if self.backend == CacheBackend.REDIS:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=self.config.get('redis_host', 'localhost'),
                    port=self.config.get('redis_port', 6379),
                    db=self.config.get('redis_db', 0)
                )
                self.logger.info("Redis cache backend initialized")
            except ImportError:
                self.logger.warning("Redis not available, falling back to memory")
                self.backend = CacheBackend.MEMORY
            except Exception as e:
                self.logger.error(f"Failed to initialize Redis: {e}")
                self.backend = CacheBackend.MEMORY
        
        elif self.backend == CacheBackend.FILE:
            self.cache_dir = Path(self.config.get('cache_dir', 'cache'))
            self.cache_dir.mkdir(exist_ok=True)
            self.logger.info(f"File cache backend initialized: {self.cache_dir}")
    
    def _start_cleanup_thread(self):
        """Start cache cleanup thread"""
        if self.cleanup_thread:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        self.logger.info("Cache cleanup thread started")
    
    def _cleanup_loop(self):
        """Cache cleanup loop"""
        while self.running:
            try:
                self._cleanup_expired()
                self._cleanup_lru()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")
                time.sleep(60)
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = datetime.now()
        expired_keys = []
        
        with self.lock:
            for key, entry in self.cache.items():
                if entry.expires_at and entry.expires_at < current_time:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
            self.stats['evictions'] += 1
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    def _cleanup_lru(self):
        """Remove least recently used entries if cache is full"""
        if len(self.cache) <= self.max_size:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at
        )
        
        # Remove oldest entries
        to_remove = len(self.cache) - self.max_size
        for i in range(to_remove):
            key, _ = sorted_entries[i]
            self.delete(key)
            self.stats['evictions'] += 1
        
        if to_remove > 0:
            self.logger.debug(f"Cleaned up {to_remove} LRU entries")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.backend == CacheBackend.REDIS:
                return await self._get_redis(key)
            elif self.backend == CacheBackend.FILE:
                return await self._get_file(key)
            else:
                return self._get_memory(key)
        except Exception as e:
            self.logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def _get_memory(self, key: str) -> Optional[Any]:
        """Get from memory cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.expires_at and entry.expires_at < datetime.now():
                    del self.cache[key]
                    self.stats['misses'] += 1
                    return None
                
                # Update access info
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                self.stats['hits'] += 1
                return entry.value
            else:
                self.stats['misses'] += 1
                return None
    
    async def _get_redis(self, key: str) -> Optional[Any]:
        """Get from Redis cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                self.stats['hits'] += 1
                return pickle.loads(value)
            else:
                self.stats['misses'] += 1
                return None
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            self.stats['misses'] += 1
            return None
    
    async def _get_file(self, key: str) -> Optional[Any]:
        """Get from file cache"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Check expiration
                if data.get('expires_at') and data['expires_at'] < datetime.now():
                    cache_file.unlink()
                    self.stats['misses'] += 1
                    return None
                
                self.stats['hits'] += 1
                return data['value']
            else:
                self.stats['misses'] += 1
                return None
        except Exception as e:
            self.logger.error(f"File cache get error: {e}")
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if self.backend == CacheBackend.REDIS:
                return await self._set_redis(key, value, ttl)
            elif self.backend == CacheBackend.FILE:
                return await self._set_file(key, value, ttl)
            else:
                return self._set_memory(key, value, ttl)
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def _set_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in memory cache"""
        try:
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Calculate size
            size_bytes = len(pickle.dumps(value))
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            with self.lock:
                self.cache[key] = entry
                self.stats['sets'] += 1
            
            return True
        except Exception as e:
            self.logger.error(f"Memory cache set error: {e}")
            return False
    
    async def _set_redis(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in Redis cache"""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                self.redis_client.setex(key, ttl, serialized)
            else:
                self.redis_client.set(key, serialized)
            
            self.stats['sets'] += 1
            return True
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
            return False
    
    async def _set_file(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in file cache"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            
            data = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(seconds=ttl) if ttl else None
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.stats['sets'] += 1
            return True
        except Exception as e:
            self.logger.error(f"File cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.backend == CacheBackend.REDIS:
                self.redis_client.delete(key)
            elif self.backend == CacheBackend.FILE:
                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    cache_file.unlink()
            else:
                with self.lock:
                    if key in self.cache:
                        del self.cache[key]
            
            self.stats['deletes'] += 1
            return True
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def clear(self):
        """Clear all cache"""
        try:
            if self.backend == CacheBackend.REDIS:
                self.redis_client.flushdb()
            elif self.backend == CacheBackend.FILE:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
            else:
                with self.lock:
                    self.cache.clear()
            
            self.logger.info("Cache cleared")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests == 0:
            return 0.0
        return self.stats['hits'] / total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'backend': self.backend.value,
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'sets': self.stats['sets'],
                'deletes': self.stats['deletes'],
                'evictions': self.stats['evictions'],
                'hit_rate': self.get_hit_rate()
            }
    
    def warm_cache(self, warmup_func: Callable[[], Dict[str, Any]]):
        """Warm cache with data from function"""
        try:
            warmup_data = warmup_func()
            for key, value in warmup_data.items():
                self.set(key, value)
            self.logger.info(f"Cache warmed with {len(warmup_data)} entries")
        except Exception as e:
            self.logger.error(f"Cache warming failed: {e}")
    
    def preload(self, keys: List[str], loader_func: Callable[[str], Any]):
        """Preload cache with specific keys"""
        for key in keys:
            try:
                value = loader_func(key)
                self.set(key, value)
            except Exception as e:
                self.logger.error(f"Failed to preload key {key}: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get cache memory usage"""
        if self.backend != CacheBackend.MEMORY:
            return {}
        
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            return {
                'total_bytes': total_size,
                'total_mb': total_size / (1024 * 1024),
                'entries': len(self.cache),
                'avg_entry_size': total_size / len(self.cache) if self.cache else 0
            }
    
    def cleanup(self):
        """Cleanup cache resources"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        if hasattr(self, 'redis_client'):
            self.redis_client.close()



