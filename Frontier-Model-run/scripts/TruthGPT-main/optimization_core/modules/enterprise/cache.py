"""
Enterprise TruthGPT Cache System
Advanced caching with LRU, LFU, and intelligent eviction
"""

from typing import Any, Optional, Dict, List
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum

class CacheStrategy(Enum):
    """Cache strategy enum."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live

@dataclass
class CacheEntry:
    """Cache entry dataclass."""
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 1
    last_access: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None
    size: int = 0

class EnterpriseCache:
    """Enterprise caching system with multiple strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[timedelta] = None,
        max_memory_mb: int = 1024
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl or timedelta(hours=1)
        self.max_memory_mb = max_memory_mb
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict[str, None] = OrderedDict()
        self.frequency: Dict[str, int] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiry
            if entry.expiry and entry.expiry < datetime.now():
                self.delete(key)
                self.misses += 1
                return None
            
            # Update access information
            entry.access_count += 1
            entry.last_access = datetime.now()
            self.access_order.move_to_end(key)
            self.hits += 1
            
            return entry.value
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in cache."""
        try:
            # Calculate entry size
            size = self._calculate_size(value)
            
            # Check memory limit
            if self._get_total_memory_mb() + size > self.max_memory_mb:
                self._evict_entries()
            
            # Create entry
            expiry = None
            if ttl or self.default_ttl:
                expiry = datetime.now() + (ttl or self.default_ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                last_access=datetime.now(),
                expiry=expiry,
                size=size
            )
            
            # Evict if necessary
            if len(self.cache) >= self.max_size:
                self._evict_one()
            
            # Add to cache
            self.cache[key] = entry
            self.access_order[key] = None
            self.frequency[key] = self.frequency.get(key, 0) + 1
            
            return True
            
        except Exception as e:
            print(f"Error setting cache entry: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_order[key]
            if key in self.frequency:
                del self.frequency[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.access_order.clear()
        self.frequency.clear()
    
    def _evict_one(self):
        """Evict one entry based on strategy."""
        if not self.cache:
            return
        
        evict_key = None
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            evict_key = next(iter(self.access_order))
            
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            evict_key = min(self.frequency.items(), key=lambda x: x[1])[0]
            
        elif self.strategy == CacheStrategy.FIFO:
            # Evict first in
            evict_key = next(iter(self.access_order))
            
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.expiry and entry.expiry < now
            ]
            if expired_keys:
                evict_key = expired_keys[0]
            else:
                evict_key = next(iter(self.access_order))
        
        if evict_key:
            self.delete(evict_key)
            self.evictions += 1
    
    def _evict_entries(self):
        """Evict multiple entries to free memory."""
        target_memory_mb = self.max_memory_mb * 0.8
        
        while self._get_total_memory_mb() > target_memory_mb and self.cache:
            self._evict_one()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            serialized = pickle.dumps(value)
            return len(serialized)
        except:
            return 1
    
    def _get_total_memory_mb(self) -> float:
        """Get total memory usage in MB."""
        total_bytes = sum(entry.size for entry in self.cache.values())
        return total_bytes / (1024 * 1024)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_mb": self._get_total_memory_mb(),
            "max_memory_mb": self.max_memory_mb,
            "strategy": self.strategy.value
        }

# Global cache instance
_cache: Optional[EnterpriseCache] = None

def get_cache() -> EnterpriseCache:
    """Get or create enterprise cache."""
    global _cache
    if _cache is None:
        _cache = EnterpriseCache()
    return _cache

# Example usage
if __name__ == "__main__":
    # Create enterprise cache
    cache = EnterpriseCache(
        max_size=1000,
        strategy=CacheStrategy.LRU,
        default_ttl=timedelta(minutes=30),
        max_memory_mb=512
    )
    
    # Set values
    for i in range(100):
        cache.set(f"key_{i}", f"value_{i}")
    
    # Get values
    value = cache.get("key_0")
    print(f"Retrieved value: {value}")
    
    # Get statistics
    stats = cache.get_stats()
    print("\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clear cache
    cache.clear()
    print("\nCache cleared")








