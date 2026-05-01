"""
Cache manager for inference results to improve performance.
"""
import logging
import hashlib
import pickle
from typing import Dict, Any, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of inference results.
    Supports both in-memory and disk caching.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_size: int = 1000,
        use_disk_cache: bool = True,
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for disk cache
            max_memory_size: Maximum number of items in memory cache
            use_disk_cache: Enable disk caching
        """
        self.memory_cache: Dict[str, Any] = {}
        self.max_memory_size = max_memory_size
        self.use_disk_cache = use_disk_cache
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """
        Generate cache key from prompt and generation parameters.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
        
        Returns:
            Cache key (hash)
        """
        # Create hash from prompt and parameters
        cache_data = {
            "prompt": prompt,
            **kwargs
        }
        cache_str = str(sorted(cache_data.items()))
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[Any]:
        """
        Get cached result.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
        
        Returns:
            Cached result or None
        """
        cache_key = self._get_cache_key(prompt, **kwargs)
        
        # Check memory cache
        if cache_key in self.memory_cache:
            logger.debug(f"Cache hit (memory): {cache_key[:8]}")
            return self.memory_cache[cache_key]
        
        # Check disk cache
        if self.use_disk_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        result = pickle.load(f)
                    logger.debug(f"Cache hit (disk): {cache_key[:8]}")
                    # Also store in memory for faster access
                    self.memory_cache[cache_key] = result
                    return result
                except Exception as e:
                    logger.warning(f"Error loading cache file: {e}")
        
        return None
    
    def set(self, prompt: str, result: Any, **kwargs) -> None:
        """
        Store result in cache.
        
        Args:
            prompt: Input prompt
            result: Result to cache
            **kwargs: Generation parameters
        """
        cache_key = self._get_cache_key(prompt, **kwargs)
        
        # Store in memory cache
        if len(self.memory_cache) >= self.max_memory_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = result
        
        # Store in disk cache
        if self.use_disk_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"Error saving cache file: {e}")
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error removing cache file {cache_file}: {e}")
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        disk_cache_size = 0
        if self.cache_dir:
            disk_cache_size = len(list(self.cache_dir.glob("*.pkl")))
        
        return {
            "memory_cache_size": len(self.memory_cache),
            "max_memory_size": self.max_memory_size,
            "disk_cache_size": disk_cache_size,
            "use_disk_cache": self.use_disk_cache,
        }



