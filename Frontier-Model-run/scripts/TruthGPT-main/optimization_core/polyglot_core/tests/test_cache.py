"""
Tests for unified KV Cache.
"""

import pytest
import numpy as np
from optimization_core.polyglot_core.cache import (
    KVCache,
    KVCacheConfig,
    EvictionStrategy,
    CacheStats,
)
from optimization_core.polyglot_core.backend import Backend


def test_kvcache_config():
    """Test KVCacheConfig."""
    config = KVCacheConfig(max_size=1000)
    assert config.max_size == 1000
    assert config.eviction_strategy == EvictionStrategy.LRU
    
    # Preset configs
    inference_config = KVCacheConfig.inference_optimized(8)
    assert inference_config.max_memory_bytes == 8 * 1024**3
    
    long_ctx_config = KVCacheConfig.long_context(32)
    assert long_ctx_config.max_memory_bytes == 32 * 1024**3


def test_kvcache_basic_operations():
    """Test basic cache operations."""
    cache = KVCache(max_size=100)
    
    # Put and get
    k = np.random.randn(64).astype(np.float32)
    v = np.random.randn(64).astype(np.float32)
    
    cache.put(layer=0, position=0, key=k, value=v)
    result = cache.get(layer=0, position=0)
    
    assert result is not None
    assert 'key' in result
    assert 'value' in result
    np.testing.assert_array_almost_equal(result['key'], k)
    np.testing.assert_array_almost_equal(result['value'], v)


def test_kvcache_missing():
    """Test getting non-existent entry."""
    cache = KVCache(max_size=100)
    result = cache.get(layer=99, position=99)
    assert result is None


def test_kvcache_remove():
    """Test removing entries."""
    cache = KVCache(max_size=100)
    
    k = np.random.randn(32).astype(np.float32)
    v = np.random.randn(32).astype(np.float32)
    
    cache.put(layer=0, position=0, key=k, value=v)
    assert cache.get(layer=0, position=0) is not None
    
    removed = cache.remove(layer=0, position=0)
    assert removed is True
    
    assert cache.get(layer=0, position=0) is None


def test_kvcache_clear():
    """Test clearing cache."""
    cache = KVCache(max_size=100)
    
    for i in range(10):
        k = np.random.randn(32).astype(np.float32)
        v = np.random.randn(32).astype(np.float32)
        cache.put(layer=0, position=i, key=k, value=v)
    
    assert cache.size > 0
    cache.clear()
    assert cache.size == 0


def test_kvcache_size():
    """Test cache size tracking."""
    cache = KVCache(max_size=100)
    
    assert cache.size == 0
    
    for i in range(5):
        k = np.random.randn(32).astype(np.float32)
        v = np.random.randn(32).astype(np.float32)
        cache.put(layer=0, position=i, key=k, value=v)
    
    assert cache.size == 5


def test_kvcache_hit_rate():
    """Test cache hit rate calculation."""
    cache = KVCache(max_size=100)
    
    k = np.random.randn(32).astype(np.float32)
    v = np.random.randn(32).astype(np.float32)
    
    cache.put(layer=0, position=0, key=k, value=v)
    
    # 2 hits
    cache.get(layer=0, position=0)
    cache.get(layer=0, position=0)
    
    # 1 miss
    cache.get(layer=0, position=1)
    
    hit_rate = cache.hit_rate
    assert 0.0 <= hit_rate <= 1.0
    assert hit_rate > 0.0  # Should have some hits


def test_kvcache_stats():
    """Test cache statistics."""
    cache = KVCache(max_size=100)
    
    k = np.random.randn(32).astype(np.float32)
    v = np.random.randn(32).astype(np.float32)
    
    cache.put(layer=0, position=0, key=k, value=v)
    cache.get(layer=0, position=0)
    
    stats = cache.stats
    assert isinstance(stats, CacheStats)
    assert stats.entry_count >= 0


def test_kvcache_backend_selection():
    """Test automatic backend selection."""
    cache = KVCache(max_size=100)
    assert cache.backend in [Backend.PYTHON, Backend.RUST, Backend.CPP, Backend.GO]


def test_kvcache_with_tag():
    """Test cache with tags."""
    cache = KVCache(max_size=100)
    
    k1 = np.random.randn(32).astype(np.float32)
    v1 = np.random.randn(32).astype(np.float32)
    k2 = np.random.randn(32).astype(np.float32)
    v2 = np.random.randn(32).astype(np.float32)
    
    cache.put(layer=0, position=0, key=k1, value=v1, tag="tag1")
    cache.put(layer=0, position=0, key=k2, value=v2, tag="tag2")
    
    result1 = cache.get(layer=0, position=0, tag="tag1")
    result2 = cache.get(layer=0, position=0, tag="tag2")
    
    assert result1 is not None
    assert result2 is not None
    np.testing.assert_array_almost_equal(result1['key'], k1)
    np.testing.assert_array_almost_equal(result2['key'], k2)


