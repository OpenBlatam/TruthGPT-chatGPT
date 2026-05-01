"""
Simple test script for K/V cache optimization
Tests the basic functionality of the implemented components
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

def test_kv_cache_basic():
    """Test basic K/V cache functionality."""
    print("Testing K/V Cache Basic Functionality...")
    
    try:
        from modules.attention.efficient_kv_cache import KVCache, KVCacheConfig
        
        # Create cache configuration
        config = KVCacheConfig(max_cache_size=100)
        
        # Create cache
        cache = KVCache(config)
        
        # Test basic operations
        test_data = {
            'key': torch.randn(1, 8, 64),
            'value': torch.randn(1, 8, 64)
        }
        
        # Test put/get
        cache.put(0, 0, test_data)
        retrieved = cache.get(0, 0)
        
        assert retrieved is not None, "Cache should return data"
        assert torch.allclose(retrieved['key'], test_data['key']), "Keys should match"
        assert torch.allclose(retrieved['value'], test_data['value']), "Values should match"
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['hit_count'] == 1, "Should have 1 hit"
        assert stats['miss_count'] == 0, "Should have 0 misses"
        
        print("✓ K/V Cache basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ K/V Cache test failed: {e}")
        return False

def test_efficient_attention():
    """Test efficient attention mechanism."""
    print("Testing Efficient Attention...")
    
    try:
        from modules.attention.efficient_kv_cache import EfficientMultiHeadAttention, KVCacheConfig
        
        # Create attention module
        attention = EfficientMultiHeadAttention(
            d_model=512,
            n_heads=8,
            dropout=0.1,
            use_kv_cache=True,
            cache_config=KVCacheConfig()
        )
        
        # Test forward pass
        batch_size, seq_len, d_model = 2, 10, 512
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, cache_info = attention(query, key, value, use_cache=True, cache_position=0)
        
        assert output.shape == (batch_size, seq_len, d_model), "Output shape should match input"
        assert cache_info is not None, "Cache info should be returned"
        
        # Test cache stats
        stats = attention.get_cache_stats()
        assert 'hit_count' in stats, "Cache stats should include hit_count"
        
        print("✓ Efficient Attention test passed")
        return True
        
    except Exception as e:
        print(f"✗ Efficient Attention test failed: {e}")
        return False

def test_decoder_config():
    """Test decoder configuration."""
    print("Testing Decoder Configuration...")
    
    try:
        from modules.transformer.efficient_decoder import DecoderConfig, create_decoder_config
        
        # Test creating config
        config = create_decoder_config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            use_kv_cache=True
        )
        
        assert config.d_model == 512, "d_model should be 512"
        assert config.n_heads == 8, "n_heads should be 8"
        assert config.n_layers == 6, "n_layers should be 6"
        assert config.use_kv_cache == True, "use_kv_cache should be True"
        
        print("✓ Decoder Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Decoder Configuration test failed: {e}")
        return False

def test_optimizer_config():
    """Test optimizer configuration."""
    print("Testing Optimizer Configuration...")
    
    try:
        from modules.optimizers.kv_cache.kv_cache_optimizer import KVCacheOptimizationConfig, create_kv_cache_config
        
        # Test creating config
        config = create_kv_cache_config(
            max_cache_size=2048,
            use_flash_attention=True,
            use_memory_efficient_attention=True
        )
        
        assert config.max_cache_size == 2048, "max_cache_size should be 2048"
        assert config.use_flash_attention == True, "use_flash_attention should be True"
        assert config.use_memory_efficient_attention == True, "use_memory_efficient_attention should be True"
        
        print("✓ Optimizer Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Optimizer Configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("Running K/V Cache Optimization Tests")
    print("=" * 50)
    
    tests = [
        test_kv_cache_basic,
        test_efficient_attention,
        test_decoder_config,
        test_optimizer_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! K/V Cache optimization is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()





