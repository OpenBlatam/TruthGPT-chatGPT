"""
Unit tests for attention optimizations
Tests efficient attention mechanisms, KV cache, and attention optimizations
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.mock_components import MockAttention, MockKVCache
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestAttentionOptimizations(unittest.TestCase):
    """Test suite for attention optimizations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        self.config = TestUtils.create_test_config()
        
    def test_efficient_attention_basic(self):
        """Test basic efficient attention functionality"""
        # Create test data
        attention_data = self.test_data.create_attention_data()
        
        # Create mock attention
        attention = MockAttention(d_model=512, n_heads=8)
        
        # Test forward pass
        output, weights = attention(attention_data['query'], 
                                  attention_data['key'], 
                                  attention_data['value'])
        
        # Assertions
        self.assertEqual(output.shape, attention_data['query'].shape)
        self.assertEqual(weights.shape, (2, 8, 128, 128))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1))))
        
    def test_attention_performance(self):
        """Test attention performance"""
        attention_data = self.test_data.create_attention_data(batch_size=4, seq_len=256)
        attention = MockAttention(d_model=512, n_heads=8)
        
        # Profile attention computation
        self.profiler.start_profile("attention_forward")
        output, weights = attention(attention_data['query'], 
                                  attention_data['key'], 
                                  attention_data['value'])
        metrics = self.profiler.end_profile()
        
        # Assert performance
        self.assertLess(metrics['execution_time'], 1.0)  # Should be fast
        self.assertLess(metrics['memory_used'], 100)  # Should use reasonable memory
        
    def test_kv_cache_functionality(self):
        """Test KV cache functionality"""
        cache = MockKVCache(max_size=100)
        
        # Test cache operations
        test_key = "test_key"
        test_value = torch.randn(1, 8, 64)
        
        # Test put
        success = cache.put(test_key, test_value)
        self.assertTrue(success)
        
        # Test get
        retrieved = cache.get(test_key)
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(retrieved, test_value))
        
        # Test cache stats
        stats = cache.get_stats()
        self.assertEqual(stats['hit_count'], 1)
        self.assertEqual(stats['miss_count'], 0)
        self.assertEqual(stats['hit_rate'], 1.0)
        
    def test_kv_cache_miss(self):
        """Test KV cache miss handling"""
        cache = MockKVCache(max_size=100)
        
        # Test get non-existent key
        retrieved = cache.get("non_existent")
        self.assertIsNone(retrieved)
        
        # Test cache stats
        stats = cache.get_stats()
        self.assertEqual(stats['hit_count'], 0)
        self.assertEqual(stats['miss_count'], 1)
        self.assertEqual(stats['hit_rate'], 0.0)
        
    def test_attention_with_kv_cache(self):
        """Test attention mechanism with KV cache"""
        attention = MockAttention(d_model=512, n_heads=8)
        cache = MockKVCache(max_size=1000)
        
        # First forward pass
        attention_data = self.test_data.create_attention_data()
        output1, weights1 = attention(attention_data['query'], 
                                   attention_data['key'], 
                                   attention_data['value'])
        
        # Store in cache
        cache_key = "attention_0"
        cache.put(cache_key, output1)
        
        # Second forward pass (should use cache)
        cached_output = cache.get(cache_key)
        self.assertIsNotNone(cached_output)
        
        # Verify cache is working
        stats = cache.get_stats()
        self.assertGreater(stats['hit_count'], 0)
        
    def test_attention_gradient_flow(self):
        """Test attention gradient flow"""
        attention = MockAttention(d_model=512, n_heads=8)
        attention_data = self.test_data.create_attention_data()
        
        # Enable gradients
        attention_data['query'].requires_grad_(True)
        attention_data['key'].requires_grad_(True)
        attention_data['value'].requires_grad_(True)
        
        # Forward pass
        output, weights = attention(attention_data['query'], 
                                  attention_data['key'], 
                                  attention_data['value'])
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(attention_data['query'].grad)
        self.assertIsNotNone(attention_data['key'].grad)
        self.assertIsNotNone(attention_data['value'].grad)
        
        # Assert gradient flow
        gradients = [attention_data['query'].grad, attention_data['key'].grad, attention_data['value'].grad]
        self.assertTrue(TestAssertions.assert_gradient_flow(gradients))
        
    def test_attention_numerical_stability(self):
        """Test attention numerical stability"""
        attention = MockAttention(d_model=512, n_heads=8)
        
        # Test with extreme values
        extreme_data = {
            'query': torch.randn(2, 128, 512) * 1000,  # Large values
            'key': torch.randn(2, 128, 512) * 1000,
            'value': torch.randn(2, 128, 512) * 1000
        }
        
        output, weights = attention(extreme_data['query'], 
                                  extreme_data['key'], 
                                  extreme_data['value'])
        
        # Check for numerical stability
        self.assertTrue(TestAssertions.assert_numerical_stability(output))
        self.assertTrue(TestAssertions.assert_numerical_stability(weights))
        
    def test_attention_memory_efficiency(self):
        """Test attention memory efficiency"""
        attention = MockAttention(d_model=512, n_heads=8)
        
        # Test with large sequence length
        large_data = self.test_data.create_attention_data(seq_len=512)
        
        # Profile memory usage
        with TestUtils.measure_execution_time(lambda: attention(
            large_data['query'], large_data['key'], large_data['value']
        )) as metrics:
            pass
        
        # Assert memory efficiency
        self.assertLess(metrics['memory_used'], 500)  # Should use reasonable memory
        
    def test_attention_batch_processing(self):
        """Test attention batch processing"""
        attention = MockAttention(d_model=512, n_heads=8)
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                data = self.test_data.create_attention_data(batch_size=batch_size)
                output, weights = attention(data['query'], data['key'], data['value'])
                
                self.assertEqual(output.shape[0], batch_size)
                self.assertEqual(weights.shape[0], batch_size)
                
    def test_attention_different_sequence_lengths(self):
        """Test attention with different sequence lengths"""
        attention = MockAttention(d_model=512, n_heads=8)
        
        # Test with different sequence lengths
        seq_lengths = [64, 128, 256, 512]
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                data = self.test_data.create_attention_data(seq_len=seq_len)
                output, weights = attention(data['query'], data['key'], data['value'])
                
                self.assertEqual(output.shape[1], seq_len)
                self.assertEqual(weights.shape[2], seq_len)
                self.assertEqual(weights.shape[3], seq_len)

class TestAttentionOptimizationsIntegration(unittest.TestCase):
    """Integration tests for attention optimizations"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_attention_with_mlp_integration(self):
        """Test attention integration with MLP"""
        from tests.fixtures.mock_components import MockMLP
        
        attention = MockAttention(d_model=512, n_heads=8)
        mlp = MockMLP(input_size=512, hidden_size=2048, output_size=512)
        
        # Create test data
        data = self.test_data.create_attention_data()
        
        # Attention forward pass
        attn_output, weights = attention(data['query'], data['key'], data['value'])
        
        # MLP forward pass
        mlp_output = mlp(attn_output)
        
        # Verify integration
        self.assertEqual(mlp_output.shape, attn_output.shape)
        self.assertIsNotNone(mlp_output)
        
    def test_attention_optimization_workflow(self):
        """Test complete attention optimization workflow"""
        attention = MockAttention(d_model=512, n_heads=8)
        cache = MockKVCache(max_size=1000)
        
        # Simulate training loop
        for epoch in range(3):
            data = self.test_data.create_attention_data()
            
            # Forward pass
            output, weights = attention(data['query'], data['key'], data['value'])
            
            # Cache output
            cache.put(f"epoch_{epoch}", output)
            
            # Verify cache is working
            cached = cache.get(f"epoch_{epoch}")
            self.assertIsNotNone(cached)
            
        # Verify final stats
        attn_stats = attention.get_attention_stats()
        cache_stats = cache.get_stats()
        
        self.assertGreater(attn_stats['attention_count'], 0)
        self.assertGreater(cache_stats['hit_count'], 0)

if __name__ == '__main__':
    unittest.main()





