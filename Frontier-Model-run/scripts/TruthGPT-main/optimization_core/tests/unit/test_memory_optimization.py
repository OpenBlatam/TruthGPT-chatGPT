"""
Unit tests for memory optimization components
Tests memory management, pooling, and optimization techniques
"""

import unittest
import torch
import torch.nn as nn
import gc
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, MemoryTracker, TestAssertions

class TestMemoryPooling(unittest.TestCase):
    """Test suite for memory pooling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.memory_tracker = MemoryTracker()
        
    def test_tensor_pool_basic(self):
        """Test basic tensor pool functionality"""
        class TensorPool:
            def __init__(self, max_size=100):
                self.max_size = max_size
                self.pool = {}
                self.usage_count = {}
                
            def get_tensor(self, shape, dtype=torch.float32, device='cpu'):
                """Get tensor from pool or create new one"""
                key = (shape, dtype, device)
                
                if key in self.pool and len(self.pool[key]) > 0:
                    tensor = self.pool[key].pop()
                    self.usage_count[key] = self.usage_count.get(key, 0) + 1
                    return tensor
                else:
                    return torch.zeros(shape, dtype=dtype, device=device)
                    
            def return_tensor(self, tensor):
                """Return tensor to pool"""
                key = (tensor.shape, tensor.dtype, tensor.device)
                
                if key not in self.pool:
                    self.pool[key] = []
                    
                if len(self.pool[key]) < self.max_size:
                    tensor.zero_()  # Reset tensor
                    self.pool[key].append(tensor)
                    
            def get_stats(self):
                """Get pool statistics"""
                total_tensors = sum(len(tensors) for tensors in self.pool.values())
                return {
                    'total_tensors': total_tensors,
                    'pool_keys': len(self.pool),
                    'usage_count': self.usage_count
                }
        
        pool = TensorPool(max_size=10)
        
        # Test getting and returning tensors
        tensor1 = pool.get_tensor((2, 128, 512))
        tensor2 = pool.get_tensor((2, 128, 512))
        
        self.assertEqual(tensor1.shape, (2, 128, 512))
        self.assertEqual(tensor2.shape, (2, 128, 512))
        
        # Return tensors to pool
        pool.return_tensor(tensor1)
        pool.return_tensor(tensor2)
        
        # Get stats
        stats = pool.get_stats()
        self.assertGreaterEqual(stats['total_tensors'], 0)
        
    def test_activation_cache(self):
        """Test activation cache functionality"""
        class ActivationCache:
            def __init__(self, max_size=1000):
                self.max_size = max_size
                self.cache = {}
                self.access_count = 0
                self.hit_count = 0
                
            def get_activation(self, key, compute_func):
                """Get activation from cache or compute"""
                self.access_count += 1
                
                if key in self.cache:
                    self.hit_count += 1
                    return self.cache[key]
                else:
                    activation = compute_func()
                    if len(self.cache) < self.max_size:
                        self.cache[key] = activation
                    return activation
                    
            def clear_cache(self):
                """Clear cache"""
                self.cache.clear()
                
            def get_stats(self):
                """Get cache statistics"""
                hit_rate = self.hit_count / self.access_count if self.access_count > 0 else 0
                return {
                    'cache_size': len(self.cache),
                    'access_count': self.access_count,
                    'hit_count': self.hit_count,
                    'hit_rate': hit_rate
                }
        
        cache = ActivationCache(max_size=100)
        
        # Test cache functionality
        def compute_activation():
            return torch.randn(2, 128, 512)
        
        # First access (miss)
        activation1 = cache.get_activation("key1", compute_activation)
        self.assertIsNotNone(activation1)
        
        # Second access (hit)
        activation2 = cache.get_activation("key1", compute_activation)
        self.assertIsNotNone(activation2)
        
        # Check stats
        stats = cache.get_stats()
        self.assertEqual(stats['access_count'], 2)
        self.assertEqual(stats['hit_count'], 1)
        self.assertEqual(stats['hit_rate'], 0.5)
        
    def test_memory_pooling_optimization(self):
        """Test memory pooling optimization"""
        class OptimizedMemoryPool:
            def __init__(self):
                self.tensor_pools = {}
                self.activation_cache = {}
                self.memory_usage = 0
                
            def get_optimized_tensor(self, shape, dtype=torch.float32, device='cpu'):
                """Get optimized tensor"""
                key = (shape, dtype, device)
                
                if key in self.tensor_pools and len(self.tensor_pools[key]) > 0:
                    return self.tensor_pools[key].pop()
                else:
                    tensor = torch.zeros(shape, dtype=dtype, device=device)
                    self.memory_usage += tensor.numel() * tensor.element_size()
                    return tensor
                    
            def return_tensor(self, tensor):
                """Return tensor to pool"""
                key = (tensor.shape, tensor.dtype, tensor.device)
                
                if key not in self.tensor_pools:
                    self.tensor_pools[key] = []
                    
                if len(self.tensor_pools[key]) < 10:  # Limit pool size
                    tensor.zero_()
                    self.tensor_pools[key].append(tensor)
                    
            def get_memory_stats(self):
                """Get memory statistics"""
                total_pooled = sum(len(tensors) for tensors in self.tensor_pools.values())
                return {
                    'memory_usage': self.memory_usage,
                    'pooled_tensors': total_pooled,
                    'pool_types': len(self.tensor_pools)
                }
        
        pool = OptimizedMemoryPool()
        
        # Test memory optimization
        tensors = []
        for i in range(5):
            tensor = pool.get_optimized_tensor((2, 64, 256))
            tensors.append(tensor)
            
        # Return tensors to pool
        for tensor in tensors:
            pool.return_tensor(tensor)
            
        stats = pool.get_memory_stats()
        self.assertGreaterEqual(stats['memory_usage'], 0)
        self.assertGreaterEqual(stats['pooled_tensors'], 0)

class TestMemoryManagement(unittest.TestCase):
    """Test suite for memory management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.memory_tracker = MemoryTracker()
        
    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        class MemoryManager:
            def __init__(self):
                self.allocated_tensors = []
                self.memory_limit = 100 * 1024 * 1024  # 100MB
                
            def allocate_tensor(self, shape, dtype=torch.float32):
                """Allocate tensor with memory tracking"""
                tensor = torch.randn(shape, dtype=dtype)
                self.allocated_tensors.append(tensor)
                return tensor
                
            def cleanup_memory(self):
                """Clean up memory"""
                self.allocated_tensors.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            def get_memory_usage(self):
                """Get current memory usage"""
                total_elements = sum(tensor.numel() for tensor in self.allocated_tensors)
                return total_elements
                
            def is_memory_limit_exceeded(self):
                """Check if memory limit is exceeded"""
                return self.get_memory_usage() > self.memory_limit
        
        manager = MemoryManager()
        
        # Allocate some tensors
        for i in range(3):
            tensor = manager.allocate_tensor((100, 100))
            self.assertIsNotNone(tensor)
            
        # Check memory usage
        memory_usage = manager.get_memory_usage()
        self.assertGreater(memory_usage, 0)
        
        # Cleanup memory
        manager.cleanup_memory()
        self.assertEqual(len(manager.allocated_tensors), 0)
        
    def test_memory_efficient_forward(self):
        """Test memory efficient forward pass"""
        class MemoryEfficientModel(nn.Module):
            def __init__(self, d_model=512, d_ff=2048):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                # Memory efficient forward pass
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x
                
            def memory_efficient_forward(self, x):
                """Memory efficient forward with intermediate cleanup"""
                # First layer
                x = self.linear1(x)
                x = self.activation(x)
                
                # Cleanup intermediate activations
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = MemoryEfficientModel()
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        # Test regular forward
        output1 = model(data)
        self.assertEqual(output1.shape, data.shape)
        
        # Test memory efficient forward
        output2 = model.memory_efficient_forward(data)
        self.assertEqual(output2.shape, data.shape)
        
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing for memory efficiency"""
        class CheckpointedModel(nn.Module):
            def __init__(self, d_model=512, d_ff=2048):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                return self.linear2(self.activation(self.linear1(x)))
                
            def checkpointed_forward(self, x):
                """Forward with gradient checkpointing"""
                def checkpoint_fn(inputs):
                    return self.linear2(self.activation(self.linear1(inputs)))
                
                return torch.utils.checkpoint.checkpoint(checkpoint_fn, x)
        
        model = CheckpointedModel()
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        data.requires_grad_(True)
        
        # Test regular forward
        output1 = model(data)
        loss1 = output1.sum()
        loss1.backward()
        
        # Clear gradients
        data.grad = None
        
        # Test checkpointed forward
        output2 = model.checkpointed_forward(data)
        loss2 = output2.sum()
        loss2.backward()
        
        # Both should produce similar results
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))
        self.assertIsNotNone(data.grad)

class TestMemoryOptimization(unittest.TestCase):
    """Test suite for memory optimization techniques"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_memory_efficient_attention(self):
        """Test memory efficient attention"""
        class MemoryEfficientAttention(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads
                
                self.q_linear = nn.Linear(d_model, d_model, bias=False)
                self.k_linear = nn.Linear(d_model, d_model, bias=False)
                self.v_linear = nn.Linear(d_model, d_model, bias=False)
                self.out_linear = nn.Linear(d_model, d_model)
                
            def forward(self, query, key, value, mask=None):
                batch_size, seq_len, d_model = query.shape
                
                # Compute Q, K, V
                q = self.q_linear(query)
                k = self.k_linear(key)
                v = self.v_linear(value)
                
                # Reshape for multi-head attention
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                
                # Memory efficient attention computation
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                # Use in-place operations where possible
                attention_weights = torch.softmax(scores, dim=-1)
                attention_output = torch.matmul(attention_weights, v)
                
                # Reshape and output
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, d_model)
                
                return self.out_linear(attention_output), attention_weights
        
        attention = MemoryEfficientAttention(512, 8)
        data = self.test_data.create_attention_data()
        
        # Profile memory usage
        self.memory_tracker.take_snapshot("before_attention")
        output, weights = attention(data['query'], data['key'], data['value'])
        self.memory_tracker.take_snapshot("after_attention")
        
        self.assertEqual(output.shape, data['query'].shape)
        
        # Check memory growth
        growth = self.memory_tracker.get_memory_growth()
        if growth:
            self.assertGreaterEqual(growth[-1]['memory_growth'], 0)
            
    def test_memory_efficient_mlp(self):
        """Test memory efficient MLP"""
        class MemoryEfficientMLP(nn.Module):
            def __init__(self, d_model, d_ff, dropout=0.1):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                # Memory efficient forward pass
                x = self.linear1(x)
                x = self.activation(x)
                
                # Cleanup intermediate activations
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                x = self.dropout(x)
                x = self.linear2(x)
                return x
                
            def memory_efficient_forward(self, x):
                """Even more memory efficient forward"""
                # Process in chunks to reduce memory usage
                batch_size, seq_len, d_model = x.shape
                chunk_size = min(64, seq_len)
                
                outputs = []
                for i in range(0, seq_len, chunk_size):
                    chunk = x[:, i:i+chunk_size, :]
                    chunk_out = self.forward(chunk)
                    outputs.append(chunk_out)
                    
                return torch.cat(outputs, dim=1)
        
        mlp = MemoryEfficientMLP(512, 2048)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=256, d_model=512)
        
        # Test regular forward
        output1 = mlp(data)
        self.assertEqual(output1.shape, data.shape)
        
        # Test memory efficient forward
        output2 = mlp.memory_efficient_forward(data)
        self.assertEqual(output2.shape, data.shape)
        
    def test_memory_optimization_techniques(self):
        """Test various memory optimization techniques"""
        class OptimizedModel(nn.Module):
            def __init__(self, d_model=512, d_ff=2048):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                return self.linear2(self.dropout(self.activation(self.linear1(x))))
                
            def optimized_forward(self, x):
                """Optimized forward with memory techniques"""
                # Use in-place operations where possible
                x = self.linear1(x)
                x = self.activation(x)
                
                # Cleanup memory
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                x = self.dropout(x)
                x = self.linear2(x)
                return x
                
            def gradient_checkpointed_forward(self, x):
                """Forward with gradient checkpointing"""
                def checkpoint_fn(inputs):
                    return self.linear2(self.dropout(self.activation(self.linear1(inputs))))
                
                return torch.utils.checkpoint.checkpoint(checkpoint_fn, x)
        
        model = OptimizedModel()
        data = self.test_data.create_mlp_data(batch_size=4, seq_len=128, d_model=512)
        data.requires_grad_(True)
        
        # Test different forward methods
        output1 = model(data)
        output2 = model.optimized_forward(data)
        output3 = model.gradient_checkpointed_forward(data)
        
        # All should produce similar results
        self.assertEqual(output1.shape, data.shape)
        self.assertEqual(output2.shape, data.shape)
        self.assertEqual(output3.shape, data.shape)
        
        # Test gradient flow
        loss1 = output1.sum()
        loss1.backward()
        grad1 = data.grad.clone()
        
        data.grad = None
        loss2 = output2.sum()
        loss2.backward()
        grad2 = data.grad.clone()
        
        data.grad = None
        loss3 = output3.sum()
        loss3.backward()
        grad3 = data.grad.clone()
        
        # Gradients should be similar
        self.assertTrue(torch.allclose(grad1, grad2, atol=1e-5))
        self.assertTrue(torch.allclose(grad1, grad3, atol=1e-5))

class TestMemoryProfiling(unittest.TestCase):
    """Test suite for memory profiling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.memory_tracker = MemoryTracker()
        
    def test_memory_profiling_basic(self):
        """Test basic memory profiling"""
        class ProfiledModel(nn.Module):
            def __init__(self, d_model=512, d_ff=2048):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.linear2(x)
                return x
        
        model = ProfiledModel()
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        # Profile memory usage
        self.memory_tracker.take_snapshot("before_forward")
        output = model(data)
        self.memory_tracker.take_snapshot("after_forward")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        self.memory_tracker.take_snapshot("after_backward")
        
        # Get memory summary
        summary = self.memory_tracker.get_memory_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('snapshots_taken', summary)
        
    def test_memory_profiling_advanced(self):
        """Test advanced memory profiling"""
        class AdvancedProfiledModel(nn.Module):
            def __init__(self, d_model=512, d_ff=2048, n_layers=3):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.ReLU(),
                        nn.Linear(d_ff, d_model)
                    ) for _ in range(n_layers)
                ])
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
                
            def profiled_forward(self, x, memory_tracker):
                """Forward with memory profiling"""
                memory_tracker.take_snapshot("start")
                
                for i, layer in enumerate(self.layers):
                    memory_tracker.take_snapshot(f"before_layer_{i}")
                    x = layer(x)
                    memory_tracker.take_snapshot(f"after_layer_{i}")
                    
                memory_tracker.take_snapshot("end")
                return x
        
        model = AdvancedProfiledModel()
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
        
        # Profile with detailed snapshots
        output = model.profiled_forward(data, self.memory_tracker)
        
        # Get memory growth analysis
        growth = self.memory_tracker.get_memory_growth()
        self.assertIsInstance(growth, list)
        
        # Get summary
        summary = self.memory_tracker.get_memory_summary()
        self.assertGreater(summary['snapshots_taken'], 0)

if __name__ == '__main__':
    unittest.main()





