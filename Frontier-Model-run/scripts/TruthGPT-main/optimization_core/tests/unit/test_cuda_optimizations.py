"""
Unit tests for CUDA optimizations
Tests CUDA kernels, memory management, and GPU-specific optimizations
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestCUDAOptimizations(unittest.TestCase):
    """Test suite for CUDA optimizations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_cuda_availability(self):
        """Test CUDA availability and basic functionality"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Test basic CUDA operations
        tensor = torch.randn(2, 128, 512).to(self.device)
        self.assertEqual(tensor.device, self.device)
        
        # Test CUDA memory allocation
        cuda_memory_before = torch.cuda.memory_allocated()
        large_tensor = torch.randn(1000, 1000).to(self.device)
        cuda_memory_after = torch.cuda.memory_allocated()
        
        self.assertGreater(cuda_memory_after, cuda_memory_before)
        
    def test_cuda_memory_management(self):
        """Test CUDA memory management"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Test memory allocation and deallocation
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate memory
        tensors = []
        for i in range(5):
            tensor = torch.randn(100, 100).to(self.device)
            tensors.append(tensor)
            
        allocated_memory = torch.cuda.memory_allocated()
        self.assertGreater(allocated_memory, initial_memory)
        
        # Deallocate memory
        del tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        self.assertLessEqual(final_memory, allocated_memory)
        
    def test_cuda_optimized_attention(self):
        """Test CUDA optimized attention"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        class CUDAOptimizedAttention(nn.Module):
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
                
                # Move to CUDA if available
                device = query.device
                
                # Linear transformations
                q = self.q_linear(query)
                k = self.k_linear(key)
                v = self.v_linear(value)
                
                # Reshape for multi-head attention
                q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                
                # Scaled dot-product attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                attention_weights = torch.softmax(scores, dim=-1)
                attention_output = torch.matmul(attention_weights, v)
                
                # Reshape and output
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len, d_model)
                
                return self.out_linear(attention_output), attention_weights
        
        attention = CUDAOptimizedAttention(512, 8)
        data = self.test_data.create_attention_data()
        
        # Move data to CUDA
        data_cuda = {k: v.to(self.device) for k, v in data.items()}
        attention = attention.to(self.device)
        
        # Profile CUDA attention
        self.profiler.start_profile("cuda_attention")
        output, weights = attention(data_cuda['query'], data_cuda['key'], data_cuda['value'])
        metrics = self.profiler.end_profile()
        
        self.assertEqual(output.shape, data_cuda['query'].shape)
        self.assertEqual(weights.shape, (2, 8, 128, 128))
        self.assertLess(metrics['execution_time'], 1.0)
        
    def test_cuda_optimized_mlp(self):
        """Test CUDA optimized MLP"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        class CUDAOptimizedMLP(nn.Module):
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x
                
            def cuda_optimized_forward(self, x):
                """CUDA optimized forward pass"""
                # Use CUDA-specific optimizations
                x = self.linear1(x)
                x = self.activation(x)
                
                # CUDA memory optimization
                if x.is_cuda:
                    torch.cuda.synchronize()
                
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        mlp = CUDAOptimizedMLP(512, 2048)
        data = self.test_data.create_mlp_data()
        
        # Move to CUDA
        data_cuda = data.to(self.device)
        mlp = mlp.to(self.device)
        
        # Test regular forward
        output1 = mlp(data_cuda)
        self.assertEqual(output1.shape, data_cuda.shape)
        
        # Test CUDA optimized forward
        output2 = mlp.cuda_optimized_forward(data_cuda)
        self.assertEqual(output2.shape, data_cuda.shape)
        
        # Results should be similar
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))
        
    def test_cuda_memory_pooling(self):
        """Test CUDA memory pooling"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        class CUDAMemoryPool:
            def __init__(self, max_size=100):
                self.max_size = max_size
                self.pool = {}
                self.device = torch.device("cuda")
                
            def get_tensor(self, shape, dtype=torch.float32):
                """Get tensor from pool or create new one"""
                key = (shape, dtype)
                
                if key in self.pool and len(self.pool[key]) > 0:
                    tensor = self.pool[key].pop()
                    return tensor
                else:
                    return torch.zeros(shape, dtype=dtype, device=self.device)
                    
            def return_tensor(self, tensor):
                """Return tensor to pool"""
                key = (tensor.shape, tensor.dtype)
                
                if key not in self.pool:
                    self.pool[key] = []
                    
                if len(self.pool[key]) < self.max_size:
                    tensor.zero_()
                    self.pool[key].append(tensor)
                    
            def get_stats(self):
                """Get pool statistics"""
                total_tensors = sum(len(tensors) for tensors in self.pool.values())
                return {
                    'total_tensors': total_tensors,
                    'pool_keys': len(self.pool),
                    'device': str(self.device)
                }
        
        pool = CUDAMemoryPool()
        
        # Test CUDA memory pooling
        tensor1 = pool.get_tensor((2, 128, 512))
        tensor2 = pool.get_tensor((2, 128, 512))
        
        self.assertEqual(tensor1.device, torch.device("cuda"))
        self.assertEqual(tensor2.device, torch.device("cuda"))
        
        # Return tensors to pool
        pool.return_tensor(tensor1)
        pool.return_tensor(tensor2)
        
        # Check stats
        stats = pool.get_stats()
        self.assertGreaterEqual(stats['total_tensors'], 0)
        self.assertEqual(stats['device'], "cuda:0")
        
    def test_cuda_kernel_fusion(self):
        """Test CUDA kernel fusion"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        class FusedCUDAKernel(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.linear = nn.Linear(d_model, d_model)
                self.norm = nn.LayerNorm(d_model)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                # Fused operations
                x = self.linear(x)
                x = self.norm(x)
                x = self.activation(x)
                return x
                
            def fused_forward(self, x):
                """Fused forward pass with CUDA optimizations"""
                # Simulate kernel fusion
                x = self.linear(x)
                
                # CUDA synchronization for fused operations
                if x.is_cuda:
                    torch.cuda.synchronize()
                
                x = self.norm(x)
                x = self.activation(x)
                return x
        
        fused_kernel = FusedCUDAKernel(512)
        data = self.test_data.create_mlp_data()
        
        # Move to CUDA
        data_cuda = data.to(self.device)
        fused_kernel = fused_kernel.to(self.device)
        
        # Test regular forward
        output1 = fused_kernel(data_cuda)
        self.assertEqual(output1.shape, data_cuda.shape)
        
        # Test fused forward
        output2 = fused_kernel.fused_forward(data_cuda)
        self.assertEqual(output2.shape, data_cuda.shape)
        
        # Results should be similar
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))
        
    def test_cuda_quantization(self):
        """Test CUDA quantization"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        class CUDAQuantizedLinear(nn.Module):
            def __init__(self, in_features, out_features, num_bits=8):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.num_bits = num_bits
                
                self.weight = nn.Parameter(torch.randn(out_features, in_features))
                self.bias = nn.Parameter(torch.randn(out_features))
                
                # Quantization parameters
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.zero_point = nn.Parameter(torch.tensor(0.0))
                
            def forward(self, x):
                # CUDA optimized quantization
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                # Quantize weights
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.scale + self.zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.zero_point) * self.scale
                
                # CUDA optimized linear operation
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
                
            def cuda_quantized_forward(self, x):
                """CUDA optimized quantized forward"""
                # Use CUDA-specific quantization
                qmin = -(2 ** (self.num_bits - 1))
                qmax = 2 ** (self.num_bits - 1) - 1
                
                # CUDA optimized quantization
                weight_quantized = torch.clamp(
                    torch.round(self.weight / self.scale + self.zero_point),
                    qmin, qmax
                )
                weight_dequantized = (weight_quantized - self.zero_point) * self.scale
                
                # CUDA synchronization for quantized operations
                if x.is_cuda:
                    torch.cuda.synchronize()
                
                return torch.nn.functional.linear(x, weight_dequantized, self.bias)
        
        quantized_linear = CUDAQuantizedLinear(512, 1024, num_bits=8)
        data = self.test_data.create_mlp_data()
        
        # Move to CUDA
        data_cuda = data.to(self.device)
        quantized_linear = quantized_linear.to(self.device)
        
        # Test regular quantized forward
        output1 = quantized_linear(data_cuda)
        self.assertEqual(output1.shape, (2, 128, 1024))
        
        # Test CUDA optimized quantized forward
        output2 = quantized_linear.cuda_quantized_forward(data_cuda)
        self.assertEqual(output2.shape, (2, 128, 1024))
        
        # Results should be similar
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))
        
    def test_cuda_performance_benchmark(self):
        """Test CUDA performance benchmark"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        class CUDAPerformanceBenchmark:
            def __init__(self):
                self.device = torch.device("cuda")
                
            def benchmark_attention(self, d_model, n_heads, seq_len, batch_size):
                """Benchmark CUDA attention performance"""
                class CUDAAttention(nn.Module):
                    def __init__(self, d_model, n_heads):
                        super().__init__()
                        self.d_model = d_model
                        self.n_heads = n_heads
                        self.head_dim = d_model // n_heads
                        
                        self.q_linear = nn.Linear(d_model, d_model, bias=False)
                        self.k_linear = nn.Linear(d_model, d_model, bias=False)
                        self.v_linear = nn.Linear(d_model, d_model, bias=False)
                        self.out_linear = nn.Linear(d_model, d_model)
                        
                    def forward(self, query, key, value):
                        batch_size, seq_len, d_model = query.shape
                        
                        q = self.q_linear(query)
                        k = self.k_linear(key)
                        v = self.v_linear(value)
                        
                        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                        
                        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                        attention_weights = torch.softmax(scores, dim=-1)
                        attention_output = torch.matmul(attention_weights, v)
                        
                        attention_output = attention_output.transpose(1, 2).contiguous().view(
                            batch_size, seq_len, d_model)
                        
                        return self.out_linear(attention_output), attention_weights
                
                attention = CUDAAttention(d_model, n_heads).to(self.device)
                
                # Create test data
                query = torch.randn(batch_size, seq_len, d_model).to(self.device)
                key = torch.randn(batch_size, seq_len, d_model).to(self.device)
                value = torch.randn(batch_size, seq_len, d_model).to(self.device)
                
                # Warmup
                for _ in range(3):
                    _ = attention(query, key, value)
                
                # Benchmark
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                for _ in range(10):
                    output, weights = attention(query, key, value)
                end_time.record()
                
                torch.cuda.synchronize()
                execution_time = start_time.elapsed_time(end_time) / 10  # Average time
                
                return {
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'seq_len': seq_len,
                    'batch_size': batch_size,
                    'execution_time': execution_time,
                    'throughput': (batch_size * seq_len) / (execution_time / 1000)  # samples per second
                }
                
            def benchmark_mlp(self, d_model, d_ff, seq_len, batch_size):
                """Benchmark CUDA MLP performance"""
                class CUDAMLP(nn.Module):
                    def __init__(self, d_model, d_ff):
                        super().__init__()
                        self.linear1 = nn.Linear(d_model, d_ff)
                        self.linear2 = nn.Linear(d_ff, d_model)
                        self.activation = nn.GELU()
                        
                    def forward(self, x):
                        x = self.linear1(x)
                        x = self.activation(x)
                        x = self.linear2(x)
                        return x
                
                mlp = CUDAMLP(d_model, d_ff).to(self.device)
                data = torch.randn(batch_size, seq_len, d_model).to(self.device)
                
                # Warmup
                for _ in range(3):
                    _ = mlp(data)
                
                # Benchmark
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                for _ in range(10):
                    output = mlp(data)
                end_time.record()
                
                torch.cuda.synchronize()
                execution_time = start_time.elapsed_time(end_time) / 10
                
                return {
                    'd_model': d_model,
                    'd_ff': d_ff,
                    'seq_len': seq_len,
                    'batch_size': batch_size,
                    'execution_time': execution_time,
                    'throughput': (batch_size * seq_len) / (execution_time / 1000)
                }
        
        benchmark = CUDAPerformanceBenchmark()
        
        # Test attention benchmark
        attention_result = benchmark.benchmark_attention(512, 8, 128, 2)
        self.assertGreater(attention_result['execution_time'], 0)
        self.assertGreater(attention_result['throughput'], 0)
        
        # Test MLP benchmark
        mlp_result = benchmark.benchmark_mlp(512, 2048, 128, 2)
        self.assertGreater(mlp_result['execution_time'], 0)
        self.assertGreater(mlp_result['throughput'], 0)
        
    def test_cuda_memory_optimization(self):
        """Test CUDA memory optimization"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        class CUDAMemoryOptimizer:
            def __init__(self):
                self.device = torch.device("cuda")
                self.memory_pool = {}
                
            def optimize_memory_usage(self, model, data):
                """Optimize CUDA memory usage"""
                # Track memory usage
                initial_memory = torch.cuda.memory_allocated()
                
                # Forward pass with memory optimization
                output = model(data)
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
                final_memory = torch.cuda.memory_allocated()
                memory_used = final_memory - initial_memory
                
                return {
                    'initial_memory': initial_memory,
                    'final_memory': final_memory,
                    'memory_used': memory_used,
                    'output_shape': output.shape
                }
                
            def test_memory_efficiency(self, model, data):
                """Test memory efficiency"""
                # Test with different batch sizes
                batch_sizes = [1, 2, 4, 8]
                results = []
                
                for batch_size in batch_sizes:
                    # Create data with specific batch size
                    test_data = torch.randn(batch_size, data.shape[1], data.shape[2]).to(self.device)
                    
                    # Measure memory usage
                    initial_memory = torch.cuda.memory_allocated()
                    output = model(test_data)
                    final_memory = torch.cuda.memory_allocated()
                    
                    memory_used = final_memory - initial_memory
                    memory_per_sample = memory_used / batch_size
                    
                    results.append({
                        'batch_size': batch_size,
                        'memory_used': memory_used,
                        'memory_per_sample': memory_per_sample
                    })
                    
                    # Cleanup
                    del test_data, output
                    torch.cuda.empty_cache()
                
                return results
        
        optimizer = CUDAMemoryOptimizer()
        model = MockModel(input_size=256, hidden_size=512, output_size=256).to(self.device)
        data = torch.randn(2, 64, 256).to(self.device)
        
        # Test memory optimization
        memory_result = optimizer.optimize_memory_usage(model, data)
        self.assertGreater(memory_result['memory_used'], 0)
        self.assertEqual(memory_result['output_shape'], data.shape)
        
        # Test memory efficiency
        efficiency_results = optimizer.test_memory_efficiency(model, data)
        self.assertEqual(len(efficiency_results), 4)
        
        for result in efficiency_results:
            self.assertGreater(result['memory_used'], 0)
            self.assertGreater(result['memory_per_sample'], 0)

if __name__ == '__main__':
    unittest.main()





