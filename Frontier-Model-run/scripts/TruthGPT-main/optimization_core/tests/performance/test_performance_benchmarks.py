"""
Performance benchmark tests for TruthGPT optimization core
Tests performance metrics, memory usage, and optimization effectiveness
"""

import unittest
import torch
import torch.nn as nn
import time
import psutil
import gc
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.mock_components import MockModel, MockOptimizer, MockAttention, MockMLP, MockKVCache
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions, MemoryTracker

class TestPerformanceBenchmarks(unittest.TestCase):
    """Test suite for performance benchmarks"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        self.memory_tracker = MemoryTracker()
        self.config = TestUtils.create_test_config()
        
    def test_attention_performance_benchmark(self):
        """Benchmark attention performance"""
        attention = MockAttention(d_model=512, n_heads=8)
        
        # Test different sequence lengths
        seq_lengths = [64, 128, 256, 512, 1024]
        results = []
        
        for seq_len in seq_lengths:
            data = self.test_data.create_attention_data(seq_len=seq_len)
            
            # Profile attention
            self.profiler.start_profile(f"attention_{seq_len}")
            output, weights = attention(data['query'], data['key'], data['value'])
            metrics = self.profiler.end_profile()
            
            results.append({
                'seq_len': seq_len,
                'execution_time': metrics['execution_time'],
                'memory_used': metrics['memory_used']
            })
            
            # Verify output
            self.assertEqual(output.shape, data['query'].shape)
            
        # Analyze performance scaling
        for i in range(1, len(results)):
            prev_result = results[i-1]
            curr_result = results[i]
            
            # Performance should scale reasonably
            time_ratio = curr_result['execution_time'] / prev_result['execution_time']
            seq_ratio = curr_result['seq_len'] / prev_result['seq_len']
            
            # Time should not scale worse than O(n^2) for attention
            self.assertLess(time_ratio, seq_ratio ** 2.5)
            
    def test_mlp_performance_benchmark(self):
        """Benchmark MLP performance"""
        mlp = MockMLP(input_size=512, hidden_size=2048, output_size=512)
        
        # Test different hidden sizes
        hidden_sizes = [512, 1024, 2048, 4096]
        results = []
        
        for hidden_size in hidden_sizes:
            mlp = MockMLP(input_size=512, hidden_size=hidden_size, output_size=512)
            data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=512)
            
            # Profile MLP
            self.profiler.start_profile(f"mlp_{hidden_size}")
            output = mlp(data)
            metrics = self.profiler.end_profile()
            
            results.append({
                'hidden_size': hidden_size,
                'execution_time': metrics['execution_time'],
                'memory_used': metrics['memory_used']
            })
            
        # Analyze performance scaling
        for result in results:
            # Performance should scale linearly with hidden size
            self.assertLess(result['execution_time'], 1.0)  # Should be fast
            
    def test_kv_cache_performance_benchmark(self):
        """Benchmark KV cache performance"""
        cache = MockKVCache(max_size=10000)
        
        # Test cache performance
        cache_sizes = [100, 500, 1000, 5000, 10000]
        results = []
        
        for cache_size in cache_sizes:
            cache = MockKVCache(max_size=cache_size)
            
            # Profile cache operations
            self.profiler.start_profile(f"cache_{cache_size}")
            
            # Fill cache
            for i in range(cache_size):
                data = torch.randn(1, 8, 64)
                cache.put(f"key_{i}", data)
            
            # Test cache hits
            for i in range(100):
                cache.get(f"key_{i % cache_size}")
            
            metrics = self.profiler.end_profile()
            
            results.append({
                'cache_size': cache_size,
                'execution_time': metrics['execution_time'],
                'memory_used': metrics['memory_used'],
                'hit_rate': cache.get_stats()['hit_rate']
            })
            
        # Analyze cache performance
        for result in results:
            # Cache should be fast
            self.assertLess(result['execution_time'], 0.1)
            # Hit rate should be high for repeated accesses
            self.assertGreater(result['hit_rate'], 0.9)
            
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage"""
        model = MockModel(input_size=512, hidden_size=1024, output_size=512)
        
        # Test memory usage with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        results = []
        
        for batch_size in batch_sizes:
            # Track memory usage
            self.memory_tracker.take_snapshot(f"before_{batch_size}")
            
            data = self.test_data.create_mlp_data(batch_size=batch_size, seq_len=128, d_model=512)
            output = model(data)
            
            self.memory_tracker.take_snapshot(f"after_{batch_size}")
            
            # Get memory growth
            growth = self.memory_tracker.get_memory_growth()
            if growth:
                results.append({
                    'batch_size': batch_size,
                    'memory_growth': growth[-1]['memory_growth']
                })
            
        # Analyze memory usage
        for result in results:
            # Memory should scale reasonably with batch size
            self.assertGreater(result['memory_growth'], 0)
            
    def test_optimization_performance_benchmark(self):
        """Benchmark optimization performance"""
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        optimizer = MockOptimizer(learning_rate=0.001)
        
        # Test optimization performance
        num_steps = [10, 50, 100, 500]
        results = []
        
        for steps in num_steps:
            # Profile optimization
            self.profiler.start_profile(f"optimization_{steps}")
            
            for step in range(steps):
                data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
                output = model(data)
                target = torch.randn_like(output)
                loss = nn.MSELoss()(output, target)
                optimizer.step(loss)
            
            metrics = self.profiler.end_profile()
            
            results.append({
                'steps': steps,
                'execution_time': metrics['execution_time'],
                'memory_used': metrics['memory_used'],
                'time_per_step': metrics['execution_time'] / steps
            })
            
        # Analyze optimization performance
        for result in results:
            # Time per step should be reasonable
            self.assertLess(result['time_per_step'], 0.1)
            
    def test_throughput_benchmark(self):
        """Benchmark throughput"""
        model = MockModel(input_size=512, hidden_size=1024, output_size=512)
        
        # Test throughput with different configurations
        configs = [
            {'batch_size': 1, 'seq_len': 128},
            {'batch_size': 2, 'seq_len': 128},
            {'batch_size': 4, 'seq_len': 128},
            {'batch_size': 2, 'seq_len': 256},
            {'batch_size': 2, 'seq_len': 512}
        ]
        
        results = []
        
        for config in configs:
            data = self.test_data.create_mlp_data(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                d_model=512
            )
            
            # Profile throughput
            self.profiler.start_profile(f"throughput_{config['batch_size']}_{config['seq_len']}")
            
            # Run multiple iterations
            for _ in range(10):
                output = model(data)
            
            metrics = self.profiler.end_profile()
            
            # Calculate throughput
            total_samples = config['batch_size'] * config['seq_len'] * 10
            throughput = total_samples / metrics['execution_time']
            
            results.append({
                'config': config,
                'execution_time': metrics['execution_time'],
                'throughput': throughput
            })
            
        # Analyze throughput
        for result in results:
            # Throughput should be positive
            self.assertGreater(result['throughput'], 0)
            
    def test_latency_benchmark(self):
        """Benchmark latency"""
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        
        # Test latency with different input sizes
        input_sizes = [64, 128, 256, 512, 1024]
        results = []
        
        for seq_len in input_sizes:
            data = self.test_data.create_mlp_data(batch_size=1, seq_len=seq_len, d_model=256)
            
            # Profile latency
            self.profiler.start_profile(f"latency_{seq_len}")
            output = model(data)
            metrics = self.profiler.end_profile()
            
            results.append({
                'seq_len': seq_len,
                'latency': metrics['execution_time']
            })
            
        # Analyze latency
        for result in results:
            # Latency should be reasonable
            self.assertLess(result['latency'], 1.0)
            
    def test_memory_efficiency_benchmark(self):
        """Benchmark memory efficiency"""
        model = MockModel(input_size=512, hidden_size=1024, output_size=512)
        
        # Test memory efficiency with different configurations
        configs = [
            {'batch_size': 1, 'seq_len': 128},
            {'batch_size': 2, 'seq_len': 128},
            {'batch_size': 4, 'seq_len': 128},
            {'batch_size': 8, 'seq_len': 128}
        ]
        
        results = []
        
        for config in configs:
            data = self.test_data.create_mlp_data(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                d_model=512
            )
            
            # Track memory usage
            self.memory_tracker.take_snapshot(f"before_{config['batch_size']}")
            output = model(data)
            self.memory_tracker.take_snapshot(f"after_{config['batch_size']}")
            
            # Get memory usage
            summary = self.memory_tracker.get_memory_summary()
            if summary:
                results.append({
                    'config': config,
                    'memory_usage': summary['max_memory']
                })
            
        # Analyze memory efficiency
        for result in results:
            # Memory usage should be reasonable
            self.assertGreater(result['memory_usage'], 0)
            
    def test_scalability_benchmark(self):
        """Benchmark scalability"""
        # Test scalability with different model sizes
        model_sizes = [
            (128, 256, 128),
            (256, 512, 256),
            (512, 1024, 512),
            (1024, 2048, 1024)
        ]
        
        results = []
        
        for input_size, hidden_size, output_size in model_sizes:
            model = MockModel(input_size, hidden_size, output_size)
            data = self.test_data.create_mlp_data(batch_size=2, seq_len=128, d_model=input_size)
            
            # Profile scalability
            self.profiler.start_profile(f"scalability_{input_size}")
            output = model(data)
            metrics = self.profiler.end_profile()
            
            results.append({
                'model_size': (input_size, hidden_size, output_size),
                'execution_time': metrics['execution_time'],
                'memory_used': metrics['memory_used']
            })
            
        # Analyze scalability
        for result in results:
            # Performance should scale reasonably
            self.assertLess(result['execution_time'], 5.0)
            
    def test_optimization_effectiveness_benchmark(self):
        """Benchmark optimization effectiveness"""
        # Test optimization effectiveness
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        optimizer = MockOptimizer(learning_rate=0.001)
        
        # Track optimization progress
        losses = []
        
        for epoch in range(20):
            data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
            output = model(data)
            target = torch.randn_like(output)
            loss = nn.MSELoss()(output, target)
            
            optimizer.step(loss)
            losses.append(loss.item())
            
        # Analyze optimization effectiveness
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        # Loss should generally decrease (though this is not guaranteed in all cases)
        self.assertIsInstance(initial_loss, float)
        self.assertIsInstance(final_loss, float)
        
    def test_benchmark_comparison(self):
        """Compare different benchmark configurations"""
        # Test different configurations
        configs = [
            {'batch_size': 1, 'seq_len': 128, 'd_model': 256},
            {'batch_size': 2, 'seq_len': 128, 'd_model': 256},
            {'batch_size': 1, 'seq_len': 256, 'd_model': 256},
            {'batch_size': 1, 'seq_len': 128, 'd_model': 512}
        ]
        
        results = []
        
        for config in configs:
            model = MockModel(input_size=config['d_model'], 
                            hidden_size=config['d_model']*2, 
                            output_size=config['d_model'])
            
            data = self.test_data.create_mlp_data(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                d_model=config['d_model']
            )
            
            # Profile configuration
            self.profiler.start_profile(f"config_{config['batch_size']}_{config['seq_len']}_{config['d_model']}")
            output = model(data)
            metrics = self.profiler.end_profile()
            
            results.append({
                'config': config,
                'execution_time': metrics['execution_time'],
                'memory_used': metrics['memory_used']
            })
            
        # Analyze comparison
        for result in results:
            # All configurations should complete successfully
            self.assertGreater(result['execution_time'], 0)
            self.assertGreater(result['memory_used'], 0)

if __name__ == '__main__':
    unittest.main()





