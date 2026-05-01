"""
Advanced performance benchmark tests for TruthGPT optimization core
Tests advanced performance metrics, scalability, and optimization effectiveness
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
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, MemoryTracker, TestAssertions

class TestAdvancedPerformanceBenchmarks(unittest.TestCase):
    """Test suite for advanced performance benchmarks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        self.memory_tracker = MemoryTracker()
        
    def test_scalability_benchmark(self):
        """Test scalability with different model sizes"""
        model_sizes = [
            (128, 256, 128),   # Small
            (256, 512, 256),   # Medium
            (512, 1024, 512),  # Large
            (1024, 2048, 1024) # Extra Large
        ]
        
        results = []
        
        for input_size, hidden_size, output_size in model_sizes:
            with self.subTest(size=(input_size, hidden_size, output_size)):
                model = MockModel(input_size, hidden_size, output_size)
                data = self.test_data.create_mlp_data(
                    batch_size=2, seq_len=64, d_model=input_size
                )
                
                # Profile performance
                self.profiler.start_profile(f"scalability_{input_size}")
                output = model(data)
                metrics = self.profiler.end_profile()
                
                # Calculate theoretical complexity
                param_count = input_size * hidden_size + hidden_size * output_size
                flops = 2 * param_count * data.numel()  # Rough FLOP estimate
                
                results.append({
                    'model_size': (input_size, hidden_size, output_size),
                    'param_count': param_count,
                    'flops': flops,
                    'execution_time': metrics['execution_time'],
                    'memory_used': metrics['memory_used'],
                    'efficiency': flops / metrics['execution_time'] if metrics['execution_time'] > 0 else 0
                })
                
                self.assertEqual(output.shape, data.shape)
        
        # Analyze scalability
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            
            # Performance should scale reasonably
            time_ratio = curr['execution_time'] / prev['execution_time']
            param_ratio = curr['param_count'] / prev['param_count']
            
            # Time should not scale worse than O(n^2)
            self.assertLess(time_ratio, param_ratio ** 2)
            
    def test_memory_efficiency_benchmark(self):
        """Test memory efficiency with different configurations"""
        configurations = [
            {'batch_size': 1, 'seq_len': 64, 'd_model': 256},
            {'batch_size': 2, 'seq_len': 64, 'd_model': 256},
            {'batch_size': 4, 'seq_len': 64, 'd_model': 256},
            {'batch_size': 2, 'seq_len': 128, 'd_model': 256},
            {'batch_size': 2, 'seq_len': 64, 'd_model': 512},
        ]
        
        results = []
        
        for config in configurations:
            with self.subTest(config=config):
                model = MockModel(
                    input_size=config['d_model'],
                    hidden_size=config['d_model'] * 2,
                    output_size=config['d_model']
                )
                
                data = self.test_data.create_mlp_data(
                    batch_size=config['batch_size'],
                    seq_len=config['seq_len'],
                    d_model=config['d_model']
                )
                
                # Track memory usage
                self.memory_tracker.take_snapshot("before_forward")
                output = model(data)
                self.memory_tracker.take_snapshot("after_forward")
                
                # Calculate memory efficiency
                total_elements = data.numel()
                memory_growth = self.memory_tracker.get_memory_growth()
                memory_used = memory_growth[-1]['memory_growth'] if memory_growth else 0
                
                memory_efficiency = total_elements / memory_used if memory_used > 0 else 0
                
                results.append({
                    'config': config,
                    'total_elements': total_elements,
                    'memory_used': memory_used,
                    'memory_efficiency': memory_efficiency
                })
                
                self.assertEqual(output.shape, data.shape)
        
        # Analyze memory efficiency
        for result in results:
            self.assertGreater(result['total_elements'], 0)
            self.assertGreaterEqual(result['memory_used'], 0)
            
    def test_throughput_benchmark(self):
        """Test throughput with different batch sizes"""
        batch_sizes = [1, 2, 4, 8, 16]
        results = []
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                model = MockModel(input_size=256, hidden_size=512, output_size=256)
                data = self.test_data.create_mlp_data(
                    batch_size=batch_size, seq_len=64, d_model=256
                )
                
                # Measure throughput
                start_time = time.time()
                iterations = 10
                
                for _ in range(iterations):
                    output = model(data)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Calculate throughput
                total_samples = batch_size * 64 * iterations
                throughput = total_samples / total_time
                
                results.append({
                    'batch_size': batch_size,
                    'total_samples': total_samples,
                    'total_time': total_time,
                    'throughput': throughput,
                    'samples_per_second': throughput
                })
                
                self.assertEqual(output.shape, data.shape)
        
        # Analyze throughput scaling
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            
            # Throughput should generally increase with batch size
            # (though not necessarily linearly due to memory constraints)
            self.assertGreater(curr['throughput'], 0)
            
    def test_latency_benchmark(self):
        """Test latency with different sequence lengths"""
        seq_lengths = [32, 64, 128, 256, 512]
        results = []
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                model = MockModel(input_size=256, hidden_size=512, output_size=256)
                data = self.test_data.create_mlp_data(
                    batch_size=1, seq_len=seq_len, d_model=256
                )
                
                # Measure latency
                latencies = []
                for _ in range(5):  # Multiple measurements
                    start_time = time.time()
                    output = model(data)
                    end_time = time.time()
                    latencies.append(end_time - start_time)
                
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                results.append({
                    'seq_len': seq_len,
                    'avg_latency': avg_latency,
                    'min_latency': min_latency,
                    'max_latency': max_latency,
                    'latency_std': (sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)) ** 0.5
                })
                
                self.assertEqual(output.shape, data.shape)
        
        # Analyze latency scaling
        for result in results:
            self.assertGreater(result['avg_latency'], 0)
            self.assertLessEqual(result['min_latency'], result['avg_latency'])
            self.assertGreaterEqual(result['max_latency'], result['avg_latency'])
            
    def test_optimization_effectiveness_benchmark(self):
        """Test optimization effectiveness"""
        class OptimizationEffectivenessTest:
            def __init__(self):
                self.baseline_model = MockModel(input_size=256, hidden_size=512, output_size=256)
                self.optimized_model = MockModel(input_size=256, hidden_size=512, output_size=256)
                self.optimizer = MockOptimizer(learning_rate=0.001)
                
            def run_baseline(self, data, target):
                """Run baseline model"""
                self.profiler.start_profile("baseline")
                output = self.baseline_model(data)
                loss = nn.MSELoss()(output, target)
                metrics = self.profiler.end_profile()
                
                return {
                    'output': output,
                    'loss': loss.item(),
                    'execution_time': metrics['execution_time'],
                    'memory_used': metrics['memory_used']
                }
                
            def run_optimized(self, data, target):
                """Run optimized model"""
                self.profiler.start_profile("optimized")
                output = self.optimized_model(data)
                loss = nn.MSELoss()(output, target)
                
                # Apply optimization
                self.optimizer.step(loss)
                
                metrics = self.profiler.end_profile()
                
                return {
                    'output': output,
                    'loss': loss.item(),
                    'execution_time': metrics['execution_time'],
                    'memory_used': metrics['memory_used']
                }
                
            def compare_effectiveness(self, data, target):
                """Compare baseline vs optimized"""
                baseline_results = self.run_baseline(data, target)
                optimized_results = self.run_optimized(data, target)
                
                # Calculate improvements
                time_improvement = baseline_results['execution_time'] / optimized_results['execution_time']
                memory_improvement = baseline_results['memory_used'] / optimized_results['memory_used'] if optimized_results['memory_used'] > 0 else 1
                
                return {
                    'baseline': baseline_results,
                    'optimized': optimized_results,
                    'time_improvement': time_improvement,
                    'memory_improvement': memory_improvement,
                    'overall_improvement': (time_improvement + memory_improvement) / 2
                }
        
        # Test optimization effectiveness
        test = OptimizationEffectivenessTest()
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        comparison = test.compare_effectiveness(data, target)
        
        # Verify results
        self.assertIsInstance(comparison['time_improvement'], float)
        self.assertIsInstance(comparison['memory_improvement'], float)
        self.assertIsInstance(comparison['overall_improvement'], float)
        
    def test_advanced_attention_benchmark(self):
        """Test advanced attention performance"""
        attention_configs = [
            {'d_model': 256, 'n_heads': 4, 'seq_len': 64},
            {'d_model': 512, 'n_heads': 8, 'seq_len': 128},
            {'d_model': 1024, 'n_heads': 16, 'seq_len': 256},
        ]
        
        results = []
        
        for config in attention_configs:
            with self.subTest(config=config):
                attention = MockAttention(d_model=config['d_model'], n_heads=config['n_heads'])
                data = self.test_data.create_attention_data(
                    batch_size=2, seq_len=config['seq_len'], d_model=config['d_model']
                )
                
                # Profile attention
                self.profiler.start_profile(f"attention_{config['d_model']}")
                output, weights = attention(data['query'], data['key'], data['value'])
                metrics = self.profiler.end_profile()
                
                # Calculate attention complexity
                seq_len = config['seq_len']
                d_model = config['d_model']
                n_heads = config['n_heads']
                
                # Attention complexity is O(seq_len^2 * d_model)
                attention_complexity = seq_len * seq_len * d_model
                
                results.append({
                    'config': config,
                    'attention_complexity': attention_complexity,
                    'execution_time': metrics['execution_time'],
                    'memory_used': metrics['memory_used'],
                    'efficiency': attention_complexity / metrics['execution_time'] if metrics['execution_time'] > 0 else 0
                })
                
                self.assertEqual(output.shape, data['query'].shape)
                self.assertEqual(weights.shape, (2, n_heads, seq_len, seq_len))
        
        # Analyze attention performance
        for result in results:
            self.assertGreater(result['attention_complexity'], 0)
            self.assertGreater(result['execution_time'], 0)
            self.assertGreater(result['efficiency'], 0)
            
    def test_memory_optimization_benchmark(self):
        """Test memory optimization techniques"""
        class MemoryOptimizationTest:
            def __init__(self):
                self.memory_tracker = MemoryTracker()
                
            def test_memory_pooling(self, model, data, iterations=10):
                """Test memory pooling effectiveness"""
                # Without pooling
                self.memory_tracker.take_snapshot("before_pooling_test")
                
                for i in range(iterations):
                    output = model(data)
                    if i % 2 == 0:  # Simulate memory cleanup
                        gc.collect()
                
                self.memory_tracker.take_snapshot("after_pooling_test")
                
                # Get memory growth
                growth = self.memory_tracker.get_memory_growth()
                memory_used = growth[-1]['memory_growth'] if growth else 0
                
                return {
                    'iterations': iterations,
                    'memory_used': memory_used,
                    'memory_per_iteration': memory_used / iterations
                }
                
            def test_memory_efficiency(self, model, data):
                """Test memory efficiency"""
                # Track memory usage
                self.memory_tracker.take_snapshot("before_forward")
                output = model(data)
                self.memory_tracker.take_snapshot("after_forward")
                
                # Calculate memory efficiency
                input_size = data.numel() * data.element_size()
                growth = self.memory_tracker.get_memory_growth()
                memory_growth = growth[-1]['memory_growth'] if growth else 0
                
                efficiency = input_size / memory_growth if memory_growth > 0 else float('inf')
                
                return {
                    'input_size': input_size,
                    'memory_growth': memory_growth,
                    'efficiency': efficiency
                }
        
        # Test memory optimization
        memory_test = MemoryOptimizationTest()
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        
        # Test memory pooling
        pooling_results = memory_test.test_memory_pooling(model, data)
        self.assertGreater(pooling_results['iterations'], 0)
        self.assertGreaterEqual(pooling_results['memory_used'], 0)
        
        # Test memory efficiency
        efficiency_results = memory_test.test_memory_efficiency(model, data)
        self.assertGreater(efficiency_results['input_size'], 0)
        self.assertGreaterEqual(efficiency_results['memory_growth'], 0)
        
    def test_quantization_benchmark(self):
        """Test quantization performance"""
        class QuantizationBenchmark:
            def __init__(self):
                self.bit_widths = [4, 6, 8, 16, 32]
                
            def test_quantization_performance(self, model, data, bit_width):
                """Test quantization performance for specific bit width"""
                # Simulate quantization
                class QuantizedModel(nn.Module):
                    def __init__(self, original_model, num_bits):
                        super().__init__()
                        self.original_model = original_model
                        self.num_bits = num_bits
                        
                    def forward(self, x):
                        # Simulate quantized forward pass
                        output = self.original_model(x)
                        
                        # Simulate quantization noise
                        noise_scale = 1.0 / (2 ** self.num_bits)
                        noise = torch.randn_like(output) * noise_scale
                        return output + noise
                
                quantized_model = QuantizedModel(model, bit_width)
                
                # Profile performance
                profiler = PerformanceProfiler()
                profiler.start_profile(f"quantized_{bit_width}bit")
                output = quantized_model(data)
                metrics = profiler.end_profile()
                
                # Calculate quantization error
                original_output = model(data)
                quantization_error = torch.mean((output - original_output) ** 2).item()
                
                return {
                    'bit_width': bit_width,
                    'execution_time': metrics['execution_time'],
                    'memory_used': metrics['memory_used'],
                    'quantization_error': quantization_error
                }
                
            def run_quantization_benchmark(self, model, data):
                """Run complete quantization benchmark"""
                results = []
                
                for bit_width in self.bit_widths:
                    result = self.test_quantization_performance(model, data, bit_width)
                    results.append(result)
                
                return results
        
        # Test quantization benchmark
        quant_benchmark = QuantizationBenchmark()
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        
        results = quant_benchmark.run_quantization_benchmark(model, data)
        
        # Analyze quantization results
        self.assertEqual(len(results), len(quant_benchmark.bit_widths))
        
        for result in results:
            self.assertIn('bit_width', result)
            self.assertGreater(result['execution_time'], 0)
            self.assertGreaterEqual(result['memory_used'], 0)
            self.assertGreaterEqual(result['quantization_error'], 0)
            
        # Verify quantization error decreases with higher bit widths
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            self.assertLessEqual(curr['quantization_error'], prev['quantization_error'])

class TestAdvancedScalabilityBenchmarks(unittest.TestCase):
    """Test suite for advanced scalability benchmarks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_multi_gpu_scalability(self):
        """Test multi-GPU scalability (simulated)"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Simulate multi-GPU scenarios
        gpu_configs = [
            {'devices': 1, 'batch_size': 2},
            {'devices': 2, 'batch_size': 4},
            {'devices': 4, 'batch_size': 8},
        ]
        
        results = []
        
        for config in gpu_configs:
            with self.subTest(config=config):
                # Simulate multi-GPU processing
                model = MockModel(input_size=256, hidden_size=512, output_size=256)
                data = self.test_data.create_mlp_data(
                    batch_size=config['batch_size'], seq_len=64, d_model=256
                )
                
                # Profile performance
                self.profiler.start_profile(f"multi_gpu_{config['devices']}")
                output = model(data)
                metrics = self.profiler.end_profile()
                
                # Calculate theoretical speedup
                theoretical_speedup = config['devices']
                actual_speedup = 1.0  # Simplified calculation
                
                results.append({
                    'config': config,
                    'execution_time': metrics['execution_time'],
                    'memory_used': metrics['memory_used'],
                    'theoretical_speedup': theoretical_speedup,
                    'actual_speedup': actual_speedup,
                    'efficiency': actual_speedup / theoretical_speedup
                })
                
                self.assertEqual(output.shape, data.shape)
        
        # Analyze multi-GPU performance
        for result in results:
            self.assertGreater(result['execution_time'], 0)
            self.assertGreater(result['theoretical_speedup'], 0)
            self.assertGreaterEqual(result['efficiency'], 0)
            
    def test_distributed_training_scalability(self):
        """Test distributed training scalability"""
        # Simulate distributed training scenarios
        worker_configs = [
            {'workers': 1, 'batch_size': 8},
            {'workers': 2, 'batch_size': 16},
            {'workers': 4, 'batch_size': 32},
        ]
        
        results = []
        
        for config in worker_configs:
            with self.subTest(config=config):
                # Simulate distributed training
                model = MockModel(input_size=256, hidden_size=512, output_size=256)
                optimizer = MockOptimizer(learning_rate=0.001)
                
                # Simulate training step
                data = self.test_data.create_mlp_data(
                    batch_size=config['batch_size'], seq_len=64, d_model=256
                )
                target = torch.randn_like(data)
                
                # Profile training step
                self.profiler.start_profile(f"distributed_{config['workers']}")
                output = model(data)
                loss = nn.MSELoss()(output, target)
                result = optimizer.step(loss)
                metrics = self.profiler.end_profile()
                
                # Calculate scalability metrics
                total_samples = config['batch_size'] * 64
                throughput = total_samples / metrics['execution_time']
                
                results.append({
                    'config': config,
                    'execution_time': metrics['execution_time'],
                    'memory_used': metrics['memory_used'],
                    'throughput': throughput,
                    'samples_per_second': throughput
                })
                
                self.assertTrue(result['optimized'])
        
        # Analyze distributed training performance
        for result in results:
            self.assertGreater(result['execution_time'], 0)
            self.assertGreater(result['throughput'], 0)
            self.assertGreater(result['samples_per_second'], 0)

if __name__ == '__main__':
    unittest.main()





