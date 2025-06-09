"""
Comprehensive benchmarking suite for Qwen models.
"""

import torch
import torch.nn as nn
import time
import psutil
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json

@dataclass
class QwenBenchmarkConfig:
    """Configuration for Qwen benchmarks."""
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_warmup_runs: int = 5
    num_benchmark_runs: int = 20
    measure_memory: bool = True
    measure_throughput: bool = True
    measure_latency: bool = True
    measure_flops: bool = True
    save_results: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16]
        if self.sequence_lengths is None:
            self.sequence_lengths = [512, 1024, 2048, 4096]

class QwenPerformanceProfiler:
    """Performance profiler for Qwen models."""
    
    def __init__(self):
        self.results = {}
        
    def profile_memory_usage(self, model, input_ids, device='cuda'):
        """Profile memory usage during forward pass."""
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()
            
            return {
                'initial_memory_mb': initial_memory / 1024 / 1024,
                'peak_memory_mb': peak_memory / 1024 / 1024,
                'final_memory_mb': final_memory / 1024 / 1024,
                'memory_increase_mb': (peak_memory - initial_memory) / 1024 / 1024
            }
        else:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            final_memory = process.memory_info().rss / 1024 / 1024
            
            return {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': final_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': final_memory - initial_memory
            }
    
    def profile_inference_speed(self, model, input_ids, num_runs=20):
        """Profile inference speed."""
        model.eval()
        
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time_ms': sum(times) / len(times) * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000
        }
    
    def profile_throughput(self, model, input_ids, num_runs=10):
        """Profile throughput (tokens/second)."""
        model.eval()
        batch_size, seq_len = input_ids.shape
        total_tokens = batch_size * seq_len
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model(input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        tokens_per_second = (total_tokens * num_runs) / total_time
        
        return {
            'tokens_per_second': tokens_per_second,
            'samples_per_second': (batch_size * num_runs) / total_time,
            'total_time_s': total_time,
            'time_per_sample_ms': (total_time / (batch_size * num_runs)) * 1000
        }

class QwenModelAnalyzer:
    """Analyzer for Qwen model characteristics."""
    
    def __init__(self):
        pass
    
    def count_parameters(self, model):
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_parameters_millions': total_params / 1e6,
            'total_parameters_billions': total_params / 1e9,
            'trainable_parameters_millions': trainable_params / 1e6,
            'trainable_parameters_billions': trainable_params / 1e9
        }
    
    def analyze_model_size(self, model):
        """Analyze model size in memory."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'parameter_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': total_size / 1024 / 1024,
            'total_size_gb': total_size / 1024 / 1024 / 1024
        }
    
    def analyze_moe_efficiency(self, model):
        """Analyze MoE efficiency metrics."""
        moe_stats = {
            'total_experts': 0,
            'active_experts_per_layer': [],
            'expert_utilization': [],
            'routing_efficiency': []
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'experts') and hasattr(module, 'gate'):
                moe_stats['total_experts'] += len(module.experts)
                
                if hasattr(module, 'num_experts_per_tok'):
                    moe_stats['active_experts_per_layer'].append(module.num_experts_per_tok)
        
        return moe_stats

class QwenBenchmarkSuite:
    """Comprehensive benchmark suite for Qwen models."""
    
    def __init__(self, config: QwenBenchmarkConfig):
        self.config = config
        self.profiler = QwenPerformanceProfiler()
        self.analyzer = QwenModelAnalyzer()
        self.results = {}
        
    def run_comprehensive_benchmark(self, model, device='cuda'):
        """Run comprehensive benchmarks on the model."""
        model.eval()
        model = model.to(device)
        
        print("🚀 Starting Qwen Model Comprehensive Benchmark")
        print("=" * 60)
        
        self.results['model_analysis'] = self.analyzer.count_parameters(model)
        self.results['model_size'] = self.analyzer.analyze_model_size(model)
        self.results['moe_analysis'] = self.analyzer.analyze_moe_efficiency(model)
        
        print(f"📊 Model Parameters: {self.results['model_analysis']['total_parameters_billions']:.2f}B")
        print(f"💾 Model Size: {self.results['model_size']['total_size_gb']:.2f} GB")
        
        performance_results = {}
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                print(f"\n🔄 Testing batch_size={batch_size}, seq_len={seq_len}")
                
                try:
                    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                    
                    for _ in range(self.config.num_warmup_runs):
                        with torch.no_grad():
                            _ = model(input_ids)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    test_key = f"batch_{batch_size}_seq_{seq_len}"
                    performance_results[test_key] = {}
                    
                    if self.config.measure_memory:
                        memory_stats = self.profiler.profile_memory_usage(model, input_ids, device)
                        performance_results[test_key]['memory'] = memory_stats
                        print(f"  💾 Peak Memory: {memory_stats['peak_memory_mb']:.1f} MB")
                    
                    if self.config.measure_latency:
                        speed_stats = self.profiler.profile_inference_speed(
                            model, input_ids, self.config.num_benchmark_runs
                        )
                        performance_results[test_key]['latency'] = speed_stats
                        print(f"  ⚡ Avg Latency: {speed_stats['mean_time_ms']:.1f} ms")
                    
                    if self.config.measure_throughput:
                        throughput_stats = self.profiler.profile_throughput(model, input_ids)
                        performance_results[test_key]['throughput'] = throughput_stats
                        print(f"  🚄 Throughput: {throughput_stats['tokens_per_second']:.0f} tokens/s")
                        print(f"  📈 Samples/s: {throughput_stats['samples_per_second']:.1f}")
                    
                except RuntimeError as e:
                    print(f"  ❌ Error: {str(e)}")
                    performance_results[test_key] = {'error': str(e)}
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        self.results['performance'] = performance_results
        
        if self.config.save_results:
            self.save_results()
        
        self.print_summary()
        
        return self.results
    
    def save_results(self, filename=None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"qwen_benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 60)
        
        model_analysis = self.results['model_analysis']
        print(f"🔢 Total Parameters: {model_analysis['total_parameters_billions']:.3f}B")
        print(f"🎯 Trainable Parameters: {model_analysis['trainable_parameters_billions']:.3f}B")
        
        model_size = self.results['model_size']
        print(f"📦 Model Size: {model_size['total_size_gb']:.2f} GB")
        
        if 'performance' in self.results:
            best_throughput = 0
            best_latency = float('inf')
            
            for test_key, test_results in self.results['performance'].items():
                if 'error' not in test_results:
                    if 'throughput' in test_results:
                        throughput = test_results['throughput']['tokens_per_second']
                        if throughput > best_throughput:
                            best_throughput = throughput
                    
                    if 'latency' in test_results:
                        latency = test_results['latency']['mean_time_ms']
                        if latency < best_latency:
                            best_latency = latency
            
            print(f"🚄 Best Throughput: {best_throughput:.0f} tokens/s")
            print(f"⚡ Best Latency: {best_latency:.1f} ms")
        
        moe_analysis = self.results['moe_analysis']
        if moe_analysis['total_experts'] > 0:
            print(f"🧠 Total Experts: {moe_analysis['total_experts']}")
            if moe_analysis['active_experts_per_layer']:
                avg_active = sum(moe_analysis['active_experts_per_layer']) / len(moe_analysis['active_experts_per_layer'])
                print(f"⚙️  Avg Active Experts/Layer: {avg_active:.1f}")
        
        print("=" * 60)

def run_qwen_benchmarks(model, config: Optional[Dict[str, Any]] = None, device='cuda'):
    """Run Qwen benchmarks with optional configuration."""
    if config is None:
        config = {}
    
    benchmark_config = QwenBenchmarkConfig(
        batch_sizes=config.get('batch_sizes', [1, 2, 4, 8]),
        sequence_lengths=config.get('sequence_lengths', [512, 1024, 2048]),
        num_warmup_runs=config.get('num_warmup_runs', 5),
        num_benchmark_runs=config.get('num_benchmark_runs', 20),
        measure_memory=config.get('measure_memory', True),
        measure_throughput=config.get('measure_throughput', True),
        measure_latency=config.get('measure_latency', True),
        measure_flops=config.get('measure_flops', False),
        save_results=config.get('save_results', True)
    )
    
    benchmark_suite = QwenBenchmarkSuite(benchmark_config)
    return benchmark_suite.run_comprehensive_benchmark(model, device)
