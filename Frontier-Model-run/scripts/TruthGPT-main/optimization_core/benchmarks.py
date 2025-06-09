"""
Comprehensive benchmarking suite for TruthGPT optimizations.
Integrated from benchmark.py optimization files.
"""

import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass
import warnings

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    warnings.warn("Matplotlib not available. Plotting features will be disabled.")
    MATPLOTLIB_AVAILABLE = False
    plt = None

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_runs: int = 10
    warmup_runs: int = 3
    measure_memory: bool = True
    measure_throughput: bool = True
    measure_latency: bool = True
    device: str = 'cuda'
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512, 1024]

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for TruthGPT models."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        
    def benchmark_model(self, model: nn.Module, model_name: str, 
                       input_generator: callable) -> Dict[str, Any]:
        """Benchmark a model with various configurations."""
        model.eval()
        model = model.to(self.config.device)
        
        results = {
            'model_name': model_name,
            'device': self.config.device,
            'batch_size_results': {},
            'sequence_length_results': {},
            'memory_usage': {},
            'throughput': {},
            'latency': {}
        }
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                key = f"bs_{batch_size}_seq_{seq_len}"
                
                try:
                    batch_results = self._benchmark_configuration(
                        model, input_generator, batch_size, seq_len
                    )
                    results['batch_size_results'][key] = batch_results
                    
                except Exception as e:
                    warnings.warn(f"Benchmark failed for {key}: {e}")
                    results['batch_size_results'][key] = {'error': str(e)}
        
        self.results[model_name] = results
        return results
    
    def _benchmark_configuration(self, model: nn.Module, input_generator: callable,
                                batch_size: int, seq_len: int) -> Dict[str, float]:
        """Benchmark a specific configuration."""
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        inputs = input_generator(batch_size, seq_len, self.config.device)
        
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model(**inputs if isinstance(inputs, dict) else inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        memory_before = self._get_memory_usage()
        
        times = []
        for _ in range(self.config.num_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model(**inputs if isinstance(inputs, dict) else inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        memory_after = self._get_memory_usage()
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = (batch_size * seq_len) / avg_time
        
        return {
            'avg_latency_ms': avg_time * 1000,
            'std_latency_ms': std_time * 1000,
            'throughput_tokens_per_sec': throughput,
            'memory_usage_mb': memory_after - memory_before,
            'batch_size': batch_size,
            'sequence_length': seq_len
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance between multiple models."""
        comparison = {
            'models': [r['model_name'] for r in model_results],
            'performance_ratios': {},
            'memory_ratios': {},
            'throughput_ratios': {}
        }
        
        if len(model_results) < 2:
            return comparison
        
        baseline = model_results[0]
        
        for i, result in enumerate(model_results[1:], 1):
            model_name = result['model_name']
            
            for config_key in baseline['batch_size_results']:
                if config_key in result['batch_size_results']:
                    baseline_latency = baseline['batch_size_results'][config_key].get('avg_latency_ms', 0)
                    current_latency = result['batch_size_results'][config_key].get('avg_latency_ms', 0)
                    
                    if baseline_latency > 0 and current_latency > 0:
                        speedup = baseline_latency / current_latency
                        comparison['performance_ratios'][f"{model_name}_{config_key}"] = speedup
                    
                    baseline_memory = baseline['batch_size_results'][config_key].get('memory_usage_mb', 0)
                    current_memory = result['batch_size_results'][config_key].get('memory_usage_mb', 0)
                    
                    if baseline_memory > 0 and current_memory > 0:
                        memory_ratio = baseline_memory / current_memory
                        comparison['memory_ratios'][f"{model_name}_{config_key}"] = memory_ratio
                    
                    baseline_throughput = baseline['batch_size_results'][config_key].get('throughput_tokens_per_sec', 0)
                    current_throughput = result['batch_size_results'][config_key].get('throughput_tokens_per_sec', 0)
                    
                    if baseline_throughput > 0 and current_throughput > 0:
                        throughput_ratio = current_throughput / baseline_throughput
                        comparison['throughput_ratios'][f"{model_name}_{config_key}"] = throughput_ratio
        
        return comparison
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("# TruthGPT Optimization Benchmark Report\n")
        
        for model_name, results in self.results.items():
            report.append(f"## {model_name}\n")
            
            avg_latencies = []
            avg_throughputs = []
            avg_memory = []
            
            for config, metrics in results['batch_size_results'].items():
                if 'error' not in metrics:
                    avg_latencies.append(metrics['avg_latency_ms'])
                    avg_throughputs.append(metrics['throughput_tokens_per_sec'])
                    avg_memory.append(metrics['memory_usage_mb'])
            
            if avg_latencies:
                report.append(f"- **Average Latency**: {np.mean(avg_latencies):.2f} ± {np.std(avg_latencies):.2f} ms")
                report.append(f"- **Average Throughput**: {np.mean(avg_throughputs):.0f} ± {np.std(avg_throughputs):.0f} tokens/sec")
                report.append(f"- **Average Memory Usage**: {np.mean(avg_memory):.2f} ± {np.std(avg_memory):.2f} MB")
            
            report.append("\n### Detailed Results\n")
            for config, metrics in results['batch_size_results'].items():
                if 'error' not in metrics:
                    report.append(f"**{config}**:")
                    report.append(f"  - Latency: {metrics['avg_latency_ms']:.2f} ms")
                    report.append(f"  - Throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/sec")
                    report.append(f"  - Memory: {metrics['memory_usage_mb']:.2f} MB")
                else:
                    report.append(f"**{config}**: Error - {metrics['error']}")
            
            report.append("\n")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_results(self, output_path: str):
        """Save benchmark results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

def create_text_input_generator():
    """Create input generator for text models."""
    def generator(batch_size: int, seq_len: int, device: str):
        return {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len), device=device),
            'attention_mask': torch.ones(batch_size, seq_len, device=device)
        }
    return generator

def create_multimodal_input_generator():
    """Create input generator for multimodal models."""
    def generator(batch_size: int, seq_len: int, device: str):
        return {
            'visual_features': torch.randn(batch_size, seq_len, 2048, device=device),
            'audio_features': torch.randn(batch_size, seq_len, 512, device=device),
            'text_features': torch.randn(batch_size, seq_len, 768, device=device),
            'engagement_features': torch.randn(batch_size, seq_len, 64, device=device)
        }
    return generator

def benchmark_optimization_impact(original_model: nn.Module, optimized_model: nn.Module,
                                input_generator: callable, model_name: str) -> Dict[str, Any]:
    """Benchmark the impact of optimizations on a model."""
    config = BenchmarkConfig(
        batch_sizes=[1, 2, 4, 8],
        sequence_lengths=[128, 256, 512],
        num_runs=10
    )
    
    benchmark = PerformanceBenchmark(config)
    
    original_results = benchmark.benchmark_model(original_model, f"{model_name}_original", input_generator)
    optimized_results = benchmark.benchmark_model(optimized_model, f"{model_name}_optimized", input_generator)
    
    comparison = benchmark.compare_models([original_results, optimized_results])
    
    return {
        'original_results': original_results,
        'optimized_results': optimized_results,
        'comparison': comparison,
        'benchmark_config': config
    }

def run_comprehensive_benchmarks(models: Dict[str, nn.Module], 
                                input_generators: Dict[str, callable]) -> Dict[str, Any]:
    """Run comprehensive benchmarks across all models."""
    all_results = {}
    
    config = BenchmarkConfig(
        batch_sizes=[1, 2, 4, 8, 16],
        sequence_lengths=[128, 256, 512, 1024],
        num_runs=15,
        warmup_runs=5
    )
    
    benchmark = PerformanceBenchmark(config)
    
    for model_name, model in models.items():
        input_gen = input_generators.get(model_name, create_text_input_generator())
        
        try:
            results = benchmark.benchmark_model(model, model_name, input_gen)
            all_results[model_name] = results
            print(f"✅ Benchmarked {model_name}")
        except Exception as e:
            print(f"❌ Failed to benchmark {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    report = benchmark.generate_report()
    
    return {
        'individual_results': all_results,
        'benchmark_report': report,
        'config': config
    }
