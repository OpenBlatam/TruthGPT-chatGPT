"""
Comprehensive benchmarking suite for model performance evaluation.
"""

import torch
import torch.nn as nn
import time
import psutil
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
from contextlib import contextmanager

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    test_name: str
    inference_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    throughput_samples_per_sec: float
    accuracy_score: Optional[float] = None
    flops: Optional[int] = None
    parameters: int = 0
    model_size_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'test_name': self.test_name,
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'accuracy_score': self.accuracy_score,
            'flops': self.flops,
            'parameters': self.parameters,
            'model_size_mb': self.model_size_mb
        }

class MemoryTracker:
    """Track memory usage during model operations."""
    
    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.current_memory = 0
        
    @contextmanager
    def track(self):
        """Context manager for tracking memory usage."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.peak_memory = self.start_memory
        
        try:
            yield self
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                self.peak_memory = max(self.peak_memory, torch.cuda.max_memory_allocated() / 1024 / 1024)
            else:
                self.current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, self.current_memory)
    
    def get_usage(self) -> Tuple[float, float]:
        """Get current and peak memory usage in MB."""
        return self.current_memory - self.start_memory, self.peak_memory - self.start_memory

class SpeedBenchmark:
    """Benchmark inference speed and throughput."""
    
    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
    
    def benchmark_inference(self, model: nn.Module, input_data: Any, batch_size: int = 1) -> Tuple[float, float]:
        """
        Benchmark model inference speed.
        
        Returns:
            Tuple of (average_inference_time_ms, throughput_samples_per_sec)
        """
        model.eval()
        
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                if isinstance(input_data, dict):
                    _ = model(**input_data)
                elif isinstance(input_data, (list, tuple)):
                    _ = model(*input_data)
                else:
                    _ = model(input_data)
        
        times = []
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                if isinstance(input_data, dict):
                    _ = model(**input_data)
                elif isinstance(input_data, (list, tuple)):
                    _ = model(*input_data)
                else:
                    _ = model(input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
        
        avg_time_ms = sum(times) / len(times)
        throughput = (batch_size * 1000) / avg_time_ms
        
        return avg_time_ms, throughput

class ModelBenchmark:
    """Benchmark individual models with comprehensive metrics."""
    
    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        self.memory_tracker = MemoryTracker()
        self.speed_benchmark = SpeedBenchmark()
        
    def get_model_stats(self) -> Tuple[int, float]:
        """Get model parameter count and size in MB."""
        param_count = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        return param_count, model_size_mb
    
    def benchmark_forward_pass(self, input_data: Any, batch_size: int = 1) -> BenchmarkResult:
        """Benchmark a single forward pass."""
        param_count, model_size_mb = self.get_model_stats()
        
        with self.memory_tracker.track():
            avg_time_ms, throughput = self.speed_benchmark.benchmark_inference(
                self.model, input_data, batch_size
            )
        
        memory_usage, peak_memory = self.memory_tracker.get_usage()
        
        return BenchmarkResult(
            model_name=self.model_name,
            test_name="forward_pass",
            inference_time_ms=avg_time_ms,
            memory_usage_mb=memory_usage,
            peak_memory_mb=peak_memory,
            throughput_samples_per_sec=throughput,
            parameters=param_count,
            model_size_mb=model_size_mb
        )
    
    def benchmark_batch_processing(self, input_generator: Callable, batch_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark different batch sizes."""
        results = []
        
        for batch_size in batch_sizes:
            try:
                input_data = input_generator(batch_size)
                result = self.benchmark_forward_pass(input_data, batch_size)
                result.test_name = f"batch_size_{batch_size}"
                results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    warnings.warn(f"OOM at batch size {batch_size} for {self.model_name}")
                    break
                else:
                    raise e
        
        return results

class BenchmarkSuite:
    """Comprehensive benchmarking suite for comparing models."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def add_model(self, model: nn.Module, model_name: str) -> ModelBenchmark:
        """Add a model to the benchmark suite."""
        return ModelBenchmark(model, model_name)
    
    def run_comparison(self, models: Dict[str, nn.Module], input_generators: Dict[str, Callable]) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive comparison between models.
        
        Args:
            models: Dictionary of model_name -> model
            input_generators: Dictionary of test_name -> input_generator_function
        """
        all_results = {}
        
        for model_name, model in models.items():
            model_benchmark = self.add_model(model, model_name)
            model_results = []
            
            for test_name, input_gen in input_generators.items():
                try:
                    input_data = input_gen(1)
                    result = model_benchmark.benchmark_forward_pass(input_data, 1)
                    result.test_name = test_name
                    model_results.append(result)
                    
                    batch_results = model_benchmark.benchmark_batch_processing(
                        input_gen, [1, 2, 4, 8, 16]
                    )
                    model_results.extend(batch_results)
                    
                except Exception as e:
                    warnings.warn(f"Failed to benchmark {model_name} on {test_name}: {e}")
            
            all_results[model_name] = model_results
            self.results.extend(model_results)
        
        return all_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report = ["# Model Performance Benchmark Report", ""]
        
        model_results = {}
        for result in self.results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)
        
        report.append("## Summary")
        report.append("| Model | Parameters | Size (MB) | Avg Inference (ms) | Throughput (samples/s) | Memory (MB) |")
        report.append("|-------|------------|-----------|-------------------|----------------------|-------------|")
        
        for model_name, results in model_results.items():
            if results:
                avg_inference = sum(r.inference_time_ms for r in results) / len(results)
                avg_throughput = sum(r.throughput_samples_per_sec for r in results) / len(results)
                avg_memory = sum(r.memory_usage_mb for r in results) / len(results)
                params = results[0].parameters
                size_mb = results[0].model_size_mb
                
                report.append(f"| {model_name} | {params:,} | {size_mb:.2f} | {avg_inference:.2f} | {avg_throughput:.2f} | {avg_memory:.2f} |")
        
        report.append("")
        
        report.append("## Detailed Results")
        for model_name, results in model_results.items():
            report.append(f"### {model_name}")
            report.append("| Test | Inference (ms) | Memory (MB) | Peak Memory (MB) | Throughput (samples/s) |")
            report.append("|------|----------------|-------------|------------------|----------------------|")
            
            for result in results:
                report.append(f"| {result.test_name} | {result.inference_time_ms:.2f} | {result.memory_usage_mb:.2f} | {result.peak_memory_mb:.2f} | {result.throughput_samples_per_sec:.2f} |")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        import json
        
        results_dict = [result.to_dict() for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def get_performance_comparison(self) -> Dict[str, Dict[str, float]]:
        """Get performance comparison metrics between models."""
        if not self.results:
            return {}
        
        model_metrics = {}
        
        for result in self.results:
            if result.model_name not in model_metrics:
                model_metrics[result.model_name] = {
                    'avg_inference_ms': [],
                    'avg_memory_mb': [],
                    'avg_throughput': [],
                    'parameters': result.parameters,
                    'model_size_mb': result.model_size_mb
                }
            
            model_metrics[result.model_name]['avg_inference_ms'].append(result.inference_time_ms)
            model_metrics[result.model_name]['avg_memory_mb'].append(result.memory_usage_mb)
            model_metrics[result.model_name]['avg_throughput'].append(result.throughput_samples_per_sec)
        
        comparison = {}
        for model_name, metrics in model_metrics.items():
            comparison[model_name] = {
                'avg_inference_ms': sum(metrics['avg_inference_ms']) / len(metrics['avg_inference_ms']),
                'avg_memory_mb': sum(metrics['avg_memory_mb']) / len(metrics['avg_memory_mb']),
                'avg_throughput': sum(metrics['avg_throughput']) / len(metrics['avg_throughput']),
                'parameters': metrics['parameters'],
                'model_size_mb': metrics['model_size_mb'],
                'efficiency_score': metrics['parameters'] / (sum(metrics['avg_inference_ms']) / len(metrics['avg_inference_ms']))
            }
        
        return comparison
