import torch
import torch.nn as nn
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Protocol, Type
from enum import Enum
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from triton_kernels import DeepSeekLayerNormModule
from modular import OptimizedLayerNorm, LayerNormConfig, PrecisionMode

class BenchmarkMode(Enum):
    """Benchmark execution modes."""
    TRAINING = "training"
    INFERENCE = "inference"
    BOTH = "both"

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    batch_sizes: List[int] = (32, 64, 128, 256)
    seq_lengths: List[int] = (512, 1024, 2048)
    hidden_sizes: List[int] = (512, 1024, 2048)
    num_runs: int = 100
    warmup_runs: int = 10
    mode: BenchmarkMode = BenchmarkMode.BOTH
    device: str = "cuda"
    precision: str = "fp32"
    use_mixed_precision: bool = False
    use_grad_checkpointing: bool = False
    use_amp: bool = False
    output_dir: str = "benchmark_results"
    save_plots: bool = True
    save_metrics: bool = True

class DataGeneratorInterface(Protocol):
    """Interface for data generators."""
    def generate_input(self, batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
        ...
    def generate_target(self, batch_size: int, seq_len: int) -> torch.Tensor:
        ...

class ModelFactoryInterface(Protocol):
    """Interface for model factories."""
    def create_models(self) -> Dict[str, nn.Module]:
        ...

class MetricsCollectorInterface(Protocol):
    """Interface for metrics collectors."""
    def collect_metrics(self, model: nn.Module, input_data: torch.Tensor, 
                       target_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        ...
    def aggregate_metrics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        ...

class VisualizerInterface(Protocol):
    """Interface for result visualizers."""
    def plot_metrics(self, metrics: Dict[str, Dict[str, float]], output_dir: str):
        ...
    def save_metrics(self, metrics: Dict[str, Dict[str, float]], output_dir: str):
        ...

class DataGenerator(DataGeneratorInterface):
    """Generates benchmark data."""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def generate_input(self, batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
        """Generate input tensor for benchmarking."""
        return torch.randn(batch_size, seq_len, hidden_size, device=self.config.device)
    
    def generate_target(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Generate target tensor for benchmarking."""
        return torch.randint(0, 1000, (batch_size, seq_len), device=self.config.device)

class ModelFactory(ModelFactoryInterface):
    """Factory for creating benchmark models."""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def create_models(self) -> Dict[str, nn.Module]:
        """Create all models for benchmarking."""
        models = {
            'PyTorch Native': nn.LayerNorm(512),
            'Optimized': OptimizedLayerNorm(LayerNormConfig(
                normalized_shape=512,
                device=self.config.device,
                precision=PrecisionMode.FP32
            )),
            'DeepSeek': DeepSeekLayerNormModule(normalized_shape=512),
            'Triton': TritonLayerNorm(normalized_shape=512)
        }
        
        # Move models to device and apply optimizations
        for model in models.values():
            model.to(self.config.device)
            if self.config.use_mixed_precision:
                model.half()
        
        return models

class MetricsCollector(MetricsCollectorInterface):
    """Collects benchmark metrics."""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def collect_metrics(self, model: nn.Module, input_data: torch.Tensor, 
                       target_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Collect metrics for a single model run."""
        # Warmup runs
        for _ in range(self.config.warmup_runs):
            if self.config.mode in [BenchmarkMode.TRAINING, BenchmarkMode.BOTH]:
                output = model(input_data)
                loss = output.mean()
                loss.backward()
            else:
                _ = model(input_data)
        
        # Synchronize CUDA
        torch.cuda.synchronize()
        
        # Timing runs
        start_time = time.time()
        for _ in range(self.config.num_runs):
            if self.config.mode in [BenchmarkMode.TRAINING, BenchmarkMode.BOTH]:
                output = model(input_data)
                loss = output.mean()
                loss.backward()
            else:
                _ = model(input_data)
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate statistics
        total_time = end_time - start_time
        avg_time = total_time / self.config.num_runs
        throughput = self.config.num_runs / total_time
        
        return {
            'total_time': total_time,
            'avg_time': avg_time,
            'throughput': throughput
        }
    
    def aggregate_metrics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple runs."""
        aggregated = {
            'avg_time': np.mean([m['avg_time'] for m in metrics]),
            'std_time': np.std([m['avg_time'] for m in metrics]),
            'min_time': np.min([m['avg_time'] for m in metrics]),
            'max_time': np.max([m['avg_time'] for m in metrics]),
            'throughput': np.mean([m['throughput'] for m in metrics])
        }
        return aggregated

class Visualizer(VisualizerInterface):
    """Visualizes benchmark results."""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.setup_output_dir()
    
    def setup_output_dir(self):
        """Setup output directory for results."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def plot_metrics(self, metrics: Dict[str, Dict[str, float]], output_dir: str):
        """Plot benchmark metrics."""
        # Convert metrics to DataFrame
        df = pd.DataFrame.from_dict(metrics, orient='index')
        
        # Plot throughput comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df.index, y='throughput', data=df)
        plt.title('Model Throughput Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison.png")
        plt.close()
        
        # Plot timing comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df.index, y='avg_time', data=df)
        plt.title('Average Execution Time Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_comparison.png")
        plt.close()
    
    def save_metrics(self, metrics: Dict[str, Dict[str, float]], output_dir: str):
        """Save metrics to file."""
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

class BenchmarkRunner:
    """Runs benchmarks for models."""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_generator = DataGenerator(config)
        self.model_factory = ModelFactory(config)
        self.metrics_collector = MetricsCollector(config)
        self.visualizer = Visualizer(config)
    
    def run_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Run benchmarks for all models and configurations."""
        models = self.model_factory.create_models()
        results = {}
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.seq_lengths:
                for hidden_size in self.config.hidden_sizes:
                    print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
                    
                    # Generate data
                    input_data = self.data_generator.generate_input(batch_size, seq_len, hidden_size)
                    target_data = self.data_generator.generate_target(batch_size, seq_len)
                    
                    # Benchmark each model
                    for name, model in models.items():
                        try:
                            # Reshape input if needed
                            if name in ['PyTorch Native', 'Optimized']:
                                x = input_data.view(-1, hidden_size)
                            else:
                                x = input_data.view(-1, hidden_size)
                            
                            # Collect metrics
                            metrics = self.metrics_collector.collect_metrics(model, x, target_data)
                            
                            # Store results
                            key = f"{name}_{batch_size}_{seq_len}_{hidden_size}"
                            results[key] = metrics
                            
                            print(f"{name}:")
                            print(f"  Average time: {metrics['avg_time']*1000:.2f} ms")
                            print(f"  Throughput: {metrics['throughput']:.2f} samples/sec")
                            
                        except Exception as e:
                            print(f"Error benchmarking {name}: {str(e)}")
        
        # Aggregate and visualize results
        if self.config.save_metrics:
            self.visualizer.save_metrics(results, self.config.output_dir)
        if self.config.save_plots:
            self.visualizer.plot_metrics(results, self.config.output_dir)
        
        return results

def main():
    # Create benchmark configuration
    config = BenchmarkConfig(
        mode=BenchmarkMode.BOTH,
        use_mixed_precision=True,
        use_amp=True,
        output_dir="benchmark_results"
    )
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(config)
    
    # Run benchmarks
    print("Starting Layer Normalization Benchmark")
    print("=" * 80)
    results = runner.run_benchmarks()
    
    print("\nBenchmark completed! Results saved in:", config.output_dir)

if __name__ == "__main__":
    main() 