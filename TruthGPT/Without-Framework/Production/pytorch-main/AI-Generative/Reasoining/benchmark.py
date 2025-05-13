import torch
import torch.nn as nn
import time
import numpy as np
from triton_kernels import DeepSeekLayerNormModule
from modular import OptimizedLayerNorm, LayerNormConfig, PrecisionMode

def benchmark_layer_norm(model, input_data, num_runs=100, warmup_runs=10):
    """Benchmark layer normalization performance."""
    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(input_data)
    
    # Synchronize CUDA
    torch.cuda.synchronize()
    
    # Timing runs
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_data)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate statistics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = num_runs / total_time
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'throughput': throughput
    }

def run_benchmarks():
    """Run benchmarks for different layer normalization implementations."""
    # Benchmark configurations
    batch_sizes = [32, 64, 128, 256]
    seq_lengths = [512, 1024, 2048]
    hidden_sizes = [512, 1024, 2048]
    
    # Initialize models
    models = {
        'PyTorch Native': nn.LayerNorm(512),
        'Optimized': OptimizedLayerNorm(LayerNormConfig(
            normalized_shape=512,
            device='cuda',
            precision=PrecisionMode.FP32
        )),
        'DeepSeek': DeepSeekLayerNormModule(normalized_shape=512),
        'Triton': TritonLayerNorm(normalized_shape=512)
    }
    
    # Move models to CUDA
    for model in models.values():
        model.cuda()
    
    # Results storage
    results = {}
    
    # Run benchmarks
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            for hidden_size in hidden_sizes:
                print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
                
                # Create input data
                input_data = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
                
                # Benchmark each implementation
                for name, model in models.items():
                    try:
                        # Reshape input if needed
                        if name in ['PyTorch Native', 'Optimized']:
                            x = input_data.view(-1, hidden_size)
                        else:
                            x = input_data.view(-1, hidden_size)
                        
                        # Run benchmark
                        stats = benchmark_layer_norm(model, x)
                        
                        # Store results
                        key = f"{name}_{batch_size}_{seq_len}_{hidden_size}"
                        results[key] = stats
                        
                        print(f"{name}:")
                        print(f"  Average time: {stats['avg_time']*1000:.2f} ms")
                        print(f"  Throughput: {stats['throughput']:.2f} samples/sec")
                        
                    except Exception as e:
                        print(f"Error benchmarking {name}: {str(e)}")
    
    return results

def print_benchmark_summary(results):
    """Print a summary of benchmark results."""
    print("\nBenchmark Summary:")
    print("=" * 80)
    
    # Group results by model type
    model_results = {}
    for key, stats in results.items():
        model_name = key.split('_')[0]
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(stats['avg_time'])
    
    # Print summary for each model
    for model_name, times in model_results.items():
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"\n{model_name}:")
        print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Average throughput: {1/avg_time:.2f} samples/sec")

if __name__ == "__main__":
    print("Starting Layer Normalization Benchmark")
    print("=" * 80)
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Print summary
    print_benchmark_summary(results) 