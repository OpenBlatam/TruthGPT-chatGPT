"""
Examples for benchmarking.

Demonstrates usage of benchmarking utilities.
"""
from optimization_core.benchmarks import (
    BenchmarkRunner,
    run_benchmark,
    compare_benchmarks,
    MetricsCollector,
    collect_metrics,
)

# Example 1: Simple Benchmark
def example_simple_benchmark():
    """Example of running a simple benchmark."""
    from inference.vllm_engine import VLLMEngine
    
    engine = VLLMEngine(model="mistral-7b")
    
    # Run benchmark
    result = run_benchmark(
        "vllm_generation",
        engine.generate,
        "test prompt",
        num_runs=10
    )
    
    print(f"Duration: {result.duration:.3f}s")
    print(f"Throughput: {result.throughput:.2f} ops/s")
    return result


# Example 2: Comparing Multiple Engines
def example_compare_engines():
    """Example of comparing multiple engines."""
    from inference.engine_factory import create_inference_engine, EngineType
    
    engines = [
        ("vllm", create_inference_engine("model", EngineType.VLLM)),
        ("tensorrt", create_inference_engine("model", EngineType.TENSORRT_LLM)),
    ]
    
    results = []
    for name, engine in engines:
        result = run_benchmark(
            f"{name}_generation",
            engine.generate,
            "test prompt"
        )
        results.append(result)
    
    # Compare
    comparison = compare_benchmarks(results)
    print(f"Best: {comparison['best']}")
    print(f"Improvements: {comparison['improvements']}")
    return comparison


# Example 3: Collecting Metrics
def example_collect_metrics():
    """Example of collecting performance metrics."""
    from inference.vllm_engine import VLLMEngine
    
    engine = VLLMEngine(model="mistral-7b")
    collector = MetricsCollector()
    
    # Collect metrics during execution
    for i in range(100):
        result = collect_metrics(
            "inference",
            engine.generate,
            f"prompt {i}",
            collector=collector
        )
    
    # Get summary
    summary = collector.get_summary()
    print(f"Average duration: {summary['inference']['avg_duration']:.3f}s")
    print(f"P95 duration: {summary['inference']['p95_duration']:.3f}s")
    return summary


# Example 4: Using BenchmarkRunner
def example_benchmark_runner():
    """Example of using BenchmarkRunner directly."""
    runner = BenchmarkRunner(warmup_runs=3, num_runs=10)
    
    def my_function():
        # Your code here
        pass
    
    result = runner.run("my_benchmark", my_function)
    return result













