"""
Example: Generating comprehensive reports.

Demonstrates how to generate HTML, Markdown, and JSON reports
from benchmarks, profiling, and metrics.
"""

import numpy as np
from optimization_core.polyglot_core import (
    KVCache,
    Attention,
    Compressor,
    Benchmark,
    Profiler,
    AttentionConfig,
    ReportGenerator,
    get_metrics_collector,
    get_profiler,
)


def main():
    print("=" * 80)
    print("Polyglot Core - Reporting Example")
    print("=" * 80)
    print()
    
    # 1. Run benchmarks
    print("1. Running benchmarks...")
    benchmark = Benchmark()
    
    # Cache benchmark
    cache = KVCache(max_size=1000)
    k = np.random.randn(64).astype(np.float32)
    v = np.random.randn(64).astype(np.float32)
    
    cache_result = benchmark.run(
        "cache_operations",
        lambda: cache.get(layer=0, position=0) if cache.get(layer=0, position=0) else cache.put(layer=0, position=0, key=k, value=v),
        iterations=100
    )
    
    # Attention benchmark
    attention = Attention(AttentionConfig(d_model=256, n_heads=4))
    q = np.random.randn(2 * 16, 256).astype(np.float32)
    
    attention_result = benchmark.run(
        "attention_forward",
        lambda: attention.forward(q, q, q, batch_size=2, seq_len=16),
        iterations=50
    )
    
    # Compression benchmark
    compressor = Compressor(algorithm="lz4")
    data = b"Compression test " * 100
    
    compression_result = benchmark.run(
        "compression",
        lambda: compressor.compress(data),
        iterations=200
    )
    
    results = {
        "cache_operations": cache_result,
        "attention_forward": attention_result,
        "compression": compression_result
    }
    
    # 2. Collect metrics
    print("\n2. Collecting metrics...")
    collector = get_metrics_collector()
    
    collector.record_latency("cache_get", 0.5, backend=cache.backend.name)
    collector.record_latency("cache_get", 0.6, backend=cache.backend.name)
    collector.record_latency("cache_get", 0.4, backend=cache.backend.name)
    
    collector.record_throughput("attention", 1000.0, backend=attention.backend.name)
    collector.record_throughput("compression", 5000.0, backend=compressor.backend.name)
    
    # 3. Profile operations
    print("\n3. Profiling operations...")
    profiler = get_profiler()
    
    with profiler.profile("cache_profile", backend=cache.backend.name):
        for i in range(10):
            cache.put(layer=0, position=i, key=k, value=v)
    
    with profiler.profile("attention_profile", backend=attention.backend.name):
        attention.forward(q, q, q, batch_size=2, seq_len=16)
    
    profiler_summary = profiler.summary()
    
    # 4. Generate reports
    print("\n4. Generating reports...")
    generator = ReportGenerator()
    
    # Benchmark report
    benchmark_report = generator.generate_benchmark_report(
        results,
        title="Polyglot Core Benchmark Report"
    )
    
    # Profiling report
    profiling_report = generator.generate_profiling_report(
        profiler_summary,
        title="Polyglot Core Profiling Report"
    )
    
    # Metrics report
    metrics_summaries = collector.get_all_summaries()
    metrics_data = {
        name: {
            'count': s.count,
            'avg': s.avg,
            'min': s.min,
            'max': s.max,
            'p95': s.p95
        }
        for name, s in metrics_summaries.items()
    }
    
    metrics_report = generator.generate_metrics_report(
        metrics_data,
        title="Polyglot Core Metrics Report"
    )
    
    # 5. Save reports
    print("\n5. Saving reports...")
    
    # Save as Markdown
    benchmark_report.save("benchmark_report", format="markdown")
    profiling_report.save("profiling_report", format="markdown")
    metrics_report.save("metrics_report", format="markdown")
    
    # Save as HTML
    benchmark_report.save("benchmark_report", format="html")
    profiling_report.save("profiling_report", format="html")
    metrics_report.save("metrics_report", format="html")
    
    # Save as JSON
    benchmark_report.save("benchmark_report", format="json")
    
    print("\n" + "=" * 80)
    print("Reports Generated Successfully!")
    print("=" * 80)
    print("\nFiles created:")
    print("  - benchmark_report.md/html/json")
    print("  - profiling_report.md/html")
    print("  - metrics_report.md/html")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()













