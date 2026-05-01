#!/usr/bin/env python3
"""
Script to run polyglot_core benchmarks.

Usage:
    python -m optimization_core.polyglot_core.scripts.run_benchmarks
    python -m optimization_core.polyglot_core.scripts.run_benchmarks --quick
    python -m optimization_core.polyglot_core.scripts.run_benchmarks --output results.json
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from optimization_core.polyglot_core import (
    Benchmark,
    Backend,
    KVCache,
    Attention,
    Compressor,
    AttentionConfig,
    ReportGenerator,
)


def main():
    parser = argparse.ArgumentParser(description="Run polyglot_core benchmarks")
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmarks (fewer iterations)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON)'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Generate HTML report'
    )
    
    args = parser.parse_args()
    
    iterations = 50 if args.quick else 1000
    
    print("=" * 80)
    print("Polyglot Core - Benchmark Suite")
    print("=" * 80)
    print(f"Iterations: {iterations}")
    print()
    
    benchmark = Benchmark(warmup_iterations=5)
    results = {}
    
    # 1. KV Cache Benchmark
    print("1. Benchmarking KV Cache...")
    def create_cache(backend):
        return KVCache(max_size=10000, backend=backend)
    
    def cache_benchmark(cache):
        k = np.random.randn(64).astype(np.float32)
        v = np.random.randn(64).astype(np.float32)
        cache.put(layer=0, position=0, key=k, value=v)
        return cache.get(layer=0, position=0)
    
    cache_results = benchmark.compare_backends(
        "kv_cache",
        lambda b: lambda: cache_benchmark(create_cache(b)),
        iterations=iterations,
        backends=[Backend.PYTHON, Backend.RUST, Backend.CPP]
    )
    results.update(cache_results)
    benchmark.print_comparison(cache_results)
    
    # 2. Attention Benchmark
    print("\n2. Benchmarking Attention...")
    def create_attention(backend):
        return Attention(AttentionConfig(d_model=256, n_heads=4), backend=backend)
    
    def attention_benchmark(attention):
        q = np.random.randn(2 * 16, 256).astype(np.float32)
        return attention.forward(q, q, q, batch_size=2, seq_len=16)
    
    attention_results = benchmark.compare_backends(
        "attention",
        lambda b: lambda: attention_benchmark(create_attention(b)),
        iterations=iterations // 10,  # Fewer iterations (slower)
        backends=[Backend.PYTHON, Backend.CPP]
    )
    results.update(attention_results)
    benchmark.print_comparison(attention_results)
    
    # 3. Compression Benchmark
    print("\n3. Benchmarking Compression...")
    def create_compressor(backend):
        return Compressor(algorithm="lz4", backend=backend)
    
    data = b"Compression benchmark data " * 1000
    
    def compression_benchmark(compressor):
        result = compressor.compress(data)
        return result.data
    
    compression_results = benchmark.compare_backends(
        "compression",
        lambda b: lambda: compression_benchmark(create_compressor(b)),
        iterations=iterations // 2,
        backends=[Backend.PYTHON, Backend.RUST]
    )
    results.update(compression_results)
    benchmark.print_comparison(compression_results)
    
    # Save results
    if args.output:
        benchmark.save_results(results, args.output)
        print(f"\nResults saved to: {args.output}")
    
    # Generate report
    if args.report:
        generator = ReportGenerator()
        report = generator.generate_benchmark_report(results)
        report.save(args.report, format="html")
        print(f"Report saved to: {args.report}.html")
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()













