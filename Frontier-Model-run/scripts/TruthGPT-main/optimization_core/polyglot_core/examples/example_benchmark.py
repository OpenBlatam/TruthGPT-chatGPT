"""
Example: Benchmarking different backends.

Compares performance of Rust, C++, and Python backends.
"""

import numpy as np
from optimization_core.polyglot_core import (
    KVCache,
    Attention,
    Compressor,
    Benchmark,
    Backend,
    get_available_backends,
    print_backend_status,
    print_device_info,
)


def main():
    print("=" * 80)
    print("Polyglot Core - Benchmarking Example")
    print("=" * 80)
    print()
    
    # Show device info
    print_device_info()
    
    # Show backend status
    print_backend_status()
    
    benchmark = Benchmark(warmup_iterations=5)
    
    # 1. KV Cache Benchmark
    print("\n1. KV Cache Benchmark:")
    print("-" * 80)
    
    def create_cache(backend):
        return KVCache(max_size=10000, backend=backend)
    
    def cache_benchmark(cache):
        k = np.random.randn(64).astype(np.float32)
        v = np.random.randn(64).astype(np.float32)
        cache.put(layer=0, position=0, key=k, value=v)
        return cache.get(layer=0, position=0)
    
    cache_results = benchmark.compare_backends(
        "kv_cache",
        create_cache,
        (),
        {},
        iterations=1000,
        backends=[Backend.PYTHON, Backend.RUST, Backend.CPP]
    )
    benchmark.print_comparison(cache_results)
    
    # 2. Attention Benchmark
    print("\n2. Attention Benchmark:")
    print("-" * 80)
    
    def create_attention(backend):
        from optimization_core.polyglot_core.attention import AttentionConfig
        return Attention(AttentionConfig(d_model=256, n_heads=4), backend=backend)
    
    def attention_benchmark(attention):
        batch_size = 2
        seq_len = 32
        d_model = 256
        q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
        return attention.forward(q, q, q, batch_size, seq_len)
    
    attention_results = benchmark.compare_backends(
        "attention",
        create_attention,
        (),
        {},
        iterations=100,
        backends=[Backend.PYTHON, Backend.CPP]
    )
    benchmark.print_comparison(attention_results)
    
    # 3. Compression Benchmark
    print("\n3. Compression Benchmark:")
    print("-" * 80)
    
    def create_compressor(backend):
        return Compressor(algorithm="lz4", backend=backend)
    
    data = b"Compression benchmark data " * 1000
    
    def compression_benchmark(compressor):
        result = compressor.compress(data)
        return result.data
    
    compression_results = benchmark.compare_backends(
        "compression",
        create_compressor,
        (),
        {},
        iterations=500,
        backends=[Backend.PYTHON, Backend.RUST]
    )
    benchmark.print_comparison(compression_results)
    
    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    
    all_results = {
        **cache_results,
        **attention_results,
        **compression_results
    }
    
    successful = {k: v for k, v in all_results.items() if v.success}
    if successful:
        fastest = max(successful.items(), key=lambda x: x[1].throughput)
        print(f"\nFastest overall: {fastest[0]} ({fastest[1].throughput:.0f} ops/s)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()













