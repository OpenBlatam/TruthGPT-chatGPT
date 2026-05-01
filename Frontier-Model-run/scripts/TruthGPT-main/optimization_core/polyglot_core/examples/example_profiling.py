"""
Example: Profiling operations with polyglot_core.

Demonstrates how to profile operations and track performance metrics.
"""

import numpy as np
from optimization_core.polyglot_core import (
    KVCache,
    Attention,
    Compressor,
    Profiler,
    AttentionConfig,
    get_profiler,
)


def main():
    print("=" * 80)
    print("Polyglot Core - Profiling Example")
    print("=" * 80)
    print()
    
    profiler = get_profiler()
    
    # Start resource monitoring
    profiler.start_monitoring(interval=0.5)
    
    # 1. Profile KV Cache
    print("1. Profiling KV Cache:")
    print("-" * 80)
    
    cache = KVCache(max_size=1000)
    k = np.random.randn(64).astype(np.float32)
    v = np.random.randn(64).astype(np.float32)
    
    with profiler.profile("kv_cache_put", backend=cache.backend.name):
        for i in range(100):
            cache.put(layer=0, position=i, key=k, value=v)
    
    with profiler.profile("kv_cache_get", backend=cache.backend.name):
        for i in range(100):
            cache.get(layer=0, position=i)
    
    # 2. Profile Attention
    print("\n2. Profiling Attention:")
    print("-" * 80)
    
    attention = Attention(AttentionConfig(d_model=256, n_heads=4))
    batch_size = 2
    seq_len = 32
    d_model = 256
    q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    
    with profiler.profile("attention_forward", backend=attention.backend.name):
        output = attention.forward(q, q, q, batch_size, seq_len)
    
    # 3. Profile Compression
    print("\n3. Profiling Compression:")
    print("-" * 80)
    
    compressor = Compressor(algorithm="lz4")
    data = b"Compression profiling test " * 100
    
    with profiler.profile("compression", backend=compressor.backend.name):
        result = compressor.compress(data)
    
    # Stop monitoring
    profiler.stop_monitoring()
    
    # Print summary
    print("\n" + "=" * 80)
    profiler.print_summary()
    
    # Resource usage
    print("\n" + "=" * 80)
    print("Resource Usage:")
    print("=" * 80)
    
    history = profiler.get_resource_history()
    if history:
        peak_memory = profiler.get_peak_memory()
        avg_memory = sum(r.memory_mb for r in history) / len(history)
        avg_cpu = sum(r.cpu_percent for r in history) / len(history)
        
        print(f"Peak Memory: {peak_memory:.2f} MB")
        print(f"Average Memory: {avg_memory:.2f} MB")
        print(f"Average CPU: {avg_cpu:.2f}%")
        print(f"Samples: {len(history)}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()













