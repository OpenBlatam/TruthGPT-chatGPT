"""
TruthGPT Polyglot Core - Comprehensive Benchmarks

Benchmarks all backends and components for performance comparison.

Usage:
    python -m benchmarks.polyglot_benchmarks
    python -m benchmarks.polyglot_benchmarks --component attention
    python -m benchmarks.polyglot_benchmarks --all --output results.json

Results are saved in JSON format for analysis.
"""

from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from pathlib import Path
import sys

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "run_all_benchmarks",
    "benchmark_attention",
    "benchmark_kv_cache",
    "benchmark_compression",
    "benchmark_inference",
]


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    component: str
    backend: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float
    memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return (
            f"{self.name} ({self.backend}): "
            f"avg={self.avg_time_ms:.2f}ms, "
            f"throughput={self.throughput:.0f}/s"
        )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    warmup_iterations: int = 3
    benchmark_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    d_models: List[int] = field(default_factory=lambda: [768, 1024, 2048, 4096])


class Timer:
    """High-resolution timer for benchmarking."""
    
    def __init__(self):
        self._start = None
        self._times: List[float] = []
    
    def start(self):
        self._start = time.perf_counter_ns()
    
    def stop(self) -> float:
        elapsed = (time.perf_counter_ns() - self._start) / 1_000_000
        self._times.append(elapsed)
        return elapsed
    
    def reset(self):
        self._times.clear()
    
    @property
    def total_ms(self) -> float:
        return sum(self._times)
    
    @property
    def avg_ms(self) -> float:
        return np.mean(self._times) if self._times else 0
    
    @property
    def min_ms(self) -> float:
        return min(self._times) if self._times else 0
    
    @property
    def max_ms(self) -> float:
        return max(self._times) if self._times else 0


def measure_memory() -> float:
    """Measure current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


class BenchmarkSuite:
    """Suite of benchmarks for polyglot components."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(
        self,
        name: str,
        component: str,
        backend: str,
        func: Callable,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        metadata: Optional[Dict] = None
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        timer = Timer()
        
        if setup:
            ctx = setup()
        else:
            ctx = None
        
        for _ in range(self.config.warmup_iterations):
            if ctx:
                func(ctx)
            else:
                func()
        
        memory_before = measure_memory()
        
        for _ in range(self.config.benchmark_iterations):
            timer.start()
            if ctx:
                func(ctx)
            else:
                func()
            timer.stop()
        
        memory_after = measure_memory()
        
        if teardown:
            teardown(ctx)
        
        ops_per_sec = self.config.benchmark_iterations / (timer.total_ms / 1000)
        
        result = BenchmarkResult(
            name=name,
            component=component,
            backend=backend,
            iterations=self.config.benchmark_iterations,
            total_time_ms=timer.total_ms,
            avg_time_ms=timer.avg_ms,
            min_time_ms=timer.min_ms,
            max_time_ms=timer.max_ms,
            throughput=ops_per_sec,
            memory_mb=max(0, memory_after - memory_before),
            metadata=metadata or {}
        )
        
        self.results.append(result)
        return result
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        by_component = {}
        for r in self.results:
            if r.component not in by_component:
                by_component[r.component] = []
            by_component[r.component].append(r)
        
        for component, results in by_component.items():
            print(f"\n{component.upper()}")
            print("-" * 60)
            print(f"{'Name':<30} {'Backend':<10} {'Avg (ms)':<12} {'Throughput':<15}")
            print("-" * 60)
            
            for r in sorted(results, key=lambda x: x.avg_time_ms):
                print(f"{r.name:<30} {r.backend:<10} {r.avg_time_ms:<12.2f} {r.throughput:<15.0f}/s")
        
        print("\n" + "=" * 80)
    
    def save_results(self, path: str):
        """Save results to JSON file."""
        data = {
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {path}")


def benchmark_attention(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Benchmark attention implementations."""
    results = []
    
    batch_size = 4
    seq_len = 512
    d_model = 768
    n_heads = 12
    
    def setup_python():
        q = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        k = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        v = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        return {"q": q, "k": k, "v": v}
    
    def python_attention(ctx):
        q, k, v = ctx["q"], ctx["k"], ctx["v"]
        scale = 1.0 / np.sqrt(d_model // n_heads)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return np.matmul(weights, v)
    
    r = suite.run_benchmark(
        name=f"attention_b{batch_size}_s{seq_len}_d{d_model}",
        component="attention",
        backend="python",
        func=python_attention,
        setup=setup_python,
        metadata={"batch_size": batch_size, "seq_len": seq_len, "d_model": d_model}
    )
    results.append(r)
    print(f"  {r}")
    
    try:
        from optimization_core.polyglot_core import Attention, AttentionConfig, Backend
        
        config = AttentionConfig(d_model=d_model, n_heads=n_heads)
        
        for backend in [Backend.CPP, Backend.RUST, Backend.PYTHON]:
            try:
                attn = Attention(config, backend=backend)
                
                def polyglot_attention(ctx):
                    return attn.forward(
                        ctx["q"], ctx["k"], ctx["v"],
                        batch_size=batch_size, seq_len=seq_len
                    )
                
                r = suite.run_benchmark(
                    name=f"attention_b{batch_size}_s{seq_len}_d{d_model}",
                    component="attention",
                    backend=backend.name.lower(),
                    func=polyglot_attention,
                    setup=setup_python,
                    metadata={"batch_size": batch_size, "seq_len": seq_len}
                )
                results.append(r)
                print(f"  {r}")
            except Exception as e:
                print(f"  Skipping {backend.name}: {e}")
    except ImportError:
        print("  polyglot_core not available")
    
    return results


def benchmark_kv_cache(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Benchmark KV cache implementations."""
    results = []
    
    cache_size = 10000
    entry_size = 1024
    
    def setup_python():
        cache = {}
        data = [np.random.bytes(entry_size) for _ in range(cache_size)]
        return {"cache": cache, "data": data}
    
    def python_cache_put(ctx):
        cache, data = ctx["cache"], ctx["data"]
        key = np.random.randint(0, cache_size)
        cache[(0, key)] = data[key % len(data)]
    
    def python_cache_get(ctx):
        cache = ctx["cache"]
        key = np.random.randint(0, cache_size)
        return cache.get((0, key))
    
    r = suite.run_benchmark(
        name=f"kv_cache_put_{cache_size}",
        component="kv_cache",
        backend="python_dict",
        func=python_cache_put,
        setup=setup_python,
        metadata={"cache_size": cache_size, "entry_size": entry_size}
    )
    results.append(r)
    print(f"  {r}")
    
    r = suite.run_benchmark(
        name=f"kv_cache_get_{cache_size}",
        component="kv_cache",
        backend="python_dict",
        func=python_cache_get,
        setup=setup_python,
        metadata={"cache_size": cache_size}
    )
    results.append(r)
    print(f"  {r}")
    
    try:
        from optimization_core.polyglot_core import KVCache, KVCacheConfig, Backend
        
        for backend in [Backend.RUST, Backend.CPP, Backend.PYTHON]:
            try:
                cache = KVCache(KVCacheConfig(max_size=cache_size), backend=backend)
                data = [np.random.bytes(entry_size) for _ in range(100)]
                
                def polyglot_cache_put():
                    key = np.random.randint(0, cache_size)
                    cache.put(0, key, data[key % len(data)])
                
                def polyglot_cache_get():
                    key = np.random.randint(0, cache_size)
                    return cache.get(0, key)
                
                r = suite.run_benchmark(
                    name=f"kv_cache_put_{cache_size}",
                    component="kv_cache",
                    backend=backend.name.lower(),
                    func=polyglot_cache_put,
                    metadata={"cache_size": cache_size}
                )
                results.append(r)
                print(f"  {r}")
                
            except Exception as e:
                print(f"  Skipping {backend.name}: {e}")
    except ImportError:
        print("  polyglot_core not available")
    
    return results


def benchmark_compression(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Benchmark compression implementations."""
    results = []
    
    data_sizes = [1024, 10240, 102400, 1048576]
    
    for size in data_sizes:
        data = np.random.bytes(size)
        
        def setup():
            return {"data": data}
        
        try:
            import zlib
            
            def zlib_compress(ctx):
                return zlib.compress(ctx["data"])
            
            r = suite.run_benchmark(
                name=f"compress_{size}B",
                component="compression",
                backend="zlib",
                func=zlib_compress,
                setup=setup,
                metadata={"data_size": size}
            )
            results.append(r)
            print(f"  {r}")
        except ImportError:
            pass
        
        try:
            import lz4.frame
            
            def lz4_compress(ctx):
                return lz4.frame.compress(ctx["data"])
            
            r = suite.run_benchmark(
                name=f"compress_{size}B",
                component="compression",
                backend="lz4",
                func=lz4_compress,
                setup=setup,
                metadata={"data_size": size}
            )
            results.append(r)
            print(f"  {r}")
        except ImportError:
            print(f"  lz4 not available")
        
        try:
            import zstandard as zstd
            
            cctx = zstd.ZstdCompressor(level=3)
            
            def zstd_compress(ctx):
                return cctx.compress(ctx["data"])
            
            r = suite.run_benchmark(
                name=f"compress_{size}B",
                component="compression",
                backend="zstd",
                func=zstd_compress,
                setup=setup,
                metadata={"data_size": size}
            )
            results.append(r)
            print(f"  {r}")
        except ImportError:
            print(f"  zstandard not available")
    
    return results


def benchmark_inference(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Benchmark inference (sampling) implementations."""
    results = []
    
    vocab_size = 32000
    
    def setup():
        logits = np.random.randn(vocab_size).astype(np.float32)
        return {"logits": logits}
    
    def greedy_sample(ctx):
        return np.argmax(ctx["logits"])
    
    r = suite.run_benchmark(
        name="greedy_sample",
        component="inference",
        backend="numpy",
        func=greedy_sample,
        setup=setup,
        metadata={"vocab_size": vocab_size}
    )
    results.append(r)
    print(f"  {r}")
    
    def softmax_numpy(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def topk_sample(ctx):
        logits = ctx["logits"]
        probs = softmax_numpy(logits)
        top_k_idx = np.argsort(probs)[-50:]
        top_k_probs = probs[top_k_idx]
        top_k_probs /= top_k_probs.sum()
        return np.random.choice(top_k_idx, p=top_k_probs)
    
    r = suite.run_benchmark(
        name="topk_sample_k50",
        component="inference",
        backend="numpy",
        func=topk_sample,
        setup=setup,
        metadata={"vocab_size": vocab_size, "k": 50}
    )
    results.append(r)
    print(f"  {r}")
    
    def topp_sample(ctx):
        logits = ctx["logits"]
        probs = softmax_numpy(logits)
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, 0.9) + 1
        top_idx = sorted_idx[:cutoff]
        top_probs = sorted_probs[:cutoff]
        top_probs /= top_probs.sum()
        return np.random.choice(top_idx, p=top_probs)
    
    r = suite.run_benchmark(
        name="topp_sample_p0.9",
        component="inference",
        backend="numpy",
        func=topp_sample,
        setup=setup,
        metadata={"vocab_size": vocab_size, "p": 0.9}
    )
    results.append(r)
    print(f"  {r}")
    
    return results


def run_all_benchmarks(config: Optional[BenchmarkConfig] = None) -> BenchmarkSuite:
    """Run all benchmarks."""
    suite = BenchmarkSuite(config)
    
    print("\n🚀 Running Polyglot Benchmarks\n")
    
    print("📊 Attention Benchmarks")
    benchmark_attention(suite)
    
    print("\n📦 KV Cache Benchmarks")
    benchmark_kv_cache(suite)
    
    print("\n🗜️ Compression Benchmarks")
    benchmark_compression(suite)
    
    print("\n🎯 Inference Benchmarks")
    benchmark_inference(suite)
    
    suite.print_results()
    
    return suite


def main():
    parser = argparse.ArgumentParser(description="TruthGPT Polyglot Benchmarks")
    parser.add_argument("--component", choices=["attention", "kv_cache", "compression", "inference", "all"],
                       default="all", help="Component to benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        benchmark_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    suite = BenchmarkSuite(config)
    
    print("\n🚀 Running Polyglot Benchmarks\n")
    
    if args.component in ["attention", "all"]:
        print("📊 Attention Benchmarks")
        benchmark_attention(suite)
    
    if args.component in ["kv_cache", "all"]:
        print("\n📦 KV Cache Benchmarks")
        benchmark_kv_cache(suite)
    
    if args.component in ["compression", "all"]:
        print("\n🗜️ Compression Benchmarks")
        benchmark_compression(suite)
    
    if args.component in ["inference", "all"]:
        print("\n🎯 Inference Benchmarks")
        benchmark_inference(suite)
    
    suite.print_results()
    
    if args.output:
        suite.save_results(args.output)


if __name__ == "__main__":
    main()













