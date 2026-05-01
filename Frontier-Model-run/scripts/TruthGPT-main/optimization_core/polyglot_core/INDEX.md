# 📑 Polyglot Core - Index

Quick navigation guide for all polyglot_core components.

## 🚀 Quick Links

### Getting Started
- [Quick Start Guide](QUICK_START.md) - 5-minute tutorial
- [README](README.md) - Complete documentation
- [API Reference](API_REFERENCE.md) - Full API documentation

### Examples
- [Complete Example](examples/example_complete.py) - Full pipeline
- [Benchmark Example](examples/example_benchmark.py) - Backend comparison
- [Profiling Example](examples/example_profiling.py) - Performance profiling
- [Reporting Example](examples/example_reporting.py) - Report generation

### Documentation
- [Changelog](CHANGELOG.md) - Version history
- [Summary](SUMMARY.md) - Refactoring summary
- [Final Summary](FINAL_REFACTORING_SUMMARY.md) - Complete overview

## 📦 Modules

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| **Backend** | `backend.py` | Backend detection & selection |
| **Cache** | `cache.py` | KV Cache (Rust > C++ > Python) |
| **Attention** | `attention.py` | Attention (C++ > Rust > Python) |
| **Compression** | `compression.py` | Compression (Rust > C++ > Python) |
| **Inference** | `inference.py` | Text generation engine |
| **Tokenization** | `tokenization.py` | Tokenization (Rust > Python) |
| **Quantization** | `quantization.py` | Quantization (C++ > Rust > Python) |

### Utility Modules

| Module | File | Purpose |
|--------|------|---------|
| **Profiling** | `profiling.py` | Performance profiling |
| **Benchmarking** | `benchmarking.py` | Backend comparison |
| **Metrics** | `metrics.py` | Metrics collection |
| **Reporting** | `reporting.py` | Report generation |
| **Utils** | `utils.py` | Common utilities |
| **Integration** | `integration.py` | Test compatibility |
| **Distributed** | `distributed.py` | Go service clients |

## 🧪 Tests

| Test | File | Coverage |
|------|------|----------|
| Backend | `tests/test_backend.py` | Backend detection |
| Cache | `tests/test_cache.py` | KV Cache operations |
| Attention | `tests/test_attention.py` | Attention mechanisms |
| Compression | `tests/test_compression.py` | Compression algorithms |
| Integration | `tests/test_integration.py` | End-to-end workflows |

## 📊 Performance

### Speedups vs Python

| Operation | Rust | C++ | Best |
|-----------|------|-----|------|
| KV Cache | 50x | 45x | Rust |
| Compression | 6.5x | 6.5x | Tie |
| Attention | 3x | 21x* | C++ |
| Tokenization | 5x | - | Rust |

*With CUDA

## 🎯 Common Use Cases

### 1. LLM Inference with KV Cache

```python
from optimization_core.polyglot_core import KVCache, Attention, InferenceEngine

cache = KVCache(max_size=100000)
attention = Attention(d_model=768, n_heads=12)
engine = InferenceEngine(seed=42)
```

### 2. Performance Profiling

```python
from optimization_core.polyglot_core import get_profiler

profiler = get_profiler()
with profiler.profile("operation"):
    # Your code
    pass
profiler.print_summary()
```

### 3. Backend Comparison

```python
from optimization_core.polyglot_core import Benchmark

benchmark = Benchmark()
results = benchmark.compare_backends("kv_cache", create_cache)
benchmark.print_comparison(results)
```

### 4. Metrics Collection

```python
from optimization_core.polyglot_core import get_metrics_collector

collector = get_metrics_collector()
collector.record_latency("operation", 10.5, backend="rust")
summary = collector.get_summary("operation")
```

### 5. Report Generation

```python
from optimization_core.polyglot_core import ReportGenerator

generator = ReportGenerator()
report = generator.generate_benchmark_report(results)
report.save("report.html", format="html")
```

## 🔧 Configuration

### KV Cache

```python
from optimization_core.polyglot_core import KVCacheConfig

# Inference optimized
config = KVCacheConfig.inference_optimized(8)  # 8GB

# Long context
config = KVCacheConfig.long_context(32)  # 32GB
```

### Attention

```python
from optimization_core.polyglot_core import AttentionConfig

# Preset configs
config = AttentionConfig.llama_7b()
config = AttentionConfig.mistral_7b()  # With sliding window
```

### Generation

```python
from optimization_core.polyglot_core import GenerationConfig

# Preset configs
config = GenerationConfig.greedy()
config = GenerationConfig.creative()
config = GenerationConfig.factual()
```

## 📚 Additional Resources

- [POLYGLOT_ARCHITECTURE.md](../POLYGLOT_ARCHITECTURE.md) - Architecture overview
- [CPP_LIBRARIES_RECOMMENDATIONS.md](../CPP_LIBRARIES_RECOMMENDATIONS.md) - C++ libraries
- [RUST_OPTIMIZATION_RECOMMENDATIONS.md](../RUST_OPTIMIZATION_RECOMMENDATIONS.md) - Rust optimizations

## 🆘 Troubleshooting

### Backend Not Available

```python
from optimization_core.polyglot_core import print_backend_status

print_backend_status()  # Check what's available
```

### Import Errors

```python
from optimization_core.polyglot_core import check_polyglot_availability

availability = check_polyglot_availability()
print(availability)
```

### Test Compatibility

```python
from optimization_core.polyglot_core import get_test_compatibility_info

info = get_test_compatibility_info()
print(info['recommendations'])
```

---

**Version**: 2.0.0  
**Last Updated**: 2025-01-XX












