# 📚 Polyglot Core - API Reference

Complete API reference for all polyglot_core modules.

## Table of Contents

1. [Backend Management](#backend-management)
2. [KV Cache](#kv-cache)
3. [Attention](#attention)
4. [Compression](#compression)
5. [Inference](#inference)
6. [Tokenization](#tokenization)
7. [Quantization](#quantization)
8. [Profiling](#profiling)
9. [Benchmarking](#benchmarking)
10. [Metrics](#metrics)
11. [Reporting](#reporting)
12. [Utilities](#utilities)

---

## Backend Management

### `Backend` (Enum)

Available backend implementations.

```python
from optimization_core.polyglot_core import Backend

Backend.PYTHON  # Pure Python fallback
Backend.RUST    # Rust via PyO3
Backend.CPP     # C++ via PyBind11
Backend.GO      # Go via gRPC/HTTP
```

### `get_available_backends() -> List[BackendInfo]`

Get list of all available backends.

```python
from optimization_core.polyglot_core import get_available_backends

for backend in get_available_backends():
    print(f"{backend.name}: {'✓' if backend.available else '✗'}")
```

### `get_best_backend(feature: str) -> Backend`

Get best available backend for a feature.

```python
from optimization_core.polyglot_core import get_best_backend

best = get_best_backend('kv_cache')  # Returns Backend.RUST
best = get_best_backend('attention')  # Returns Backend.CPP
```

### `print_backend_status()`

Print formatted backend status.

```python
from optimization_core.polyglot_core import print_backend_status

print_backend_status()
```

---

## KV Cache

### `KVCache`

High-performance KV cache with automatic backend selection.

```python
from optimization_core.polyglot_core import KVCache, KVCacheConfig

# Basic usage
cache = KVCache(max_size=10000)
cache.put(layer=0, position=0, key=k_tensor, value=v_tensor)
result = cache.get(layer=0, position=0)

# With config
config = KVCacheConfig.inference_optimized(8)  # 8GB
cache = KVCache(config)

# Force backend
cache = KVCache(max_size=10000, backend=Backend.RUST)
```

**Methods:**
- `put(layer, position, key, value, tag="", priority=1.0)` - Store KV state
- `get(layer, position, tag="")` - Retrieve KV state
- `remove(layer, position, tag="")` - Remove entry
- `clear()` - Clear all entries
- `contains(layer, position, tag="")` - Check if exists

**Properties:**
- `size` - Number of entries
- `hit_rate` - Cache hit rate (0.0-1.0)
- `stats` - CacheStats object
- `backend` - Current backend

### `KVCacheConfig`

Configuration for KV cache.

```python
from optimization_core.polyglot_core import KVCacheConfig, EvictionStrategy

config = KVCacheConfig(
    max_size=100000,
    max_memory_bytes=8 * 1024**3,  # 8GB
    eviction_strategy=EvictionStrategy.S3FIFO,
    enable_compression=True,
    num_shards=64
)

# Preset configs
config = KVCacheConfig.inference_optimized(8)  # 8GB
config = KVCacheConfig.long_context(32)       # 32GB
```

---

## Attention

### `Attention`

Unified attention with automatic backend selection.

```python
from optimization_core.polyglot_core import Attention, AttentionConfig

# Basic usage
attention = Attention(d_model=768, n_heads=12)
output = attention.forward(q, k, v, batch_size=4, seq_len=512)

# With config
config = AttentionConfig.llama_7b()
attention = Attention(config)

# Flash Attention
from optimization_core.polyglot_core import FlashAttention
flash_attn = FlashAttention(AttentionConfig(d_model=768, n_heads=12))
```

**Methods:**
- `forward(query, key, value, batch_size, seq_len, attention_mask=None, return_attention_weights=False)` - Forward pass

**Returns:** `AttentionOutput` with:
- `output` - Output tensor
- `attention_weights` - Optional attention weights
- `compute_time_ms` - Computation time
- `memory_bytes` - Memory used

### `AttentionConfig`

Configuration for attention.

```python
from optimization_core.polyglot_core import AttentionConfig, AttentionPattern, PositionEncoding

config = AttentionConfig(
    d_model=4096,
    n_heads=32,
    n_kv_heads=8,  # GQA
    pattern=AttentionPattern.CAUSAL,
    position_encoding=PositionEncoding.ROPE,
    use_causal_mask=True,
    block_size=64  # For Flash Attention
)

# Preset configs
config = AttentionConfig.llama_7b()
config = AttentionConfig.llama_70b()  # With GQA
config = AttentionConfig.mistral_7b()  # With sliding window
```

---

## Compression

### `Compressor`

High-performance compressor with automatic backend selection.

```python
from optimization_core.polyglot_core import Compressor, CompressionAlgorithm

# Basic usage
compressor = Compressor(algorithm="lz4")
result = compressor.compress(data)
decompressed = compressor.decompress(result.data)

# With config
from optimization_core.polyglot_core import CompressionConfig
config = CompressionConfig(algorithm=CompressionAlgorithm.ZSTD, level=9)
compressor = Compressor(config)
```

**Methods:**
- `compress(data: bytes) -> CompressionResult`
- `decompress(data: bytes) -> bytes`
- `compress_with_stats(data: bytes) -> Tuple[bytes, CompressionStats]`

### Convenience Functions

```python
from optimization_core.polyglot_core import compress, decompress

compressed = compress(data, algorithm="lz4")
original = decompress(compressed, algorithm="lz4")
```

---

## Inference

### `InferenceEngine`

Text generation engine with various sampling strategies.

```python
from optimization_core.polyglot_core import InferenceEngine, GenerationConfig

engine = InferenceEngine(seed=42)

# Mock forward function
def model_forward(tokens):
    vocab_size = 50257
    logits = np.random.randn(vocab_size).astype(np.float32)
    return logits

# Generate
config = GenerationConfig.sampling(temperature=0.7, top_p=0.9)
result = engine.generate(prompt_ids, model_forward, config)

print(f"Generated: {result.tokens_generated} tokens")
print(f"Speed: {result.tokens_per_second:.0f} tok/s")
```

**Methods:**
- `generate(input_ids, forward_fn, config, **kwargs) -> GenerationResult`
- `generate_batch(batch_input_ids, forward_fn, config) -> List[GenerationResult]`

### `GenerationConfig`

Configuration for text generation.

```python
from optimization_core.polyglot_core import GenerationConfig

# Preset configs
config = GenerationConfig.greedy()              # Deterministic
config = GenerationConfig.sampling(0.7, 0.9)    # Temperature + top-p
config = GenerationConfig.beam_search(4)       # Beam search
config = GenerationConfig.creative()            # Creative/diverse
config = GenerationConfig.factual()             # Factual/deterministic

# Custom config
config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True
)
```

---

## Tokenization

### `Tokenizer`

Unified tokenizer with automatic backend selection.

```python
from optimization_core.polyglot_core import Tokenizer

tokenizer = Tokenizer(model_name="gpt2")
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)
tokens_list = tokenizer.tokenize("Hello, world!")
```

**Methods:**
- `encode(text, ...) -> List[int]` - Encode to token IDs
- `decode(token_ids, ...) -> str` - Decode to text
- `tokenize(text) -> List[str]` - Tokenize to tokens

**Properties:**
- `vocab_size` - Vocabulary size
- `pad_token_id` - Padding token ID
- `eos_token_id` - End-of-sequence token ID
- `backend` - Current backend

---

## Quantization

### `Quantizer`

Unified quantizer with automatic backend selection.

```python
from optimization_core.polyglot_core import Quantizer, QuantizationType

quantizer = Quantizer(quantization_type="int8")
quantized, stats = quantizer.quantize(weights)
dequantized = quantizer.dequantize(quantized)

# Quantize PyTorch model
quantized_model = quantizer.quantize_model(model, calibration_data)
```

**Methods:**
- `quantize(weights, calibration_data=None) -> Tuple[np.ndarray, QuantizationStats]`
- `dequantize(quantized, scale=None) -> np.ndarray`
- `quantize_model(model, calibration_data) -> Any`

---

## Profiling

### `Profiler`

Performance profiler for operations.

```python
from optimization_core.polyglot_core import get_profiler

profiler = get_profiler()

# Context manager
with profiler.profile("operation", backend="rust"):
    result = cache.get(layer=0, position=0)

metrics = profiler.get_metrics("operation")
print(f"Time: {metrics.duration_ms:.2f}ms")

# Function profiling
metrics = profiler.profile_function(cache.get, (0, 0), iterations=100)

# Resource monitoring
profiler.start_monitoring(interval=1.0)
# ... do work ...
profiler.stop_monitoring()
peak_memory = profiler.get_peak_memory()

# Summary
profiler.print_summary()
```

---

## Benchmarking

### `Benchmark`

Comprehensive benchmarking tool.

```python
from optimization_core.polyglot_core import Benchmark, Backend

benchmark = Benchmark()

# Single benchmark
result = benchmark.run(
    "kv_cache_get",
    cache.get,
    (0, 0),
    iterations=1000
)

# Compare backends
def create_cache(backend):
    return KVCache(max_size=10000, backend=backend)

results = benchmark.compare_backends(
    "kv_cache",
    create_cache,
    iterations=1000
)

benchmark.print_comparison(results)
benchmark.save_results(results, "results.json")
```

---

## Metrics

### `MetricsCollector`

Collects and aggregates metrics.

```python
from optimization_core.polyglot_core import get_metrics_collector, record_metric

collector = get_metrics_collector()

# Record metrics
collector.record("cache_hit", 1.0, tags={"backend": "rust"})
collector.record_latency("operation", 10.5, backend="rust")
collector.record_throughput("operation", 1000.0, backend="rust")

# Get summary
summary = collector.get_summary("cache_hit")
print(f"Average: {summary.avg}")

# Export
json_str = collector.export_json()
collector.print_summary()
```

---

## Reporting

### `ReportGenerator`

Generates comprehensive reports.

```python
from optimization_core.polyglot_core import ReportGenerator

generator = ReportGenerator()

# Benchmark report
report = generator.generate_benchmark_report(benchmark_results)
report.save("report", format="html")  # or "markdown", "json"

# Profiling report
report = generator.generate_profiling_report(profiler_summary)
report.save("profiling_report", format="html")

# Metrics report
report = generator.generate_metrics_report(metrics_summaries)
report.save("metrics_report", format="html")
```

---

## Utilities

### Formatting

```python
from optimization_core.polyglot_core import format_bytes, format_time

formatted = format_bytes(1024 * 1024 * 1024)  # "1.00 GB"
formatted = format_time(0.001)                # "1.00 ms"
```

### Device Info

```python
from optimization_core.polyglot_core import get_device_info, print_device_info

info = get_device_info()
print_device_info()
```

### Tensor Utilities

```python
from optimization_core.polyglot_core import (
    pad_sequence, truncate_sequence, batch_tensors,
    ensure_contiguous, validate_shape
)

padded = pad_sequence(sequences, max_length=512)
truncated = truncate_sequence(sequence, max_length=256)
batched = batch_tensors(tensors, pad=True)
contiguous = ensure_contiguous(tensor)
validate_shape(tensor, (4, 512, 768), name="input")
```

---

## Integration

### Compatibility Helpers

```python
from optimization_core.polyglot_core import (
    check_polyglot_availability,
    print_polyglot_status,
    get_test_compatibility_info,
    UnifiedKVCache,  # Backward compatibility
    UnifiedCompressor  # Backward compatibility
)

# Check availability
availability = check_polyglot_availability()
print_polyglot_status()

# Test compatibility
info = get_test_compatibility_info()

# Backward compatibility
cache = UnifiedKVCache(max_size=10000)  # Maps to KVCache
```

### Test Helpers

```python
from optimization_core.polyglot_core import get_test_helper

helper = get_test_helper()

# Skip if unavailable
helper.skip_if_unavailable("cache")

# Require backend
helper.require_backend("rust")

# Create test components
cache = helper.create_test_cache()
attention = helper.create_test_attention(d_model=256, n_heads=4)
```

---

For more examples, see the [examples/](examples/) directory.












