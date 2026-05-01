# 🚀 Polyglot Core - Quick Start Guide

## Installation

```bash
# Install with all backends
pip install optimization-core[all]

# Or install specific backends
pip install optimization-core[rust]  # Rust backend
pip install optimization-core[cpp]  # C++ backend
```

## 5-Minute Tutorial

### 1. Check Available Backends

```python
from optimization_core.polyglot_core import print_backend_status

print_backend_status()
```

### 2. Use KV Cache

```python
from optimization_core.polyglot_core import KVCache
import numpy as np

# Auto-selects best backend (Rust > C++ > Python)
cache = KVCache(max_size=10000)

# Store data
k = np.random.randn(64).astype(np.float32)
v = np.random.randn(64).astype(np.float32)
cache.put(layer=0, position=0, key=k, value=v)

# Retrieve data
result = cache.get(layer=0, position=0)
print(f"Hit rate: {cache.hit_rate:.2%}")
```

### 3. Use Attention

```python
from optimization_core.polyglot_core import Attention, AttentionConfig
import numpy as np

# Auto-selects C++ (CUDA) > C++ (CPU) > Rust > Python
attention = Attention(AttentionConfig(d_model=768, n_heads=12))

q = np.random.randn(4 * 512, 768).astype(np.float32)
output = attention.forward(q, q, q, batch_size=4, seq_len=512)
print(f"Compute time: {output.compute_time_ms:.2f}ms")
```

### 4. Use Compression

```python
from optimization_core.polyglot_core import Compressor

# Auto-selects Rust > C++ > Python
compressor = Compressor(algorithm="lz4")
data = b"Hello, world! " * 1000

result = compressor.compress(data)
print(f"Compression ratio: {result.stats.compression_ratio:.2%}")
print(f"Throughput: {result.stats.compression_throughput_mbps:.0f} MB/s")
```

### 5. Generate Text

```python
from optimization_core.polyglot_core import InferenceEngine, GenerationConfig

engine = InferenceEngine(seed=42)

# Mock model forward function
def model_forward(tokens):
    vocab_size = 50257
    logits = np.random.randn(vocab_size).astype(np.float32)
    logits[42] = 10.0  # Favor token 42
    return logits

# Generate
config = GenerationConfig.sampling(temperature=0.7, top_p=0.9)
result = engine.generate([1, 2, 3], model_forward, config)

print(f"Generated {result.tokens_generated} tokens")
print(f"Speed: {result.tokens_per_second:.0f} tokens/sec")
```

## Common Patterns

### Force Specific Backend

```python
from optimization_core.polyglot_core import KVCache, Backend

# Force Rust backend
cache = KVCache(max_size=10000, backend=Backend.RUST)

# Force C++ backend
cache = KVCache(max_size=10000, backend=Backend.CPP)
```

### Profile Operations

```python
from optimization_core.polyglot_core import get_profiler

profiler = get_profiler()

with profiler.profile("my_operation"):
    # Your code here
    result = cache.get(layer=0, position=0)

metrics = profiler.get_metrics("my_operation")
print(f"Time: {metrics.duration_ms:.2f}ms")
```

### Benchmark Backends

```python
from optimization_core.polyglot_core import Benchmark, Backend

benchmark = Benchmark()

def create_cache(backend):
    return KVCache(max_size=10000, backend=backend)

results = benchmark.compare_backends(
    "kv_cache",
    create_cache,
    iterations=1000
)

benchmark.print_comparison(results)
```

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [examples/](examples/) for more examples
- Run tests: `pytest polyglot_core/tests/`












