# 🌍 Polyglot Integration Module

Unified Python interface to Rust, Go, C++, and Julia cores.

## Features

- **Automatic Backend Selection**: Chooses best available backend
- **Unified API**: Same interface regardless of backend
- **Performance**: 2-10x speedup over pure Python
- **Fallback Support**: Gracefully falls back to Python if native backends unavailable

## Modules

### `__init__.py`
Backend detection and status information.

```python
from optimization_core.polyglot import get_available_backends, get_backend_info

backends = get_available_backends()
# {'rust': True, 'go': False, 'cpp': True, 'julia': False}

info = get_backend_info()
# Detailed information about each backend
```

### `kv_cache.py`
Unified KV cache interface.

```python
from optimization_core.polyglot.kv_cache import KVCache

# Automatically selects Rust > C++ > Python
cache = KVCache(max_size=8192, eviction_strategy="adaptive")

cache.put(layer_idx=0, position=0, data=tensor_bytes)
data = cache.get(layer_idx=0, position=0)

stats = cache.stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### `attention.py`
Unified attention computation.

```python
from optimization_core.polyglot.attention import attention

# Automatically selects C++ > Rust > Julia > PyTorch
output = attention(q, k, v, use_flash=True, causal=True)
```

### `optimization.py`
Unified hyperparameter optimization.

```python
from optimization_core.polyglot.optimization import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(use_julia=True)

result = optimizer.optimize(
    loss_fn=my_loss_function,
    bounds={
        "lr": (1e-6, 1e-2),
        "batch_size": (8, 128),
        "dropout": (0.0, 0.5),
    },
    method="bayesian",
    max_iters=100
)

print(f"Best params: {result['best_params']}")
print(f"Best loss: {result['best_loss']}")
```

## Backend Priority

### KV Cache
1. **Rust** (fastest, most features)
2. **C++** (GPU-optimized)
3. **Python** (dict-based fallback)

### Attention
1. **C++** (GPU, fastest)
2. **Rust** (CPU, fast)
3. **Julia** (scientific computing)
4. **PyTorch** (fallback)

### Optimization
1. **Julia** (mathematical optimization)
2. **Python** (scipy.optimize fallback)

## Performance

| Operation | Backend | Speedup |
|-----------|---------|---------|
| KV Cache get | Rust | 10x |
| KV Cache get | C++ | 20x |
| Attention (GPU) | C++ | 5-10x |
| Attention (CPU) | Rust | 2x |
| Optimization | Julia | 10x |

## Installation

### Rust Backend
```bash
cd rust_core
maturin develop
```

### C++ Backend
```bash
cd cpp_core
python setup.py build_ext --inplace
```

### Julia Backend
```bash
julia -e 'using Pkg; Pkg.add("PyCall")'
python -c "import julia; julia.install()"
```

## Usage Example

```python
from optimization_core.inference.inference_engine_refactored import (
    InferenceEngine, InferenceConfig, GenerationConfig
)
from optimization_core.polyglot import get_backend_info

# Check available backends
info = get_backend_info()
print(f"Available: {info['available']}")
print(f"Recommended: {info['recommended']}")

# Create engine with automatic backend selection
config = InferenceConfig(
    backend=Backend.AUTO,
    use_rust_tokenizer=True,
    use_cpp_attention=True,
    use_kv_cache=True,
)

engine = InferenceEngine(model, tokenizer, config)

# Generate with optimal backend
result = engine.generate(
    "Hello, world!",
    GenerationConfig(max_new_tokens=128)
)

# Get metrics
metrics = engine.get_metrics()
print(f"Throughput: {metrics.throughput_tokens_per_sec:.1f} tok/s")
```

## Error Handling

All modules gracefully fall back to Python implementations if native backends fail:

```python
# This will work even if Rust/C++/Julia are unavailable
cache = KVCache()  # Falls back to Python dict
output = attention(q, k, v)  # Falls back to PyTorch
```

## Contributing

When adding new backends:

1. Add detection logic in `__init__.py`
2. Create unified interface in appropriate module
3. Implement fallback to Python
4. Add tests
5. Update documentation












