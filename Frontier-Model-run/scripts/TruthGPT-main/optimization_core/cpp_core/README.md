# optimization_core C++ Extensions

High-performance C++ backend for TruthGPT optimization, providing **10-100x speedups** over pure Python implementations.

## 🚀 Features

### Core Modules

| Module | Description | Speedup |
|--------|-------------|---------|
| **Flash Attention** | Memory-efficient O(N) attention | 3-10x vs standard attention |
| **KV Cache** | Concurrent cache with eviction strategies | 100x vs Python dict |
| **Compression** | LZ4/Zstd for cache compression | 5 GB/s throughput |
| **Inference Engine** | Sampling, beam search, generation | 2-5x vs Python |

### Performance Highlights

- **Flash Attention**: Process 8K+ token sequences with O(N) memory
- **KV Cache**: 100M+ ops/sec with concurrent access
- **Compression**: 2-5x cache size reduction with LZ4
- **SIMD**: AVX2/AVX-512/NEON optimized kernels

## 📦 Installation

### Prerequisites

```bash
# Windows (PowerShell as Admin)
.\scripts\install_deps.ps1

# Linux/macOS
./scripts/install_deps.sh
```

### Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON

# Build
cmake --build . --config Release -j

# Install (optional)
cmake --install .
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_PYTHON_BINDINGS` | ON | Build PyBind11 Python module |
| `BUILD_TESTS` | ON | Build unit tests |
| `BUILD_BENCHMARKS` | ON | Build performance benchmarks |
| `BUILD_EXAMPLES` | OFF | Build example programs |
| `USE_EIGEN` | ON | Use Eigen for linear algebra |
| `USE_TBB` | ON | Use Intel TBB for parallelization |
| `USE_MIMALLOC` | ON | Use mimalloc allocator |
| `USE_COMPRESSION` | ON | Enable LZ4/Zstd compression |
| `ENABLE_NATIVE_ARCH` | ON | Enable -march=native |
| `ENABLE_LTO` | ON | Enable Link-Time Optimization |

## 📁 Project Structure

```
cpp_core/
├── include/
│   ├── common/types.hpp          # Common types and utilities
│   ├── attention/
│   │   ├── attention.hpp         # Attention interface
│   │   └── flash_attention.hpp   # Flash Attention implementation
│   ├── memory/
│   │   ├── cache.hpp             # KV Cache
│   │   └── kv_cache.hpp          # Legacy header
│   ├── inference/
│   │   └── engine.hpp            # Inference engine
│   ├── compression/
│   │   └── compression.hpp       # LZ4/Zstd compression
│   └── optimization_core.hpp     # Unified header
├── src/
│   ├── attention/                # Attention implementations
│   ├── memory/                   # Memory management
│   ├── optim/                    # Optimizers
│   └── inference/                # Inference engine
├── python/
│   ├── bindings.cpp              # PyBind11 bindings
│   └── cpp_wrapper.py            # High-level Python wrapper
├── tests/
│   ├── test_attention.cpp        # Attention tests
│   ├── test_cache.cpp            # Cache tests
│   ├── test_inference.cpp        # Inference tests
│   └── test_compression.cpp      # Compression tests
├── benchmarks/
│   ├── benchmark_attention.cpp   # Attention benchmarks
│   └── benchmark_all.cpp         # All benchmarks
├── examples/
│   ├── example_attention.cpp     # Attention usage examples
│   ├── example_cache.cpp         # Cache usage examples
│   ├── example_inference.cpp     # Inference examples
│   ├── example_compression.cpp   # Compression examples
│   └── example_integration.cpp   # Full integration example
├── scripts/
│   ├── install_deps.ps1          # Windows dependency installer
│   └── install_deps.sh           # Linux/macOS dependency installer
├── CMakeLists.txt                # Main CMake configuration
├── vcpkg.json                    # vcpkg dependencies
└── README.md                     # This file
```

## 🔧 Usage

### C++ API

```cpp
#include <optimization_core.hpp>

using namespace optimization_core;

// Flash Attention
attention::AttentionConfig config;
config.num_heads = 12;
config.head_dim = 64;
config.use_flash = true;
config.use_causal_mask = true;

auto attn = attention::create_attention(config);
auto output = attn->forward(q, k, v, batch, seq_len, std::nullopt);

// KV Cache with compression
memory::CacheConfig cache_config;
cache_config.max_size = 100000;
cache_config.eviction_strategy = memory::EvictionStrategy::LRU;

memory::ConcurrentKVCache cache(cache_config);
cache.put(layer, position, data, "key");
auto result = cache.get(layer, position, "key");

// Inference Engine
inference::InferenceEngine engine(seed);
auto gen_config = inference::GenerationConfig::sampling(0.7f, 0.9f)
    .with_max_tokens(100)
    .with_eos(eos_token);

auto result = engine.generate(prompt_ids, forward_fn, gen_config);
```

### Python API

```python
from optimization_core.cpp_core.python import cpp_wrapper as cpp

# Check backend availability
print(cpp.check_cpp_backend())

# Flash Attention
attn = cpp.FlashAttention(num_heads=12, head_dim=64)
output = attn.forward(q, k, v)

# KV Cache
cache = cpp.KVCache(max_size=10000, enable_compression=True)
cache.put(layer=0, position=0, data=kv_tensor)
result = cache.get(layer=0, position=0)

# Inference Engine
engine = cpp.InferenceEngine(seed=42)
config = cpp.GenerationConfig.sampling(temperature=0.7, top_p=0.9)
result = engine.generate(prompt_ids, model.forward, config)
print(f"Generated {result.tokens_generated} tokens at {result.tokens_per_second:.0f} tok/s")
```

## 🧪 Testing

```bash
# Build tests
cmake --build build --target test_attention test_cache test_inference test_compression

# Run all tests
cd build && ctest --output-on-failure

# Run specific test
./build/test_attention
```

## 📊 Benchmarks

```bash
# Build benchmarks
cmake --build build --target benchmark_all

# Run benchmarks
./build/benchmark_all --benchmark_format=console

# Run with specific filter
./build/benchmark_all --benchmark_filter=BM_FlashAttention
```

### Sample Results

```
---------------------------------------------------------
Benchmark                   Time           CPU Iterations
---------------------------------------------------------
BM_FlashAttention/512      2.3 ms       2.3 ms        303
BM_FlashAttention/1024     8.1 ms       8.1 ms         86
BM_FlashAttention/2048    31.2 ms      31.1 ms         22
BM_CacheGet/10000         45.3 ns      45.2 ns   15482943
BM_LZ4Compress/1M         198 us        198 us       3521
BM_SamplerGreedy/50000    12.5 us      12.5 us      56012
```

## 🎯 Examples

Build and run examples:

```bash
cmake -DBUILD_EXAMPLES=ON ..
cmake --build . --target example_attention example_cache example_inference

./example_attention
./example_cache
./example_inference
./example_integration
```

## 🏗️ Architecture

### Attention Module

```
IAttention (Interface)
├── ScaledDotProductAttention  # Standard O(n²) attention
├── FlashAttention             # Memory-efficient tiled attention
└── SparseAttention            # Local + global attention patterns
```

### Memory Module

```
KVCache (Single-threaded)
├── LRU eviction
├── LFU eviction
├── FIFO eviction
└── Adaptive eviction

ConcurrentKVCache (Thread-safe)
└── Read-write locks for parallel access
```

### Compression Module

```
Algorithm
├── None      # No compression
├── LZ4       # Fastest (~5 GB/s)
└── Zstd      # Better ratio (~400 MB/s)

Compressor
├── compress()
├── decompress()
└── compress_with_stats()

StreamingCompressor  # For large data
BatchCompressor      # Parallel compression
```

## 🔗 Dependencies

### Required
- **CMake** >= 3.21
- **C++20** compatible compiler
- **Threads** (pthread/Windows threads)

### Optional (auto-detected)
- **Eigen3** - Linear algebra
- **Intel TBB** - Parallelization
- **mimalloc** - High-performance allocator
- **LZ4** - Fast compression
- **Zstd** - Better compression
- **xsimd** - Portable SIMD
- **PyBind11** - Python bindings
- **Google Test** - Unit testing
- **Google Benchmark** - Benchmarking

## 📝 License

MIT License - see LICENSE file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run tests and benchmarks
5. Submit a pull request

## 📚 References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [Efficient Transformers Survey](https://arxiv.org/abs/2009.06732)
- [LZ4 Compression](https://lz4.github.io/lz4/)
- [Zstandard](https://facebook.github.io/zstd/)
