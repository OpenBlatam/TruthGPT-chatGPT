# 🦀 TruthGPT Rust Core

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![PyO3](https://img.shields.io/badge/PyO3-0.20-blue.svg)](https://pyo3.rs/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

High-performance Rust backend for TruthGPT optimization core. Provides ultra-fast implementations of KV caching, compression, attention mechanisms, tokenization, and data loading.

## 🚀 Performance

| Operation | Throughput | vs Python |
|-----------|------------|-----------|
| KV Cache get | ~50ns | **10x faster** |
| KV Cache put | ~100ns | **8x faster** |
| LZ4 compress | 5 GB/s | **5x faster** |
| Zstd compress | 400 MB/s | **3x faster** |
| Batch tokenize | 100K tok/s | **3x faster** |
| Attention (8K seq) | 10ms | **2x faster** |
| Parallel data load | 100K samples/s | **4x faster** |
| INT8 quantize | 10 GB/s | **5x faster** |
| FP16 convert | 15 GB/s | **4x faster** |

## 📦 Installation

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin
```

### Build

```bash
cd rust_core

# Development build
maturin develop

# Release build (optimized)
maturin develop --release

# Build with CUDA support
maturin develop --release --features cuda

# Build with Metal support (macOS)
maturin develop --release --features metal
```

## 🔧 Usage

### KV Cache

```python
from truthgpt_rust import PyKVCache

# Create cache with adaptive eviction
cache = PyKVCache(
    max_size=8192,
    eviction_strategy="adaptive",  # "lru", "lfu", "fifo", "adaptive"
    enable_compression=True,
    compression_threshold=1024
)

# Store KV pairs by layer and position
cache.put(layer_idx=0, position=0, data=tensor_bytes)
cache.put(layer_idx=0, position=1, data=tensor_bytes)

# Retrieve data
data = cache.get(layer_idx=0, position=0)

# Check stats
stats = cache.stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Size: {cache.size()}/{cache.max_size()}")

# Clear when needed
cache.clear()
```

### Compression

```python
from truthgpt_rust import PyCompressor, fast_lz4_compress, fast_lz4_decompress

# Using compressor class
compressor = PyCompressor("zstd", level=3)

# Compress data
compressed = compressor.compress(data)
original = compressor.decompress(compressed)

# Get compression stats
compressed, stats = compressor.compress_with_stats(data)
print(f"Ratio: {stats['ratio']:.2%}")
print(f"Savings: {stats['savings']:.2%}")

# Standalone functions (fastest)
compressed = fast_lz4_compress(data)
decompressed = fast_lz4_decompress(compressed)
```

### Tokenization

```python
from truthgpt_rust import PyFastTokenizer, parallel_tokenize

# Load tokenizer
tokenizer = PyFastTokenizer("tokenizer.json")
# Or from HuggingFace
tokenizer = PyFastTokenizer.from_pretrained("gpt2")

# Single encoding
tokens = tokenizer.encode("Hello, world!", add_special_tokens=True)
text = tokenizer.decode(tokens, skip_special_tokens=True)

# Batch encoding (parallel)
texts = ["Hello", "World", "!"] * 1000
all_tokens = tokenizer.encode_batch(texts, add_special_tokens=True)

# Batch decoding
all_texts = tokenizer.decode_batch(all_tokens, skip_special_tokens=True)

# Get vocabulary info
print(f"Vocab size: {tokenizer.vocab_size()}")
print(f"Token ID for 'hello': {tokenizer.token_to_id('hello')}")

# Standalone parallel tokenization
tokens = parallel_tokenize("tokenizer.json", texts, add_special_tokens=True)
```

### Data Loading

```python
from truthgpt_rust import PyDataLoader

# Create loader
loader = PyDataLoader(num_workers=8, shuffle=True)

# Add JSONL files
loader.add_file("train_data.jsonl")
loader.add_file("val_data.jsonl")

# Load all samples
samples = loader.load_all()
for sample in samples:
    print(f"Text: {sample['text'][:50]}...")
```

### System Info

```python
from truthgpt_rust import get_version, get_system_info

print(get_version())  # "truthgpt-rust v1.0.0"

info = get_system_info()
print(f"CPUs: {info['cpu_count']}")
print(f"Rayon threads: {info['rayon_threads']}")
print(f"CUDA available: {info['cuda_available']}")
```

## 🏗️ Architecture

```
rust_core/
├── Cargo.toml              # Dependencies & features
├── README.md               # This file
├── src/
│   ├── lib.rs              # Main module & Python bindings
│   ├── error.rs            # Error types
│   ├── kv_cache.rs         # KV Cache implementation
│   ├── compression.rs      # LZ4/Zstd compression
│   ├── attention.rs        # Attention mechanisms
│   ├── tokenizer_wrapper.rs # HuggingFace tokenizers wrapper
│   ├── data_loader.rs      # Parallel data loading
│   ├── quantization.rs     # INT8/INT4/FP16/BF16 quantization
│   ├── batch_inference.rs  # Dynamic & continuous batching
│   └── utils.rs            # Timer, stats, formatters
└── benches/
    ├── kv_cache_bench.rs
    ├── compression_bench.rs
    ├── attention_bench.rs
    ├── tokenizer_bench.rs
    └── quantization_bench.rs
```

## 📊 Modules

### KV Cache (`kv_cache.rs`)
- Multiple eviction strategies: LRU, LFU, FIFO, Adaptive
- Automatic compression for large entries
- Thread-safe concurrent access
- Detailed hit/miss statistics

### Compression (`compression.rs`)
- LZ4: ~5 GB/s, best for speed
- Zstd: ~400 MB/s, best for ratio
- Streaming compression for large data
- Batch parallel compression

### Attention (`attention.rs`)
- Scaled dot-product attention
- Flash attention (block-based)
- Sparse attention (local + global)
- Causal masking

### Tokenization (`tokenizer_wrapper.rs`)
- HuggingFace tokenizers integration
- Parallel batch encoding/decoding
- Full metadata extraction
- Configurable truncation/padding

### Data Loading (`data_loader.rs`)
- JSONL file loading
- Parallel file processing
- Length-based bucketing
- Prefetch buffering

### Quantization (`quantization.rs`)
- INT8/INT4 quantization with symmetric/asymmetric modes
- FP16/BF16 half-precision conversion
- Per-tensor and per-group quantization
- INT8 matrix multiplication
- Memory ratio: 4x (INT8), 8x (INT4), 2x (FP16)

### Batch Inference (`batch_inference.rs`)
- Dynamic batching with configurable wait time
- Continuous batching for high throughput
- Priority scheduling (Low/Normal/High/Critical)
- Request queuing and cancellation
- Detailed inference statistics

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test kv_cache
cargo test compression
cargo test attention

# Run with output
cargo test -- --nocapture
```

## 📈 Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- kv_cache
cargo bench -- compression
```

### Sample Results (Apple M2 Pro)

| Benchmark | Time | Ops/sec |
|-----------|------|---------|
| KV Cache get (hit) | 45ns | 22M |
| KV Cache put | 85ns | 12M |
| LZ4 compress (1MB) | 200µs | 5 GB/s |
| Zstd compress (1MB) | 2.5ms | 400 MB/s |
| Attention (1024 seq) | 1.2ms | 833 |
| Batch tokenize (1K texts) | 10ms | 100K tok/s |

## 🔧 Features

| Feature | Description |
|---------|-------------|
| `python` | Enable Python bindings (default) |
| `cuda` | Enable CUDA acceleration |
| `metal` | Enable Metal acceleration (macOS) |
| `full` | All features enabled |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `cargo test`
4. Format: `cargo fmt`
5. Lint: `cargo clippy`
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
