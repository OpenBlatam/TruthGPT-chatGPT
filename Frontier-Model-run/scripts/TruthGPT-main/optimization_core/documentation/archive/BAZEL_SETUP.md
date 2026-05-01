# 🚀 Bazel Setup Guide - TruthGPT Optimization Core

## Overview

This project uses Bazel to build all modules including:
- **Python** modules (core, factories, trainers, modules, optimizers, etc.)
- **C++** core (cpp_core)
- **Rust** core (rust_core)
- **Go** core (go_core)
- **Julia** core (julia_core)
- **Scala** core (scala_core)
- **Elixir** core (elixir_core)

## Prerequisites

1. **Install Bazel** (version 7.0.0 or higher)
   ```bash
   # macOS
   brew install bazel
   
   # Linux
   curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel-archive-keyring.gpg
   sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
   echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
   sudo apt update && sudo apt install bazel
   
   # Windows (via Chocolatey)
   choco install bazel
   ```

2. **Install Language Runtimes**
   - Python 3.10+
   - Rust 1.75+ (for rust_core)
   - Go 1.22+ (for go_core)
   - Julia 1.9+ (for julia_core, optional)
   - Scala 2.13+ (for scala_core, optional)
   - Elixir 1.14+ (for elixir_core, optional)

## Quick Start

### 1. Build Everything

```bash
# Build all targets
bazel build //...

# Build specific module
bazel build //core:core
bazel build //cpp_core:cpp_core
bazel build //rust_core:rust_core
bazel build //go_core:go_core
```

### 2. Run Python Targets

```bash
# Run training script
bazel run //:train_llm -- --config configs/llm_default.yaml

# Run inference server
bazel run //inference:inference_server
```

### 3. Run Tests

```bash
# Run all tests
bazel test //...

# Run specific test
bazel test //cpp_core:test_attention
```

## Build Modes

### Debug Build
```bash
bazel build --config=debug //...
```

### Release Build (default)
```bash
bazel build --config=release //...
```

### Fast Build (development)
```bash
bazel build --config=fast //...
```

## Module-Specific Builds

### Python Modules
```bash
# Build all Python modules
bazel build //core:core //factories:factories //trainers:trainers

# Build with specific Python version
bazel build --python_version=3.11 //...
```

### C++ Core
```bash
# Build C++ library
bazel build //cpp_core:cpp_core

# Build Python bindings
bazel build //cpp_core:_cpp_core

# Build with CUDA support
bazel build --config=cuda //cpp_core:cpp_core
```

### Rust Core
```bash
# Build Rust library
bazel build //rust_core:rust_core

# Build Python bindings
bazel build //rust_core:rust_core_so
```

### Go Core
```bash
# Build Go binaries
bazel build //go_core:inference_server
bazel build //go_core:cache_service
bazel build //go_core:data_pipeline
bazel build //go_core:coordinator
```

### Julia Core
```bash
# Build Julia package
bazel build //julia_core:julia_core_pkg
```

### Scala Core
```bash
# Build Scala library
bazel build //scala_core:scala_core

# Build Scala binary
bazel build //scala_core:inference_service
```

### Elixir Core
```bash
# Build Elixir release
bazel build //elixir_core:elixir_core_release
```

## Dependencies

### Python Dependencies

Python dependencies are managed via `pip_parse` in `WORKSPACE.bazel`. The lock file is `requirements_lock.txt`.

To update dependencies:
```bash
# Install pip-tools
pip install pip-tools

# Update lock file
pip-compile requirements.txt -o requirements_lock.txt

# Update Bazel dependencies
bazel sync --only=@pip
```

### C++ Dependencies

C++ dependencies (Eigen, fmt, spdlog, LZ4, Zstd) are fetched via `http_archive` in `WORKSPACE.bazel`.

### Rust Dependencies

Rust dependencies are managed via `Cargo.toml` in `rust_core/`. Bazel will use the Cargo build system.

### Go Dependencies

Go dependencies are managed via `go.mod` in `go_core/`. Bazel will automatically resolve them.

## Project Structure

```
optimization_core/
├── WORKSPACE.bazel          # Bazel workspace configuration
├── BUILD.bazel              # Root BUILD file
├── .bazelrc                  # Bazel configuration
├── requirements_lock.txt     # Python dependencies lock file
│
├── core/                     # Python core module
│   └── BUILD.bazel
│
├── cpp_core/                 # C++ core
│   ├── BUILD.bazel
│   └── BUILD.*               # External dependency BUILD files
│
├── rust_core/               # Rust core
│   └── BUILD.bazel
│
├── go_core/                  # Go core
│   └── BUILD.bazel
│
├── julia_core/               # Julia core
│   └── BUILD.bazel
│
├── scala_core/               # Scala core
│   └── BUILD.bazel
│
└── elixir_core/              # Elixir core
    └── BUILD.bazel
```

## Troubleshooting

### Python Dependencies Not Found

```bash
# Sync Python dependencies
bazel sync --only=@pip

# Clean and rebuild
bazel clean
bazel build //...
```

### C++ Compilation Errors

```bash
# Check C++ toolchain
bazel info cpp

# Build with verbose output
bazel build --verbose_failures //cpp_core:cpp_core
```

### Rust Build Issues

```bash
# Check Rust toolchain
bazel info rust

# Build with Cargo directly (for debugging)
cd rust_core && cargo build --release
```

### Go Build Issues

```bash
# Check Go toolchain
bazel info go

# Build with Go directly (for debugging)
cd go_core && go build ./cmd/inference-server
```

## Advanced Usage

### Remote Caching

To enable remote caching, add to `.bazelrc`:
```
build --remote_cache=https://your-cache-server.com
```

### Remote Execution

To enable remote execution, add to `.bazelrc`:
```
build --remote_executor=grpc://your-executor.com:8980
```

### Custom Build Flags

Add to `.bazelrc`:
```
build:custom --copt=-DCUSTOM_FLAG
```

Then use:
```bash
bazel build --config=custom //...
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build with Bazel

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: bazel-contrib/setup-bazel@v0.1.0
      - run: bazel build //...
      - run: bazel test //...
```

## Resources

- [Bazel Documentation](https://bazel.build/docs)
- [Rules Python](https://github.com/bazelbuild/rules_python)
- [Rules Rust](https://github.com/bazelbuild/rules_rust)
- [Rules Go](https://github.com/bazelbuild/rules_go)
- [Rules Scala](https://github.com/bazelbuild/rules_scala)












