# 🚀 Bazel Quick Start Guide - TruthGPT Optimization Core

[![Bazel](https://img.shields.io/badge/Bazel-7.0+-4285F4?style=flat-square&logo=bazel)](https://bazel.build)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)]()

Complete guide for building and testing the TruthGPT polyglot optimization core with Bazel.

```
╔══════════════════════════════════════════════════════════════════════╗
║                    TruthGPT Bazel Build System                        ║
║                                                                      ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              ║
║  │   Python     │  │     Rust     │  │      C++     │              ║
║  │   Modules    │  │     Core     │  │     Core     │              ║
║  └──────────────┘  └──────────────┘  └──────────────┘              ║
║         │                  │                  │                      ║
║         └──────────────────┼──────────────────┘                     ║
║                            │                                         ║
║                    ┌───────▼────────┐                                ║
║                    │  Bazel Build   │                                ║
║                    │   System       │                                ║
║                    └────────────────┘                                ║
╚══════════════════════════════════════════════════════════════════════╝
```

## 📋 Table of Contents

- [Installation](#installation)
- [First Run](#first-run)
- [Build Commands](#build-commands)
- [Testing](#testing)
- [Common Issues](#common-issues)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [CI/CD Integration](#cicd-integration)

## 🔧 Installation

### Windows

```powershell
# Option 1: Using Chocolatey (Recommended)
choco install bazel

# Option 2: Using Scoop
scoop install bazel

# Option 3: Manual Installation
# Download from: https://github.com/bazelbuild/bazel/releases
# Extract and add to PATH
```

**Verify Installation:**
```powershell
bazel version
# Should show: bazel 7.0.0 or higher
```

### Linux

```bash
# Ubuntu/Debian
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel

# Fedora/RHEL
sudo dnf install bazel

# Arch Linux
yay -S bazel
```

**Verify Installation:**
```bash
bazel version
```

### macOS

```bash
# Using Homebrew (Recommended)
brew install bazel

# Or using Bazelisk (version manager)
brew install bazelisk
```

**Verify Installation:**
```bash
bazel version
```

## 🎯 First Run

### Step 1: Validate Workspace

```bash
cd optimization_core
bazel query //...
```

This will:
- Parse all BUILD files
- Download dependencies
- Validate the workspace structure

### Step 2: Fix SHA256 Issues (If Any)

When Bazel downloads dependencies, it may report SHA256 mismatches. This is normal on first run.

**Example Error:**
```
ERROR: /path/to/WORKSPACE.bazel:123:45: 
SHA256 mismatch for @some_dependency:
  Expected: abc123...
  Actual:   def456...
```

**Solution:**
1. Copy the **Actual** SHA256 from the error message
2. Update `WORKSPACE.bazel` with the correct SHA256
3. Re-run the command

```bash
# Example: Update WORKSPACE.bazel
# Change:
#   sha256 = "abc123..."
# To:
#   sha256 = "def456..."
```

### Step 3: Sync Python Dependencies

```bash
# Sync pip dependencies
bazel sync --only=@pip

# Or sync everything
bazel sync
```

### Step 4: Build a Test Target

```bash
# Build a simple Python module
bazel build //core:core

# If successful, you're ready to go! 🎉
```

## 🏗️ Build Commands

### Validate Configuration

```bash
# Check if workspace parses correctly
bazel query //...

# List all targets
bazel query //... --output=package

# List targets with their types
bazel query //... --output=label_kind

# Show dependency graph
bazel query --output=graph //core:core | dot -Tpng > deps.png
```

### Build Individual Modules

```bash
# Python modules
bazel build //core:core
bazel build //factories:factories
bazel build //trainers:trainers
bazel build //optimizers:optimizers
bazel build //inference:inference

# C++ core
bazel build //cpp_core:cpp_core

# Rust core (if configured)
bazel build //rust_core:rust_core

# Go core (if configured)
bazel build //go_core:inference_server
```

### Build Everything

```bash
# Build all targets
bazel build //...

# Build with specific configuration
bazel build //... --config=release

# Build with verbose output
bazel build //... --verbose_failures
```

### Build Specific Configurations

```bash
# Debug build
bazel build //... --compilation_mode=dbg

# Release build (optimized)
bazel build //... --compilation_mode=opt

# Fast build (less optimization)
bazel build //... --compilation_mode=fastbuild
```

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
bazel test //...

# Run tests with output
bazel test //... --test_output=all

# Run tests in parallel
bazel test //... --jobs=8
```

### Run Specific Test Suites

```bash
# Python tests
bazel test //core:core_test
bazel test //inference:inference_test

# C++ tests
bazel test //cpp_core:cpp_core_test

# Integration tests
bazel test //tests/integration:integration_test
```

### Test Filtering

```bash
# Run specific test
bazel test //core:core_test --test_filter=TestClassName.test_method

# Run tests matching pattern
bazel test //core:core_test --test_filter="*test_*"
```

## 🔍 Common Issues & Solutions

### Issue: SHA256 Mismatch

**Symptom:**
```
ERROR: SHA256 mismatch for @dependency
```

**Solution:**
1. Update `WORKSPACE.bazel` with the correct SHA256 from the error
2. Re-run the command

**Prevention:**
- Use `bazel sync` regularly to update dependencies
- Pin dependency versions in `WORKSPACE.bazel`

### Issue: Python Dependencies Not Found

**Symptom:**
```
ERROR: Could not find dependency '@pip//package'
```

**Solution:**
```bash
# Sync pip dependencies
bazel sync --only=@pip

# Clean and rebuild
bazel clean
bazel build //...
```

### Issue: C++ Compilation Errors

**Symptom:**
```
ERROR: C++ compilation failed
```

**Solution:**
1. **Check C++ Compiler:**
   ```bash
   # Linux/macOS
   g++ --version  # Should be GCC 9+ or Clang 10+
   
   # Windows
   cl  # Should have MSVC 2019+
   ```

2. **Check C++20 Support:**
   ```bash
   # Test C++20 support
   echo 'int main() { return 0; }' > test.cpp
   g++ -std=c++20 test.cpp -o test && ./test
   ```

3. **Install Required Libraries:**
   ```bash
   # Ubuntu/Debian
   sudo apt install build-essential libeigen3-dev
   
   # macOS
   brew install eigen
   ```

### Issue: Rust Build Fails

**Symptom:**
```
ERROR: Rust compilation failed
```

**Solution:**
1. **Install Rust:**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Install Maturin (for PyO3):**
   ```bash
   pip install maturin
   ```

3. **Configure rules_rust:**
   - Ensure `WORKSPACE.bazel` has proper `rules_rust` setup
   - For production, use `maturin` directly instead of genrules

### Issue: Out of Memory

**Symptom:**
```
ERROR: Out of memory
```

**Solution:**
```bash
# Reduce parallel jobs
bazel build //... --jobs=2

# Increase JVM heap size (edit .bazelrc)
echo "startup --host_jvm_args=-Xmx4g" >> .bazelrc
```

### Issue: Build Cache Issues

**Symptom:**
```
ERROR: Stale build artifacts
```

**Solution:**
```bash
# Clean build
bazel clean

# Clean and expunge
bazel clean --expunge

# Rebuild
bazel build //...
```

## 📁 Project Structure

```
optimization_core/
├── WORKSPACE.bazel          # Workspace root configuration
├── BUILD.bazel              # Root BUILD file
├── .bazelrc                 # Bazel configuration flags
├── requirements_lock.txt    # Python dependencies (exact versions)
│
├── core/
│   └── BUILD.bazel         # Python core module
│
├── factories/
│   └── BUILD.bazel         # Factory pattern implementations
│
├── trainers/
│   └── BUILD.bazel         # Training components
│
├── optimizers/
│   └── BUILD.bazel         # Optimization algorithms
│
├── inference/
│   └── BUILD.bazel         # Inference engines (vLLM, TensorRT-LLM)
│
├── data/
│   └── BUILD.bazel         # Data loaders and processors
│
├── utils/
│   └── BUILD.bazel         # Utility functions
│
├── cpp_core/               # C++ High-Performance Core
│   ├── BUILD.bazel         # Main C++ build
│   ├── BUILD.eigen         # Eigen linear algebra
│   ├── BUILD.fmt           # fmt formatting library
│   ├── BUILD.spdlog        # spdlog logging
│   ├── BUILD.lz4           # LZ4 compression
│   ├── BUILD.zstd          # Zstd compression
│   └── BUILD.pybind11      # Python bindings
│
├── rust_core/              # Rust Core
│   └── BUILD.bazel         # Rust build (uses genrules)
│
├── go_core/                # Go Core
│   └── BUILD.bazel         # Go build
│
├── julia_core/             # Julia Core
│   └── BUILD.bazel         # Julia build
│
├── scala_core/             # Scala Core
│   └── BUILD.bazel         # Scala build
│
└── elixir_core/            # Elixir Core
    └── BUILD.bazel         # Elixir build
```

## 🚀 Advanced Usage

### Custom Build Configurations

Create `.bazelrc` configurations:

```bash
# .bazelrc
# Release configuration
build:release --compilation_mode=opt
build:release --copt=-O3
build:release --cxxopt=-std=c++20

# Debug configuration
build:debug --compilation_mode=dbg
build:debug --copt=-g
build:debug --strip=never

# GPU configuration
build:gpu --action_env=CUDA_PATH=/usr/local/cuda
build:gpu --copt=-DCUDA_ENABLED
```

**Usage:**
```bash
bazel build //... --config=release
bazel build //... --config=debug
bazel build //... --config=gpu
```

### Remote Caching

Set up remote build cache for faster builds:

```bash
# .bazelrc
build --remote_cache=https://your-cache-server.com
build --remote_upload_local_results=true
```

### Build Profiles

Profile build performance:

```bash
# Generate build profile
bazel build //... --profile=profile.json

# Analyze profile
bazel analyze-profile profile.json
```

### Query Examples

```bash
# Find all Python targets
bazel query "kind(py_binary, //...)"

# Find all dependencies of a target
bazel query "deps(//core:core)"

# Find reverse dependencies
bazel query "rdeps(//..., //core:core)"

# Find all test targets
bazel query "kind(test, //...)"
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Bazel CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Bazel
        uses: bazel-contrib/setup-bazel@v0
        with:
          bazelisk-version: '1.x'
      
      - name: Build
        run: bazel build //...
      
      - name: Test
        run: bazel test //...
```

### GitLab CI Example

```yaml
build:
  image: l.gcr.io/google/bazel:latest
  script:
    - bazel build //...
    - bazel test //...
```

### Jenkins Example

```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'bazel build //...'
            }
        }
        stage('Test') {
            steps {
                sh 'bazel test //...'
            }
        }
    }
}
```

## 📊 Build Performance Tips

1. **Use Remote Caching**: Share build cache across team
2. **Incremental Builds**: Only rebuild changed targets
3. **Parallel Execution**: Use `--jobs=N` for parallel builds
4. **Build Profiles**: Identify slow build steps
5. **Target Filtering**: Build only what you need

## 🛠️ Troubleshooting Checklist

- [ ] Bazel version is 7.0+ (`bazel version`)
- [ ] All SHA256 values in `WORKSPACE.bazel` are correct
- [ ] Python dependencies synced (`bazel sync --only=@pip`)
- [ ] C++ compiler supports C++20
- [ ] Required system libraries installed
- [ ] Sufficient disk space (Bazel cache can be large)
- [ ] Network access for downloading dependencies

## 📚 Additional Resources

- [Bazel Documentation](https://bazel.build/docs)
- [Bazel Best Practices](https://bazel.build/configure/best-practices)
- [Rules Python](https://github.com/bazelbuild/rules_python)
- [Rules Rust](https://github.com/bazelbuild/rules_rust)
- [Rules Go](https://github.com/bazelbuild/rules_go)
- [Bazel Query Reference](https://bazel.build/query/language)

## 🎯 Next Steps

1. ✅ Install Bazel
2. ✅ Run `bazel query //...` to validate
3. ✅ Fix any SHA256 errors
4. ✅ Build individual modules
5. ✅ Run tests
6. ✅ Set up CI/CD
7. ✅ Configure remote caching (optional)
8. ✅ Optimize build performance

---

**Version:** 2.0.0  
**Last Updated:** 2024  
**Status:** ✅ Production Ready

**Need Help?** Check the [Bazel Status](./BAZEL_STATUS.md) or [Bazel Setup](./BAZEL_SETUP.md) guides.
