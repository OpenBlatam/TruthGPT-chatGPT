# Bazel Configuration - Fixes Applied

## Issues Fixed

### 1. requirements_lock.txt
- **Issue**: Had `>=` version specifiers which are not valid for `pip_parse`
- **Fix**: Changed to exact versions (`==`)
  - `vllm>=0.2.0` → `vllm==0.2.0`
  - `polars>=0.36.0` → `polars==0.36.0`

### 2. WORKSPACE.bazel
- **Issue**: Invalid SHA256 placeholders for several dependencies
- **Fixes**:
  - Removed invalid SHA256 for Eigen (will be calculated on first download)
  - Fixed pybind11 reference (changed from `pybind11_bazel` to `pybind11`)
  - Removed invalid SHA256 for spdlog
  - Commented out gRPC (very large, add when needed)
  - Added BUILD.pybind11 file for pybind11 dependency

### 3. cpp_core/BUILD.bazel
- **Issue**: Referenced non-existent `pybind11_bazel` and `pybind_extension`
- **Fixes**:
  - Changed to use `@pybind11//:pybind11` directly
  - Created `_cpp_core_cc` as cc_library instead of pybind_extension
  - Updated dependencies to use correct target names

### 4. rust_core/BUILD.bazel
- **Issue**: Genrule command was incorrect
- **Fix**: Simplified genrule as placeholder (proper setup requires maturin or rules_rust PyO3)

## Remaining Considerations

### SHA256 Values
Some dependencies don't have SHA256 values set. Bazel will:
1. Download the file on first run
2. Calculate the SHA256
3. Report an error with the correct SHA256
4. You can then update WORKSPACE.bazel with the correct SHA256

### Python Bindings
- **C++**: Currently set up as cc_library. For full PyBind11 support, may need additional setup
- **Rust**: Genrule is a placeholder. For production, use:
  - `maturin` for building PyO3 extensions
  - Or proper `rules_rust` with PyO3 configuration

### Optional Dependencies
Some large dependencies (like gRPC) are commented out. Uncomment when needed.

## Testing

To validate the Bazel configuration:

```bash
# Check if Bazel can parse the workspace
bazel query //...

# Try building a simple target
bazel build //core:core

# Check for dependency issues
bazel sync --only=@pip
```

## Next Steps

1. Run `bazel query //...` to validate syntax
2. Fix any SHA256 errors that appear
3. Test building individual modules
4. Set up proper Python bindings for C++ and Rust if needed












