# 🚀 Quick Start Guide - TruthGPT Tests

## Prerequisites Check

### 1. Quick Environment Check
```bash
python quick_check.py
```

This will verify:
- ✅ Python version (3.7+)
- ✅ Directory structure
- ✅ Required dependencies
- ✅ Core module imports

### 2. Setup Environment (if needed)
```bash
python setup_environment.py
```

This will:
- Check Python installation
- Install missing dependencies (torch, numpy)
- Verify setup is correct

## Running Tests

### Basic Usage

```bash
# Run all tests
python run_unified_tests.py

# Quick check before running
python run_unified_tests.py --check

# Run specific category
python run_unified_tests.py core

# Stop on first failure
python run_unified_tests.py --failfast

# Verbose output
python run_unified_tests.py --verbose

# Quiet mode (for CI/CD)
python run_unified_tests.py --quiet

# Export results to JSON
python run_unified_tests.py --json results.json

# List all categories
python run_unified_tests.py --list
```

### Using Helper Scripts

#### Windows (Batch)
```cmd
run_tests.bat                    # All tests
run_tests.bat core               # Core tests only
run_tests.bat --failfast         # Stop on first failure
run_tests.bat --json report.json # Export to JSON
```

#### Windows (PowerShell)
```powershell
.\run_tests.ps1                    # All tests
.\run_tests.ps1 core              # Core tests only
.\run_tests.ps1 --failfast         # Stop on first failure
.\run_tests.ps1 --json report.json # Export to JSON
```

## Test Categories

Available test categories:

- `core` - Core component tests
- `optimization` - Optimization engine tests
- `models` - Model management tests
- `training` - Training system tests
- `inference` - Inference engine tests
- `monitoring` - Monitoring system tests
- `integration` - Integration tests
- `edge` - Edge cases and stress tests
- `performance` - Performance and benchmark tests
- `security` - Security and validation tests
- `compatibility` - Compatibility tests
- `regression` - Regression tests
- `validation` - Validation tests
- `benchmarks` - Benchmark tests

## Troubleshooting

### Python Not Found
1. Install Python from https://www.python.org/downloads/
2. Check "Add Python to PATH" during installation
3. Or disable Microsoft Store Python redirect in Windows Settings

### Missing Dependencies
```bash
pip install torch numpy
```

Or use the setup script:
```bash
python setup_environment.py
```

### Import Errors
Make sure you're in the TruthGPT-main directory:
```bash
cd "C:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run\scripts\TruthGPT-main"
```

### Windows Issues
The code automatically handles Windows-specific issues (DataLoader, multiprocessing, etc.)

## Advanced Features

### Check Dependencies
```bash
python check_dependencies.py
```

### Quick Environment Check
```bash
python quick_check.py
```

### Setup Environment
```bash
python setup_environment.py
```

## Expected Output

When tests run successfully, you'll see:

```
🧪 TruthGPT Unified Test Runner
============================================================
🧪 Adding all test classes to unified test suite...
✅ Added core component tests
✅ Added optimization tests
...
📊 Total test classes added: 14

🚀 Starting unified test suite...
📦 Test suite contains 204 test cases
...
✅ Tests completed: 204 passed, 0 failed, 0 errors, 0 skipped
⏱️  Execution time: 45.23s (4.5 tests/sec)

🎉 All tests passed!
```

## Next Steps

1. ✅ Run `quick_check.py` to verify environment
2. ✅ Run `setup_environment.py` if needed
3. ✅ Run `run_unified_tests.py` to execute tests
4. ✅ Review results and fix any failures
5. ✅ Use `--json` to export results for analysis

## Help

For more information:
- `python run_unified_tests.py --help` - Full help message
- `python run_unified_tests.py --list` - List all categories
- See `IMPROVEMENTS.md` and `MORE_IMPROVEMENTS.md` for detailed documentation







