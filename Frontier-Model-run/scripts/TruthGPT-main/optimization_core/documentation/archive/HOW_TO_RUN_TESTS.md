# How to Run Tests

## Prerequisites

Make sure Python is installed and available in your PATH. You can verify by running:
```bash
python --version
```

If Python is not found, you may need to:
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Add Python to your system PATH
3. Or use a Python launcher like `py` (Windows) or `python3` (Linux/Mac)

## Running Tests

### Option 1: Run All Tests
```bash
cd "C:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run\scripts\TruthGPT-main"
python run_unified_tests.py
```

### Option 2: Run Specific Test Category
```bash
python run_unified_tests.py core
python run_unified_tests.py optimization
python run_unified_tests.py models
python run_unified_tests.py training
python run_unified_tests.py inference
python run_unified_tests.py monitoring
python run_unified_tests.py integration
```

### Option 3: Run Individual Test Files
```bash
python -m unittest tests.test_core
python -m unittest tests.test_optimization
python -m unittest tests.test_models
python -m unittest tests.test_training
python -m unittest tests.test_inference
python -m unittest tests.test_monitoring
python -m unittest tests.test_integration
```

### Option 4: Run with Verbose Output
```bash
python -m unittest discover tests -v
```

## Expected Output

When tests run successfully, you should see:
- ✅ All test classes loaded
- ✅ Test execution progress
- ✅ Success/failure counts
- ✅ Performance metrics
- ✅ Final test report

## Troubleshooting

If you encounter import errors:
1. Make sure you're in the correct directory (TruthGPT-main)
2. Verify Python can find the `core` and `tests` modules
3. Check that all dependencies are installed (torch, numpy, etc.)

## All Fixes Applied ✅

All test fixes have been completed:
- Core exports fixed
- Validation added
- Imports fixed
- Device handling fixed
- Error handling added
- Windows compatibility fixed
- Early stopping implemented

The tests are ready to run once Python is available!









