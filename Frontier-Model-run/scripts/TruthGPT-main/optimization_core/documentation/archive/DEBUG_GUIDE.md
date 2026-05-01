# TruthGPT Test Debug Guide

## Quick Debug Commands

### Option 1: Run Debug Script (Recommended)
```bash
# Windows
debug.bat

# Or directly
python debug_tests.py
```

### Option 2: Manual Debug Steps

1. **Check Python Installation:**
   ```bash
   python --version
   # Should show Python 3.7 or higher
   ```

2. **Check Dependencies:**
   ```bash
   python -c "import torch; print('torch:', torch.__version__)"
   python -c "import numpy; print('numpy:', numpy.__version__)"
   python -c "import psutil; print('psutil OK')"
   ```

3. **Test Imports:**
   ```bash
   python -c "from core import OptimizationEngine; print('Core imports OK')"
   ```

4. **Run Tests:**
   ```bash
   python run_unified_tests.py
   ```

## Common Issues

### Issue: Python Not Found

**Symptoms:**
- `python: command not found`
- `Python was not found; run without arguments to install from the Microsoft Store`

**Solutions:**

1. **Install Python:**
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"
   - Restart terminal after installation

2. **Fix Microsoft Store Redirect:**
   - Open Windows Settings
   - Go to Apps > App execution aliases
   - Disable Python app execution aliases

3. **Use Python Launcher:**
   ```bash
   py --version
   py -m pip install torch numpy
   ```

### Issue: Missing Dependencies

**Symptoms:**
- `ModuleNotFoundError: No module named 'torch'`
- `ImportError: cannot import name 'OptimizationEngine'`

**Solution:**
```bash
pip install torch numpy psutil
```

Or install from requirements file if available:
```bash
pip install -r requirements.txt
```

### Issue: Import Errors

**Symptoms:**
- `ModuleNotFoundError: No module named 'core'`
- `ImportError: cannot import name 'OptimizationEngine' from 'core'`

**Solution:**
1. Make sure you're in the correct directory:
   ```bash
   cd "C:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run\scripts\TruthGPT-main"
   ```

2. Check if core module exists:
   ```bash
   dir core
   # Should show __init__.py and other core files
   ```

3. Add path if needed:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent))
   ```

### Issue: Unittest Compatibility

**Symptoms:**
- `AttributeError: module 'unittest' has no attribute 'makeSuite'`
- This should be fixed in the latest version

**Solution:**
The code has been updated to use `unittest.TestLoader().loadTestsFromTestCase()` instead of deprecated `makeSuite()`. If you see this error, update your `run_unified_tests.py` file.

### Issue: Test Failures

**Symptoms:**
- Tests fail with errors
- Tests timeout

**Debug Steps:**

1. **Run individual test categories:**
   ```bash
   python run_unified_tests.py core
   python run_unified_tests.py optimization
   python run_unified_tests.py models
   ```

2. **Run specific test file:**
   ```bash
   python -m unittest tests.test_core
   ```

3. **Run with verbose output:**
   ```bash
   python run_unified_tests.py -v
   ```

## Debug Script Output

The `debug_tests.py` script provides:
- Python version check
- Dependencies status
- Import verification
- Core module exports check
- Test files verification
- Test runner validation

**Expected output:**
```
✅ ALL CHECKS PASSED - Tests should run successfully
```

## Files Created

- `debug_tests.py` - Python diagnostic script
- `debug.bat` - Windows batch file for quick debug
- `DEBUG_GUIDE.md` - This guide

## Next Steps After Debug

1. **If all checks pass:**
   ```bash
   python run_unified_tests.py
   ```

2. **If dependencies missing:**
   ```bash
   pip install torch numpy psutil
   ```

3. **If Python not found:**
   - Install Python from python.org
   - Restart terminal
   - Run debug again

## Getting Help

If issues persist:
1. Check the debug script output
2. Review error messages carefully
3. Verify all requirements are met
4. Check READY_TO_TEST.md for known issues









