# ⚡ Quick Start Guide

Get started with the TruthGPT Test Framework in 5 minutes!

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (1 min)

```bash
# Install Python packages
pip install torch numpy psutil pytest pytest-cov pytest-xdist

# Or if you have a requirements file
pip install -r requirements.txt
```

### Step 2: Run Tests (2 min)

```bash
# Navigate to optimization_core
cd agents/backend/onyx/server/features/Frontier-Model-run/scripts/TruthGPT-main/optimization_core

# Run all tests
python tests/run_all_tests.py

# That's it! ✅
```

### Step 3: View Results (1 min)

```bash
# Results appear in console
# Also saved to: test_results.json

# Check results
cat test_results.json
```

### Step 4: Generate HTML Report (1 min)

```bash
# Run the demo (includes HTML report)
python tests/demo_framework.py

# Or generate manually
python -c "
from tests.report_generator import HTMLReportGenerator
import json

with open('test_results.json') as f:
    results = json.load(f)

HTMLReportGenerator().generate_report(results, 'test_report.html')
"

# Open test_report.html in browser
```

## 🎯 Common Use Cases

### Run Specific Test Category

```bash
# Unit tests only
python tests/run_all_tests.py --pattern unit

# Integration tests only
python tests/run_all_tests.py --integration

# Performance tests only
python tests/run_all_tests.py --performance
```

### Get Detailed Output

```bash
# Verbose mode
python tests/run_all_tests.py --verbose

# Save results to file
python tests/run_all_tests.py --save-results
```

### Run Tests in Parallel

```python
# In your test file
from tests.fixtures.test_utils import ParallelTestRunner

runner = ParallelTestRunner(max_workers=4)
results = runner.run_tests_parallel([test1, test2, test3, test4])
```

### Use Advanced Decorators

```python
# In your test file
from tests.fixtures.test_utils import AdvancedTestDecorators

@AdvancedTestDecorators.retry(max_attempts=3)
def test_flaky_connection():
    # Your test code
    pass

@AdvancedTestDecorators.timeout(seconds=60)
def test_with_timeout():
    # Your test code
    pass
```

## 📊 What You'll See

### Console Output

```
🧪 Running TruthGPT Optimization Core Tests
============================================================

📁 Running tests from: test_optimizer_core.py
   ✅ PASSED - 15 tests, 0 failures, 0 errors

📊 Test Summary:
  Total Tests: 250
  Success Rate: 100.0%
  Total Time: 15.23s

🎉 All tests passed successfully!
```

### HTML Report

A beautiful, professional HTML report with:
- 📊 Visual statistics
- 📈 Progress bars
- 🎨 Professional design
- 📝 Detailed breakdown

### JSON Results

```json
{
  "total_tests": 250,
  "total_failures": 0,
  "total_errors": 0,
  "success_rate": 100.0,
  "performance_metrics": {...},
  "memory_summary": {...}
}
```

## 🎯 Next Steps

### 1. Explore Advanced Features

```bash
# Run the comprehensive demo
python tests/demo_framework.py

# See all features in action!
```

### 2. Read the Documentation

- [README.md](README.md) - Main documentation
- [RUN_TESTS.md](RUN_TESTS.md) - How to run tests
- [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) - Advanced features
- [CICD_GUIDE.md](CICD_GUIDE.md) - CI/CD integration

### 3. Integrate with CI/CD

```yaml
# .github/workflows/tests.yml (already included!)
# Automatically runs on push/PR
```

### 4. Customize for Your Needs

```python
# Use the advanced utilities
from tests.fixtures.test_utils import (
    TestCoverageTracker,
    AdvancedTestDecorators,
    ParallelTestRunner,
    TestVisualizer
)
```

## ❓ Troubleshooting

### Python Not Found

```bash
# Check if Python is installed
python --version

# If not, install from python.org or Microsoft Store
```

### Import Errors

```bash
# Make sure you're in the right directory
pwd  # Should be in optimization_core

# Add to Python path
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Tests Not Running

```bash
# Check if test files exist
ls tests/unit/
ls tests/integration/
ls tests/performance/

# Run with verbose output
python tests/run_all_tests.py --verbose
```

## ✅ Success Checklist

- [ ] Dependencies installed
- [ ] Tests run successfully
- [ ] HTML report generated
- [ ] Results saved to JSON
- [ ] Documentation read
- [ ] CI/CD integrated (optional)

## 🎉 You're Done!

You now have a world-class test framework running!

**Quick Reference Card:**

```bash
# Most common commands
python tests/run_all_tests.py              # Run all tests
python tests/run_all_tests.py --verbose   # Detailed output
python tests/demo_framework.py            # Full demo
```

**Need Help?**

- 📖 Check [README.md](README.md)
- 📖 Read [RUN_TESTS.md](RUN_TESTS.md)
- 📖 Explore [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)

**Happy Testing! 🧪✨**


