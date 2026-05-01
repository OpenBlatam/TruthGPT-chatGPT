# Advanced Test Features

This document describes the advanced features added to the TruthGPT Optimization Core test framework.

## New Advanced Features

### 1. Test Coverage Tracker

Track and analyze test coverage across your test suite.

```python
from tests.fixtures.test_utils import TestCoverageTracker

# Create tracker
tracker = TestCoverageTracker()

# Record test results
tracker.record_test("test_attention", passed=True, duration=0.5, coverage=85.0)
tracker.record_test("test_optimizer", passed=False, duration=1.2, coverage=75.0)

# Calculate total coverage
summary = tracker.calculate_total_coverage()
print(f"Total Coverage: {summary['total_coverage']:.1f}%")
print(f"Success Rate: {summary['success_rate']:.1f}%")
```

### 2. Advanced Test Decorators

Powerful decorators for testing edge cases and reliability.

#### Retry Failed Tests

Automatically retry flaky tests:

```python
from tests.fixtures.test_utils import AdvancedTestDecorators

@AdvancedTestDecorators.retry(max_attempts=3, delay=1.0)
def test_flaky_connection():
    """This test will retry up to 3 times if it fails"""
    # Your test code here
    pass
```

#### Timeout for Tests

Add timeout to prevent hanging tests:

```python
@AdvancedTestDecorators.timeout(seconds=60)
def test_long_operation():
    """This test will timeout after 60 seconds"""
    # Your test code here
    pass
```

#### Performance Regression Tests

Compare against baseline performance:

```python
@AdvancedTestDecorators.performance_test(baseline_time=1.0)
def test_fast_operation():
    """This test will fail if it takes more than 1.2x the baseline"""
    # Your test code here
    pass
```

### 3. Parallel Test Execution

Run multiple tests in parallel for speed:

```python
from tests.fixtures.test_utils import ParallelTestRunner

# Create runner with 4 parallel workers
runner = ParallelTestRunner(max_workers=4)

# Define test functions
test_functions = [
    lambda: test_attention(),
    lambda: test_optimizer(),
    lambda: test_transformer(),
    lambda: test_quantization()
]

# Run tests in parallel
results = runner.run_tests_parallel(test_functions)
```

### 4. Test Visualization

Create visual summaries and graphs of your test results.

#### Results Summary

```python
from tests.fixtures.test_utils import TestVisualizer

results = {
    'total_tests': 250,
    'total_failures': 5,
    'total_errors': 3,
    'success_rate': 96.8,
    'performance_metrics': {
        'total_execution_time': 125.5,
        'total_memory_used': 1024.0
    }
}

# Create visual summary
summary = TestVisualizer.create_results_summary(results)
print(summary)
```

Output:
```
================================================================================
TruthGPT Optimization Core Test Results
================================================================================

Total Tests: 250
Passed: 242
Failed: 8
Errors: 0
Success Rate: 96.8%

Performance Metrics:
  Execution Time: 125.50s
  Memory Used: 1024.00MB

Status: ████████████████████████████████████████

================================================================================
```

#### Performance Graphs

```python
profiles = [
    {'name': 'test_attention', 'execution_time': 0.5},
    {'name': 'test_optimizer', 'execution_time': 1.2},
    {'name': 'test_transformer', 'execution_time': 2.8}
]

graph = TestVisualizer.create_performance_graph(profiles)
print(graph)
```

Output:
```
Performance Profile (Execution Time in seconds)

test_attention                       0.500s ████████
test_optimizer                       1.200s ████████████████
test_transformer                     2.800s ███████████████████████████████████████
```

## Usage Examples

### Complete Test Suite with Advanced Features

```python
import unittest
from tests.fixtures.test_utils import (
    TestCoverageTracker,
    ParallelTestRunner,
    TestVisualizer,
    AdvancedTestDecorators
)

class AdvancedTestSuite(unittest.TestCase):
    
    def setUp(self):
        self.coverage_tracker = TestCoverageTracker()
        self.test_start_time = time.time()
    
    @AdvancedTestDecorators.retry(max_attempts=3)
    def test_reliable_connection(self):
        """Test with auto-retry for flaky connections"""
        # Your test code
        pass
    
    @AdvancedTestDecorators.timeout(seconds=30)
    def test_timeout_prevention(self):
        """Test with timeout protection"""
        # Your test code
        pass
    
    @AdvancedTestDecorators.performance_test(baseline_time=1.0)
    def test_performance_regression(self):
        """Test for performance regressions"""
        # Your test code
        pass
    
    def tearDown(self):
        duration = time.time() - self.test_start_time
        passed = len(self._outcome.result.failures) == 0
        self.coverage_tracker.record_test(
            self._testMethodName, 
            passed, 
            duration
        )
    
    @classmethod
    def tearDownClass(cls):
        # Get coverage summary
        summary = cls.coverage_tracker.calculate_total_coverage()
        
        # Create visual summary
        visualizer = TestVisualizer()
        summary_text = visualizer.create_results_summary(summary)
        print(summary_text)
```

### Parallel Test Execution

```python
from tests.fixtures.test_utils import ParallelTestRunner

# Define independent test functions
def test_unit_1():
    # Unit test 1
    pass

def test_unit_2():
    # Unit test 2
    pass

def test_unit_3():
    # Unit test 3
    pass

def test_unit_4():
    # Unit test 4
    pass

# Run tests in parallel
runner = ParallelTestRunner(max_workers=4)
tests = [test_unit_1, test_unit_2, test_unit_3, test_unit_4]
results = runner.run_tests_parallel(tests)
```

## Benefits

### 1. Better Reliability

- **Auto-retry**: Automatically retry flaky tests without manual intervention
- **Timeout protection**: Prevent tests from hanging indefinitely
- **Performance regression detection**: Catch performance issues early

### 2. Faster Execution

- **Parallel execution**: Run independent tests simultaneously
- **Reduced total time**: Complete test suite faster with parallelization

### 3. Better Reporting

- **Visual summaries**: ASCII-based visualizations
- **Coverage tracking**: Monitor test coverage over time
- **Performance graphs**: Identify slow tests

### 4. Easier Debugging

- **Clear visualizations**: Quickly identify issues
- **Detailed metrics**: Track memory, CPU, and timing
- **Trend analysis**: Monitor improvements over time

## Advanced Configuration

### Custom Coverage Thresholds

```python
tracker = TestCoverageTracker()
tracker.min_coverage = 80.0  # Minimum coverage percentage
tracker.min_success_rate = 95.0  # Minimum success rate
```

### Parallel Execution Strategies

```python
# Conservative (safer, slower)
runner = ParallelTestRunner(max_workers=2)

# Balanced (default)
runner = ParallelTestRunner(max_workers=4)

# Aggressive (faster, may cause resource contention)
runner = ParallelTestRunner(max_workers=8)
```

### Performance Baselines

```python
# Track baselines over time
baselines = {
    'test_attention': 0.5,
    'test_optimizer': 1.2,
    'test_transformer': 2.8,
    'test_quantization': 0.8
}
```

## Best Practices

1. **Use retry for network/IO tests**: Flaky connections benefit from auto-retry
2. **Use timeout for long-running tests**: Prevent hanging on infinite loops
3. **Use performance tests for critical paths**: Catch regressions early
4. **Run unit tests in parallel**: Most unit tests are independent
5. **Visualize results regularly**: Track trends and improvements
6. **Monitor coverage over time**: Ensure comprehensive testing

## Integration with CI/CD

```yaml
# Example GitHub Actions workflow
name: TruthGPT Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests with coverage
        run: |
          python tests/run_all_tests.py --save-results
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_results.json
```

## Summary

The advanced test features provide:

✅ **Reliability**: Auto-retry, timeouts, performance checks  
✅ **Speed**: Parallel execution for faster results  
✅ **Insight**: Visual summaries and coverage tracking  
✅ **Automation**: Automated regression detection  
✅ **Quality**: Better test organization and reporting

These features make the TruthGPT Optimization Core test suite more robust, faster, and easier to use!


