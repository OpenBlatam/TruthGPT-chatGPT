# TruthGPT Optimization Core Test Suite

## Overview

Comprehensive test suite for the TruthGPT optimization core, covering unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── __init__.py                 # Test suite module
├── run_all_tests.py           # Main test runner
├── conftest.py                # Pytest configuration
├── fixtures/                  # Test fixtures and utilities
│   ├── __init__.py
│   ├── test_data.py           # Test data factory
│   ├── mock_components.py    # Mock components for testing
│   └── test_utils.py          # Test utilities
├── unit/                      # Unit tests
│   ├── __init__.py
│   ├── test_attention_optimizations.py
│   ├── test_optimizer_core.py
│   ├── test_transformer_components.py
│   ├── test_quantization.py
│   ├── test_memory_optimization.py
│   ├── test_cuda_optimizations.py
│   ├── test_advanced_optimizations.py
│   ├── test_advanced_workflows.py
│   ├── test_automated_ml.py
│   ├── test_federated_optimization.py
│   ├── test_hyperparameter_optimization.py
│   ├── test_meta_learning_optimization.py
│   ├── test_neural_architecture_search.py
│   ├── test_optimization_ai.py
│   ├── test_optimization_analytics.py
│   ├── test_optimization_automation.py
│   ├── test_optimization_benchmarks.py
│   ├── test_optimization_research.py
│   ├── test_optimization_validation.py
│   ├── test_optimization_visualization.py
│   └── test_quantum_optimization.py
├── integration/              # Integration tests
│   ├── __init__.py
│   ├── test_end_to_end.py
│   └── test_advanced_workflows.py
└── performance/              # Performance benchmarks
    ├── __init__.py
    ├── test_performance_benchmarks.py
    └── test_advanced_benchmarks.py
```

## Running Tests

### Run All Tests

```bash
# From the optimization_core directory
python tests/run_all_tests.py
```

### Run Specific Test Categories

```bash
# Run only unit tests
python tests/run_all_tests.py --pattern unit

# Run only integration tests
python tests/run_all_tests.py --integration

# Run only performance tests
python tests/run_all_tests.py --performance
```

### Run with Options

```bash
# Verbose output
python tests/run_all_tests.py --verbose

# Save results to JSON file
python tests/run_all_tests.py --save-results

# Run specific test files
python tests/run_all_tests.py --pattern attention optimizer
```

### Using pytest (recommended)

```bash
# Run all tests
pytest tests/

# Run specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run in parallel
pytest tests/ -n auto
```

## Test Configuration

The test suite uses several key components:

### Test Data Factory

Generates synthetic datasets for testing:

```python
from tests.fixtures.test_data import TestDataFactory

factory = TestDataFactory()

# Create attention test data
attention_data = factory.create_attention_data(batch_size=2, seq_len=128, d_model=512)

# Create optimization test data
opt_data = factory.create_optimization_data(num_params=1000, num_epochs=10)
```

### Mock Components

Ready-to-use mock components for testing:

```python
from tests.fixtures.mock_components import MockModel, MockOptimizer, MockAttention

model = MockModel(input_size=512, hidden_size=1024, output_size=512)
optimizer = MockOptimizer(learning_rate=0.001)
attention = MockAttention(d_model=512, n_heads=8)
```

### Test Utilities

Performance profiling and memory tracking:

```python
from tests.fixtures.test_utils import PerformanceProfiler, MemoryTracker

profiler = PerformanceProfiler()
profiler.start_profile("test_name")
# ... run test code ...
metrics = profiler.end_profile()

memory_tracker = MemoryTracker()
memory_tracker.take_snapshot("before")
# ... run test code ...
memory_tracker.take_snapshot("after")
```

## Test Coverage

### Unit Tests

- **Optimizer Core**: Advanced optimization algorithms (Adam, SGD, AdaGrad, etc.)
- **Attention Optimizations**: KV cache, efficient attention mechanisms
- **Transformer Components**: Blocks, normalization, positional encoding
- **Quantization**: INT8, FP16, dynamic quantization
- **Memory Optimization**: Memory pooling, efficient memory management
- **CUDA Optimizations**: GPU kernels, fused operations
- **Advanced Techniques**: Meta-learning, federated optimization, NAS
- **AI-Driven Optimization**: Automated optimization strategies

### Integration Tests

- **End-to-End Workflows**: Complete training pipelines
- **Advanced Workflows**: Complex multi-stage optimization

### Performance Tests

- **Benchmarks**: Performance comparisons and metrics
- **Advanced Benchmarks**: Scalability and throughput tests

## Expected Results

### Success Criteria

- All unit tests should pass with ≥95% success rate
- Integration tests should verify complete workflows
- Performance benchmarks should show improvements
- Memory usage should be monitored and optimized
- No memory leaks or resource issues

### Troubleshooting

**Import Errors**: Ensure all dependencies are installed:
```bash
pip install torch numpy psutil pytest pytest-cov
```

**Missing Modules**: The test runner includes fallback implementations for missing utilities.

**Path Issues**: Make sure you're running tests from the correct directory (optimization_core).

## Advanced Features

### Performance Profiling

```python
# Automatic performance profiling during tests
profiler = PerformanceProfiler()
profiler.start_profile("operation_name")
# ... operation ...
metrics = profiler.end_profile()
print(f"Execution time: {metrics['execution_time']}s")
```

### Memory Tracking

```python
# Track memory usage during tests
memory_tracker = MemoryTracker()
memory_tracker.take_snapshot("start")
# ... operations ...
memory_tracker.take_snapshot("end")
summary = memory_tracker.get_memory_summary()
print(f"Peak memory: {summary['peak_memory']}MB")
```

## Contributing

When adding new tests:

1. Create test file in appropriate category (unit/integration/performance)
2. Use fixtures from `tests/fixtures/` for consistency
3. Add proper docstrings and assertions
4. Update this documentation if adding new test categories

## License

Part of the TruthGPT Optimization Core test suite.


