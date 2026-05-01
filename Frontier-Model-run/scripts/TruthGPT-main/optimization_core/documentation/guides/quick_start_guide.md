# Quick Start Guide — Enhanced Test Framework

## Prerequisites

- Python 3.8+
- Optimization core system installed
- Dependencies: `pip install numpy pandas psutil pyyaml matplotlib seaborn`

## Verify Installation

```python
from test_framework.test_runner_enhanced import EnhancedTestRunner
print("Enhanced Test Framework installed successfully!")
```

## Basic Usage

```python
from test_framework.test_runner_enhanced import EnhancedTestRunner
from test_framework.test_config import TestConfig

config = TestConfig(max_workers=2, timeout=60, log_level="INFO")
runner = EnhancedTestRunner(config)

results = runner.run_enhanced_tests()
print(f"Tests executed: {results['results']['total_tests']}")
print(f"Success rate:   {results['results']['success_rate']:.2f}%")
```

## Configuration

### Minimal

```python
config = TestConfig(
    max_workers=4,
    timeout=300,
    log_level="INFO",
    output_dir="results",
)
```

### Full-featured

```python
import os

config = TestConfig(
    max_workers=min(8, os.cpu_count() or 4),
    timeout=600,
    log_level="DEBUG",
    output_dir="test_results",
    parallel_execution=True,
    intelligent_scheduling=True,
    adaptive_timeout=True,
    quality_gates=True,
    performance_monitoring=True,
)
```

## Test Categories

| Category | Classes | Purpose |
|----------|---------|---------|
| **Integration** | `TestModuleIntegration`, `TestComponentIntegration` | Component interactions & end-to-end |
| **Performance** | `TestLoadPerformance`, `TestStressPerformance` | Load, stress, and scalability |
| **Automation** | `TestUnitAutomation`, `TestIntegrationAutomation` | Continuous automated execution |
| **Validation** | `TestInputValidation`, `TestOutputValidation` | Input/output and config validation |
| **Quality** | `TestCodeQuality`, `TestPerformanceQuality` | Code quality and security |

## Running Specific Tests

```python
# Run a single category
test_suite = runner.discover_tests()
categorized = runner.categorize_tests(test_suite)
integration_results = runner.execute_tests_parallel(categorized["integration"])

# Run a single test class
from test_framework.test_integration import TestModuleIntegration

test = TestModuleIntegration()
test.setUp()
test.test_advanced_libraries_integration()
```

## Understanding Results

```python
results = {
    "results": {
        "total_tests": 150,
        "success_rate": 95.5,
        "execution_time": 45.2,
        "total_failures": 5,
        "total_errors": 2,
    },
    "metrics": {
        "execution_metrics": {...},
        "performance_metrics": {...},
        "quality_metrics": {...},
    },
    "analytics": {
        "trend_analysis": {...},
        "pattern_analysis": {...},
        "predictive_analysis": {...},
    },
}
```

| Metric | Description |
|--------|-------------|
| Success Rate | Percentage of passing tests |
| Execution Time | Total wall-clock time |
| Test Coverage | Lines covered by tests |
| Quality Score | Composite quality assessment |
| Performance Score | Composite performance evaluation |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Ensure `optimization_core` is on `sys.path` |
| Timeout failures | Increase `timeout` in `TestConfig` |
| Resource exhaustion | Lower `max_workers` (e.g. `2`) |
| Verbose debugging | Set `log_level="DEBUG"` |

## Next Steps

1. Explore different test categories (integration, performance, automation, validation, quality)
2. Customize `TestConfig` for your environment
3. Use the analytics features to identify failure patterns
4. Extend the framework with custom test classes

## Resources

- [Enhanced Framework Docs](../README_ENHANCED_FRAMEWORK.md)
- [Examples directory](../../examples/)
- [API Reference](api_reference.md)