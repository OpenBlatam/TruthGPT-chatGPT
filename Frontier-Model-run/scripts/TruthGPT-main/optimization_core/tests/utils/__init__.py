"""
Testing utilities for optimization_core.

Provides shared testing utilities, fixtures, and helpers for all test modules.
"""
from .test_helpers import (
    create_mock_engine,
    create_mock_processor,
    create_test_config,
    assert_metrics_valid,
    assert_performance_improvement,
)
from .test_fixtures import (
    MockInferenceEngine,
    MockDataProcessor,
    TestConfig,
    TestDataGenerator,
)
from .test_assertions import (
    assert_engine_works,
    assert_processor_works,
    assert_config_valid,
    assert_error_handled,
)

__all__ = [
    # Helpers
    "create_mock_engine",
    "create_mock_processor",
    "create_test_config",
    "assert_metrics_valid",
    "assert_performance_improvement",
    # Fixtures
    "MockInferenceEngine",
    "MockDataProcessor",
    "TestConfig",
    "TestDataGenerator",
    # Assertions
    "assert_engine_works",
    "assert_processor_works",
    "assert_config_valid",
    "assert_error_handled",
]













