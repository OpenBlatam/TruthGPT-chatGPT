"""
Testing Framework for TruthGPT Optimization Core
Provides comprehensive testing utilities and test suites
"""

from .test_runner import (
    TestRunner,
    TestResult,
    TestSuite,
    create_test_runner,
    run_tests
)

from .unit_tests import (
    UnitTestSuite,
    create_unit_test_suite
)

from .integration_tests import (
    IntegrationTestSuite,
    create_integration_test_suite
)

from .performance_tests import (
    PerformanceTestSuite,
    create_performance_test_suite
)

from .test_utils import (
    TestUtils,
    MockModel,
    create_test_utils
)

__all__ = [
    # Test Runner
    'TestRunner',
    'TestResult',
    'TestSuite',
    'create_test_runner',
    'run_tests',
    
    # Unit Tests
    'UnitTestSuite',
    'create_unit_test_suite',
    
    # Integration Tests
    'IntegrationTestSuite',
    'create_integration_test_suite',
    
    # Performance Tests
    'PerformanceTestSuite',
    'create_performance_test_suite',
    
    # Test Utils
    'TestUtils',
    'MockModel',
    'create_test_utils'
]
