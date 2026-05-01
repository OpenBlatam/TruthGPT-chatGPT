"""
Test Runner for TruthGPT Optimization Core
Provides comprehensive testing framework with test discovery and execution
"""

import unittest
import time
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback

logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Result of a test execution."""
    name: str
    status: TestStatus
    duration: float
    message: Optional[str] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestSuite:
    """Test suite containing multiple tests."""
    name: str
    tests: List[Callable]
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TestRunner:
    """
    Test runner for executing test suites and individual tests.
    
    Provides comprehensive testing capabilities including test discovery,
    execution, reporting, and result analysis.
    """
    
    def __init__(self, verbose: bool = True, fail_fast: bool = False):
        """
        Initialize test runner.
        
        Args:
            verbose: Whether to use verbose output
            fail_fast: Whether to stop on first failure
        """
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.results: List[TestResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def run_test(self, test_func: Callable, test_name: Optional[str] = None) -> TestResult:
        """
        Run a single test function.
        
        Args:
            test_func: Test function to run
            test_name: Optional name for the test
            
        Returns:
            Test result
        """
        name = test_name or test_func.__name__
        start_time = time.time()
        
        try:
            # Run the test
            test_func()
            duration = time.time() - start_time
            
            result = TestResult(
                name=name,
                status=TestStatus.PASSED,
                duration=duration
            )
            
            if self.verbose:
                self.logger.info(f"✓ {name} passed ({duration:.3f}s)")
            
        except AssertionError as e:
            duration = time.time() - start_time
            result = TestResult(
                name=name,
                status=TestStatus.FAILED,
                duration=duration,
                message=str(e),
                traceback=traceback.format_exc()
            )
            
            if self.verbose:
                self.logger.error(f"✗ {name} failed ({duration:.3f}s): {e}")
            
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                name=name,
                status=TestStatus.ERROR,
                duration=duration,
                message=str(e),
                traceback=traceback.format_exc()
            )
            
            if self.verbose:
                self.logger.error(f"✗ {name} error ({duration:.3f}s): {e}")
        
        self.results.append(result)
        return result
    
    def run_suite(self, suite: TestSuite) -> List[TestResult]:
        """
        Run a test suite.
        
        Args:
            suite: Test suite to run
            
        Returns:
            List of test results
        """
        self.logger.info(f"Running test suite: {suite.name}")
        
        # Setup
        if suite.setup:
            try:
                suite.setup()
            except Exception as e:
                self.logger.error(f"Setup failed for suite {suite.name}: {e}")
                return []
        
        # Run tests
        suite_results = []
        for test_func in suite.tests:
            result = self.run_test(test_func)
            suite_results.append(result)
            
            # Stop on failure if fail_fast is enabled
            if self.fail_fast and result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                break
        
        # Teardown
        if suite.teardown:
            try:
                suite.teardown()
            except Exception as e:
                self.logger.error(f"Teardown failed for suite {suite.name}: {e}")
        
        return suite_results
    
    def run_tests(self, tests: List[Union[Callable, TestSuite]]) -> List[TestResult]:
        """
        Run multiple tests or test suites.
        
        Args:
            tests: List of test functions or test suites
            
        Returns:
            List of test results
        """
        all_results = []
        
        for test in tests:
            if isinstance(test, TestSuite):
                results = self.run_suite(test)
                all_results.extend(results)
            else:
                result = self.run_test(test)
                all_results.append(result)
        
        return all_results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of test results.
        
        Returns:
            Dictionary containing test summary
        """
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        skipped_tests = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        
        total_duration = sum(r.duration for r in self.results)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'skipped_tests': skipped_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_duration': total_duration,
            'average_duration': total_duration / total_tests if total_tests > 0 else 0
        }
    
    def print_summary(self) -> None:
        """Print test summary."""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Errors: {summary['error_tests']}")
        print(f"Skipped: {summary['skipped_tests']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Total duration: {summary['total_duration']:.3f}s")
        print(f"Average duration: {summary['average_duration']:.3f}s")
        print("="*50)
    
    def get_failed_tests(self) -> List[TestResult]:
        """Get list of failed tests."""
        return [r for r in self.results if r.status in [TestStatus.FAILED, TestStatus.ERROR]]
    
    def get_passed_tests(self) -> List[TestResult]:
        """Get list of passed tests."""
        return [r for r in self.results if r.status == TestStatus.PASSED]
    
    def clear_results(self) -> None:
        """Clear all test results."""
        self.results.clear()

class TestDiscovery:
    """Test discovery utilities."""
    
    @staticmethod
    def discover_tests(module_path: str, pattern: str = "test_*.py") -> List[str]:
        """
        Discover test files in a module.
        
        Args:
            module_path: Path to the module
            pattern: Pattern to match test files
            
        Returns:
            List of test file paths
        """
        import os
        import glob
        
        test_files = []
        for root, dirs, files in os.walk(module_path):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(os.path.join(root, file))
        
        return test_files
    
    @staticmethod
    def load_test_module(module_path: str) -> Any:
        """
        Load a test module.
        
        Args:
            module_path: Path to the test module
            
        Returns:
            Loaded test module
        """
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_module"] = module
        spec.loader.exec_module(module)
        
        return module

# Factory functions
def create_test_runner(verbose: bool = True, fail_fast: bool = False) -> TestRunner:
    """Create a test runner instance."""
    return TestRunner(verbose, fail_fast)

def run_tests(
    tests: List[Union[Callable, TestSuite]], 
    verbose: bool = True, 
    fail_fast: bool = False
) -> List[TestResult]:
    """
    Run tests using a test runner.
    
    Args:
        tests: List of test functions or test suites
        verbose: Whether to use verbose output
        fail_fast: Whether to stop on first failure
        
    Returns:
        List of test results
    """
    runner = TestRunner(verbose, fail_fast)
    return runner.run_tests(tests)

def create_test_suite(
    name: str, 
    tests: List[Callable], 
    setup: Optional[Callable] = None, 
    teardown: Optional[Callable] = None,
    **metadata
) -> TestSuite:
    """Create a test suite."""
    return TestSuite(name, tests, setup, teardown, metadata)
