"""
Comprehensive Integration Testing Framework
==========================================

A comprehensive testing framework for TruthGPT optimization systems
with automated testing, performance validation, and integration testing.

Author: TruthGPT Optimization Team
Version: 41.3.0-INTEGRATION-TESTING-FRAMEWORK
"""

import asyncio
import logging
import time
import unittest
import pytest
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import json
import pickle
from datetime import datetime, timedelta
import threading
import queue
import subprocess
import tempfile
import shutil
import os
import sys
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Test type enumeration"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    STRESS = "stress"
    REGRESSION = "regression"
    COMPATIBILITY = "compatibility"
    SECURITY = "security"
    LOAD = "load"
    END_TO_END = "end_to_end"

class TestStatus(Enum):
    """Test status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class TestPriority(Enum):
    """Test priority enumeration"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TestCase(BaseModel):
    """Test case data structure"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    test_id: str
    test_name: str
    test_type: TestType
    priority: TestPriority
    description: str
    test_function: Any  # Callable
    setup_function: Optional[Any] = None  # Optional[Callable]
    teardown_function: Optional[Any] = None  # Optional[Callable]
    timeout: float = 300.0
    retry_count: int = 0
    dependencies: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    expected_result: Optional[Any] = None
    performance_thresholds: Dict[str, float] = Field(default_factory=dict)


class TestResult(BaseModel):
    """Test result data structure"""
    test_id: str
    test_name: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    output: str = ""
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestSuite(BaseModel):
    """Test suite data structure"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    suite_id: str
    suite_name: str
    description: str
    test_cases: List[Any]  # List[TestCase]
    setup_function: Optional[Any] = None  # Optional[Callable]
    teardown_function: Optional[Any] = None  # Optional[Callable]
    parallel_execution: bool = True
    max_parallel_tests: int = 4

class ComprehensiveIntegrationTestingFramework:
    """
    Comprehensive Integration Testing Framework
    
    Provides automated testing, performance validation, and integration testing
    for TruthGPT optimization systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Testing Framework
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.test_suites = {}
        self.test_results = {}
        self.test_queue = queue.Queue()
        self.running_tests = {}
        self.test_executor = None
        
        # Test configuration
        self.max_parallel_tests = self.config.get('max_parallel_tests', 4)
        self.default_timeout = self.config.get('default_timeout', 300.0)
        self.retry_failed_tests = self.config.get('retry_failed_tests', True)
        self.max_retries = self.config.get('max_retries', 3)
        
        # Performance thresholds
        self.performance_thresholds = self.config.get('performance_thresholds', {
            'min_speedup': 1.0,
            'max_memory_usage': 16.0,  # GB
            'max_execution_time': 300.0,  # seconds
            'min_accuracy': 0.95
        })
        
        # Test environment
        self.test_environment = self._setup_test_environment()
        
        # Initialize test suites
        self._initialize_test_suites()
        
        logger.info("Comprehensive Integration Testing Framework initialized")
    
    def _setup_test_environment(self) -> Dict[str, Any]:
        """Setup test environment"""
        # Create temporary directory for test artifacts
        temp_dir = tempfile.mkdtemp(prefix="truthgpt_test_")
        
        # Setup PyTorch for testing
        torch.manual_seed(42)
        np.random.seed(42)
        
        return {
            'temp_dir': temp_dir,
            'test_data_dir': os.path.join(temp_dir, 'test_data'),
            'results_dir': os.path.join(temp_dir, 'results'),
            'logs_dir': os.path.join(temp_dir, 'logs'),
            'torch_device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'test_model': self._create_test_model()
        }
    
    def _create_test_model(self) -> nn.Module:
        """Create a test model for optimization testing"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(50, 10)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        return TestModel()
    
    def _initialize_test_suites(self):
        """Initialize test suites"""
        # Unit Tests Suite
        self._create_unit_tests_suite()
        
        # Integration Tests Suite
        self._create_integration_tests_suite()
        
        # Performance Tests Suite
        self._create_performance_tests_suite()
        
        # Stress Tests Suite
        self._create_stress_tests_suite()
        
        # End-to-End Tests Suite
        self._create_end_to_end_tests_suite()
        
        logger.info(f"Initialized {len(self.test_suites)} test suites")
    
    def _create_unit_tests_suite(self):
        """Create unit tests suite"""
        suite = TestSuite(
            suite_id="unit_tests",
            suite_name="Unit Tests",
            description="Unit tests for individual components",
            test_cases=[]
        )
        
        # Test optimizer initialization
        def test_optimizer_initialization():
            from .ultra_master_orchestration_system import UltraMasterOrchestrationSystem
            orchestration_system = UltraMasterOrchestrationSystem()
            assert orchestration_system is not None
            assert len(orchestration_system.optimizers) > 0
        
        suite.test_cases.append(TestCase(
            test_id="test_optimizer_init",
            test_name="Test Optimizer Initialization",
            test_type=TestType.UNIT,
            priority=TestPriority.HIGH,
            description="Test that optimizers can be initialized correctly",
            test_function=test_optimizer_initialization
        ))
        
        # Test adaptive strategies
        def test_adaptive_strategies():
            from .adaptive_optimization_strategies import AdaptiveOptimizationStrategies
            adaptive_system = AdaptiveOptimizationStrategies()
            assert adaptive_system is not None
            assert len(adaptive_system.adaptation_rules) > 0
        
        suite.test_cases.append(TestCase(
            test_id="test_adaptive_strategies",
            test_name="Test Adaptive Strategies",
            test_type=TestType.UNIT,
            priority=TestPriority.HIGH,
            description="Test that adaptive strategies can be initialized",
            test_function=test_adaptive_strategies
        ))
        
        # Test performance monitor
        def test_performance_monitor():
            from .real_time_performance_monitor import RealTimePerformanceMonitor
            monitor = RealTimePerformanceMonitor()
            assert monitor is not None
            assert monitor.monitoring_active
        
        suite.test_cases.append(TestCase(
            test_id="test_performance_monitor",
            test_name="Test Performance Monitor",
            test_type=TestType.UNIT,
            priority=TestPriority.HIGH,
            description="Test that performance monitor can be initialized",
            test_function=test_performance_monitor
        ))
        
        self.test_suites["unit_tests"] = suite
    
    def _create_integration_tests_suite(self):
        """Create integration tests suite"""
        suite = TestSuite(
            suite_id="integration_tests",
            suite_name="Integration Tests",
            description="Integration tests for component interactions",
            test_cases=[]
        )
        
        # Test orchestration with optimization
        def test_orchestration_optimization():
            from .ultra_master_orchestration_system import UltraMasterOrchestrationSystem, OptimizationRequest, OptimizationStrategy, OptimizationLevel
            from .adaptive_optimization_strategies import AdaptationContext
            
            orchestration_system = UltraMasterOrchestrationSystem()
            
            # Create test data
            test_data = torch.randn(100, 100)
            
            # Create optimization request
            request = OptimizationRequest(
                input_data=test_data,
                strategy=OptimizationStrategy.PERFORMANCE_FOCUSED,
                level=OptimizationLevel.ULTRA,
                timeout=60.0
            )
            
            # Run optimization
            async def run_optimization():
                result = await orchestration_system.optimize(request)
                assert result.success
                assert result.optimizer_used is not None
                assert result.optimization_time > 0
            
            asyncio.run(run_optimization())
        
        suite.test_cases.append(TestCase(
            test_id="test_orchestration_optimization",
            test_name="Test Orchestration Optimization",
            test_type=TestType.INTEGRATION,
            priority=TestPriority.CRITICAL,
            description="Test orchestration system with optimization",
            test_function=test_orchestration_optimization,
            timeout=120.0
        ))
        
        # Test adaptive strategies with orchestration
        def test_adaptive_orchestration():
            from .ultra_master_orchestration_system import UltraMasterOrchestrationSystem
            from .adaptive_optimization_strategies import AdaptiveOptimizationStrategies, AdaptationContext
            
            orchestration_system = UltraMasterOrchestrationSystem()
            adaptive_system = AdaptiveOptimizationStrategies()
            
            # Create adaptation context
            context = AdaptationContext(
                current_optimizer="UltimateTruthGPTOptimizer",
                task_metrics={'performance': 0.7, 'memory_usage': 0.8},
                system_metrics={'cpu_usage': 85.0, 'memory_usage': 90.0},
                user_preferences={'priority': 'performance'},
                task_history=[],
                available_optimizers=list(orchestration_system.optimizers.keys()),
                resource_constraints={'max_memory': 16.0},
                time_constraints={'timeout': 300.0},
                quality_requirements={'min_quality': 0.9}
            )
            
            # Evaluate adaptation needs
            decisions = adaptive_system.evaluate_adaptation_need(context)
            assert isinstance(decisions, list)
        
        suite.test_cases.append(TestCase(
            test_id="test_adaptive_orchestration",
            test_name="Test Adaptive Orchestration",
            test_type=TestType.INTEGRATION,
            priority=TestPriority.HIGH,
            description="Test adaptive strategies with orchestration",
            test_function=test_adaptive_orchestration
        ))
        
        self.test_suites["integration_tests"] = suite
    
    def _create_performance_tests_suite(self):
        """Create performance tests suite"""
        suite = TestSuite(
            suite_id="performance_tests",
            suite_name="Performance Tests",
            description="Performance validation tests",
            test_cases=[]
        )
        
        # Test optimization speedup
        def test_optimization_speedup():
            from .ultra_master_orchestration_system import UltraMasterOrchestrationSystem, OptimizationRequest, OptimizationStrategy, OptimizationLevel
            
            orchestration_system = UltraMasterOrchestrationSystem()
            
            # Create test data
            test_data = torch.randn(1000, 1000)
            
            # Measure baseline performance
            start_time = time.time()
            baseline_result = torch.matmul(test_data, test_data.t())
            baseline_time = time.time() - start_time
            
            # Create optimization request
            request = OptimizationRequest(
                input_data=test_data,
                strategy=OptimizationStrategy.PERFORMANCE_FOCUSED,
                level=OptimizationLevel.ULTRA,
                timeout=60.0
            )
            
            # Run optimization
            async def run_optimization():
                result = await orchestration_system.optimize(request)
                assert result.success
                
                # Calculate speedup
                speedup = baseline_time / result.optimization_time
                assert speedup >= self.performance_thresholds['min_speedup']
            
            asyncio.run(run_optimization())
        
        suite.test_cases.append(TestCase(
            test_id="test_optimization_speedup",
            test_name="Test Optimization Speedup",
            test_type=TestType.PERFORMANCE,
            priority=TestPriority.CRITICAL,
            description="Test that optimization provides speedup",
            test_function=test_optimization_speedup,
            performance_thresholds={'min_speedup': 1.0}
        ))
        
        # Test memory efficiency
        def test_memory_efficiency():
            import psutil
            from .ultra_master_orchestration_system import UltraMasterOrchestrationSystem, OptimizationRequest, OptimizationStrategy, OptimizationLevel
            
            orchestration_system = UltraMasterOrchestrationSystem()
            
            # Create large test data
            test_data = torch.randn(5000, 5000)
            
            # Measure memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**3)  # GB
            
            # Create optimization request
            request = OptimizationRequest(
                input_data=test_data,
                strategy=OptimizationStrategy.MEMORY_FOCUSED,
                level=OptimizationLevel.ULTRA,
                timeout=60.0
            )
            
            # Run optimization
            async def run_optimization():
                result = await orchestration_system.optimize(request)
                assert result.success
                
                # Check memory usage
                final_memory = process.memory_info().rss / (1024**3)  # GB
                memory_increase = final_memory - initial_memory
                assert memory_increase <= self.performance_thresholds['max_memory_usage']
            
            asyncio.run(run_optimization())
        
        suite.test_cases.append(TestCase(
            test_id="test_memory_efficiency",
            test_name="Test Memory Efficiency",
            test_type=TestType.PERFORMANCE,
            priority=TestPriority.HIGH,
            description="Test that optimization is memory efficient",
            test_function=test_memory_efficiency,
            performance_thresholds={'max_memory_usage': 16.0}
        ))
        
        self.test_suites["performance_tests"] = suite
    
    def _create_stress_tests_suite(self):
        """Create stress tests suite"""
        suite = TestSuite(
            suite_id="stress_tests",
            suite_name="Stress Tests",
            description="Stress tests for system stability",
            test_cases=[]
        )
        
        # Test concurrent optimizations
        def test_concurrent_optimizations():
            from .ultra_master_orchestration_system import UltraMasterOrchestrationSystem, OptimizationRequest, OptimizationStrategy, OptimizationLevel
            
            orchestration_system = UltraMasterOrchestrationSystem()
            
            # Create multiple optimization requests
            requests = []
            for i in range(10):
                test_data = torch.randn(100, 100)
                request = OptimizationRequest(
                    input_data=test_data,
                    strategy=OptimizationStrategy.PERFORMANCE_FOCUSED,
                    level=OptimizationLevel.ULTRA,
                    timeout=30.0
                )
                requests.append(request)
            
            # Run concurrent optimizations
            async def run_concurrent_optimizations():
                tasks = [orchestration_system.optimize(req) for req in requests]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check results
                successful_results = [r for r in results if not isinstance(r, Exception) and r.success]
                assert len(successful_results) >= 8  # At least 80% success rate
            
            asyncio.run(run_concurrent_optimizations())
        
        suite.test_cases.append(TestCase(
            test_id="test_concurrent_optimizations",
            test_name="Test Concurrent Optimizations",
            test_type=TestType.STRESS,
            priority=TestPriority.HIGH,
            description="Test system stability under concurrent load",
            test_function=test_concurrent_optimizations,
            timeout=180.0
        ))
        
        # Test long-running optimization
        def test_long_running_optimization():
            from .ultra_master_orchestration_system import UltraMasterOrchestrationSystem, OptimizationRequest, OptimizationStrategy, OptimizationLevel
            
            orchestration_system = UltraMasterOrchestrationSystem()
            
            # Create large test data
            test_data = torch.randn(10000, 1000)
            
            # Create optimization request
            request = OptimizationRequest(
                input_data=test_data,
                strategy=OptimizationStrategy.QUALITY_FOCUSED,
                level=OptimizationLevel.ULTIMATE,
                timeout=600.0
            )
            
            # Run long optimization
            async def run_long_optimization():
                result = await orchestration_system.optimize(request)
                assert result.success
                assert result.optimization_time <= self.performance_thresholds['max_execution_time']
            
            asyncio.run(run_long_optimization())
        
        suite.test_cases.append(TestCase(
            test_id="test_long_running_optimization",
            test_name="Test Long Running Optimization",
            test_type=TestType.STRESS,
            priority=TestPriority.MEDIUM,
            description="Test system stability for long-running operations",
            test_function=test_long_running_optimization,
            timeout=900.0
        ))
        
        self.test_suites["stress_tests"] = suite
    
    def _create_end_to_end_tests_suite(self):
        """Create end-to-end tests suite"""
        suite = TestSuite(
            suite_id="end_to_end_tests",
            suite_name="End-to-End Tests",
            description="End-to-end workflow tests",
            test_cases=[]
        )
        
        # Test complete optimization workflow
        def test_complete_optimization_workflow():
            from .ultra_master_orchestration_system import UltraMasterOrchestrationSystem, OptimizationRequest, OptimizationStrategy, OptimizationLevel
            from .adaptive_optimization_strategies import AdaptiveOptimizationStrategies
            from .real_time_performance_monitor import RealTimePerformanceMonitor, MetricType
            
            # Initialize all components
            orchestration_system = UltraMasterOrchestrationSystem()
            adaptive_system = AdaptiveOptimizationStrategies()
            monitor = RealTimePerformanceMonitor()
            
            # Create test data
            test_data = torch.randn(1000, 1000)
            
            # Create optimization request
            request = OptimizationRequest(
                input_data=test_data,
                strategy=OptimizationStrategy.ADAPTIVE,
                level=OptimizationLevel.ULTRA,
                timeout=120.0
            )
            
            # Run complete workflow
            async def run_complete_workflow():
                # Start monitoring
                monitor.record_metric("workflow_start", MetricType.PERFORMANCE, 1.0)
                
                # Run optimization
                result = await orchestration_system.optimize(request)
                assert result.success
                
                # Record performance metrics
                monitor.record_metric("optimization_performance", MetricType.PERFORMANCE, result.performance_metrics.get('speedup', 1.0))
                monitor.record_metric("optimization_memory", MetricType.MEMORY, result.memory_usage.get('peak_memory', 0.0))
                
                # Test adaptive strategies
                context = adaptive_system.evaluate_adaptation_need(
                    AdaptationContext(
                        current_optimizer=result.optimizer_used,
                        task_metrics=result.performance_metrics,
                        system_metrics=monitor.system_metrics.__dict__,
                        user_preferences={'priority': 'performance'},
                        task_history=[],
                        available_optimizers=list(orchestration_system.optimizers.keys()),
                        resource_constraints={'max_memory': 16.0},
                        time_constraints={'timeout': 300.0},
                        quality_requirements={'min_quality': 0.9}
                    )
                )
                
                # Verify workflow completion
                assert result.success
                assert result.optimizer_used is not None
                assert len(context) >= 0  # Adaptation decisions
                
                # Record completion
                monitor.record_metric("workflow_complete", MetricType.PERFORMANCE, 1.0)
            
            asyncio.run(run_complete_workflow())
        
        suite.test_cases.append(TestCase(
            test_id="test_complete_workflow",
            test_name="Test Complete Optimization Workflow",
            test_type=TestType.END_TO_END,
            priority=TestPriority.CRITICAL,
            description="Test complete optimization workflow with all components",
            test_function=test_complete_optimization_workflow,
            timeout=300.0
        ))
        
        self.test_suites["end_to_end_tests"] = suite
    
    def run_test_suite(self, suite_id: str) -> Dict[str, TestResult]:
        """Run a specific test suite"""
        if suite_id not in self.test_suites:
            raise ValueError(f"Test suite {suite_id} not found")
        
        suite = self.test_suites[suite_id]
        results = {}
        
        logger.info(f"Running test suite: {suite.suite_name}")
        
        # Setup suite
        if suite.setup_function:
            try:
                suite.setup_function()
            except Exception as e:
                logger.error(f"Suite setup failed: {e}")
                return results
        
        # Run test cases
        for test_case in suite.test_cases:
            try:
                result = self._run_test_case(test_case)
                results[test_case.test_id] = result
            except Exception as e:
                logger.error(f"Test case {test_case.test_id} failed: {e}")
                results[test_case.test_id] = TestResult(
                    test_id=test_case.test_id,
                    test_name=test_case.test_name,
                    status=TestStatus.ERROR,
                    start_time=datetime.now(),
                    error_message=str(e)
                )
        
        # Teardown suite
        if suite.teardown_function:
            try:
                suite.teardown_function()
            except Exception as e:
                logger.error(f"Suite teardown failed: {e}")
        
        return results
    
    def _run_test_case(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        start_time = datetime.now()
        
        try:
            # Setup
            if test_case.setup_function:
                test_case.setup_function()
            
            # Run test
            test_case.test_function()
            
            # Teardown
            if test_case.teardown_function:
                test_case.teardown_function()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_id=test_case.test_id,
                test_name=test_case.test_name,
                status=TestStatus.PASSED,
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_id=test_case.test_id,
                test_name=test_case.test_name,
                status=TestStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                error_message=str(e)
            )
    
    def run_all_tests(self) -> Dict[str, Dict[str, TestResult]]:
        """Run all test suites"""
        all_results = {}
        
        for suite_id in self.test_suites.keys():
            try:
                results = self.run_test_suite(suite_id)
                all_results[suite_id] = results
            except Exception as e:
                logger.error(f"Failed to run test suite {suite_id}: {e}")
                all_results[suite_id] = {}
        
        return all_results
    
    def generate_test_report(self, results: Dict[str, Dict[str, TestResult]]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        
        suite_summaries = {}
        
        for suite_id, suite_results in results.items():
            suite_total = len(suite_results)
            suite_passed = len([r for r in suite_results.values() if r.status == TestStatus.PASSED])
            suite_failed = len([r for r in suite_results.values() if r.status == TestStatus.FAILED])
            suite_error = len([r for r in suite_results.values() if r.status == TestStatus.ERROR])
            
            total_tests += suite_total
            passed_tests += suite_passed
            failed_tests += suite_failed
            error_tests += suite_error
            
            suite_summaries[suite_id] = {
                'total': suite_total,
                'passed': suite_passed,
                'failed': suite_failed,
                'error': suite_error,
                'success_rate': suite_passed / max(suite_total, 1)
            }
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': passed_tests / max(total_tests, 1),
                'timestamp': datetime.now()
            },
            'suite_summaries': suite_summaries,
            'detailed_results': results,
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Dict[str, TestResult]]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze failure patterns
        failed_tests = []
        for suite_results in results.values():
            for test_result in suite_results.values():
                if test_result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    failed_tests.append(test_result)
        
        if failed_tests:
            recommendations.append("Review failed tests and fix underlying issues")
            
            # Performance-related failures
            perf_failures = [t for t in failed_tests if 'performance' in t.test_name.lower()]
            if perf_failures:
                recommendations.append("Optimize performance-critical components")
            
            # Memory-related failures
            memory_failures = [t for t in failed_tests if 'memory' in t.test_name.lower()]
            if memory_failures:
                recommendations.append("Implement memory optimization strategies")
        
        # Success rate recommendations
        total_tests = sum(len(suite_results) for suite_results in results.values())
        passed_tests = sum(len([r for r in suite_results.values() if r.status == TestStatus.PASSED]) 
                          for suite_results in results.values())
        
        if total_tests > 0:
            success_rate = passed_tests / total_tests
            if success_rate < 0.8:
                recommendations.append("Improve overall test success rate")
            elif success_rate < 0.95:
                recommendations.append("Fine-tune optimization parameters")
            else:
                recommendations.append("System is performing well, consider adding more test cases")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup test environment"""
        try:
            shutil.rmtree(self.test_environment['temp_dir'])
            logger.info("Test environment cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup test environment: {e}")

# Factory function
def create_integration_testing_framework(config: Optional[Dict[str, Any]] = None) -> ComprehensiveIntegrationTestingFramework:
    """
    Create a Comprehensive Integration Testing Framework instance
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ComprehensiveIntegrationTestingFramework instance
    """
    return ComprehensiveIntegrationTestingFramework(config)

# Example usage
if __name__ == "__main__":
    # Create testing framework
    testing_framework = create_integration_testing_framework()
    
    # Run all tests
    results = testing_framework.run_all_tests()
    
    # Generate report
    report = testing_framework.generate_test_report(results)
    
    print(f"Test Report Summary:")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.2%}")
    
    # Cleanup
    testing_framework.cleanup()

