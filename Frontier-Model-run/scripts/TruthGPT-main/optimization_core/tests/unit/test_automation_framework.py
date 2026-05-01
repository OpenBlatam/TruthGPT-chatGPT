"""
Advanced Test Automation Framework for TruthGPT Optimization Core
=================================================================

This module implements advanced test automation capabilities including:
- Automated test discovery and execution
- Dynamic test configuration
- Test result analysis and reporting
- Continuous test optimization
- Test environment management
"""

import unittest
import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestConfiguration:
    """Test configuration for automated execution"""
    test_patterns: List[str]
    execution_mode: str
    parallel_workers: int
    timeout_seconds: int
    retry_count: int
    environment_vars: Dict[str, str]
    resource_limits: Dict[str, Any]

@dataclass
class TestExecutionResult:
    """Result of test execution"""
    test_name: str
    status: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    output: str
    error_message: Optional[str]
    coverage_data: Dict[str, Any]

class AutomatedTestDiscovery:
    """Automated test discovery and organization"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.test_patterns = [
            "test_*.py",
            "*_test.py", 
            "tests/test_*.py",
            "tests/*_test.py"
        ]
        self.discovered_tests = {}
    
    def discover_tests(self) -> Dict[str, List[str]]:
        """Discover all tests in the project"""
        logger.info(f"Discovering tests in {self.base_path}")
        
        discovered = {
            "unit_tests": [],
            "integration_tests": [],
            "performance_tests": [],
            "other_tests": []
        }
        
        for pattern in self.test_patterns:
            test_files = list(self.base_path.rglob(pattern))
            
            for test_file in test_files:
                test_type = self._classify_test(test_file)
                discovered[test_type].append(str(test_file))
        
        self.discovered_tests = discovered
        logger.info(f"Discovered {sum(len(tests) for tests in discovered.values())} test files")
        
        return discovered
    
    def _classify_test(self, test_file: Path) -> str:
        """Classify test file by type"""
        path_str = str(test_file).lower()
        
        if "unit" in path_str:
            return "unit_tests"
        elif "integration" in path_str:
            return "integration_tests"
        elif "performance" in path_str:
            return "performance_tests"
        else:
            return "other_tests"
    
    def get_test_dependencies(self, test_file: str) -> List[str]:
        """Get dependencies for a test file"""
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            dependencies = []
            
            # Extract import statements
            import_lines = [line.strip() for line in content.split('\n') 
                          if line.strip().startswith('import ') or 
                             line.strip().startswith('from ')]
            
            for line in import_lines:
                if 'import' in line:
                    module = line.split('import')[1].strip().split()[0]
                    dependencies.append(module)
            
            return dependencies
        except Exception as e:
            logger.warning(f"Could not analyze dependencies for {test_file}: {e}")
            return []

class DynamicTestConfiguration:
    """Dynamic test configuration management"""
    
    def __init__(self):
        self.configurations = {}
        self.environment_profiles = {
            "development": {
                "debug": True,
                "log_level": "DEBUG",
                "timeout": 30,
                "retry_count": 1
            },
            "staging": {
                "debug": False,
                "log_level": "INFO", 
                "timeout": 60,
                "retry_count": 2
            },
            "production": {
                "debug": False,
                "log_level": "WARNING",
                "timeout": 120,
                "retry_count": 3
            }
        }
    
    def create_configuration(self, environment: str, 
                           custom_settings: Dict[str, Any] = None) -> TestConfiguration:
        """Create test configuration for environment"""
        base_config = self.environment_profiles.get(environment, 
                                                  self.environment_profiles["development"])
        
        if custom_settings:
            base_config.update(custom_settings)
        
        config = TestConfiguration(
            test_patterns=["test_*.py"],
            execution_mode="parallel",
            parallel_workers=min(4, os.cpu_count() or 1),
            timeout_seconds=base_config["timeout"],
            retry_count=base_config["retry_count"],
            environment_vars={
                "DEBUG": str(base_config["debug"]),
                "LOG_LEVEL": base_config["log_level"]
            },
            resource_limits={
                "max_memory_mb": 2048,
                "max_cpu_percent": 80
            }
        )
        
        self.configurations[environment] = config
        return config
    
    def optimize_configuration(self, test_results: List[TestExecutionResult]) -> TestConfiguration:
        """Optimize configuration based on test results"""
        logger.info("Optimizing test configuration based on results")
        
        # Analyze execution times
        avg_execution_time = sum(r.execution_time for r in test_results) / len(test_results)
        
        # Analyze resource usage
        avg_memory = sum(r.memory_usage for r in test_results) / len(test_results)
        avg_cpu = sum(r.cpu_usage for r in test_results) / len(test_results)
        
        # Optimize based on analysis
        optimized_config = TestConfiguration(
            test_patterns=["test_*.py"],
            execution_mode="parallel" if avg_execution_time > 1.0 else "sequential",
            parallel_workers=self._calculate_optimal_workers(avg_cpu),
            timeout_seconds=int(avg_execution_time * 2),
            retry_count=2 if any(r.status == "FAILED" for r in test_results) else 1,
            environment_vars={},
            resource_limits={
                "max_memory_mb": int(avg_memory * 1.5),
                "max_cpu_percent": 85
            }
        )
        
        return optimized_config
    
    def _calculate_optimal_workers(self, avg_cpu: float) -> int:
        """Calculate optimal number of parallel workers"""
        cpu_count = os.cpu_count() or 1
        
        if avg_cpu > 70:
            return max(1, cpu_count // 2)
        elif avg_cpu > 50:
            return max(1, int(cpu_count * 0.75))
        else:
            return cpu_count

class TestExecutionEngine:
    """Advanced test execution engine"""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.execution_results = []
        self.resource_monitor = ResourceMonitor()
    
    def execute_tests(self, test_files: List[str]) -> List[TestExecutionResult]:
        """Execute tests with advanced monitoring"""
        logger.info(f"Executing {len(test_files)} test files")
        
        if self.config.execution_mode == "parallel":
            return self._execute_parallel(test_files)
        else:
            return self._execute_sequential(test_files)
    
    def _execute_parallel(self, test_files: List[str]) -> List[TestExecutionResult]:
        """Execute tests in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_file = {
                executor.submit(self._execute_single_test, test_file): test_file
                for test_file in test_files
            }
            
            for future in as_completed(future_to_file):
                test_file = future_to_file[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Test execution failed for {test_file}: {e}")
                    results.append(TestExecutionResult(
                        test_name=test_file,
                        status="ERROR",
                        execution_time=0.0,
                        memory_usage=0.0,
                        cpu_usage=0.0,
                        output="",
                        error_message=str(e),
                        coverage_data={}
                    ))
        
        return results
    
    def _execute_sequential(self, test_files: List[str]) -> List[TestExecutionResult]:
        """Execute tests sequentially"""
        results = []
        
        for test_file in test_files:
            result = self._execute_single_test(test_file)
            results.append(result)
        
        return results
    
    def _execute_single_test(self, test_file: str) -> TestExecutionResult:
        """Execute a single test file"""
        logger.info(f"Executing test: {test_file}")
        
        start_time = time.time()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Execute test
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v"],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env={**os.environ, **self.config.environment_vars}
            )
            
            execution_time = time.time() - start_time
            
            # Stop resource monitoring
            resource_stats = self.resource_monitor.stop_monitoring()
            
            return TestExecutionResult(
                test_name=test_file,
                status="PASSED" if result.returncode == 0 else "FAILED",
                execution_time=execution_time,
                memory_usage=resource_stats.get("memory_usage", 0.0),
                cpu_usage=resource_stats.get("cpu_usage", 0.0),
                output=result.stdout,
                error_message=result.stderr if result.returncode != 0 else None,
                coverage_data={}
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            resource_stats = self.resource_monitor.stop_monitoring()
            
            return TestExecutionResult(
                test_name=test_file,
                status="TIMEOUT",
                execution_time=execution_time,
                memory_usage=resource_stats.get("memory_usage", 0.0),
                cpu_usage=resource_stats.get("cpu_usage", 0.0),
                output="",
                error_message=f"Test timed out after {self.config.timeout_seconds} seconds",
                coverage_data={}
            )
        except Exception as e:
            execution_time = time.time() - start_time
            resource_stats = self.resource_monitor.stop_monitoring()
            
            return TestExecutionResult(
                test_name=test_file,
                status="ERROR",
                execution_time=execution_time,
                memory_usage=resource_stats.get("memory_usage", 0.0),
                cpu_usage=resource_stats.get("cpu_usage", 0.0),
                output="",
                error_message=str(e),
                coverage_data={}
            )

class ResourceMonitor:
    """Monitor system resources during test execution"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.resource_data = []
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.resource_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop resource monitoring and return stats"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.resource_data:
            return {"memory_usage": 0.0, "cpu_usage": 0.0}
        
        # Calculate averages
        avg_memory = sum(d["memory"] for d in self.resource_data) / len(self.resource_data)
        avg_cpu = sum(d["cpu"] for d in self.resource_data) / len(self.resource_data)
        
        return {
            "memory_usage": avg_memory,
            "cpu_usage": avg_cpu,
            "peak_memory": max(d["memory"] for d in self.resource_data),
            "peak_cpu": max(d["cpu"] for d in self.resource_data)
        }
    
    def _monitor_resources(self):
        """Monitor resources in background thread"""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                self.resource_data.append({
                    "memory": memory_info.rss / 1024 / 1024,  # MB
                    "cpu": cpu_percent,
                    "timestamp": time.time()
                })
                
                time.sleep(0.1)  # Monitor every 100ms
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break

class TestResultAnalyzer:
    """Analyze test results and generate insights"""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_results(self, results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Analyze test execution results"""
        logger.info(f"Analyzing {len(results)} test results")
        
        analysis = {
            "summary": self._generate_summary(results),
            "performance_analysis": self._analyze_performance(results),
            "failure_analysis": self._analyze_failures(results),
            "resource_analysis": self._analyze_resources(results),
            "recommendations": self._generate_recommendations(results)
        }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _generate_summary(self, results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Generate test execution summary"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == "PASSED")
        failed_tests = sum(1 for r in results if r.status == "FAILED")
        error_tests = sum(1 for r in results if r.status == "ERROR")
        timeout_tests = sum(1 for r in results if r.status == "TIMEOUT")
        
        total_time = sum(r.execution_time for r in results)
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "timeouts": timeout_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": avg_time
        }
    
    def _analyze_performance(self, results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Analyze test performance"""
        execution_times = [r.execution_time for r in results]
        
        return {
            "fastest_test": min(execution_times) if execution_times else 0,
            "slowest_test": max(execution_times) if execution_times else 0,
            "median_time": sorted(execution_times)[len(execution_times)//2] if execution_times else 0,
            "performance_distribution": {
                "fast": sum(1 for t in execution_times if t < 0.1),
                "medium": sum(1 for t in execution_times if 0.1 <= t < 1.0),
                "slow": sum(1 for t in execution_times if t >= 1.0)
            }
        }
    
    def _analyze_failures(self, results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Analyze test failures"""
        failures = [r for r in results if r.status in ["FAILED", "ERROR", "TIMEOUT"]]
        
        failure_patterns = {}
        for failure in failures:
            error_type = failure.status
            if error_type not in failure_patterns:
                failure_patterns[error_type] = []
            failure_patterns[error_type].append(failure.test_name)
        
        return {
            "total_failures": len(failures),
            "failure_patterns": failure_patterns,
            "common_error_messages": self._extract_common_errors(failures)
        }
    
    def _analyze_resources(self, results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Analyze resource usage"""
        memory_usage = [r.memory_usage for r in results if r.memory_usage > 0]
        cpu_usage = [r.cpu_usage for r in results if r.cpu_usage > 0]
        
        return {
            "average_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "peak_memory_usage": max(memory_usage) if memory_usage else 0,
            "average_cpu_usage": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
            "peak_cpu_usage": max(cpu_usage) if cpu_usage else 0
        }
    
    def _generate_recommendations(self, results: List[TestExecutionResult]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Check for slow tests
        slow_tests = [r for r in results if r.execution_time > 5.0]
        if slow_tests:
            recommendations.append(f"Consider optimizing {len(slow_tests)} slow tests (>5s)")
        
        # Check for high memory usage
        high_memory_tests = [r for r in results if r.memory_usage > 500]
        if high_memory_tests:
            recommendations.append(f"Review memory usage in {len(high_memory_tests)} tests (>500MB)")
        
        # Check for failures
        failures = [r for r in results if r.status != "PASSED"]
        if failures:
            recommendations.append(f"Investigate {len(failures)} failed tests")
        
        return recommendations
    
    def _extract_common_errors(self, failures: List[TestExecutionResult]) -> List[str]:
        """Extract common error patterns"""
        error_messages = [f.error_message for f in failures if f.error_message]
        
        # Simple error pattern extraction
        common_patterns = []
        for error in error_messages:
            if "AssertionError" in error:
                common_patterns.append("AssertionError")
            elif "TimeoutError" in error:
                common_patterns.append("TimeoutError")
            elif "ImportError" in error:
                common_patterns.append("ImportError")
        
        return list(set(common_patterns))

class TestAutomationTestGenerator(unittest.TestCase):
    """Test cases for Test Automation Framework"""
    
    def setUp(self):
        self.test_dir = Path(__file__).parent
        self.discovery = AutomatedTestDiscovery(str(self.test_dir))
        self.config_manager = DynamicTestConfiguration()
        self.analyzer = TestResultAnalyzer()
    
    def test_test_discovery(self):
        """Test automated test discovery"""
        discovered = self.discovery.discover_tests()
        
        self.assertIsInstance(discovered, dict)
        self.assertIn("unit_tests", discovered)
        self.assertIn("integration_tests", discovered)
        self.assertIn("performance_tests", discovered)
        self.assertIn("other_tests", discovered)
    
    def test_test_classification(self):
        """Test test file classification"""
        unit_test = self.test_dir / "unit" / "test_example.py"
        integration_test = self.test_dir / "integration" / "test_example.py"
        
        unit_type = self.discovery._classify_test(unit_test)
        integration_type = self.discovery._classify_test(integration_test)
        
        self.assertEqual(unit_type, "unit_tests")
        self.assertEqual(integration_type, "integration_tests")
    
    def test_configuration_creation(self):
        """Test configuration creation"""
        config = self.config_manager.create_configuration("development")
        
        self.assertIsInstance(config, TestConfiguration)
        self.assertIn("test_patterns", config.__dict__)
        self.assertIn("execution_mode", config.__dict__)
        self.assertIn("parallel_workers", config.__dict__)
        self.assertIn("timeout_seconds", config.__dict__)
    
    def test_configuration_optimization(self):
        """Test configuration optimization"""
        # Create mock results
        mock_results = [
            TestExecutionResult("test1", "PASSED", 1.0, 100.0, 50.0, "", None, {}),
            TestExecutionResult("test2", "FAILED", 2.0, 200.0, 70.0, "", "Error", {}),
            TestExecutionResult("test3", "PASSED", 0.5, 50.0, 30.0, "", None, {})
        ]
        
        optimized_config = self.config_manager.optimize_configuration(mock_results)
        
        self.assertIsInstance(optimized_config, TestConfiguration)
        self.assertGreater(optimized_config.timeout_seconds, 0)
        self.assertGreater(optimized_config.parallel_workers, 0)
    
    def test_result_analysis(self):
        """Test result analysis"""
        mock_results = [
            TestExecutionResult("test1", "PASSED", 1.0, 100.0, 50.0, "", None, {}),
            TestExecutionResult("test2", "FAILED", 2.0, 200.0, 70.0, "", "Error", {}),
            TestExecutionResult("test3", "PASSED", 0.5, 50.0, 30.0, "", None, {})
        ]
        
        analysis = self.analyzer.analyze_results(mock_results)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("summary", analysis)
        self.assertIn("performance_analysis", analysis)
        self.assertIn("failure_analysis", analysis)
        self.assertIn("resource_analysis", analysis)
        self.assertIn("recommendations", analysis)
    
    def test_summary_generation(self):
        """Test summary generation"""
        mock_results = [
            TestExecutionResult("test1", "PASSED", 1.0, 100.0, 50.0, "", None, {}),
            TestExecutionResult("test2", "FAILED", 2.0, 200.0, 70.0, "", "Error", {}),
            TestExecutionResult("test3", "PASSED", 0.5, 50.0, 30.0, "", None, {})
        ]
        
        summary = self.analyzer._generate_summary(mock_results)
        
        self.assertEqual(summary["total_tests"], 3)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertAlmostEqual(summary["success_rate"], 2/3, places=2)

def run_automation_tests():
    """Run all test automation tests"""
    logger.info("Running test automation tests")
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAutomationTestGenerator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log results
    logger.info(f"Automation tests completed: {result.testsRun} tests run")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    return result

if __name__ == "__main__":
    run_automation_tests()



