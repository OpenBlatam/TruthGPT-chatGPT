#!/usr/bin/env python3
"""
Enhanced Test Runner for Optimization Core
Advanced test runner with improved features, better reporting, and comprehensive coverage
"""

import unittest
import sys
import os
import time
import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import threading
import queue
import concurrent.futures
from datetime import datetime
import traceback
import psutil
import gc

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all test modules
from test_production_config import (
    TestProductionConfig, TestProductionConfigFileLoading, TestProductionConfigEnvironment,
    TestProductionConfigValidation, TestProductionConfigHotReload, TestProductionConfigThreadSafety
)
from test_production_optimizer import (
    TestProductionOptimizer, TestProductionOptimizerConfig, TestProductionOptimizerPerformance,
    TestProductionOptimizerCircuitBreaker, TestProductionOptimizerOptimization,
    TestProductionOptimizerCaching, TestProductionOptimizerPersistence
)
from test_optimization_core import (
    TestOptimizationCore, TestOptimizationCoreComponents, TestOptimizationCoreIntegration,
    TestOptimizationCorePerformance, TestOptimizationCoreEdgeCases
)
from test_integration import (
    TestIntegration, TestIntegrationEndToEnd, TestIntegrationConfiguration,
    TestIntegrationPerformance, TestIntegrationConcurrency, TestIntegrationErrorHandling,
    TestIntegrationPersistence
)
from test_performance import (
    TestPerformance, TestPerformanceBenchmarks, TestPerformanceScalability,
    TestPerformanceMemory, TestPerformanceSystemResources
)
from test_advanced_components import (
    TestUltraEnhancedOptimizationCore, TestMegaEnhancedOptimizationCore,
    TestSupremeOptimizationCore, TestTranscendentOptimizationCore,
    TestHybridOptimizationCore, TestEnhancedParameterOptimizer,
    TestRLPruning, TestOlympiadBenchmarks, TestAdvancedIntegration
)
from test_edge_cases import (
    TestEdgeCases, TestStressScenarios, TestBoundaryConditions,
    TestErrorRecovery, TestResourceLimits
)
from test_security import (
    TestInputValidation, TestDataProtection, TestAccessControl,
    TestInjectionAttacks, TestCryptographicSecurity, TestNetworkSecurity,
    TestLoggingSecurity
)
from test_compatibility import (
    TestPlatformCompatibility, TestPythonVersionCompatibility,
    TestPyTorchCompatibility, TestDependencyCompatibility,
    TestHardwareCompatibility, TestVersionCompatibility,
    TestBackwardCompatibility, TestForwardCompatibility
)
from test_ultra_advanced_optimizer import (
    TestQuantumState, TestNeuralArchitecture, TestHyperparameterSpace,
    TestQuantumOptimizer, TestNeuralArchitectureSearch, TestHyperparameterOptimizer,
    TestUltraAdvancedOptimizer, TestUltraAdvancedOptimizerIntegration,
    TestUltraAdvancedOptimizerPerformance
)
from test_advanced_optimizations import (
    TestOptimizationTechnique, TestOptimizationMetrics, TestNeuralArchitectureSearch,
    TestQuantumInspiredOptimizer, TestEvolutionaryOptimizer, TestMetaLearningOptimizer,
    TestAdvancedOptimizationEngine, TestFactoryFunctions, TestAdvancedOptimizationContext,
    TestAdvancedOptimizationsIntegration, TestAdvancedOptimizationsPerformance
)
from test_quantum_optimization import (
    TestQuantumStateAdvanced, TestQuantumOptimizerAdvanced, TestQuantumInspiredOptimizerAdvanced,
    TestQuantumOptimizationIntegration, TestQuantumOptimizationEdgeCases
)

class EnhancedTestResult(unittest.TestResult):
    """Enhanced test result with detailed reporting and metrics."""
    
    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super().__init__(stream, descriptions, verbosity)
        self.start_time = None
        self.end_time = None
        self.test_times = {}
        self.memory_usage = {}
        self.cpu_usage = {}
        self.detailed_results = []
        self.performance_metrics = {}
        
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
        
        # Record initial memory and CPU usage
        process = psutil.Process()
        self.memory_usage[test] = process.memory_info().rss / 1024 / 1024  # MB
        self.cpu_usage[test] = process.cpu_percent()
        
    def stopTest(self, test):
        super().stopTest(test)
        self.end_time = time.time()
        
        # Record test execution time
        if self.start_time:
            self.test_times[test] = self.end_time - self.start_time
        
        # Record final memory and CPU usage
        process = psutil.Process()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        if test in self.memory_usage:
            self.memory_usage[test] = final_memory - self.memory_usage[test]
        else:
            self.memory_usage[test] = final_memory
            
        if test in self.cpu_usage:
            self.cpu_usage[test] = final_cpu - self.cpu_usage[test]
        else:
            self.cpu_usage[test] = final_cpu
        
        # Store detailed result
        result_info = {
            'test': str(test),
            'time': self.test_times.get(test, 0),
            'memory_mb': self.memory_usage.get(test, 0),
            'cpu_percent': self.cpu_usage.get(test, 0),
            'status': 'PASS' if test not in self.failures and test not in self.errors else 'FAIL'
        }
        self.detailed_results.append(result_info)
        
    def addSuccess(self, test):
        super().addSuccess(test)
        
    def addError(self, test, err):
        super().addError(test, err)
        
    def addFailure(self, test, err):
        super().addFailure(test, err)
        
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        
    def get_summary(self):
        """Get comprehensive test summary."""
        total_time = sum(self.test_times.values())
        total_memory = sum(self.memory_usage.values())
        avg_cpu = sum(self.cpu_usage.values()) / len(self.cpu_usage) if self.cpu_usage else 0
        
        return {
            'total_tests': self.testsRun,
            'passed': self.testsRun - len(self.failures) - len(self.errors),
            'failed': len(self.failures),
            'errors': len(self.errors),
            'skipped': len(self.skipped),
            'success_rate': (self.testsRun - len(self.failures) - len(self.errors)) / self.testsRun * 100 if self.testsRun > 0 else 0,
            'total_time': total_time,
            'total_memory_mb': total_memory,
            'avg_cpu_percent': avg_cpu,
            'detailed_results': self.detailed_results
        }

class EnhancedTestRunner:
    """Enhanced test runner with advanced features."""
    
    def __init__(self, verbosity=2, parallel=False, max_workers=None, 
                 output_file=None, performance_mode=False, coverage_mode=False):
        self.verbosity = verbosity
        self.parallel = parallel
        self.max_workers = max_workers or os.cpu_count()
        self.output_file = output_file
        self.performance_mode = performance_mode
        self.coverage_mode = coverage_mode
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging for the test runner."""
        logger = logging.getLogger('EnhancedTestRunner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_test_suites(self):
        """Get all test suites organized by category."""
        return {
            "Production Configuration Tests": [
                TestProductionConfig, TestProductionConfigFileLoading, 
                TestProductionConfigEnvironment, TestProductionConfigValidation,
                TestProductionConfigHotReload, TestProductionConfigThreadSafety
            ],
            "Production Optimizer Tests": [
                TestProductionOptimizer, TestProductionOptimizerConfig,
                TestProductionOptimizerPerformance, TestProductionOptimizerCircuitBreaker,
                TestProductionOptimizerOptimization, TestProductionOptimizerCaching,
                TestProductionOptimizerPersistence
            ],
            "Optimization Core Tests": [
                TestOptimizationCore, TestOptimizationCoreComponents,
                TestOptimizationCoreIntegration, TestOptimizationCorePerformance,
                TestOptimizationCoreEdgeCases
            ],
            "Integration Tests": [
                TestIntegration, TestIntegrationEndToEnd, TestIntegrationConfiguration,
                TestIntegrationPerformance, TestIntegrationConcurrency,
                TestIntegrationErrorHandling, TestIntegrationPersistence
            ],
            "Performance Tests": [
                TestPerformance, TestPerformanceBenchmarks, TestPerformanceScalability,
                TestPerformanceMemory, TestPerformanceSystemResources
            ],
            "Advanced Component Tests": [
                TestUltraEnhancedOptimizationCore, TestMegaEnhancedOptimizationCore,
                TestSupremeOptimizationCore, TestTranscendentOptimizationCore,
                TestHybridOptimizationCore, TestEnhancedParameterOptimizer,
                TestRLPruning, TestOlympiadBenchmarks, TestAdvancedIntegration
            ],
            "Edge Cases and Stress Tests": [
                TestEdgeCases, TestStressScenarios, TestBoundaryConditions,
                TestErrorRecovery, TestResourceLimits
            ],
            "Security Tests": [
                TestInputValidation, TestDataProtection, TestAccessControl,
                TestInjectionAttacks, TestCryptographicSecurity, TestNetworkSecurity,
                TestLoggingSecurity
            ],
            "Compatibility Tests": [
                TestPlatformCompatibility, TestPythonVersionCompatibility,
                TestPyTorchCompatibility, TestDependencyCompatibility,
                TestHardwareCompatibility, TestVersionCompatibility,
                TestBackwardCompatibility, TestForwardCompatibility
            ],
            "Ultra Advanced Optimizer Tests": [
                TestQuantumState, TestNeuralArchitecture, TestHyperparameterSpace,
                TestQuantumOptimizer, TestNeuralArchitectureSearch, TestHyperparameterOptimizer,
                TestUltraAdvancedOptimizer, TestUltraAdvancedOptimizerIntegration,
                TestUltraAdvancedOptimizerPerformance
            ],
            "Advanced Optimizations Tests": [
                TestOptimizationTechnique, TestOptimizationMetrics, TestNeuralArchitectureSearch,
                TestQuantumInspiredOptimizer, TestEvolutionaryOptimizer, TestMetaLearningOptimizer,
                TestAdvancedOptimizationEngine, TestFactoryFunctions, TestAdvancedOptimizationContext,
                TestAdvancedOptimizationsIntegration, TestAdvancedOptimizationsPerformance
            ],
            "Quantum Optimization Tests": [
                TestQuantumStateAdvanced, TestQuantumOptimizerAdvanced, TestQuantumInspiredOptimizerAdvanced,
                TestQuantumOptimizationIntegration, TestQuantumOptimizationEdgeCases
            ]
        }
    
    def _run_test_suite(self, test_class, category_name):
        """Run a single test suite."""
        try:
            self.logger.info(f"Running {category_name}: {test_class.__name__}")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Create enhanced result
            result = EnhancedTestResult()
            
            # Run tests
            start_time = time.time()
            suite.run(result)
            end_time = time.time()
            
            return {
                'category': category_name,
                'test_class': test_class.__name__,
                'result': result,
                'execution_time': end_time - start_time,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error running {test_class.__name__}: {e}")
            return {
                'category': category_name,
                'test_class': test_class.__name__,
                'result': None,
                'execution_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def _run_tests_parallel(self, test_suites):
        """Run tests in parallel."""
        self.logger.info(f"Running tests in parallel with {self.max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for category_name, test_classes in test_suites.items():
                for test_class in test_classes:
                    future = executor.submit(self._run_test_suite, test_class, category_name)
                    futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel execution: {e}")
                    results.append({
                        'category': 'Unknown',
                        'test_class': 'Unknown',
                        'result': None,
                        'execution_time': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            return results
    
    def _run_tests_sequential(self, test_suites):
        """Run tests sequentially."""
        self.logger.info("Running tests sequentially")
        
        results = []
        for category_name, test_classes in test_suites.items():
            for test_class in test_classes:
                result = self._run_test_suite(test_class, category_name)
                results.append(result)
        
        return results
    
    def _generate_report(self, results):
        """Generate comprehensive test report."""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_skipped = 0
        total_time = 0
        total_memory = 0
        category_stats = {}
        
        for result in results:
            if result['success'] and result['result']:
                test_result = result['result']
                total_tests += test_result.testsRun
                total_passed += test_result.testsRun - len(test_result.failures) - len(test_result.errors)
                total_failed += len(test_result.failures)
                total_errors += len(test_result.errors)
                total_skipped += len(test_result.skipped)
                total_time += result['execution_time']
                
                # Category statistics
                category = result['category']
                if category not in category_stats:
                    category_stats[category] = {
                        'tests': 0, 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0
                    }
                
                category_stats[category]['tests'] += test_result.testsRun
                category_stats[category]['passed'] += test_result.testsRun - len(test_result.failures) - len(test_result.errors)
                category_stats[category]['failed'] += len(test_result.failures)
                category_stats[category]['errors'] += len(test_result.errors)
                category_stats[category]['skipped'] += len(test_result.skipped)
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'errors': total_errors,
                'skipped': total_skipped,
                'success_rate': success_rate,
                'total_time': total_time,
                'total_memory_mb': total_memory
            },
            'category_stats': category_stats,
            'detailed_results': results,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': os.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        return report
    
    def _print_report(self, report):
        """Print formatted test report."""
        print("\n" + "="*80)
        print("ENHANCED OPTIMIZATION CORE TEST REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Errors: {summary['errors']}")
        print(f"  Skipped: {summary['skipped']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Total Time: {summary['total_time']:.2f}s")
        print(f"  Total Memory: {summary['total_memory_mb']:.2f}MB")
        
        print(f"\nCATEGORY BREAKDOWN:")
        for category, stats in report['category_stats'].items():
            category_success_rate = (stats['passed'] / stats['tests'] * 100) if stats['tests'] > 0 else 0
            print(f"  {category}:")
            print(f"    Tests: {stats['tests']}, Passed: {stats['passed']}, "
                  f"Failed: {stats['failed']}, Errors: {stats['errors']}, "
                  f"Success Rate: {category_success_rate:.1f}%")
        
        print(f"\nSYSTEM INFORMATION:")
        system_info = report['system_info']
        print(f"  Python Version: {system_info['python_version']}")
        print(f"  Platform: {system_info['platform']}")
        print(f"  CPU Count: {system_info['cpu_count']}")
        print(f"  Memory: {system_info['memory_gb']:.1f}GB")
        
        # Print failures and errors
        if summary['failed'] > 0 or summary['errors'] > 0:
            print(f"\nFAILURES AND ERRORS:")
            for result in report['detailed_results']:
                if result['success'] and result['result']:
                    test_result = result['result']
                    if test_result.failures:
                        print(f"\n  Failures in {result['test_class']}:")
                        for test, traceback in test_result.failures:
                            print(f"    - {test}: {traceback}")
                    
                    if test_result.errors:
                        print(f"\n  Errors in {result['test_class']}:")
                        for test, traceback in test_result.errors:
                            print(f"    - {test}: {traceback}")
        
        print("\n" + "="*80)
    
    def _save_report(self, report):
        """Save test report to file."""
        if self.output_file:
            try:
                with open(self.output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                self.logger.info(f"Test report saved to {self.output_file}")
            except Exception as e:
                self.logger.error(f"Error saving report: {e}")
    
    def run_tests(self, categories=None, test_classes=None):
        """Run tests with specified options."""
        self.logger.info("Starting enhanced test runner")
        
        # Get test suites
        all_test_suites = self._get_test_suites()
        
        # Filter test suites if specified
        if categories:
            test_suites = {k: v for k, v in all_test_suites.items() if k in categories}
        else:
            test_suites = all_test_suites
        
        if test_classes:
            filtered_suites = {}
            for category, classes in test_suites.items():
                filtered_classes = [c for c in classes if c.__name__ in test_classes]
                if filtered_classes:
                    filtered_suites[category] = filtered_classes
            test_suites = filtered_suites
        
        # Run tests
        start_time = time.time()
        
        if self.parallel:
            results = self._run_tests_parallel(test_suites)
        else:
            results = self._run_tests_sequential(test_suites)
        
        end_time = time.time()
        
        # Generate report
        report = self._generate_report(results)
        report['summary']['total_execution_time'] = end_time - start_time
        
        # Print report
        self._print_report(report)
        
        # Save report
        self._save_report(report)
        
        # Return success status
        return report['summary']['success_rate'] >= 80.0

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Enhanced Test Runner for Optimization Core')
    parser.add_argument('--verbosity', type=int, default=2, help='Test verbosity level')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--output', type=str, help='Output file for test report')
    parser.add_argument('--categories', nargs='+', help='Test categories to run')
    parser.add_argument('--test-classes', nargs='+', help='Specific test classes to run')
    parser.add_argument('--performance', action='store_true', help='Enable performance mode')
    parser.add_argument('--coverage', action='store_true', help='Enable coverage mode')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = EnhancedTestRunner(
        verbosity=args.verbosity,
        parallel=args.parallel,
        max_workers=args.workers,
        output_file=args.output,
        performance_mode=args.performance,
        coverage_mode=args.coverage
    )
    
    # Run tests
    success = runner.run_tests(
        categories=args.categories,
        test_classes=args.test_classes
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

