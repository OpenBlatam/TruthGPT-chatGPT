"""
Test Runner for Optimization Core
Comprehensive test runner for all optimization core tests
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import all test modules
from test_production_config import (
    TestProductionConfig, TestFactoryFunctions, TestValidationRules,
    TestConfigValidationRule, TestConfigMetadata, TestEnvironmentEnum,
    TestConfigSourceEnum
)
from test_production_optimizer import (
    TestProductionOptimizationConfig, TestPerformanceMetrics, TestCircuitBreaker,
    TestProductionOptimizer, TestFactoryFunctions as TestOptimizerFactoryFunctions,
    TestOptimizationStrategies, TestIntegration as TestOptimizerIntegration
)
from test_optimization_core import (
    TestCUDAOptimizations, TestTritonOptimizations, TestEnhancedGRPO,
    TestMCTSOptimization, TestParallelTraining, TestExperienceBuffer,
    TestAdvancedLosses, TestRewardFunctions, TestAdvancedNormalization,
    TestPositionalEncodings, TestEnhancedMLP, TestAdvancedKernelFusion,
    TestAdvancedQuantization, TestMemoryPooling, TestEnhancedCUDAKernels,
    TestOptimizationRegistry, TestAdvancedOptimizationRegistry,
    TestMemoryOptimizations, TestComputationalOptimizations,
    TestOptimizationProfiles, TestIntegration as TestCoreIntegration
)
from test_integration import (
    TestCompleteOptimizationPipeline, TestConfigurationIntegration,
    TestPerformanceIntegration, TestConcurrencyIntegration,
    TestErrorHandlingIntegration, TestPersistenceIntegration
)
from test_performance import (
    TestPerformanceBenchmarks, TestScalabilityBenchmarks, TestMemoryBenchmarks,
    TestOptimizationBenchmarks, TestConcurrencyBenchmarks, TestSystemResourceBenchmarks
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


class TestResultCollector:
    """Collects and summarizes test results."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing."""
        self.end_time = time.time()
    
    def add_result(self, test_suite_name, result):
        """Add test result."""
        self.results[test_suite_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': getattr(result, 'skipped', 0),
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
        }
    
    def get_summary(self):
        """Get overall summary."""
        total_tests = sum(r['tests_run'] for r in self.results.values())
        total_failures = sum(r['failures'] for r in self.results.values())
        total_errors = sum(r['errors'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())
        
        overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'overall_success_rate': overall_success_rate,
            'duration': self.end_time - self.start_time if self.end_time and self.start_time else 0
        }


def run_test_suite(test_suite_name, test_classes, verbosity=2):
    """Run a specific test suite."""
    print(f"\n{'='*60}")
    print(f"Running {test_suite_name}")
    print(f"{'='*60}")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)
    
    return result


def main():
    """Main test runner function."""
    print("🚀 Optimization Core Test Suite")
    print("=" * 60)
    
    # Initialize result collector
    collector = TestResultCollector()
    collector.start()
    
    # Define test suites
    test_suites = {
        "Production Configuration Tests": [
            TestProductionConfig, TestFactoryFunctions, TestValidationRules,
            TestConfigValidationRule, TestConfigMetadata, TestEnvironmentEnum,
            TestConfigSourceEnum
        ],
        "Production Optimizer Tests": [
            TestProductionOptimizationConfig, TestPerformanceMetrics, TestCircuitBreaker,
            TestProductionOptimizer, TestOptimizerFactoryFunctions,
            TestOptimizationStrategies, TestOptimizerIntegration
        ],
        "Optimization Core Tests": [
            TestCUDAOptimizations, TestTritonOptimizations, TestEnhancedGRPO,
            TestMCTSOptimization, TestParallelTraining, TestExperienceBuffer,
            TestAdvancedLosses, TestRewardFunctions, TestAdvancedNormalization,
            TestPositionalEncodings, TestEnhancedMLP, TestAdvancedKernelFusion,
            TestAdvancedQuantization, TestMemoryPooling, TestEnhancedCUDAKernels,
            TestOptimizationRegistry, TestAdvancedOptimizationRegistry,
            TestMemoryOptimizations, TestComputationalOptimizations,
            TestOptimizationProfiles, TestCoreIntegration
        ],
        "Integration Tests": [
            TestCompleteOptimizationPipeline, TestConfigurationIntegration,
            TestPerformanceIntegration, TestConcurrencyIntegration,
            TestErrorHandlingIntegration, TestPersistenceIntegration
        ],
        "Performance Tests": [
            TestPerformanceBenchmarks, TestScalabilityBenchmarks, TestMemoryBenchmarks,
            TestOptimizationBenchmarks, TestConcurrencyBenchmarks, TestSystemResourceBenchmarks
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
        ]
    }
    
    # Run each test suite
    for suite_name, test_classes in test_suites.items():
        try:
            result = run_test_suite(suite_name, test_classes)
            collector.add_result(suite_name, result)
        except Exception as e:
            print(f"❌ Error running {suite_name}: {e}")
            collector.add_result(suite_name, type('MockResult', (), {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0
            })())
    
    # Stop timing
    collector.stop()
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("📊 OVERALL TEST SUMMARY")
    print(f"{'='*60}")
    
    summary = collector.get_summary()
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Failures: {summary['total_failures']}")
    print(f"Errors: {summary['total_errors']}")
    print(f"Skipped: {summary['total_skipped']}")
    print(f"Success Rate: {summary['overall_success_rate']:.1f}%")
    print(f"Duration: {summary['duration']:.2f}s")
    
    # Print detailed results
    print(f"\n{'='*60}")
    print("📋 DETAILED RESULTS")
    print(f"{'='*60}")
    
    for suite_name, result in collector.results.items():
        print(f"\n{suite_name}:")
        print(f"  Tests: {result['tests_run']}")
        print(f"  Failures: {result['failures']}")
        print(f"  Errors: {result['errors']}")
        print(f"  Skipped: {result['skipped']}")
        print(f"  Success Rate: {result['success_rate']:.1f}%")
    
    # Print recommendations
    print(f"\n{'='*60}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if summary['total_failures'] > 0:
        print("❌ Some tests failed. Please review the failures above.")
    
    if summary['total_errors'] > 0:
        print("❌ Some tests had errors. Please check the error messages above.")
    
    if summary['overall_success_rate'] >= 95:
        print("✅ Excellent test coverage! All critical functionality is working.")
    elif summary['overall_success_rate'] >= 90:
        print("✅ Good test coverage. Minor issues to address.")
    elif summary['overall_success_rate'] >= 80:
        print("⚠️  Moderate test coverage. Some issues need attention.")
    else:
        print("❌ Low test coverage. Significant issues need to be resolved.")
    
    # Return exit code
    if summary['total_failures'] > 0 or summary['total_errors'] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

