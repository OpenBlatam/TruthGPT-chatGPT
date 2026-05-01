"""
Comprehensive test runner for TruthGPT optimization core
Runs all tests with detailed reporting and performance analysis
"""

import unittest
import sys
import time
import os
from pathlib import Path
import argparse
import json
from typing import Dict, Any, List

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

try:
    from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, MemoryTracker
except ImportError:
    # If import fails, create minimal versions
    import psutil
    class PerformanceProfiler:
        def __init__(self):
            self.profiles = []
            self.current_profile = None
        def start_profile(self, name): pass
        def end_profile(self): return {}
        def get_profile_summary(self): return {}
    
    class MemoryTracker:
        def __init__(self):
            self.snapshots = []
            self.peak_memory = 0
        def take_snapshot(self, label): pass
        def get_memory_summary(self): return {}

class TruthGPTTestRunner:
    """Comprehensive test runner for TruthGPT optimization core"""
    
    def __init__(self, verbose: bool = True, parallel: bool = False, 
                 coverage: bool = True, performance: bool = True):
        self.verbose = verbose
        self.parallel = parallel
        self.coverage = coverage
        self.performance = performance
        
        self.profiler = PerformanceProfiler()
        self.memory_tracker = MemoryTracker()
        self.test_results = {}
        
    def discover_tests(self) -> List[str]:
        """Discover all test files"""
        test_dir = Path(__file__).parent
        test_files = []
        
        # Discover unit tests
        unit_dir = test_dir / "unit"
        if unit_dir.exists():
            for test_file in unit_dir.glob("test_*.py"):
                test_files.append(str(test_file))
        
        # Discover integration tests
        integration_dir = test_dir / "integration"
        if integration_dir.exists():
            for test_file in integration_dir.glob("test_*.py"):
                test_files.append(str(test_file))
        
        # Discover performance tests
        performance_dir = test_dir / "performance"
        if performance_dir.exists():
            for test_file in performance_dir.glob("test_*.py"):
                test_files.append(str(test_file))
        
        return test_files
    
    def run_test_suite(self, test_files: List[str]) -> Dict[str, Any]:
        """Run test suite and collect results"""
        print("🧪 Running TruthGPT Optimization Core Tests")
        print("=" * 60)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_time = 0
        
        suite_results = {}
        
        for test_file in test_files:
            print(f"\n📁 Running tests from: {Path(test_file).name}")
            
            # Start profiling
            self.profiler.start_profile(f"test_file_{Path(test_file).stem}")
            self.memory_tracker.take_snapshot(f"before_{Path(test_file).stem}")
            
            # Run tests
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(test_file.replace('.py', '').replace('/', '.'))
            
            runner = unittest.TextTestRunner(
                verbosity=2 if self.verbose else 1,
                stream=sys.stdout,
                descriptions=True,
                failfast=False
            )
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            # End profiling
            metrics = self.profiler.end_profile()
            self.memory_tracker.take_snapshot(f"after_{Path(test_file).stem}")
            
            # Collect results
            file_results = {
                'file': test_file,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'execution_time': end_time - start_time,
                'memory_used': metrics.get('memory_used', 0),
                'success': result.wasSuccessful(),
                'failure_details': [str(f[1]) for f in result.failures],
                'error_details': [str(e[1]) for e in result.errors]
            }
            
            suite_results[Path(test_file).name] = file_results
            
            # Update totals
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_time += end_time - start_time
            
            # Print file summary
            status = "✅ PASSED" if result.wasSuccessful() else "❌ FAILED"
            print(f"   {status} - {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        
        # Overall results
        self.test_results = {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_time': total_time,
            'success_rate': ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0,
            'suite_results': suite_results,
            'performance_metrics': self.profiler.get_profile_summary(),
            'memory_summary': self.memory_tracker.get_memory_summary()
        }
        
        return self.test_results
    
    def run_specific_tests(self, test_patterns: List[str]) -> Dict[str, Any]:
        """Run specific tests based on patterns"""
        print("🎯 Running specific tests")
        print("=" * 40)
        
        all_test_files = self.discover_tests()
        filtered_files = []
        
        for pattern in test_patterns:
            for test_file in all_test_files:
                if pattern.lower() in Path(test_file).name.lower():
                    filtered_files.append(test_file)
        
        if not filtered_files:
            print(f"❌ No test files found matching patterns: {test_patterns}")
            return {}
        
        print(f"📋 Found {len(filtered_files)} test files matching patterns")
        return self.run_test_suite(filtered_files)
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance-focused tests"""
        print("⚡ Running Performance Tests")
        print("=" * 40)
        
        performance_files = []
        test_dir = Path(__file__).parent
        
        # Find performance tests
        performance_dir = test_dir / "performance"
        if performance_dir.exists():
            for test_file in performance_dir.glob("test_*.py"):
                performance_files.append(str(test_file))
        
        if not performance_files:
            print("❌ No performance test files found")
            return {}
        
        return self.run_test_suite(performance_files)
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("🔗 Running Integration Tests")
        print("=" * 40)
        
        integration_files = []
        test_dir = Path(__file__).parent
        
        # Find integration tests
        integration_dir = test_dir / "integration"
        if integration_dir.exists():
            for test_file in integration_dir.glob("test_*.py"):
                integration_files.append(str(test_file))
        
        if not integration_files:
            print("❌ No integration test files found")
            return {}
        
        return self.run_test_suite(integration_files)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        if not results:
            return "❌ No test results to report"
        
        report = []
        report.append("📊 TruthGPT Optimization Core Test Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("📈 Test Summary:")
        report.append(f"  Total Tests: {results['total_tests']}")
        report.append(f"  Failures: {results['total_failures']}")
        report.append(f"  Errors: {results['total_errors']}")
        report.append(f"  Success Rate: {results['success_rate']:.1f}%")
        report.append(f"  Total Time: {results['total_time']:.2f}s")
        report.append("")
        
        # Performance metrics
        if 'performance_metrics' in results and results['performance_metrics']:
            perf = results['performance_metrics']
            report.append("⚡ Performance Metrics:")
            report.append(f"  Total Profiles: {perf.get('total_profiles', 0)}")
            report.append(f"  Total Execution Time: {perf.get('total_execution_time', 0):.2f}s")
            report.append(f"  Average Execution Time: {perf.get('average_execution_time', 0):.2f}s")
            report.append(f"  Total Memory Used: {perf.get('total_memory_used', 0):.2f}MB")
            report.append("")
        
        # Memory summary
        if 'memory_summary' in results and results['memory_summary']:
            mem = results['memory_summary']
            report.append("💾 Memory Summary:")
            report.append(f"  Snapshots Taken: {mem.get('snapshots_taken', 0)}")
            report.append(f"  Peak Memory: {mem.get('peak_memory', 0):.2f}MB")
            report.append(f"  Average Memory: {mem.get('average_memory', 0):.2f}MB")
            report.append("")
        
        # Individual test results
        if 'suite_results' in results:
            report.append("📋 Individual Test Results:")
            for file_name, file_results in results['suite_results'].items():
                status = "✅" if file_results['success'] else "❌"
                report.append(f"  {status} {file_name}: {file_results['tests_run']} tests, "
                            f"{file_results['failures']} failures, {file_results['errors']} errors, "
                            f"{file_results['execution_time']:.2f}s")
            report.append("")
        
        # Recommendations
        report.append("💡 Recommendations:")
        if results['total_failures'] > 0:
            report.append("  - Fix failing tests to improve reliability")
        if results['total_errors'] > 0:
            report.append("  - Address test errors to ensure proper functionality")
        if results['success_rate'] < 90:
            report.append("  - Improve test coverage and fix issues")
        if results['total_time'] > 60:
            report.append("  - Consider optimizing slow tests")
        report.append("")
        
        # Final status
        if results['success_rate'] >= 95:
            report.append("🎉 Excellent! All tests are passing with high success rate.")
        elif results['success_rate'] >= 80:
            report.append("✅ Good! Most tests are passing.")
        else:
            report.append("⚠️  Attention needed! Some tests are failing.")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = "test_results.json"):
        """Save test results to file"""
        results_file = Path(__file__).parent / filename
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Test results saved to: {results_file}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        test_files = self.discover_tests()
        
        if not test_files:
            print("❌ No test files found")
            return {}
        
        print(f"📁 Found {len(test_files)} test files")
        return self.run_test_suite(test_files)

def main():
    """Main function for running tests"""
    parser = argparse.ArgumentParser(description="TruthGPT Optimization Core Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--coverage", "-c", action="store_true", help="Enable coverage reporting")
    parser.add_argument("--performance", "-perf", action="store_true", help="Run performance tests only")
    parser.add_argument("--integration", "-int", action="store_true", help="Run integration tests only")
    parser.add_argument("--pattern", "-pat", nargs="+", help="Run tests matching pattern")
    parser.add_argument("--save-results", "-s", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TruthGPTTestRunner(
        verbose=args.verbose,
        parallel=args.parallel,
        coverage=args.coverage,
        performance=args.performance
    )
    
    # Run tests based on arguments
    if args.performance:
        results = runner.run_performance_tests()
    elif args.integration:
        results = runner.run_integration_tests()
    elif args.pattern:
        results = runner.run_specific_tests(args.pattern)
    else:
        results = runner.run_all_tests()
    
    # Generate and print report
    if results:
        report = runner.generate_report(results)
        print("\n" + report)
        
        # Save results if requested
        if args.save_results:
            runner.save_results(results)
    
    # Exit with appropriate code
    if results and results.get('total_failures', 0) == 0 and results.get('total_errors', 0) == 0:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the results above.")
        sys.exit(1)

if __name__ == "__main__":
    main()



