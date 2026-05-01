"""
Comprehensive test runner for TruthGPT optimization core
Runs all tests with detailed reporting and analysis
"""

import unittest
import sys
import time
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, MemoryTracker

class ComprehensiveTestRunner:
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
        self.performance_metrics = {}
        
    def discover_all_tests(self) -> Dict[str, List[str]]:
        """Discover all test files organized by category"""
        test_dir = Path(__file__).parent
        test_categories = {
            'unit': [],
            'integration': [],
            'performance': []
        }
        
        # Discover unit tests
        unit_dir = test_dir / "unit"
        if unit_dir.exists():
            for test_file in unit_dir.glob("test_*.py"):
                test_categories['unit'].append(str(test_file))
        
        # Discover integration tests
        integration_dir = test_dir / "integration"
        if integration_dir.exists():
            for test_file in integration_dir.glob("test_*.py"):
                test_categories['integration'].append(str(test_file))
        
        # Discover performance tests
        performance_dir = test_dir / "performance"
        if performance_dir.exists():
            for test_file in performance_dir.glob("test_*.py"):
                test_categories['performance'].append(str(test_file))
        
        return test_categories
    
    def run_test_category(self, category: str, test_files: List[str]) -> Dict[str, Any]:
        """Run tests for a specific category"""
        print(f"\n🧪 Running {category.title()} Tests")
        print("=" * 60)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_time = 0
        
        category_results = {}
        
        for test_file in test_files:
            print(f"\n📁 Running: {Path(test_file).name}")
            
            # Start profiling
            self.profiler.start_profile(f"{category}_{Path(test_file).stem}")
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
                'category': category,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'execution_time': end_time - start_time,
                'memory_used': metrics.get('memory_used', 0),
                'success': result.wasSuccessful(),
                'failure_details': [str(f[1]) for f in result.failures],
                'error_details': [str(e[1]) for e in result.errors]
            }
            
            category_results[Path(test_file).name] = file_results
            
            # Update totals
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_time += end_time - start_time
            
            # Print file summary
            status = "✅ PASSED" if result.wasSuccessful() else "❌ FAILED"
            print(f"   {status} - {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        
        # Category summary
        category_summary = {
            'category': category,
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_time': total_time,
            'success_rate': ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0,
            'file_results': category_results,
            'performance_metrics': self.profiler.get_profile_summary(),
            'memory_summary': self.memory_tracker.get_memory_summary()
        }
        
        return category_summary
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests comprehensively"""
        print("🚀 TruthGPT Optimization Core Comprehensive Test Suite")
        print("=" * 70)
        
        # Discover all tests
        test_categories = self.discover_all_tests()
        
        # Run tests by category
        category_results = {}
        overall_stats = {
            'total_tests': 0,
            'total_failures': 0,
            'total_errors': 0,
            'total_time': 0,
            'categories_passed': 0,
            'categories_failed': 0
        }
        
        for category, test_files in test_categories.items():
            if not test_files:
                print(f"⚠️  No {category} tests found")
                continue
                
            print(f"\n📋 Found {len(test_files)} {category} test files")
            
            # Run category tests
            category_result = self.run_test_category(category, test_files)
            category_results[category] = category_result
            
            # Update overall stats
            overall_stats['total_tests'] += category_result['total_tests']
            overall_stats['total_failures'] += category_result['total_failures']
            overall_stats['total_errors'] += category_result['total_errors']
            overall_stats['total_time'] += category_result['total_time']
            
            if category_result['success_rate'] >= 95:
                overall_stats['categories_passed'] += 1
            else:
                overall_stats['categories_failed'] += 1
        
        # Calculate overall success rate
        overall_stats['success_rate'] = (
            (overall_stats['total_tests'] - overall_stats['total_failures'] - overall_stats['total_errors']) / 
            overall_stats['total_tests'] * 100
        ) if overall_stats['total_tests'] > 0 else 0
        
        # Compile comprehensive results
        self.test_results = {
            'overall_stats': overall_stats,
            'category_results': category_results,
            'test_categories': test_categories,
            'performance_metrics': self.profiler.get_all_profiles(),
            'memory_summary': self.memory_tracker.get_memory_summary(),
            'timestamp': time.time()
        }
        
        return self.test_results
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.test_results:
            return "❌ No test results to report"
        
        report = []
        report.append("📊 TruthGPT Optimization Core Comprehensive Test Report")
        report.append("=" * 70)
        report.append("")
        
        # Overall summary
        overall = self.test_results['overall_stats']
        report.append("📈 Overall Summary:")
        report.append(f"  Total Tests: {overall['total_tests']}")
        report.append(f"  Failures: {overall['total_failures']}")
        report.append(f"  Errors: {overall['total_errors']}")
        report.append(f"  Success Rate: {overall['success_rate']:.1f}%")
        report.append(f"  Total Time: {overall['total_time']:.2f}s")
        report.append(f"  Categories Passed: {overall['categories_passed']}")
        report.append(f"  Categories Failed: {overall['categories_failed']}")
        report.append("")
        
        # Category breakdown
        report.append("📋 Category Breakdown:")
        for category, results in self.test_results['category_results'].items():
            status = "✅" if results['success_rate'] >= 95 else "❌"
            report.append(f"  {status} {category.title()}: {results['total_tests']} tests, "
                         f"{results['success_rate']:.1f}% success, {results['total_time']:.2f}s")
        report.append("")
        
        # Performance metrics
        if 'performance_metrics' in self.test_results and self.test_results['performance_metrics']:
            perf = self.test_results['performance_metrics']
            report.append("⚡ Performance Metrics:")
            report.append(f"  Total Profiles: {len(perf)}")
            total_execution_time = sum(p.get('execution_time', 0) for p in perf)
            total_memory_used = sum(p.get('memory_used', 0) for p in perf)
            report.append(f"  Total Execution Time: {total_execution_time:.2f}s")
            report.append(f"  Total Memory Used: {total_memory_used:.2f}MB")
            report.append("")
        
        # Memory summary
        if 'memory_summary' in self.test_results and self.test_results['memory_summary']:
            mem = self.test_results['memory_summary']
            report.append("💾 Memory Summary:")
            report.append(f"  Snapshots Taken: {mem.get('snapshots_taken', 0)}")
            report.append(f"  Peak Memory: {mem.get('peak_memory', 0):.2f}MB")
            report.append(f"  Average Memory: {mem.get('average_memory', 0):.2f}MB")
            report.append("")
        
        # Detailed results by category
        report.append("📋 Detailed Results by Category:")
        for category, results in self.test_results['category_results'].items():
            report.append(f"\n  {category.title()} Tests:")
            for file_name, file_results in results['file_results'].items():
                status = "✅" if file_results['success'] else "❌"
                report.append(f"    {status} {file_name}: {file_results['tests_run']} tests, "
                            f"{file_results['failures']} failures, {file_results['errors']} errors, "
                            f"{file_results['execution_time']:.2f}s")
        report.append("")
        
        # Recommendations
        report.append("💡 Recommendations:")
        if overall['total_failures'] > 0:
            report.append("  - Fix failing tests to improve reliability")
        if overall['total_errors'] > 0:
            report.append("  - Address test errors to ensure proper functionality")
        if overall['success_rate'] < 90:
            report.append("  - Improve test coverage and fix issues")
        if overall['total_time'] > 300:
            report.append("  - Consider optimizing slow tests")
        if overall['categories_failed'] > 0:
            report.append("  - Focus on improving failed test categories")
        report.append("")
        
        # Final status
        if overall['success_rate'] >= 95 and overall['categories_failed'] == 0:
            report.append("🎉 Excellent! All tests are passing with high success rate.")
        elif overall['success_rate'] >= 80:
            report.append("✅ Good! Most tests are passing.")
        else:
            report.append("⚠️  Attention needed! Some tests are failing.")
        
        return "\n".join(report)
    
    def save_comprehensive_results(self, filename: str = "comprehensive_test_results.json"):
        """Save comprehensive test results to file"""
        results_file = Path(__file__).parent / filename
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"💾 Comprehensive test results saved to: {results_file}")
    
    def run_specific_category(self, category: str) -> Dict[str, Any]:
        """Run tests for a specific category only"""
        test_categories = self.discover_all_tests()
        
        if category not in test_categories:
            print(f"❌ Category '{category}' not found")
            return {}
        
        test_files = test_categories[category]
        if not test_files:
            print(f"❌ No test files found for category '{category}'")
            return {}
        
        print(f"🎯 Running {category} tests only")
        return self.run_test_category(category, test_files)
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Run performance analysis on test results"""
        if not self.test_results:
            return {}
        
        analysis = {
            'performance_analysis': {},
            'memory_analysis': {},
            'efficiency_analysis': {}
        }
        
        # Performance analysis
        if 'performance_metrics' in self.test_results:
            perf_metrics = self.test_results['performance_metrics']
            if perf_metrics:
                total_time = sum(p.get('execution_time', 0) for p in perf_metrics)
                total_memory = sum(p.get('memory_used', 0) for p in perf_metrics)
                
                analysis['performance_analysis'] = {
                    'total_execution_time': total_time,
                    'total_memory_used': total_memory,
                    'average_execution_time': total_time / len(perf_metrics),
                    'average_memory_used': total_memory / len(perf_metrics),
                    'slowest_test': max(perf_metrics, key=lambda x: x.get('execution_time', 0)),
                    'most_memory_intensive': max(perf_metrics, key=lambda x: x.get('memory_used', 0))
                }
        
        # Memory analysis
        if 'memory_summary' in self.test_results:
            mem_summary = self.test_results['memory_summary']
            analysis['memory_analysis'] = {
                'peak_memory': mem_summary.get('peak_memory', 0),
                'average_memory': mem_summary.get('average_memory', 0),
                'memory_growth': mem_summary.get('memory_growth', []),
                'snapshots_taken': mem_summary.get('snapshots_taken', 0)
            }
        
        # Efficiency analysis
        overall = self.test_results['overall_stats']
        analysis['efficiency_analysis'] = {
            'success_rate': overall['success_rate'],
            'tests_per_second': overall['total_tests'] / overall['total_time'] if overall['total_time'] > 0 else 0,
            'efficiency_score': overall['success_rate'] * (overall['total_tests'] / overall['total_time']) if overall['total_time'] > 0 else 0
        }
        
        return analysis

def main():
    """Main function for running comprehensive tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TruthGPT Optimization Core Comprehensive Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--coverage", "-c", action="store_true", help="Enable coverage reporting")
    parser.add_argument("--performance", "-perf", action="store_true", help="Run performance tests only")
    parser.add_argument("--category", "-cat", choices=['unit', 'integration', 'performance'], 
                       help="Run specific category of tests")
    parser.add_argument("--save-results", "-s", action="store_true", help="Save results to file")
    parser.add_argument("--analysis", "-a", action="store_true", help="Run performance analysis")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ComprehensiveTestRunner(
        verbose=args.verbose,
        parallel=args.parallel,
        coverage=args.coverage,
        performance=args.performance
    )
    
    # Run tests based on arguments
    if args.category:
        results = runner.run_specific_category(args.category)
    else:
        results = runner.run_all_tests()
    
    # Generate and print report
    if results:
        report = runner.generate_comprehensive_report()
        print("\n" + report)
        
        # Run performance analysis if requested
        if args.analysis:
            analysis = runner.run_performance_analysis()
            print("\n📊 Performance Analysis:")
            print(json.dumps(analysis, indent=2, default=str))
        
        # Save results if requested
        if args.save_results:
            runner.save_comprehensive_results()
    
    # Exit with appropriate code
    if results and 'overall_stats' in results:
        overall = results['overall_stats']
        if overall['total_failures'] == 0 and overall['total_errors'] == 0:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Please check the results above.")
            sys.exit(1)
    else:
        print("\n❌ No test results generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()





