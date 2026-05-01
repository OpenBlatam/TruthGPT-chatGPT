"""
Unit tests for optimization benchmarks and performance testing
Tests benchmarking frameworks, performance metrics, and optimization comparisons
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import statistics

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestOptimizationBenchmarks(unittest.TestCase):
    """Test suite for optimization benchmarks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_optimization_benchmark(self):
        """Test optimization benchmarking framework"""
        class OptimizationBenchmark:
            def __init__(self, algorithms, test_functions):
                self.algorithms = algorithms
                self.test_functions = test_functions
                self.benchmark_results = {}
                self.performance_metrics = {}
                
            def run_benchmark(self, n_runs=10, max_iterations=100):
                """Run complete optimization benchmark"""
                for algorithm_name, algorithm in self.algorithms.items():
                    algorithm_results = {}
                    
                    for function_name, test_function in self.test_functions.items():
                        function_results = []
                        
                        for run in range(n_runs):
                            # Run algorithm on test function
                            start_time = time.time()
                            result = self._run_algorithm(algorithm, test_function, max_iterations)
                            end_time = time.time()
                            
                            result['execution_time'] = end_time - start_time
                            result['run'] = run
                            function_results.append(result)
                            
                        algorithm_results[function_name] = function_results
                        
                    self.benchmark_results[algorithm_name] = algorithm_results
                    
                # Calculate performance metrics
                self._calculate_performance_metrics()
                
                return self.benchmark_results
                
            def _run_algorithm(self, algorithm, test_function, max_iterations):
                """Run single algorithm on test function"""
                # Simulate algorithm execution
                result = {
                    'best_solution': np.random.uniform(-5, 5, 5),
                    'best_fitness': test_function(np.random.uniform(-5, 5, 5)),
                    'convergence_time': np.random.uniform(10, 100),
                    'function_evaluations': np.random.randint(100, 1000),
                    'success': np.random.uniform(0, 1) > 0.2,
                    'iterations': np.random.randint(10, max_iterations)
                }
                return result
                
            def _calculate_performance_metrics(self):
                """Calculate performance metrics"""
                for algorithm_name, algorithm_results in self.benchmark_results.items():
                    algorithm_metrics = {}
                    
                    for function_name, function_results in algorithm_results.items():
                        # Calculate metrics for this function
                        fitness_values = [r['best_fitness'] for r in function_results]
                        success_rates = [r['success'] for r in function_results]
                        execution_times = [r['execution_time'] for r in function_results]
                        
                        function_metrics = {
                            'avg_fitness': np.mean(fitness_values),
                            'std_fitness': np.std(fitness_values),
                            'best_fitness': np.min(fitness_values),
                            'worst_fitness': np.max(fitness_values),
                            'success_rate': np.mean(success_rates),
                            'avg_execution_time': np.mean(execution_times),
                            'std_execution_time': np.std(execution_times),
                            'total_runs': len(function_results)
                        }
                        
                        algorithm_metrics[function_name] = function_metrics
                        
                    self.performance_metrics[algorithm_name] = algorithm_metrics
                    
            def get_benchmark_summary(self):
                """Get benchmark summary"""
                return {
                    'total_algorithms': len(self.algorithms),
                    'total_functions': len(self.test_functions),
                    'benchmark_results': self.benchmark_results,
                    'performance_metrics': self.performance_metrics,
                    'overall_winner': self._calculate_overall_winner()
                }
                
            def _calculate_overall_winner(self):
                """Calculate overall winner across all functions"""
                algorithm_scores = {}
                
                for algorithm_name, algorithm_metrics in self.performance_metrics.items():
                    total_score = 0
                    for function_name, function_metrics in algorithm_metrics.items():
                        # Score based on fitness and success rate
                        fitness_score = 1.0 / (function_metrics['avg_fitness'] + 1e-8)
                        success_score = function_metrics['success_rate']
                        total_score += fitness_score * success_score
                        
                    algorithm_scores[algorithm_name] = total_score
                    
                return max(algorithm_scores.items(), key=lambda x: x[1])[0]
        
        # Test optimization benchmark
        algorithms = {
            'genetic_algorithm': {'type': 'evolutionary'},
            'particle_swarm': {'type': 'swarm'},
            'differential_evolution': {'type': 'evolutionary'}
        }
        
        test_functions = {
            'sphere': lambda x: np.sum(x**2),
            'rosenbrock': lambda x: np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2),
            'rastrigin': lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
        }
        
        benchmark = OptimizationBenchmark(algorithms, test_functions)
        
        # Test benchmark
        results = benchmark.run_benchmark(n_runs=5, max_iterations=50)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn('genetic_algorithm', results)
        self.assertIn('particle_swarm', results)
        self.assertIn('differential_evolution', results)
        
        # Check benchmark summary
        summary = benchmark.get_benchmark_summary()
        self.assertEqual(summary['total_algorithms'], 3)
        self.assertEqual(summary['total_functions'], 3)
        self.assertEqual(len(summary['performance_metrics']), 3)
        self.assertIn('overall_winner', summary)
        
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        class PerformanceMetrics:
            def __init__(self):
                self.metrics_history = []
                self.current_metrics = {}
                
            def calculate_metrics(self, model, data, target, loss, execution_time):
                """Calculate performance metrics"""
                # Model complexity metrics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Performance metrics
                with torch.no_grad():
                    output = model(data)
                    mse = nn.MSELoss()(output, target).item()
                    mae = nn.L1Loss()(output, target).item()
                    
                # Gradient metrics
                grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.data.norm(2).item())
                        
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
                max_grad_norm = np.max(grad_norms) if grad_norms else 0
                
                # Memory metrics
                memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Throughput metrics
                batch_size = data.shape[0]
                throughput = batch_size / execution_time if execution_time > 0 else 0
                
                metrics = {
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'mse': mse,
                    'mae': mae,
                    'avg_grad_norm': avg_grad_norm,
                    'max_grad_norm': max_grad_norm,
                    'memory_usage': memory_usage,
                    'execution_time': execution_time,
                    'throughput': throughput,
                    'loss': loss.item() if hasattr(loss, 'item') else loss
                }
                
                self.current_metrics = metrics
                self.metrics_history.append(metrics.copy())
                
                return metrics
                
            def get_metrics_summary(self):
                """Get metrics summary"""
                if not self.metrics_history:
                    return {}
                    
                # Calculate summary statistics
                mse_values = [m['mse'] for m in self.metrics_history]
                mae_values = [m['mae'] for m in self.metrics_history]
                execution_times = [m['execution_time'] for m in self.metrics_history]
                throughputs = [m['throughput'] for m in self.metrics_history]
                
                return {
                    'total_iterations': len(self.metrics_history),
                    'final_mse': mse_values[-1],
                    'final_mae': mae_values[-1],
                    'mse_improvement': mse_values[0] - mse_values[-1] if len(mse_values) > 1 else 0,
                    'mae_improvement': mae_values[0] - mae_values[-1] if len(mae_values) > 1 else 0,
                    'avg_execution_time': np.mean(execution_times),
                    'avg_throughput': np.mean(throughputs),
                    'total_params': self.metrics_history[-1]['total_params'],
                    'trainable_params': self.metrics_history[-1]['trainable_params']
                }
        
        # Test performance metrics
        metrics_calculator = PerformanceMetrics()
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test metrics calculation
        start_time = time.time()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        end_time = time.time()
        
        execution_time = end_time - start_time
        metrics = metrics_calculator.calculate_metrics(model, data, target, loss, execution_time)
        
        # Verify metrics
        self.assertIn('total_params', metrics)
        self.assertIn('trainable_params', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('avg_grad_norm', metrics)
        self.assertIn('max_grad_norm', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('execution_time', metrics)
        self.assertIn('throughput', metrics)
        self.assertIn('loss', metrics)
        
        # Check metrics summary
        summary = metrics_calculator.get_metrics_summary()
        self.assertEqual(summary['total_iterations'], 1)
        self.assertGreater(summary['total_params'], 0)
        self.assertGreater(summary['trainable_params'], 0)
        self.assertGreater(summary['final_mse'], 0)
        self.assertGreater(summary['final_mae'], 0)
        self.assertGreater(summary['avg_execution_time'], 0)
        self.assertGreater(summary['avg_throughput'], 0)

class TestOptimizationComparison(unittest.TestCase):
    """Test suite for optimization comparison"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_optimization_comparison(self):
        """Test optimization comparison framework"""
        class OptimizationComparison:
            def __init__(self, optimizers):
                self.optimizers = optimizers
                self.comparison_results = {}
                self.performance_rankings = {}
                
            def compare_optimizers(self, model, data, target, max_iterations=100):
                """Compare multiple optimizers"""
                for optimizer_name, optimizer in self.optimizers.items():
                    # Create model copy for each optimizer
                    model_copy = self._copy_model(model)
                    
                    # Run optimization
                    result = self._run_optimization(optimizer, model_copy, data, target, max_iterations)
                    self.comparison_results[optimizer_name] = result
                    
                # Calculate performance rankings
                self._calculate_rankings()
                
                return self.comparison_results
                
            def _copy_model(self, model):
                """Create model copy"""
                # Simple model copying simulation
                return model
                
            def _run_optimization(self, optimizer, model, data, target, max_iterations):
                """Run single optimizer"""
                start_time = time.time()
                
                # Simulate optimization
                for iteration in range(max_iterations):
                    output = model(data)
                    loss = nn.MSELoss()(output, target)
                    loss.backward()
                    
                    # Simulate optimizer step
                    if hasattr(optimizer, 'step'):
                        optimizer.step()
                    else:
                        # Simulate optimization step
                        for param in model.parameters():
                            if param.grad is not None:
                                param.data -= 0.001 * param.grad.data
                                
                end_time = time.time()
                
                result = {
                    'optimizer': optimizer,
                    'final_loss': loss.item(),
                    'execution_time': end_time - start_time,
                    'iterations': max_iterations,
                    'converged': loss.item() < 0.1,
                    'final_accuracy': np.random.uniform(0.8, 0.99)
                }
                
                return result
                
            def _calculate_rankings(self):
                """Calculate performance rankings"""
                # Rank by final loss
                loss_rankings = sorted(self.comparison_results.items(), 
                                     key=lambda x: x[1]['final_loss'])
                
                # Rank by execution time
                time_rankings = sorted(self.comparison_results.items(), 
                                    key=lambda x: x[1]['execution_time'])
                
                # Rank by accuracy
                accuracy_rankings = sorted(self.comparison_results.items(), 
                                        key=lambda x: x[1]['final_accuracy'], reverse=True)
                
                self.performance_rankings = {
                    'loss_rankings': loss_rankings,
                    'time_rankings': time_rankings,
                    'accuracy_rankings': accuracy_rankings
                }
                
            def get_comparison_summary(self):
                """Get comparison summary"""
                return {
                    'total_optimizers': len(self.optimizers),
                    'comparison_results': self.comparison_results,
                    'performance_rankings': self.performance_rankings,
                    'best_optimizer': self._get_best_optimizer()
                }
                
            def _get_best_optimizer(self):
                """Get best optimizer based on combined metrics"""
                best_score = float('inf')
                best_optimizer = None
                
                for optimizer_name, result in self.comparison_results.items():
                    # Combined score: lower loss, higher accuracy, lower time
                    score = (result['final_loss'] / result['final_accuracy'] + 
                            result['execution_time'] / 100)
                    
                    if score < best_score:
                        best_score = score
                        best_optimizer = optimizer_name
                        
                return best_optimizer
        
        # Test optimization comparison
        optimizers = {
            'adam': {'learning_rate': 0.001},
            'sgd': {'learning_rate': 0.01},
            'rmsprop': {'learning_rate': 0.005}
        }
        
        comparison = OptimizationComparison(optimizers)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test optimization comparison
        results = comparison.compare_optimizers(model, data, target, max_iterations=10)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn('adam', results)
        self.assertIn('sgd', results)
        self.assertIn('rmsprop', results)
        
        for optimizer_name, result in results.items():
            self.assertIn('final_loss', result)
            self.assertIn('execution_time', result)
            self.assertIn('iterations', result)
            self.assertIn('converged', result)
            self.assertIn('final_accuracy', result)
            
        # Check comparison summary
        summary = comparison.get_comparison_summary()
        self.assertEqual(summary['total_optimizers'], 3)
        self.assertEqual(len(summary['comparison_results']), 3)
        self.assertEqual(len(summary['performance_rankings']), 3)
        self.assertIn('best_optimizer', summary)
        
    def test_optimization_statistics(self):
        """Test optimization statistics calculation"""
        class OptimizationStatistics:
            def __init__(self):
                self.optimization_history = []
                self.statistics = {}
                
            def add_optimization_result(self, result):
                """Add optimization result"""
                self.optimization_history.append(result)
                
            def calculate_statistics(self):
                """Calculate optimization statistics"""
                if not self.optimization_history:
                    return {}
                    
                # Extract metrics
                losses = [r['loss'] for r in self.optimization_history]
                execution_times = [r['execution_time'] for r in self.optimization_history]
                accuracies = [r['accuracy'] for r in self.optimization_history]
                
                # Calculate statistics
                self.statistics = {
                    'total_optimizations': len(self.optimization_history),
                    'loss_stats': {
                        'mean': np.mean(losses),
                        'std': np.std(losses),
                        'min': np.min(losses),
                        'max': np.max(losses),
                        'median': np.median(losses)
                    },
                    'time_stats': {
                        'mean': np.mean(execution_times),
                        'std': np.std(execution_times),
                        'min': np.min(execution_times),
                        'max': np.max(execution_times),
                        'median': np.median(execution_times)
                    },
                    'accuracy_stats': {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies),
                        'min': np.min(accuracies),
                        'max': np.max(accuracies),
                        'median': np.median(accuracies)
                    }
                }
                
                return self.statistics
                
            def get_performance_trends(self):
                """Get performance trends"""
                if len(self.optimization_history) < 2:
                    return {}
                    
                # Calculate trends
                losses = [r['loss'] for r in self.optimization_history]
                accuracies = [r['accuracy'] for r in self.optimization_history]
                
                # Linear regression for trends
                x = np.arange(len(losses))
                loss_trend = np.polyfit(x, losses, 1)[0]
                accuracy_trend = np.polyfit(x, accuracies, 1)[0]
                
                return {
                    'loss_trend': loss_trend,
                    'accuracy_trend': accuracy_trend,
                    'improving_loss': loss_trend < 0,
                    'improving_accuracy': accuracy_trend > 0
                }
        
        # Test optimization statistics
        stats_calculator = OptimizationStatistics()
        
        # Add optimization results
        for i in range(10):
            result = {
                'loss': np.random.uniform(0, 1),
                'execution_time': np.random.uniform(0.1, 1.0),
                'accuracy': np.random.uniform(0.8, 0.99)
            }
            stats_calculator.add_optimization_result(result)
            
        # Calculate statistics
        statistics = stats_calculator.calculate_statistics()
        
        # Verify statistics
        self.assertEqual(statistics['total_optimizations'], 10)
        self.assertIn('loss_stats', statistics)
        self.assertIn('time_stats', statistics)
        self.assertIn('accuracy_stats', statistics)
        
        # Check performance trends
        trends = stats_calculator.get_performance_trends()
        self.assertIn('loss_trend', trends)
        self.assertIn('accuracy_trend', trends)
        self.assertIn('improving_loss', trends)
        self.assertIn('improving_accuracy', trends)

class TestBenchmarkingUtilities(unittest.TestCase):
    """Test suite for benchmarking utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_benchmark_timer(self):
        """Test benchmark timer utility"""
        class BenchmarkTimer:
            def __init__(self):
                self.timings = {}
                self.current_timing = None
                
            def start_timing(self, name):
                """Start timing a benchmark"""
                self.current_timing = {
                    'name': name,
                    'start_time': time.time()
                }
                
            def end_timing(self):
                """End timing a benchmark"""
                if self.current_timing is None:
                    return None
                    
                end_time = time.time()
                duration = end_time - self.current_timing['start_time']
                
                timing_result = {
                    'name': self.current_timing['name'],
                    'duration': duration,
                    'start_time': self.current_timing['start_time'],
                    'end_time': end_time
                }
                
                self.timings[self.current_timing['name']] = timing_result
                self.current_timing = None
                
                return timing_result
                
            def get_timing_stats(self):
                """Get timing statistics"""
                if not self.timings:
                    return {}
                    
                durations = [timing['duration'] for timing in self.timings.values()]
                
                return {
                    'total_timings': len(self.timings),
                    'avg_duration': np.mean(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'total_duration': np.sum(durations),
                    'timings': self.timings
                }
        
        # Test benchmark timer
        timer = BenchmarkTimer()
        
        # Test timing
        timer.start_timing('test_operation')
        time.sleep(0.1)  # Simulate operation
        result = timer.end_timing()
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], 'test_operation')
        self.assertGreater(result['duration'], 0)
        self.assertIn('start_time', result)
        self.assertIn('end_time', result)
        
        # Check timing stats
        stats = timer.get_timing_stats()
        self.assertEqual(stats['total_timings'], 1)
        self.assertGreater(stats['avg_duration'], 0)
        self.assertGreater(stats['min_duration'], 0)
        self.assertGreater(stats['max_duration'], 0)
        self.assertGreater(stats['total_duration'], 0)
        
    def test_benchmark_reporter(self):
        """Test benchmark reporter utility"""
        class BenchmarkReporter:
            def __init__(self):
                self.reports = []
                self.current_report = None
                
            def start_report(self, benchmark_name):
                """Start benchmark report"""
                self.current_report = {
                    'benchmark_name': benchmark_name,
                    'start_time': time.time(),
                    'metrics': {},
                    'results': []
                }
                
            def add_metric(self, metric_name, value):
                """Add metric to current report"""
                if self.current_report is not None:
                    self.current_report['metrics'][metric_name] = value
                    
            def add_result(self, result_name, result_data):
                """Add result to current report"""
                if self.current_report is not None:
                    self.current_report['results'].append({
                        'name': result_name,
                        'data': result_data,
                        'timestamp': time.time()
                    })
                    
            def end_report(self):
                """End benchmark report"""
                if self.current_report is None:
                    return None
                    
                self.current_report['end_time'] = time.time()
                self.current_report['duration'] = (self.current_report['end_time'] - 
                                                 self.current_report['start_time'])
                
                report = self.current_report.copy()
                self.reports.append(report)
                self.current_report = None
                
                return report
                
            def get_report_summary(self):
                """Get report summary"""
                if not self.reports:
                    return {}
                    
                return {
                    'total_reports': len(self.reports),
                    'report_names': [report['benchmark_name'] for report in self.reports],
                    'avg_duration': np.mean([report['duration'] for report in self.reports]),
                    'total_duration': np.sum([report['duration'] for report in self.reports])
                }
        
        # Test benchmark reporter
        reporter = BenchmarkReporter()
        
        # Test report creation
        reporter.start_report('test_benchmark')
        reporter.add_metric('iterations', 100)
        reporter.add_metric('accuracy', 0.95)
        reporter.add_result('final_loss', 0.01)
        reporter.add_result('convergence_time', 50)
        
        report = reporter.end_report()
        
        # Verify results
        self.assertIsNotNone(report)
        self.assertEqual(report['benchmark_name'], 'test_benchmark')
        self.assertGreater(report['duration'], 0)
        self.assertIn('iterations', report['metrics'])
        self.assertIn('accuracy', report['metrics'])
        self.assertEqual(len(report['results']), 2)
        
        # Check report summary
        summary = reporter.get_report_summary()
        self.assertEqual(summary['total_reports'], 1)
        self.assertEqual(len(summary['report_names']), 1)
        self.assertGreater(summary['avg_duration'], 0)
        self.assertGreater(summary['total_duration'], 0)

if __name__ == '__main__':
    unittest.main()





