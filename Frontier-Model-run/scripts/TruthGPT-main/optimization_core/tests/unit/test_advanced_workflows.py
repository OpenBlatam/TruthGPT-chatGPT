"""
Unit tests for advanced optimization workflows
Tests complex optimization pipelines, multi-objective optimization, and workflow orchestration
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import itertools
import threading
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestMultiObjectiveOptimization(unittest.TestCase):
    """Test suite for multi-objective optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_pareto_optimization(self):
        """Test Pareto optimization for multiple objectives"""
        class ParetoOptimizer:
            def __init__(self, objectives):
                self.objectives = objectives
                self.pareto_front = []
                self.optimization_history = []
                self.best_solutions = []
                
            def evaluate_objectives(self, solution, data, target):
                """Evaluate multiple objectives"""
                objective_values = {}
                for obj_name, obj_func in self.objectives.items():
                    objective_values[obj_name] = obj_func(solution, data, target)
                return objective_values
                
            def is_pareto_optimal(self, new_solution):
                """Check if solution is Pareto optimal"""
                for existing_solution in self.pareto_front:
                    if self._dominates(existing_solution, new_solution):
                        return False
                return True
                
            def _dominates(self, solution1, solution2):
                """Check if solution1 dominates solution2"""
                obj1 = solution1['objectives']
                obj2 = solution2['objectives']
                
                # Check if solution1 is better in at least one objective
                # and not worse in any objective
                better_in_one = False
                for obj_name in self.objectives:
                    if obj1[obj_name] > obj2[obj_name]:
                        better_in_one = True
                    elif obj1[obj_name] < obj2[obj_name]:
                        return False
                        
                return better_in_one
                
            def update_pareto_front(self, solution):
                """Update Pareto front"""
                if self.is_pareto_optimal(solution):
                    # Remove dominated solutions
                    self.pareto_front = [s for s in self.pareto_front 
                                       if not self._dominates(solution, s)]
                    self.pareto_front.append(solution)
                    
            def optimize(self, data, target, n_iterations=100):
                """Run multi-objective optimization"""
                for iteration in range(n_iterations):
                    # Generate random solution
                    solution = self._generate_solution()
                    
                    # Evaluate objectives
                    objectives = self.evaluate_objectives(solution, data, target)
                    
                    # Create solution record
                    solution_record = {
                        'solution': solution,
                        'objectives': objectives,
                        'iteration': iteration
                    }
                    
                    # Update Pareto front
                    self.update_pareto_front(solution_record)
                    
                    # Record optimization step
                    self.optimization_history.append(solution_record)
                    
                return self.pareto_front
                
            def _generate_solution(self):
                """Generate random solution"""
                return {
                    'learning_rate': np.random.uniform(0.0001, 0.1),
                    'batch_size': np.random.randint(16, 129),
                    'dropout': np.random.uniform(0.1, 0.5),
                    'hidden_size': np.random.choice([128, 256, 512, 1024])
                }
                
            def get_pareto_stats(self):
                """Get Pareto front statistics"""
                if not self.pareto_front:
                    return {}
                    
                # Calculate objective statistics
                obj_stats = {}
                for obj_name in self.objectives:
                    values = [s['objectives'][obj_name] for s in self.pareto_front]
                    obj_stats[obj_name] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                    
                return {
                    'pareto_size': len(self.pareto_front),
                    'total_iterations': len(self.optimization_history),
                    'objective_stats': obj_stats,
                    'diversity': self._calculate_diversity()
                }
                
            def _calculate_diversity(self):
                """Calculate diversity of Pareto front"""
                if len(self.pareto_front) < 2:
                    return 0
                    
                # Calculate pairwise distances
                distances = []
                for i in range(len(self.pareto_front)):
                    for j in range(i + 1, len(self.pareto_front)):
                        dist = self._calculate_solution_distance(
                            self.pareto_front[i], self.pareto_front[j]
                        )
                        distances.append(dist)
                        
                return np.mean(distances) if distances else 0
                
            def _calculate_solution_distance(self, sol1, sol2):
                """Calculate distance between solutions"""
                # Normalize objective values
                obj1 = sol1['objectives']
                obj2 = sol2['objectives']
                
                distance = 0
                for obj_name in self.objectives:
                    distance += (obj1[obj_name] - obj2[obj_name]) ** 2
                    
                return np.sqrt(distance)
        
        # Test Pareto optimization
        objectives = {
            'accuracy': lambda sol, data, target: np.random.uniform(0, 1),
            'speed': lambda sol, data, target: np.random.uniform(0, 1),
            'memory': lambda sol, data, target: np.random.uniform(0, 1)
        }
        
        pareto_opt = ParetoOptimizer(objectives)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test Pareto optimization
        pareto_front = pareto_opt.optimize(data, target, n_iterations=20)
        
        # Verify results
        self.assertGreater(len(pareto_front), 0)
        self.assertEqual(len(pareto_opt.optimization_history), 20)
        
        # Check Pareto stats
        stats = pareto_opt.get_pareto_stats()
        self.assertGreater(stats['pareto_size'], 0)
        self.assertEqual(stats['total_iterations'], 20)
        self.assertIn('accuracy', stats['objective_stats'])
        self.assertIn('speed', stats['objective_stats'])
        self.assertIn('memory', stats['objective_stats'])
        self.assertGreaterEqual(stats['diversity'], 0)
        
    def test_weighted_sum_optimization(self):
        """Test weighted sum optimization"""
        class WeightedSumOptimizer:
            def __init__(self, objectives, weights):
                self.objectives = objectives
                self.weights = weights
                self.optimization_history = []
                self.best_solution = None
                self.best_score = float('inf')
                
            def evaluate_weighted_score(self, solution, data, target):
                """Evaluate weighted objective score"""
                objective_values = {}
                for obj_name, obj_func in self.objectives.items():
                    objective_values[obj_name] = obj_func(solution, data, target)
                    
                # Calculate weighted sum
                weighted_score = 0
                for obj_name, weight in self.weights.items():
                    weighted_score += weight * objective_values[obj_name]
                    
                return weighted_score, objective_values
                
            def optimize(self, data, target, n_iterations=100):
                """Run weighted sum optimization"""
                for iteration in range(n_iterations):
                    # Generate random solution
                    solution = self._generate_solution()
                    
                    # Evaluate weighted score
                    score, objectives = self.evaluate_weighted_score(solution, data, target)
                    
                    # Update best solution
                    if score < self.best_score:
                        self.best_score = score
                        self.best_solution = solution.copy()
                        
                    # Record optimization step
                    self.optimization_history.append({
                        'solution': solution,
                        'score': score,
                        'objectives': objectives,
                        'iteration': iteration
                    })
                    
                return self.best_solution, self.best_score
                
            def _generate_solution(self):
                """Generate random solution"""
                return {
                    'learning_rate': np.random.uniform(0.0001, 0.1),
                    'batch_size': np.random.randint(16, 129),
                    'dropout': np.random.uniform(0.1, 0.5)
                }
                
            def get_optimization_stats(self):
                """Get optimization statistics"""
                if not self.optimization_history:
                    return {}
                    
                scores = [step['score'] for step in self.optimization_history]
                
                return {
                    'total_iterations': len(self.optimization_history),
                    'best_score': self.best_score,
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'improvement': max(scores) - self.best_score
                }
        
        # Test weighted sum optimization
        objectives = {
            'accuracy': lambda sol, data, target: np.random.uniform(0, 1),
            'speed': lambda sol, data, target: np.random.uniform(0, 1),
            'memory': lambda sol, data, target: np.random.uniform(0, 1)
        }
        
        weights = {'accuracy': 0.5, 'speed': 0.3, 'memory': 0.2}
        
        weighted_opt = WeightedSumOptimizer(objectives, weights)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test weighted sum optimization
        best_solution, best_score = weighted_opt.optimize(data, target, n_iterations=20)
        
        # Verify results
        self.assertIsNotNone(best_solution)
        self.assertGreater(best_score, 0)
        self.assertIn('learning_rate', best_solution)
        self.assertIn('batch_size', best_solution)
        self.assertIn('dropout', best_solution)
        
        # Check optimization stats
        stats = weighted_opt.get_optimization_stats()
        self.assertEqual(stats['total_iterations'], 20)
        self.assertGreater(stats['best_score'], 0)
        self.assertGreaterEqual(stats['improvement'], 0)

class TestWorkflowOrchestration(unittest.TestCase):
    """Test suite for workflow orchestration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_optimization_pipeline(self):
        """Test optimization pipeline orchestration"""
        class OptimizationPipeline:
            def __init__(self, stages):
                self.stages = stages
                self.pipeline_history = []
                self.current_stage = 0
                self.pipeline_results = {}
                
            def run_pipeline(self, data, target):
                """Run complete optimization pipeline"""
                current_data = data
                current_target = target
                
                for stage_name, stage_func in self.stages.items():
                    # Run stage
                    stage_result = stage_func(current_data, current_target)
                    
                    # Record stage result
                    self.pipeline_results[stage_name] = stage_result
                    self.pipeline_history.append({
                        'stage': stage_name,
                        'result': stage_result,
                        'timestamp': len(self.pipeline_history)
                    })
                    
                    # Update data for next stage
                    if 'output_data' in stage_result:
                        current_data = stage_result['output_data']
                    if 'output_target' in stage_result:
                        current_target = stage_result['output_target']
                        
                return self.pipeline_results
                
            def get_pipeline_stats(self):
                """Get pipeline statistics"""
                return {
                    'total_stages': len(self.stages),
                    'completed_stages': len(self.pipeline_history),
                    'stage_names': list(self.stages.keys()),
                    'pipeline_results': self.pipeline_results
                }
        
        # Test optimization pipeline
        def preprocessing_stage(data, target):
            """Preprocessing stage"""
            return {
                'output_data': data * 0.9,  # Simulate preprocessing
                'output_target': target,
                'preprocessing_applied': True
            }
            
        def feature_engineering_stage(data, target):
            """Feature engineering stage"""
            return {
                'output_data': torch.cat([data, data**2], dim=-1),  # Add polynomial features
                'output_target': target,
                'features_added': data.shape[-1]
            }
            
        def optimization_stage(data, target):
            """Optimization stage"""
            return {
                'output_data': data,
                'output_target': target,
                'optimization_completed': True,
                'final_loss': np.random.uniform(0, 1)
            }
            
        stages = {
            'preprocessing': preprocessing_stage,
            'feature_engineering': feature_engineering_stage,
            'optimization': optimization_stage
        }
        
        pipeline = OptimizationPipeline(stages)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test pipeline execution
        results = pipeline.run_pipeline(data, target)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn('preprocessing', results)
        self.assertIn('feature_engineering', results)
        self.assertIn('optimization', results)
        
        # Check pipeline stats
        stats = pipeline.get_pipeline_stats()
        self.assertEqual(stats['total_stages'], 3)
        self.assertEqual(stats['completed_stages'], 3)
        self.assertEqual(len(stats['stage_names']), 3)
        
    def test_parallel_optimization(self):
        """Test parallel optimization execution"""
        class ParallelOptimizer:
            def __init__(self, optimizers):
                self.optimizers = optimizers
                self.parallel_results = {}
                self.execution_times = {}
                
            def run_parallel_optimization(self, data, target):
                """Run optimizers in parallel"""
                threads = []
                results = {}
                
                def run_optimizer(name, optimizer):
                    start_time = time.time()
                    result = self._run_single_optimizer(optimizer, data, target)
                    end_time = time.time()
                    
                    results[name] = result
                    self.execution_times[name] = end_time - start_time
                    
                # Start parallel threads
                for name, optimizer in self.optimizers.items():
                    thread = threading.Thread(target=run_optimizer, args=(name, optimizer))
                    threads.append(thread)
                    thread.start()
                    
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                    
                self.parallel_results = results
                return results
                
            def _run_single_optimizer(self, optimizer, data, target):
                """Run single optimizer"""
                # Simulate optimization
                time.sleep(0.1)  # Simulate computation time
                
                result = {
                    'optimizer': optimizer,
                    'performance': np.random.uniform(0, 1),
                    'converged': np.random.uniform(0, 1) > 0.5,
                    'iterations': np.random.randint(10, 100)
                }
                
                return result
                
            def get_parallel_stats(self):
                """Get parallel execution statistics"""
                if not self.parallel_results:
                    return {}
                    
                total_time = max(self.execution_times.values()) if self.execution_times else 0
                sequential_time = sum(self.execution_times.values()) if self.execution_times else 0
                speedup = sequential_time / total_time if total_time > 0 else 0
                
                return {
                    'total_optimizers': len(self.optimizers),
                    'parallel_results': self.parallel_results,
                    'execution_times': self.execution_times,
                    'total_time': total_time,
                    'sequential_time': sequential_time,
                    'speedup': speedup
                }
        
        # Test parallel optimization
        optimizers = {
            'adam': {'learning_rate': 0.001},
            'sgd': {'learning_rate': 0.01},
            'rmsprop': {'learning_rate': 0.005}
        }
        
        parallel_opt = ParallelOptimizer(optimizers)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test parallel optimization
        results = parallel_opt.run_parallel_optimization(data, target)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn('adam', results)
        self.assertIn('sgd', results)
        self.assertIn('rmsprop', results)
        
        for name, result in results.items():
            self.assertIn('performance', result)
            self.assertIn('converged', result)
            self.assertIn('iterations', result)
            
        # Check parallel stats
        stats = parallel_opt.get_parallel_stats()
        self.assertEqual(stats['total_optimizers'], 3)
        self.assertEqual(len(stats['execution_times']), 3)
        self.assertGreater(stats['total_time'], 0)
        self.assertGreater(stats['speedup'], 0)
        
    def test_adaptive_workflow(self):
        """Test adaptive workflow execution"""
        class AdaptiveWorkflow:
            def __init__(self, workflow_stages, adaptation_rules):
                self.workflow_stages = workflow_stages
                self.adaptation_rules = adaptation_rules
                self.workflow_history = []
                self.adaptations_applied = []
                
            def run_adaptive_workflow(self, data, target):
                """Run adaptive workflow"""
                current_data = data
                current_target = target
                current_stages = self.workflow_stages.copy()
                
                for stage_name, stage_func in current_stages.items():
                    # Run stage
                    stage_result = stage_func(current_data, current_target)
                    
                    # Record stage result
                    self.workflow_history.append({
                        'stage': stage_name,
                        'result': stage_result,
                        'timestamp': len(self.workflow_history)
                    })
                    
                    # Check adaptation rules
                    adaptation = self._check_adaptation_rules(stage_name, stage_result)
                    if adaptation:
                        self.adaptations_applied.append(adaptation)
                        current_stages = self._apply_adaptation(current_stages, adaptation)
                        
                    # Update data for next stage
                    if 'output_data' in stage_result:
                        current_data = stage_result['output_data']
                    if 'output_target' in stage_result:
                        current_target = stage_result['output_target']
                        
                return {
                    'workflow_history': self.workflow_history,
                    'adaptations_applied': self.adaptations_applied
                }
                
            def _check_adaptation_rules(self, stage_name, stage_result):
                """Check if adaptation should be applied"""
                for rule_name, rule_func in self.adaptation_rules.items():
                    if rule_func(stage_name, stage_result):
                        return {
                            'rule_name': rule_name,
                            'stage': stage_name,
                            'result': stage_result
                        }
                return None
                
            def _apply_adaptation(self, stages, adaptation):
                """Apply adaptation to workflow"""
                # Simulate adaptation
                adapted_stages = stages.copy()
                
                if adaptation['rule_name'] == 'add_regularization':
                    # Add regularization stage
                    adapted_stages['regularization'] = self._create_regularization_stage()
                elif adaptation['rule_name'] == 'add_validation':
                    # Add validation stage
                    adapted_stages['validation'] = self._create_validation_stage()
                    
                return adapted_stages
                
            def _create_regularization_stage(self):
                """Create regularization stage"""
                def regularization_stage(data, target):
                    return {
                        'output_data': data * 0.95,  # Simulate regularization
                        'output_target': target,
                        'regularization_applied': True
                    }
                return regularization_stage
                
            def _create_validation_stage(self):
                """Create validation stage"""
                def validation_stage(data, target):
                    return {
                        'output_data': data,
                        'output_target': target,
                        'validation_score': np.random.uniform(0, 1)
                    }
                return validation_stage
                
            def get_adaptive_stats(self):
                """Get adaptive workflow statistics"""
                return {
                    'total_stages': len(self.workflow_stages),
                    'completed_stages': len(self.workflow_history),
                    'adaptations_applied': len(self.adaptations_applied),
                    'adaptation_rate': len(self.adaptations_applied) / len(self.workflow_history) if self.workflow_history else 0
                }
        
        # Test adaptive workflow
        def preprocessing_stage(data, target):
            return {'output_data': data, 'output_target': target, 'preprocessing_done': True}
            
        def optimization_stage(data, target):
            return {'output_data': data, 'output_target': target, 'optimization_done': True}
        
        workflow_stages = {
            'preprocessing': preprocessing_stage,
            'optimization': optimization_stage
        }
        
        adaptation_rules = {
            'add_regularization': lambda stage, result: stage == 'optimization' and result.get('optimization_done', False),
            'add_validation': lambda stage, result: stage == 'preprocessing' and result.get('preprocessing_done', False)
        }
        
        adaptive_workflow = AdaptiveWorkflow(workflow_stages, adaptation_rules)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test adaptive workflow
        results = adaptive_workflow.run_adaptive_workflow(data, target)
        
        # Verify results
        self.assertIn('workflow_history', results)
        self.assertIn('adaptations_applied', results)
        self.assertGreater(len(results['workflow_history']), 0)
        
        # Check adaptive stats
        stats = adaptive_workflow.get_adaptive_stats()
        self.assertEqual(stats['total_stages'], 2)
        self.assertGreater(stats['completed_stages'], 0)
        self.assertGreaterEqual(stats['adaptations_applied'], 0)
        self.assertGreaterEqual(stats['adaptation_rate'], 0)

class TestOptimizationMonitoring(unittest.TestCase):
    """Test suite for optimization monitoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_real_time_monitoring(self):
        """Test real-time optimization monitoring"""
        class RealTimeMonitor:
            def __init__(self):
                self.metrics_history = []
                self.alerts = []
                self.monitoring_active = False
                
            def start_monitoring(self):
                """Start real-time monitoring"""
                self.monitoring_active = True
                self.metrics_history = []
                self.alerts = []
                
            def stop_monitoring(self):
                """Stop real-time monitoring"""
                self.monitoring_active = False
                
            def update_metrics(self, metrics):
                """Update monitoring metrics"""
                if not self.monitoring_active:
                    return
                    
                # Record metrics
                self.metrics_history.append({
                    'metrics': metrics,
                    'timestamp': len(self.metrics_history)
                })
                
                # Check for alerts
                self._check_alerts(metrics)
                
            def _check_alerts(self, metrics):
                """Check for monitoring alerts"""
                # Check for performance degradation
                if 'loss' in metrics and metrics['loss'] > 1.0:
                    self.alerts.append({
                        'type': 'performance_degradation',
                        'message': 'Loss is too high',
                        'value': metrics['loss'],
                        'timestamp': len(self.metrics_history)
                    })
                    
                # Check for gradient explosion
                if 'grad_norm' in metrics and metrics['grad_norm'] > 10.0:
                    self.alerts.append({
                        'type': 'gradient_explosion',
                        'message': 'Gradient norm is too high',
                        'value': metrics['grad_norm'],
                        'timestamp': len(self.metrics_history)
                    })
                    
                # Check for convergence
                if len(self.metrics_history) > 10:
                    recent_losses = [m['metrics'].get('loss', 0) for m in self.metrics_history[-10:]]
                    if np.std(recent_losses) < 0.01:
                        self.alerts.append({
                            'type': 'convergence',
                            'message': 'Optimization has converged',
                            'value': np.mean(recent_losses),
                            'timestamp': len(self.metrics_history)
                        })
                        
            def get_monitoring_stats(self):
                """Get monitoring statistics"""
                if not self.metrics_history:
                    return {}
                    
                return {
                    'total_metrics': len(self.metrics_history),
                    'monitoring_active': self.monitoring_active,
                    'alerts_count': len(self.alerts),
                    'alert_types': list(set(alert['type'] for alert in self.alerts)),
                    'latest_metrics': self.metrics_history[-1]['metrics'] if self.metrics_history else {}
                }
        
        # Test real-time monitoring
        monitor = RealTimeMonitor()
        
        # Test monitoring start
        monitor.start_monitoring()
        self.assertTrue(monitor.monitoring_active)
        
        # Test metrics updates
        for i in range(5):
            metrics = {
                'loss': np.random.uniform(0, 2),
                'grad_norm': np.random.uniform(0, 15),
                'learning_rate': 0.001
            }
            monitor.update_metrics(metrics)
            
        # Test monitoring stop
        monitor.stop_monitoring()
        self.assertFalse(monitor.monitoring_active)
        
        # Check monitoring stats
        stats = monitor.get_monitoring_stats()
        self.assertEqual(stats['total_metrics'], 5)
        self.assertFalse(stats['monitoring_active'])
        self.assertGreaterEqual(stats['alerts_count'], 0)
        self.assertIn('latest_metrics', stats)
        
    def test_optimization_dashboard(self):
        """Test optimization dashboard"""
        class OptimizationDashboard:
            def __init__(self):
                self.dashboard_data = {}
                self.visualization_data = {}
                self.dashboard_history = []
                
            def update_dashboard(self, optimization_data):
                """Update dashboard with optimization data"""
                # Process optimization data
                processed_data = self._process_optimization_data(optimization_data)
                
                # Update dashboard data
                self.dashboard_data.update(processed_data)
                
                # Generate visualizations
                self._generate_visualizations(processed_data)
                
                # Record dashboard update
                self.dashboard_history.append({
                    'data': processed_data,
                    'timestamp': len(self.dashboard_history)
                })
                
            def _process_optimization_data(self, data):
                """Process optimization data for dashboard"""
                processed = {}
                
                # Process metrics
                if 'metrics' in data:
                    processed['metrics'] = data['metrics']
                    
                # Process performance
                if 'performance' in data:
                    processed['performance'] = data['performance']
                    
                # Process convergence
                if 'convergence' in data:
                    processed['convergence'] = data['convergence']
                    
                return processed
                
            def _generate_visualizations(self, data):
                """Generate visualization data"""
                # Simulate visualization generation
                self.visualization_data = {
                    'loss_curve': np.random.uniform(0, 1, 10),
                    'gradient_norm': np.random.uniform(0, 5, 10),
                    'learning_rate': np.random.uniform(0.001, 0.01, 10),
                    'performance_metrics': {
                        'accuracy': np.random.uniform(0, 1),
                        'precision': np.random.uniform(0, 1),
                        'recall': np.random.uniform(0, 1)
                    }
                }
                
            def get_dashboard_stats(self):
                """Get dashboard statistics"""
                return {
                    'total_updates': len(self.dashboard_history),
                    'dashboard_data_keys': list(self.dashboard_data.keys()),
                    'visualization_data_keys': list(self.visualization_data.keys()),
                    'latest_update': self.dashboard_history[-1] if self.dashboard_history else None
                }
        
        # Test optimization dashboard
        dashboard = OptimizationDashboard()
        
        # Test dashboard updates
        for i in range(3):
            optimization_data = {
                'metrics': {
                    'loss': np.random.uniform(0, 1),
                    'grad_norm': np.random.uniform(0, 5)
                },
                'performance': {
                    'accuracy': np.random.uniform(0, 1),
                    'speed': np.random.uniform(0, 1)
                },
                'convergence': {
                    'converged': np.random.uniform(0, 1) > 0.5
                }
            }
            dashboard.update_dashboard(optimization_data)
            
        # Check dashboard stats
        stats = dashboard.get_dashboard_stats()
        self.assertEqual(stats['total_updates'], 3)
        self.assertGreater(len(stats['dashboard_data_keys']), 0)
        self.assertGreater(len(stats['visualization_data_keys']), 0)
        self.assertIsNotNone(stats['latest_update'])

if __name__ == '__main__':
    unittest.main()





