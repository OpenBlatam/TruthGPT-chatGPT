"""
Unit tests for hyperparameter optimization
Tests hyperparameter tuning, grid search, random search, and Bayesian optimization
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import itertools

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestGridSearchOptimization(unittest.TestCase):
    """Test suite for grid search hyperparameter optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_grid_search_basic(self):
        """Test basic grid search functionality"""
        class GridSearchOptimizer:
            def __init__(self, param_grid):
                self.param_grid = param_grid
                self.results = []
                self.best_params = None
                self.best_score = float('inf')
                
            def generate_param_combinations(self):
                """Generate all parameter combinations"""
                param_names = list(self.param_grid.keys())
                param_values = list(self.param_grid.values())
                
                combinations = []
                for combination in itertools.product(*param_values):
                    param_dict = dict(zip(param_names, combination))
                    combinations.append(param_dict)
                    
                return combinations
                
            def evaluate_params(self, params, model, data, target):
                """Evaluate parameter combination"""
                # Simulate parameter evaluation
                score = np.random.uniform(0, 1)
                
                result = {
                    'params': params,
                    'score': score,
                    'timestamp': len(self.results)
                }
                
                self.results.append(result)
                
                # Update best parameters
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
                return score
                
            def search(self, model, data, target):
                """Run grid search"""
                combinations = self.generate_param_combinations()
                
                for params in combinations:
                    score = self.evaluate_params(params, model, data, target)
                    
                return self.best_params, self.best_score
                
            def get_search_stats(self):
                """Get search statistics"""
                if not self.results:
                    return {}
                    
                scores = [result['score'] for result in self.results]
                
                return {
                    'total_combinations': len(self.results),
                    'best_score': self.best_score,
                    'worst_score': max(scores),
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'improvement': max(scores) - self.best_score
                }
        
        # Test grid search
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'dropout': [0.1, 0.2, 0.3]
        }
        
        grid_search = GridSearchOptimizer(param_grid)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test grid search
        best_params, best_score = grid_search.search(model, data, target)
        
        # Verify results
        self.assertIsNotNone(best_params)
        self.assertGreater(best_score, 0)
        self.assertIn('learning_rate', best_params)
        self.assertIn('batch_size', best_params)
        self.assertIn('dropout', best_params)
        
        # Check search stats
        stats = grid_search.get_search_stats()
        self.assertEqual(stats['total_combinations'], 27)  # 3^3
        self.assertGreater(stats['best_score'], 0)
        self.assertGreaterEqual(stats['improvement'], 0)
        
    def test_grid_search_with_validation(self):
        """Test grid search with validation"""
        class GridSearchWithValidation:
            def __init__(self, param_grid, validation_split=0.2):
                self.param_grid = param_grid
                self.validation_split = validation_split
                self.results = []
                self.best_params = None
                self.best_score = float('inf')
                
            def split_data(self, data, target):
                """Split data into train and validation"""
                n_samples = data.shape[0]
                n_val = int(n_samples * self.validation_split)
                
                indices = torch.randperm(n_samples)
                train_indices = indices[n_val:]
                val_indices = indices[:n_val]
                
                train_data = data[train_indices]
                train_target = target[train_indices]
                val_data = data[val_indices]
                val_target = target[val_indices]
                
                return train_data, train_target, val_data, val_target
                
            def evaluate_with_validation(self, params, model, train_data, train_target, val_data, val_target):
                """Evaluate parameters with validation"""
                # Simulate training
                train_score = np.random.uniform(0, 1)
                
                # Simulate validation
                val_score = np.random.uniform(0, 1)
                
                result = {
                    'params': params,
                    'train_score': train_score,
                    'val_score': val_score,
                    'timestamp': len(self.results)
                }
                
                self.results.append(result)
                
                # Update best parameters based on validation score
                if val_score < self.best_score:
                    self.best_score = val_score
                    self.best_params = params.copy()
                    
                return val_score
                
            def search(self, model, data, target):
                """Run grid search with validation"""
                train_data, train_target, val_data, val_target = self.split_data(data, target)
                
                combinations = self._generate_combinations()
                
                for params in combinations:
                    score = self.evaluate_with_validation(params, model, train_data, train_target, val_data, val_target)
                    
                return self.best_params, self.best_score
                
            def _generate_combinations(self):
                """Generate parameter combinations"""
                param_names = list(self.param_grid.keys())
                param_values = list(self.param_grid.values())
                
                combinations = []
                for combination in itertools.product(*param_values):
                    param_dict = dict(zip(param_names, combination))
                    combinations.append(param_dict)
                    
                return combinations
                
            def get_validation_stats(self):
                """Get validation statistics"""
                if not self.results:
                    return {}
                    
                val_scores = [result['val_score'] for result in self.results]
                train_scores = [result['train_score'] for result in self.results]
                
                return {
                    'total_combinations': len(self.results),
                    'best_val_score': self.best_score,
                    'average_val_score': np.mean(val_scores),
                    'average_train_score': np.mean(train_scores),
                    'overfitting': np.mean(train_scores) - np.mean(val_scores)
                }
        
        # Test grid search with validation
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32]
        }
        
        val_grid_search = GridSearchWithValidation(param_grid)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=10, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test grid search with validation
        best_params, best_score = val_grid_search.search(model, data, target)
        
        # Verify results
        self.assertIsNotNone(best_params)
        self.assertGreater(best_score, 0)
        
        # Check validation stats
        stats = val_grid_search.get_validation_stats()
        self.assertEqual(stats['total_combinations'], 4)  # 2^2
        self.assertGreater(stats['best_val_score'], 0)

class TestRandomSearchOptimization(unittest.TestCase):
    """Test suite for random search hyperparameter optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_random_search_basic(self):
        """Test basic random search functionality"""
        class RandomSearchOptimizer:
            def __init__(self, param_distributions, n_iter=10):
                self.param_distributions = param_distributions
                self.n_iter = n_iter
                self.results = []
                self.best_params = None
                self.best_score = float('inf')
                
            def sample_params(self):
                """Sample random parameters from distributions"""
                params = {}
                for param_name, distribution in self.param_distributions.items():
                    if distribution['type'] == 'uniform':
                        params[param_name] = np.random.uniform(
                            distribution['low'], distribution['high']
                        )
                    elif distribution['type'] == 'log_uniform':
                        params[param_name] = np.exp(np.random.uniform(
                            np.log(distribution['low']), np.log(distribution['high'])
                        ))
                    elif distribution['type'] == 'choice':
                        params[param_name] = np.random.choice(distribution['choices'])
                    elif distribution['type'] == 'int_uniform':
                        params[param_name] = np.random.randint(
                            distribution['low'], distribution['high'] + 1
                        )
                        
                return params
                
            def evaluate_params(self, params, model, data, target):
                """Evaluate parameter combination"""
                # Simulate parameter evaluation
                score = np.random.uniform(0, 1)
                
                result = {
                    'params': params,
                    'score': score,
                    'iteration': len(self.results)
                }
                
                self.results.append(result)
                
                # Update best parameters
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
                return score
                
            def search(self, model, data, target):
                """Run random search"""
                for iteration in range(self.n_iter):
                    params = self.sample_params()
                    score = self.evaluate_params(params, model, data, target)
                    
                return self.best_params, self.best_score
                
            def get_search_stats(self):
                """Get search statistics"""
                if not self.results:
                    return {}
                    
                scores = [result['score'] for result in self.results]
                
                return {
                    'total_iterations': len(self.results),
                    'best_score': self.best_score,
                    'worst_score': max(scores),
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'improvement': max(scores) - self.best_score
                }
        
        # Test random search
        param_distributions = {
            'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'batch_size': {'type': 'int_uniform', 'low': 16, 'high': 128},
            'dropout': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
            'optimizer': {'type': 'choice', 'choices': ['adam', 'sgd', 'rmsprop']}
        }
        
        random_search = RandomSearchOptimizer(param_distributions, n_iter=5)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test random search
        best_params, best_score = random_search.search(model, data, target)
        
        # Verify results
        self.assertIsNotNone(best_params)
        self.assertGreater(best_score, 0)
        self.assertIn('learning_rate', best_params)
        self.assertIn('batch_size', best_params)
        self.assertIn('dropout', best_params)
        self.assertIn('optimizer', best_params)
        
        # Check search stats
        stats = random_search.get_search_stats()
        self.assertEqual(stats['total_iterations'], 5)
        self.assertGreater(stats['best_score'], 0)
        self.assertGreaterEqual(stats['improvement'], 0)

class TestBayesianOptimization(unittest.TestCase):
    """Test suite for Bayesian optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_bayesian_optimization_basic(self):
        """Test basic Bayesian optimization"""
        class BayesianOptimizer:
            def __init__(self, param_bounds, n_iter=10, n_initial=3):
                self.param_bounds = param_bounds
                self.n_iter = n_iter
                self.n_initial = n_initial
                self.X = []  # Parameter combinations
                self.y = []  # Objective values
                self.best_params = None
                self.best_score = float('inf')
                
            def random_sample(self):
                """Sample random parameters"""
                params = {}
                for param_name, (low, high) in self.param_bounds.items():
                    params[param_name] = np.random.uniform(low, high)
                return params
                
            def acquisition_function(self, params):
                """Simple acquisition function (Upper Confidence Bound)"""
                # Simulate acquisition function
                return np.random.uniform(0, 1)
                
            def optimize_acquisition(self):
                """Optimize acquisition function to get next parameters"""
                best_params = None
                best_acquisition = -float('inf')
                
                # Random search for acquisition optimization
                for _ in range(100):
                    params = self.random_sample()
                    acquisition = self.acquisition_function(params)
                    
                    if acquisition > best_acquisition:
                        best_acquisition = acquisition
                        best_params = params
                        
                return best_params
                
            def evaluate_params(self, params, model, data, target):
                """Evaluate parameter combination"""
                # Simulate parameter evaluation
                score = np.random.uniform(0, 1)
                
                # Store for Bayesian optimization
                self.X.append(list(params.values()))
                self.y.append(score)
                
                # Update best parameters
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    
                return score
                
            def search(self, model, data, target):
                """Run Bayesian optimization"""
                # Initial random sampling
                for _ in range(self.n_initial):
                    params = self.random_sample()
                    score = self.evaluate_params(params, model, data, target)
                
                # Bayesian optimization iterations
                for iteration in range(self.n_iter - self.n_initial):
                    # Optimize acquisition function
                    params = self.optimize_acquisition()
                    score = self.evaluate_params(params, model, data, target)
                    
                return self.best_params, self.best_score
                
            def get_optimization_stats(self):
                """Get optimization statistics"""
                if not self.y:
                    return {}
                    
                return {
                    'total_evaluations': len(self.y),
                    'best_score': self.best_score,
                    'worst_score': max(self.y),
                    'average_score': np.mean(self.y),
                    'score_std': np.std(self.y),
                    'improvement': max(self.y) - self.best_score,
                    'convergence': self._check_convergence()
                }
                
            def _check_convergence(self):
                """Check if optimization has converged"""
                if len(self.y) < 5:
                    return False
                    
                recent_scores = self.y[-5:]
                return np.std(recent_scores) < 0.01
        
        # Test Bayesian optimization
        param_bounds = {
            'learning_rate': (1e-4, 1e-1),
            'batch_size': (16, 128),
            'dropout': (0.1, 0.5)
        }
        
        bayesian_opt = BayesianOptimizer(param_bounds, n_iter=8, n_initial=3)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test Bayesian optimization
        best_params, best_score = bayesian_opt.search(model, data, target)
        
        # Verify results
        self.assertIsNotNone(best_params)
        self.assertGreater(best_score, 0)
        self.assertIn('learning_rate', best_params)
        self.assertIn('batch_size', best_params)
        self.assertIn('dropout', best_params)
        
        # Check optimization stats
        stats = bayesian_opt.get_optimization_stats()
        self.assertEqual(stats['total_evaluations'], 8)
        self.assertGreater(stats['best_score'], 0)
        self.assertGreaterEqual(stats['improvement'], 0)

class TestHyperparameterOptimizationIntegration(unittest.TestCase):
    """Test suite for hyperparameter optimization integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_multi_objective_optimization(self):
        """Test multi-objective hyperparameter optimization"""
        class MultiObjectiveOptimizer:
            def __init__(self, param_distributions, objectives):
                self.param_distributions = param_distributions
                self.objectives = objectives
                self.results = []
                self.pareto_front = []
                
            def evaluate_objectives(self, params, model, data, target):
                """Evaluate multiple objectives"""
                # Simulate objective evaluation
                objective_values = {}
                for obj_name in self.objectives:
                    objective_values[obj_name] = np.random.uniform(0, 1)
                    
                return objective_values
                
            def is_pareto_optimal(self, new_result):
                """Check if result is Pareto optimal"""
                for existing_result in self.pareto_front:
                    # Check if existing result dominates new result
                    dominated = True
                    for obj_name in self.objectives:
                        if new_result['objectives'][obj_name] < existing_result['objectives'][obj_name]:
                            dominated = False
                            break
                    if dominated:
                        return False
                        
                return True
                
            def update_pareto_front(self, result):
                """Update Pareto front"""
                if self.is_pareto_optimal(result):
                    # Remove dominated solutions
                    self.pareto_front = [r for r in self.pareto_front 
                                       if not self._dominates(result, r)]
                    self.pareto_front.append(result)
                    
            def _dominates(self, result1, result2):
                """Check if result1 dominates result2"""
                for obj_name in self.objectives:
                    if result1['objectives'][obj_name] >= result2['objectives'][obj_name]:
                        return False
                return True
                
            def search(self, model, data, target, n_iter=10):
                """Run multi-objective optimization"""
                for iteration in range(n_iter):
                    # Sample parameters
                    params = self._sample_params()
                    
                    # Evaluate objectives
                    objectives = self.evaluate_objectives(params, model, data, target)
                    
                    result = {
                        'params': params,
                        'objectives': objectives,
                        'iteration': iteration
                    }
                    
                    self.results.append(result)
                    self.update_pareto_front(result)
                    
                return self.pareto_front
                
            def _sample_params(self):
                """Sample random parameters"""
                params = {}
                for param_name, distribution in self.param_distributions.items():
                    if distribution['type'] == 'uniform':
                        params[param_name] = np.random.uniform(
                            distribution['low'], distribution['high']
                        )
                    elif distribution['type'] == 'choice':
                        params[param_name] = np.random.choice(distribution['choices'])
                return params
                
            def get_pareto_stats(self):
                """Get Pareto front statistics"""
                if not self.pareto_front:
                    return {}
                    
                return {
                    'pareto_size': len(self.pareto_front),
                    'total_evaluations': len(self.results),
                    'objectives': self.objectives
                }
        
        # Test multi-objective optimization
        param_distributions = {
            'learning_rate': {'type': 'uniform', 'low': 0.001, 'high': 0.1},
            'batch_size': {'type': 'choice', 'choices': [16, 32, 64, 128]}
        }
        
        objectives = ['accuracy', 'speed', 'memory']
        
        multi_obj_opt = MultiObjectiveOptimizer(param_distributions, objectives)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test multi-objective optimization
        pareto_front = multi_obj_opt.search(model, data, target, n_iter=5)
        
        # Verify results
        self.assertGreater(len(pareto_front), 0)
        self.assertGreater(len(multi_obj_opt.results), 0)
        
        # Check Pareto front stats
        stats = multi_obj_opt.get_pareto_stats()
        self.assertGreater(stats['pareto_size'], 0)
        self.assertEqual(stats['total_evaluations'], 5)
        
    def test_hyperparameter_optimization_workflow(self):
        """Test complete hyperparameter optimization workflow"""
        class HyperparameterOptimizationWorkflow:
            def __init__(self):
                self.optimization_history = []
                self.best_configurations = []
                
            def run_optimization_pipeline(self, model, data, target, optimization_configs):
                """Run complete optimization pipeline"""
                for config in optimization_configs:
                    opt_type = config['type']
                    param_space = config['param_space']
                    n_iter = config.get('n_iter', 10)
                    
                    # Run optimization
                    best_params, best_score = self._run_optimization(
                        opt_type, param_space, model, data, target, n_iter
                    )
                    
                    # Record results
                    result = {
                        'optimization_type': opt_type,
                        'best_params': best_params,
                        'best_score': best_score,
                        'n_iter': n_iter
                    }
                    
                    self.optimization_history.append(result)
                    self.best_configurations.append(result)
                    
                return self.best_configurations
                
            def _run_optimization(self, opt_type, param_space, model, data, target, n_iter):
                """Run specific optimization type"""
                if opt_type == 'grid_search':
                    return self._grid_search(param_space, model, data, target)
                elif opt_type == 'random_search':
                    return self._random_search(param_space, model, data, target, n_iter)
                elif opt_type == 'bayesian':
                    return self._bayesian_optimization(param_space, model, data, target, n_iter)
                else:
                    raise ValueError(f"Unknown optimization type: {opt_type}")
                    
            def _grid_search(self, param_space, model, data, target):
                """Grid search optimization"""
                best_params = None
                best_score = float('inf')
                
                # Simple grid search implementation
                for lr in param_space['learning_rate']:
                    for batch_size in param_space['batch_size']:
                        params = {'learning_rate': lr, 'batch_size': batch_size}
                        score = np.random.uniform(0, 1)
                        
                        if score < best_score:
                            best_score = score
                            best_params = params
                            
                return best_params, best_score
                
            def _random_search(self, param_space, model, data, target, n_iter):
                """Random search optimization"""
                best_params = None
                best_score = float('inf')
                
                for _ in range(n_iter):
                    params = {
                        'learning_rate': np.random.choice(param_space['learning_rate']),
                        'batch_size': np.random.choice(param_space['batch_size'])
                    }
                    score = np.random.uniform(0, 1)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
                        
                return best_params, best_score
                
            def _bayesian_optimization(self, param_space, model, data, target, n_iter):
                """Bayesian optimization"""
                best_params = None
                best_score = float('inf')
                
                for _ in range(n_iter):
                    params = {
                        'learning_rate': np.random.uniform(0.001, 0.1),
                        'batch_size': np.random.randint(16, 129)
                    }
                    score = np.random.uniform(0, 1)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
                        
                return best_params, best_score
                
            def get_workflow_stats(self):
                """Get workflow statistics"""
                if not self.optimization_history:
                    return {}
                    
                return {
                    'total_optimizations': len(self.optimization_history),
                    'optimization_types': [opt['optimization_type'] for opt in self.optimization_history],
                    'best_scores': [opt['best_score'] for opt in self.optimization_history],
                    'average_score': np.mean([opt['best_score'] for opt in self.optimization_history])
                }
        
        # Test hyperparameter optimization workflow
        workflow = HyperparameterOptimizationWorkflow()
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Define optimization configurations
        optimization_configs = [
            {
                'type': 'grid_search',
                'param_space': {
                    'learning_rate': [0.001, 0.01],
                    'batch_size': [16, 32]
                }
            },
            {
                'type': 'random_search',
                'param_space': {
                    'learning_rate': [0.001, 0.01, 0.1],
                    'batch_size': [16, 32, 64]
                },
                'n_iter': 5
            },
            {
                'type': 'bayesian',
                'param_space': {},
                'n_iter': 3
            }
        ]
        
        # Run optimization workflow
        results = workflow.run_optimization_pipeline(model, data, target, optimization_configs)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(len(workflow.optimization_history), 3)
        
        # Check workflow stats
        stats = workflow.get_workflow_stats()
        self.assertEqual(stats['total_optimizations'], 3)
        self.assertEqual(len(stats['optimization_types']), 3)
        self.assertGreater(stats['average_score'], 0)

if __name__ == '__main__':
    unittest.main()





