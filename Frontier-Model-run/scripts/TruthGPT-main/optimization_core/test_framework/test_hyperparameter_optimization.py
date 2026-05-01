#!/usr/bin/env python3
"""
Test Suite for Hyperparameter Optimization
Comprehensive tests for hyperparameter optimization with multiple algorithms and techniques
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import json
import pickle
from pathlib import Path
import math
import random

# Import hyperparameter optimization components
import sys
sys.path.append('..')
from bulk.ultra_advanced_optimizer import (
    HyperparameterOptimizer, HyperparameterSpace, UltraAdvancedOptimizer
)
from modules.base.core_system.core.advanced_optimizations import (
    AdvancedOptimizationEngine, OptimizationTechnique, OptimizationMetrics
)

class TestHyperparameterSpaceAdvanced(unittest.TestCase):
    """Advanced tests for hyperparameter search space."""
    
    def setUp(self):
        self.space = HyperparameterSpace()
    
    def test_hyperparameter_space_validation(self):
        """Test hyperparameter space validation."""
        # Test valid ranges
        self.assertGreater(self.space.learning_rate[1], self.space.learning_rate[0])
        self.assertGreater(self.space.batch_size[1], self.space.batch_size[0])
        self.assertGreater(self.space.dropout_rate[1], self.space.dropout_rate[0])
        self.assertGreater(self.space.weight_decay[1], self.space.weight_decay[0])
        self.assertGreater(self.space.momentum[1], self.space.momentum[0])
        self.assertGreater(self.space.beta1[1], self.space.beta1[0])
        self.assertGreater(self.space.beta2[1], self.space.beta2[0])
        self.assertGreater(self.space.epsilon[1], self.space.epsilon[0])
    
    def test_hyperparameter_space_custom_ranges(self):
        """Test hyperparameter space with custom ranges."""
        custom_space = HyperparameterSpace(
            learning_rate=(1e-4, 1e-2),
            batch_size=(16, 64),
            dropout_rate=(0.1, 0.3),
            weight_decay=(1e-5, 1e-3),
            momentum=(0.8, 0.95),
            beta1=(0.9, 0.99),
            beta2=(0.99, 0.999),
            epsilon=(1e-8, 1e-6)
        )
        
        self.assertEqual(custom_space.learning_rate, (1e-4, 1e-2))
        self.assertEqual(custom_space.batch_size, (16, 64))
        self.assertEqual(custom_space.dropout_rate, (0.1, 0.3))
        self.assertEqual(custom_space.weight_decay, (1e-5, 1e-3))
        self.assertEqual(custom_space.momentum, (0.8, 0.95))
        self.assertEqual(custom_space.beta1, (0.9, 0.99))
        self.assertEqual(custom_space.beta2, (0.99, 0.999))
        self.assertEqual(custom_space.epsilon, (1e-8, 1e-6))
    
    def test_hyperparameter_space_boundary_values(self):
        """Test hyperparameter space with boundary values."""
        boundary_space = HyperparameterSpace(
            learning_rate=(0.0, 1.0),
            batch_size=(1, 1000),
            dropout_rate=(0.0, 1.0),
            weight_decay=(0.0, 1.0),
            momentum=(0.0, 1.0),
            beta1=(0.0, 1.0),
            beta2=(0.0, 1.0),
            epsilon=(0.0, 1.0)
        )
        
        # Test boundary values
        self.assertEqual(boundary_space.learning_rate, (0.0, 1.0))
        self.assertEqual(boundary_space.batch_size, (1, 1000))
        self.assertEqual(boundary_space.dropout_rate, (0.0, 1.0))
        self.assertEqual(boundary_space.weight_decay, (0.0, 1.0))
        self.assertEqual(boundary_space.momentum, (0.0, 1.0))
        self.assertEqual(boundary_space.beta1, (0.0, 1.0))
        self.assertEqual(boundary_space.beta2, (0.0, 1.0))
        self.assertEqual(boundary_space.epsilon, (0.0, 1.0))
    
    def test_hyperparameter_space_serialization(self):
        """Test hyperparameter space serialization."""
        space_dict = {
            'learning_rate': self.space.learning_rate,
            'batch_size': self.space.batch_size,
            'dropout_rate': self.space.dropout_rate,
            'weight_decay': self.space.weight_decay,
            'momentum': self.space.momentum,
            'beta1': self.space.beta1,
            'beta2': self.space.beta2,
            'epsilon': self.space.epsilon
        }
        
        # Test JSON serialization
        json_str = json.dumps(space_dict)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized['learning_rate'], list(self.space.learning_rate))
        self.assertEqual(deserialized['batch_size'], list(self.space.batch_size))
        self.assertEqual(deserialized['dropout_rate'], list(self.space.dropout_rate))
        self.assertEqual(deserialized['weight_decay'], list(self.space.weight_decay))
        self.assertEqual(deserialized['momentum'], list(self.space.momentum))
        self.assertEqual(deserialized['beta1'], list(self.space.beta1))
        self.assertEqual(deserialized['beta2'], list(self.space.beta2))
        self.assertEqual(deserialized['epsilon'], list(self.space.epsilon))

class TestHyperparameterOptimizerAdvanced(unittest.TestCase):
    """Advanced tests for hyperparameter optimizer."""
    
    def setUp(self):
        self.search_space = HyperparameterSpace()
        self.optimizer = HyperparameterOptimizer(self.search_space)
    
    def test_hyperparameter_optimizer_initialization(self):
        """Test hyperparameter optimizer initialization."""
        self.assertEqual(self.optimizer.search_space, self.search_space)
        self.assertEqual(len(self.optimizer.optimization_history), 0)
        self.assertIsInstance(self.optimizer.logger, logging.Logger)
    
    def test_bayesian_optimization_advanced(self):
        """Test advanced Bayesian optimization."""
        def objective_function(params):
            # Complex objective function
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            # Simulate complex optimization landscape
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            result = self.optimizer.bayesian_optimization(objective_function, n_trials=50)
            
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
            self.assertIsInstance(result['best_params'], dict)
            self.assertIsInstance(result['best_value'], float)
            self.assertIsInstance(result['optimization_history'], list)
    
    def test_tree_structured_parzen_estimator_advanced(self):
        """Test advanced Tree-structured Parzen Estimator optimization."""
        def objective_function(params):
            # Multi-modal objective function
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            # Multiple local minima
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        with patch('hyperopt.fmin') as mock_fmin:
            mock_fmin.return_value = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            
            with patch('hyperopt.Trials') as mock_trials:
                mock_trials.return_value.trials = [{'result': {'loss': 0.01}}]
                
                result = self.optimizer.tree_structured_parzen_estimator(
                    objective_function, max_evals=50
                )
                
                self.assertIn('best_params', result)
                self.assertIn('best_value', result)
                self.assertIn('optimization_history', result)
                self.assertIsInstance(result['best_params'], dict)
                self.assertIsInstance(result['best_value'], float)
                self.assertIsInstance(result['optimization_history'], list)
    
    def test_differential_evolution_optimization_advanced(self):
        """Test advanced differential evolution optimization."""
        def objective_function(params):
            # High-dimensional objective function
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            # Complex optimization landscape
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        with patch('scipy.optimize.differential_evolution') as mock_de:
            mock_result = Mock()
            mock_result.x = [0.001, 32, 0.2, 1e-4, 0.9]
            mock_result.fun = 0.01
            mock_de.return_value = mock_result
            
            result = self.optimizer.differential_evolution_optimization(
                objective_function, max_iterations=50
            )
            
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
            self.assertIsInstance(result['best_params'], dict)
            self.assertIsInstance(result['best_value'], float)
            self.assertIsInstance(result['optimization_history'], float)
    
    def test_hyperparameter_optimization_with_constraints(self):
        """Test hyperparameter optimization with constraints."""
        def constrained_objective(params):
            # Objective function with constraints
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            # Add constraint penalties
            penalty = 0
            if lr < 1e-5 or lr > 1e-1:
                penalty += 1000
            if bs < 8 or bs > 128:
                penalty += 1000
            if dr < 0.0 or dr > 0.5:
                penalty += 1000
            if wd < 1e-6 or wd > 1e-2:
                penalty += 1000
            if mom < 0.0 or mom > 0.99:
                penalty += 1000
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2 + penalty
        
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            result = self.optimizer.bayesian_optimization(constrained_objective, n_trials=30)
            
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
    
    def test_hyperparameter_optimization_with_noise(self):
        """Test hyperparameter optimization with noisy objective function."""
        def noisy_objective(params):
            # Objective function with noise
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            base_value = (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
            
            # Add noise
            noise = random.gauss(0, 0.01)
            return base_value + noise
        
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            result = self.optimizer.bayesian_optimization(noisy_objective, n_trials=30)
            
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
    
    def test_hyperparameter_optimization_convergence(self):
        """Test hyperparameter optimization convergence."""
        def objective_function(params):
            # Simple objective function for convergence testing
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            result = self.optimizer.bayesian_optimization(objective_function, n_trials=100)
            
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
            
            # Check that optimization history is non-empty
            self.assertGreater(len(result['optimization_history']), 0)
    
    def test_hyperparameter_optimization_parallel(self):
        """Test parallel hyperparameter optimization."""
        def objective_function(params):
            # Objective function for parallel testing
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        # Test multiple optimization runs in parallel
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            # Run multiple optimizations
            results = []
            for _ in range(3):
                result = self.optimizer.bayesian_optimization(objective_function, n_trials=20)
                results.append(result)
            
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIn('best_params', result)
                self.assertIn('best_value', result)
                self.assertIn('optimization_history', result)

class TestHyperparameterOptimizationIntegration(unittest.TestCase):
    """Integration tests for hyperparameter optimization."""
    
    def setUp(self):
        self.search_space = HyperparameterSpace()
        self.optimizer = HyperparameterOptimizer(self.search_space)
    
    def test_hyperparameter_optimization_workflow(self):
        """Test complete hyperparameter optimization workflow."""
        def objective_function(params):
            # Simulate model training and evaluation
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            # Simulate training time
            time.sleep(0.01)
            
            # Simulate model performance
            performance = 1.0 / (1.0 + (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2)
            return -performance  # Minimize negative performance
        
        # Test Bayesian optimization
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = -0.95
            mock_study.return_value.trials = []
            
            bayesian_result = self.optimizer.bayesian_optimization(objective_function, n_trials=30)
            
            self.assertIn('best_params', bayesian_result)
            self.assertIn('best_value', bayesian_result)
            self.assertIn('optimization_history', bayesian_result)
        
        # Test TPE optimization
        with patch('hyperopt.fmin') as mock_fmin:
            mock_fmin.return_value = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            
            with patch('hyperopt.Trials') as mock_trials:
                mock_trials.return_value.trials = [{'result': {'loss': -0.95}}]
                
                tpe_result = self.optimizer.tree_structured_parzen_estimator(
                    objective_function, max_evals=30
                )
                
                self.assertIn('best_params', tpe_result)
                self.assertIn('best_value', tpe_result)
                self.assertIn('optimization_history', tpe_result)
        
        # Test Differential Evolution optimization
        with patch('scipy.optimize.differential_evolution') as mock_de:
            mock_result = Mock()
            mock_result.x = [0.001, 32, 0.2, 1e-4, 0.9]
            mock_result.fun = -0.95
            mock_de.return_value = mock_result
            
            de_result = self.optimizer.differential_evolution_optimization(
                objective_function, max_iterations=30
            )
            
            self.assertIn('best_params', de_result)
            self.assertIn('best_value', de_result)
            self.assertIn('optimization_history', de_result)
    
    def test_hyperparameter_optimization_with_model(self):
        """Test hyperparameter optimization with actual model."""
        class TestModel(nn.Module):
            def __init__(self, input_size=100, hidden_size=50, output_size=10):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        def model_objective_function(params):
            # Create model with hyperparameters
            model = TestModel()
            
            # Simulate model evaluation
            total_params = sum(p.numel() for p in model.parameters())
            memory_usage = total_params * 4 / (1024 * 1024)  # MB
            
            # Performance based on hyperparameters
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            # Simulate performance metric
            performance = 1.0 / (1.0 + (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2)
            
            # Add memory efficiency
            memory_efficiency = 1.0 / (1.0 + memory_usage / 100)
            
            return -(performance + memory_efficiency) / 2
        
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = -0.9
            mock_study.return_value.trials = []
            
            result = self.optimizer.bayesian_optimization(model_objective_function, n_trials=20)
            
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
    
    def test_hyperparameter_optimization_comparison(self):
        """Test comparison of different hyperparameter optimization methods."""
        def objective_function(params):
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        results = {}
        
        # Test Bayesian optimization
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            results['bayesian'] = self.optimizer.bayesian_optimization(objective_function, n_trials=20)
        
        # Test TPE optimization
        with patch('hyperopt.fmin') as mock_fmin:
            mock_fmin.return_value = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            
            with patch('hyperopt.Trials') as mock_trials:
                mock_trials.return_value.trials = [{'result': {'loss': 0.01}}]
                
                results['tpe'] = self.optimizer.tree_structured_parzen_estimator(
                    objective_function, max_evals=20
                )
        
        # Test Differential Evolution optimization
        with patch('scipy.optimize.differential_evolution') as mock_de:
            mock_result = Mock()
            mock_result.x = [0.001, 32, 0.2, 1e-4, 0.9]
            mock_result.fun = 0.01
            mock_de.return_value = mock_result
            
            results['differential_evolution'] = self.optimizer.differential_evolution_optimization(
                objective_function, max_iterations=20
            )
        
        # Compare results
        self.assertEqual(len(results), 3)
        for method, result in results.items():
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
    
    def test_hyperparameter_optimization_persistence(self):
        """Test hyperparameter optimization persistence."""
        def objective_function(params):
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            result = self.optimizer.bayesian_optimization(objective_function, n_trials=20)
            
            # Test result serialization
            result_json = json.dumps(result, default=str)
            self.assertIsInstance(result_json, str)
            
            # Test result deserialization
            deserialized_result = json.loads(result_json)
            self.assertIn('best_params', deserialized_result)
            self.assertIn('best_value', deserialized_result)
            self.assertIn('optimization_history', deserialized_result)

class TestHyperparameterOptimizationPerformance(unittest.TestCase):
    """Performance tests for hyperparameter optimization."""
    
    def setUp(self):
        self.search_space = HyperparameterSpace()
        self.optimizer = HyperparameterOptimizer(self.search_space)
    
    def test_hyperparameter_optimization_speed(self):
        """Test hyperparameter optimization speed."""
        def objective_function(params):
            # Simple objective function for speed testing
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            start_time = time.time()
            result = self.optimizer.bayesian_optimization(objective_function, n_trials=50)
            end_time = time.time()
            
            optimization_time = end_time - start_time
            
            # Should complete within reasonable time
            self.assertLess(optimization_time, 5.0)
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
    
    def test_hyperparameter_optimization_memory_usage(self):
        """Test hyperparameter optimization memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def objective_function(params):
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            result = self.optimizer.bayesian_optimization(objective_function, n_trials=30)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            self.assertLess(memory_increase, 100.0)
            self.assertIn('best_params', result)
            self.assertIn('best_value', result)
            self.assertIn('optimization_history', result)
    
    def test_hyperparameter_optimization_scalability(self):
        """Test hyperparameter optimization scalability."""
        def objective_function(params):
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        # Test with different numbers of trials
        trial_counts = [10, 20, 50, 100]
        
        for n_trials in trial_counts:
            with patch('optuna.create_study') as mock_study:
                mock_study.return_value.optimize.return_value = None
                mock_study.return_value.best_params = {
                    'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                    'weight_decay': 1e-4, 'momentum': 0.9
                }
                mock_study.return_value.best_value = 0.01
                mock_study.return_value.trials = []
                
                start_time = time.time()
                result = self.optimizer.bayesian_optimization(objective_function, n_trials=n_trials)
                end_time = time.time()
                
                optimization_time = end_time - start_time
                
                # Time should scale reasonably with number of trials
                self.assertLess(optimization_time, n_trials * 0.1)
                self.assertIn('best_params', result)
                self.assertIn('best_value', result)
                self.assertIn('optimization_history', result)
    
    def test_hyperparameter_optimization_concurrent(self):
        """Test concurrent hyperparameter optimization."""
        def objective_function(params):
            lr = params['learning_rate']
            bs = params['batch_size']
            dr = params['dropout_rate']
            wd = params['weight_decay']
            mom = params['momentum']
            
            return (lr - 0.001)**2 + (bs - 32)**2/1000 + (dr - 0.2)**2 + (wd - 1e-4)**2*1000 + (mom - 0.9)**2
        
        # Test multiple concurrent optimizations
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value.optimize.return_value = None
            mock_study.return_value.best_params = {
                'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.2,
                'weight_decay': 1e-4, 'momentum': 0.9
            }
            mock_study.return_value.best_value = 0.01
            mock_study.return_value.trials = []
            
            start_time = time.time()
            
            results = []
            for _ in range(3):
                result = self.optimizer.bayesian_optimization(objective_function, n_trials=20)
                results.append(result)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # All optimizations should complete
            self.assertEqual(len(results), 3)
            
            for result in results:
                self.assertIn('best_params', result)
                self.assertIn('best_value', result)
                self.assertIn('optimization_history', result)
            
            # Total time should be reasonable
            self.assertLess(total_time, 10.0)

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHyperparameterSpaceAdvanced,
        TestHyperparameterOptimizerAdvanced,
        TestHyperparameterOptimizationIntegration,
        TestHyperparameterOptimizationPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Hyperparameter Optimization Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    print(f"\n{'='*60}")

