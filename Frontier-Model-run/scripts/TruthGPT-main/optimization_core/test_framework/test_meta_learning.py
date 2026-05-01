#!/usr/bin/env python3
"""
Test Suite for Meta-Learning Optimization
Comprehensive tests for meta-learning optimization techniques and algorithms
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional, Callable
import tempfile
import os
import json
import pickle
from pathlib import Path
import random
import math

# Import meta-learning optimization components
import sys
sys.path.append('..')
from modules.base.core_system.core.advanced_optimizations import (
    MetaLearningOptimizer, AdvancedOptimizationEngine, OptimizationTechnique,
    OptimizationMetrics, create_meta_learning_optimizer
)

class TestMetaLearningOptimizerAdvanced(unittest.TestCase):
    """Advanced tests for meta-learning optimizer."""
    
    def setUp(self):
        self.meta_optimizer = MetaLearningOptimizer()
    
    def test_meta_optimizer_initialization_advanced(self):
        """Test advanced meta-learning optimizer initialization."""
        self.assertIsInstance(self.meta_optimizer.config, dict)
        self.assertEqual(len(self.meta_optimizer.meta_parameters), 0)
        self.assertEqual(len(self.meta_optimizer.adaptation_history), 0)
        self.assertIsInstance(self.meta_optimizer.logger, logging.Logger)
    
    def test_meta_optimization_advanced(self):
        """Test advanced meta-learning optimization."""
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
        
        model = TestModel()
        
        # Create diverse adaptation tasks
        adaptation_tasks = [
            {
                'adaptation_steps': 5,
                'task_type': 'classification',
                'dataset': 'mnist',
                'learning_rate': 0.001,
                'batch_size': 32
            },
            {
                'adaptation_steps': 3,
                'task_type': 'regression',
                'dataset': 'boston',
                'learning_rate': 0.01,
                'batch_size': 16
            },
            {
                'adaptation_steps': 7,
                'task_type': 'classification',
                'dataset': 'cifar10',
                'learning_rate': 0.0001,
                'batch_size': 64
            }
        ]
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)
        self.assertEqual(type(meta_optimized_model), type(model))
    
    def test_meta_parameters_initialization_advanced(self):
        """Test advanced meta-parameters initialization."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        self.meta_optimizer._initialize_meta_parameters(model)
        
        self.assertGreater(len(self.meta_optimizer.meta_parameters), 0)
        
        for name, params in self.meta_optimizer.meta_parameters.items():
            self.assertIn('learning_rate', params)
            self.assertIn('momentum', params)
            self.assertIn('adaptation_rate', params)
            
            # Test parameter types and ranges
            self.assertIsInstance(params['learning_rate'], float)
            self.assertIsInstance(params['momentum'], float)
            self.assertIsInstance(params['adaptation_rate'], float)
            
            self.assertGreater(params['learning_rate'], 0.0)
            self.assertGreaterEqual(params['momentum'], 0.0)
            self.assertLess(params['momentum'], 1.0)
            self.assertGreater(params['adaptation_rate'], 0.0)
    
    def test_fast_adaptation_advanced(self):
        """Test advanced fast adaptation."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        
        # Test different task types
        task_types = [
            {
                'adaptation_steps': 3,
                'task_type': 'classification',
                'learning_rate': 0.001
            },
            {
                'adaptation_steps': 5,
                'task_type': 'regression',
                'learning_rate': 0.01
            },
            {
                'adaptation_steps': 7,
                'task_type': 'classification',
                'learning_rate': 0.0001
            }
        ]
        
        for task in task_types:
            adapted_model = self.meta_optimizer._fast_adapt(model, task)
            
            self.assertIsInstance(adapted_model, nn.Module)
            self.assertEqual(type(adapted_model), type(model))
    
    def test_meta_parameters_update_advanced(self):
        """Test advanced meta-parameters update."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        self.meta_optimizer._initialize_meta_parameters(model)
        
        # Test different task types
        tasks = [
            {
                'adaptation_steps': 3,
                'task_type': 'classification',
                'performance': 0.85
            },
            {
                'adaptation_steps': 5,
                'task_type': 'regression',
                'performance': 0.92
            }
        ]
        
        for task in tasks:
            # This should not raise an exception
            self.meta_optimizer._update_meta_parameters(model, task)
            
            # Test that meta-parameters are updated
            for name, params in self.meta_optimizer.meta_parameters.items():
                self.assertIn('learning_rate', params)
                self.assertIn('momentum', params)
                self.assertIn('adaptation_rate', params)
    
    def test_meta_parameters_application_advanced(self):
        """Test advanced meta-parameters application."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        self.meta_optimizer._initialize_meta_parameters(model)
        
        applied_model = self.meta_optimizer._apply_meta_parameters(model)
        
        self.assertIsInstance(applied_model, nn.Module)
        self.assertEqual(type(applied_model), type(model))
    
    def test_adaptation_history_tracking(self):
        """Test adaptation history tracking."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        adaptation_tasks = [
            {'adaptation_steps': 3, 'task_type': 'classification'},
            {'adaptation_steps': 5, 'task_type': 'regression'},
            {'adaptation_steps': 7, 'task_type': 'classification'}
        ]
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        # Test that adaptation history is populated
        self.assertGreater(len(self.meta_optimizer.adaptation_history), 0)
        self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_convergence(self):
        """Test meta-learning convergence."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create tasks with increasing complexity
        adaptation_tasks = []
        for i in range(10):
            task = {
                'adaptation_steps': 3 + i,
                'task_type': 'classification',
                'complexity': i * 0.1
            }
            adaptation_tasks.append(task)
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)
        
        # Test that adaptation history shows convergence
        if len(self.meta_optimizer.adaptation_history) > 1:
            # Adaptation history should show improvement over time
            self.assertGreaterEqual(
                len(self.meta_optimizer.adaptation_history),
                len(adaptation_tasks)
            )
    
    def test_meta_learning_with_different_models(self):
        """Test meta-learning with different model types."""
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.fc = nn.Linear(64 * 6 * 6, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        models = [LinearModel(), ConvModel()]
        
        for model in models:
            adaptation_tasks = [
                {'adaptation_steps': 3, 'task_type': 'classification'},
                {'adaptation_steps': 5, 'task_type': 'regression'}
            ]
            
            meta_optimized_model = self.meta_optimizer.meta_optimize(
                model, adaptation_tasks, meta_learning_rate=0.01
            )
            
            self.assertIsInstance(meta_optimized_model, nn.Module)
            self.assertEqual(type(meta_optimized_model), type(model))

class TestMetaLearningOptimizationIntegration(unittest.TestCase):
    """Integration tests for meta-learning optimization."""
    
    def setUp(self):
        self.meta_optimizer = MetaLearningOptimizer()
    
    def test_meta_learning_workflow_complete(self):
        """Test complete meta-learning workflow."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        
        # Step 1: Initialize meta-parameters
        self.meta_optimizer._initialize_meta_parameters(model)
        self.assertGreater(len(self.meta_optimizer.meta_parameters), 0)
        
        # Step 2: Create adaptation tasks
        adaptation_tasks = [
            {
                'adaptation_steps': 3,
                'task_type': 'classification',
                'dataset': 'mnist',
                'learning_rate': 0.001
            },
            {
                'adaptation_steps': 5,
                'task_type': 'regression',
                'dataset': 'boston',
                'learning_rate': 0.01
            }
        ]
        
        # Step 3: Fast adaptation to tasks
        adapted_models = []
        for task in adaptation_tasks:
            adapted_model = self.meta_optimizer._fast_adapt(model, task)
            adapted_models.append(adapted_model)
            self.assertIsInstance(adapted_model, nn.Module)
        
        # Step 4: Update meta-parameters
        for task in adaptation_tasks:
            self.meta_optimizer._update_meta_parameters(model, task)
        
        # Step 5: Apply meta-parameters
        meta_optimized_model = self.meta_optimizer._apply_meta_parameters(model)
        self.assertIsInstance(meta_optimized_model, nn.Module)
        
        # Step 6: Run complete meta-optimization
        final_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(final_model, nn.Module)
        self.assertEqual(type(final_model), type(model))
    
    def test_meta_learning_with_advanced_engine(self):
        """Test meta-learning with advanced optimization engine."""
        engine = AdvancedOptimizationEngine()
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        adaptation_tasks = [
            {'adaptation_steps': 3, 'task_type': 'classification'},
            {'adaptation_steps': 5, 'task_type': 'regression'}
        ]
        
        optimized_model, metrics = engine.optimize_model_advanced(
            model, OptimizationTechnique.META_LEARNING,
            adaptation_tasks=adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(optimized_model, nn.Module)
        self.assertIsInstance(metrics, OptimizationMetrics)
        self.assertEqual(metrics.technique, OptimizationTechnique.META_LEARNING)
    
    def test_meta_learning_performance_tracking(self):
        """Test meta-learning performance tracking."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        adaptation_tasks = [
            {'adaptation_steps': 3, 'task_type': 'classification'},
            {'adaptation_steps': 5, 'task_type': 'regression'},
            {'adaptation_steps': 7, 'task_type': 'classification'}
        ]
        
        # Run meta-learning with performance tracking
        start_time = time.time()
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        end_time = time.time()
        
        meta_learning_time = end_time - start_time
        
        # Verify performance tracking
        self.assertLess(meta_learning_time, 10.0)  # Should complete within 10 seconds
        self.assertIsInstance(meta_optimized_model, nn.Module)
        self.assertGreater(len(self.meta_optimizer.adaptation_history), 0)
    
    def test_meta_learning_memory_usage(self):
        """Test meta-learning memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        adaptation_tasks = [
            {'adaptation_steps': 3, 'task_type': 'classification'},
            {'adaptation_steps': 5, 'task_type': 'regression'}
        ]
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        self.assertLess(memory_increase, 50.0)
        self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_with_constraints(self):
        """Test meta-learning with constraints."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create constrained adaptation tasks
        adaptation_tasks = [
            {
                'adaptation_steps': 3,
                'task_type': 'classification',
                'max_parameters': 10000,  # Constraint
                'learning_rate': 0.001
            },
            {
                'adaptation_steps': 5,
                'task_type': 'regression',
                'max_parameters': 5000,  # Constraint
                'learning_rate': 0.01
            }
        ]
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)
        
        # Test that constraints are satisfied
        param_count = sum(p.numel() for p in meta_optimized_model.parameters())
        self.assertLessEqual(param_count, 10000)  # Should satisfy max constraint
    
    def test_meta_learning_with_noise(self):
        """Test meta-learning with noisy adaptation tasks."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create noisy adaptation tasks
        adaptation_tasks = []
        for i in range(5):
            task = {
                'adaptation_steps': 3 + i,
                'task_type': 'classification',
                'noise_level': random.random() * 0.1,  # Random noise
                'learning_rate': 0.001 + random.random() * 0.01
            }
            adaptation_tasks.append(task)
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_with_multiple_objectives(self):
        """Test meta-learning with multiple objectives."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create tasks with multiple objectives
        adaptation_tasks = [
            {
                'adaptation_steps': 3,
                'task_type': 'classification',
                'objectives': ['accuracy', 'efficiency', 'robustness'],
                'weights': [0.4, 0.3, 0.3]
            },
            {
                'adaptation_steps': 5,
                'task_type': 'regression',
                'objectives': ['mse', 'speed', 'memory'],
                'weights': [0.5, 0.3, 0.2]
            }
        ]
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)

class TestMetaLearningOptimizationPerformance(unittest.TestCase):
    """Performance tests for meta-learning optimization."""
    
    def setUp(self):
        self.meta_optimizer = MetaLearningOptimizer()
    
    def test_meta_learning_speed(self):
        """Test meta-learning speed."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        adaptation_tasks = [
            {'adaptation_steps': 3, 'task_type': 'classification'},
            {'adaptation_steps': 5, 'task_type': 'regression'}
        ]
        
        start_time = time.time()
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        end_time = time.time()
        
        meta_learning_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(meta_learning_time, 5.0)
        self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_scalability(self):
        """Test meta-learning scalability with different numbers of tasks."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Test with different numbers of tasks
        task_counts = [1, 3, 5, 10]
        
        for task_count in task_counts:
            adaptation_tasks = []
            for i in range(task_count):
                task = {
                    'adaptation_steps': 3 + i,
                    'task_type': 'classification',
                    'learning_rate': 0.001
                }
                adaptation_tasks.append(task)
            
            start_time = time.time()
            
            meta_optimized_model = self.meta_optimizer.meta_optimize(
                model, adaptation_tasks, meta_learning_rate=0.01
            )
            
            end_time = time.time()
            meta_learning_time = end_time - start_time
            
            # Time should scale reasonably with number of tasks
            self.assertLess(meta_learning_time, task_count * 0.5)
            self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_concurrent(self):
        """Test concurrent meta-learning optimization."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Test multiple concurrent meta-learning runs
        start_time = time.time()
        
        results = []
        for _ in range(3):
            adaptation_tasks = [
                {'adaptation_steps': 3, 'task_type': 'classification'},
                {'adaptation_steps': 5, 'task_type': 'regression'}
            ]
            
            meta_optimized_model = self.meta_optimizer.meta_optimize(
                model, adaptation_tasks, meta_learning_rate=0.01
            )
            results.append(meta_optimized_model)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All meta-learning runs should complete
        self.assertEqual(len(results), 3)
        
        for meta_optimized_model in results:
            self.assertIsInstance(meta_optimized_model, nn.Module)
        
        # Total time should be reasonable
        self.assertLess(total_time, 10.0)
    
    def test_meta_learning_memory_efficiency(self):
        """Test meta-learning memory efficiency."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1000, 1000)  # Larger model
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        adaptation_tasks = [
            {'adaptation_steps': 3, 'task_type': 'classification'},
            {'adaptation_steps': 5, 'task_type': 'regression'}
        ]
        
        # Test memory efficiency with larger models
        start_time = time.time()
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        end_time = time.time()
        
        meta_learning_time = end_time - start_time
        
        # Should complete within reasonable time even with larger models
        self.assertLess(meta_learning_time, 10.0)
        self.assertIsInstance(meta_optimized_model, nn.Module)

class TestMetaLearningOptimizationAdvanced(unittest.TestCase):
    """Advanced tests for meta-learning optimization."""
    
    def setUp(self):
        self.meta_optimizer = MetaLearningOptimizer()
    
    def test_meta_learning_with_adaptive_parameters(self):
        """Test meta-learning with adaptive parameters."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create tasks with adaptive parameters
        adaptation_tasks = []
        for i in range(5):
            task = {
                'adaptation_steps': 3 + i,
                'task_type': 'classification',
                'learning_rate': 0.001 * (1.0 + i * 0.1),  # Adaptive learning rate
                'momentum': 0.9 * (1.0 - i * 0.05)  # Adaptive momentum
            }
            adaptation_tasks.append(task)
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_with_curriculum(self):
        """Test meta-learning with curriculum learning."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create curriculum tasks (increasing difficulty)
        adaptation_tasks = []
        for i in range(5):
            task = {
                'adaptation_steps': 3 + i,
                'task_type': 'classification',
                'difficulty': i * 0.2,  # Increasing difficulty
                'learning_rate': 0.001 / (1.0 + i * 0.1)  # Decreasing learning rate
            }
            adaptation_tasks.append(task)
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_with_transfer(self):
        """Test meta-learning with transfer learning."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create transfer learning tasks
        adaptation_tasks = [
            {
                'adaptation_steps': 3,
                'task_type': 'classification',
                'source_domain': 'mnist',
                'target_domain': 'cifar10',
                'transfer_strength': 0.8
            },
            {
                'adaptation_steps': 5,
                'task_type': 'regression',
                'source_domain': 'boston',
                'target_domain': 'california',
                'transfer_strength': 0.6
            }
        ]
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_with_continual_learning(self):
        """Test meta-learning with continual learning."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create continual learning tasks
        adaptation_tasks = []
        for i in range(5):
            task = {
                'adaptation_steps': 3 + i,
                'task_type': 'classification',
                'task_id': i,
                'forgetting_factor': 0.1 * i,  # Increasing forgetting
                'learning_rate': 0.001 / (1.0 + i * 0.1)
            }
            adaptation_tasks.append(task)
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)
    
    def test_meta_learning_with_uncertainty(self):
        """Test meta-learning with uncertainty quantification."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Create tasks with uncertainty
        adaptation_tasks = []
        for i in range(5):
            task = {
                'adaptation_steps': 3 + i,
                'task_type': 'classification',
                'uncertainty': random.random() * 0.5,  # Random uncertainty
                'confidence': 1.0 - random.random() * 0.3  # Random confidence
            }
            adaptation_tasks.append(task)
        
        meta_optimized_model = self.meta_optimizer.meta_optimize(
            model, adaptation_tasks, meta_learning_rate=0.01
        )
        
        self.assertIsInstance(meta_optimized_model, nn.Module)

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMetaLearningOptimizerAdvanced,
        TestMetaLearningOptimizationIntegration,
        TestMetaLearningOptimizationPerformance,
        TestMetaLearningOptimizationAdvanced
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Meta-Learning Optimization Test Results")
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

