"""
Unit tests for meta-learning optimization
Tests meta-learning algorithms, few-shot learning, and adaptive optimization
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestMetaLearningOptimization(unittest.TestCase):
    """Test suite for meta-learning optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_model_agnostic_meta_learning(self):
        """Test Model-Agnostic Meta-Learning (MAML)"""
        class MAMLOptimizer:
            def __init__(self, meta_learning_rate=0.01, inner_learning_rate=0.01, inner_steps=5):
                self.meta_learning_rate = meta_learning_rate
                self.inner_learning_rate = inner_learning_rate
                self.inner_steps = inner_steps
                self.meta_parameters = None
                self.adaptation_history = []
                self.meta_optimization_history = []
                
            def initialize_meta_parameters(self, model_template):
                """Initialize meta-parameters"""
                self.meta_parameters = {
                    'weights': np.random.uniform(-0.1, 0.1, 10),
                    'biases': np.random.uniform(-0.1, 0.1, 5)
                }
                
            def meta_learning_step(self, support_tasks, query_tasks):
                """Execute one meta-learning step"""
                # Compute meta-gradient
                meta_gradient = self._compute_meta_gradient(support_tasks, query_tasks)
                
                # Update meta-parameters
                for param_name, param_value in self.meta_parameters.items():
                    self.meta_parameters[param_name] -= self.meta_learning_rate * meta_gradient[param_name]
                    
                # Record meta-optimization step
                self.meta_optimization_history.append({
                    'step': len(self.meta_optimization_history),
                    'meta_gradient_norm': np.linalg.norm(meta_gradient['weights']),
                    'meta_parameters': self.meta_parameters.copy()
                })
                
                return self.meta_parameters
                
            def _compute_meta_gradient(self, support_tasks, query_tasks):
                """Compute meta-gradient using MAML"""
                meta_gradient = {
                    'weights': np.zeros_like(self.meta_parameters['weights']),
                    'biases': np.zeros_like(self.meta_parameters['biases'])
                }
                
                for support_task, query_task in zip(support_tasks, query_tasks):
                    # Adapt to support task
                    adapted_parameters = self._adapt_to_task(support_task)
                    
                    # Evaluate on query task
                    query_loss = self._evaluate_task(adapted_parameters, query_task)
                    
                    # Compute gradient with respect to meta-parameters
                    task_gradient = self._compute_task_gradient(adapted_parameters, query_loss)
                    
                    # Accumulate meta-gradient
                    meta_gradient['weights'] += task_gradient['weights']
                    meta_gradient['biases'] += task_gradient['biases']
                    
                # Average over tasks
                n_tasks = len(support_tasks)
                meta_gradient['weights'] /= n_tasks
                meta_gradient['biases'] /= n_tasks
                
                return meta_gradient
                
            def _adapt_to_task(self, task):
                """Adapt meta-parameters to specific task"""
                adapted_parameters = self.meta_parameters.copy()
                
                # Simulate inner loop adaptation
                for step in range(self.inner_steps):
                    # Compute task-specific gradient
                    task_gradient = self._compute_task_gradient(adapted_parameters, task)
                    
                    # Update adapted parameters
                    adapted_parameters['weights'] -= self.inner_learning_rate * task_gradient['weights']
                    adapted_parameters['biases'] -= self.inner_learning_rate * task_gradient['biases']
                    
                # Record adaptation
                self.adaptation_history.append({
                    'task_id': task.get('id', len(self.adaptation_history)),
                    'adapted_parameters': adapted_parameters.copy(),
                    'adaptation_steps': self.inner_steps
                })
                
                return adapted_parameters
                
            def _evaluate_task(self, parameters, task):
                """Evaluate parameters on task"""
                # Simulate task evaluation
                loss = np.random.uniform(0, 1)
                return loss
                
            def _compute_task_gradient(self, parameters, loss):
                """Compute gradient for task"""
                # Simulate gradient computation
                gradient = {
                    'weights': np.random.uniform(-1, 1, 10),
                    'biases': np.random.uniform(-1, 1, 5)
                }
                return gradient
                
            def get_maml_stats(self):
                """Get MAML statistics"""
                return {
                    'meta_learning_rate': self.meta_learning_rate,
                    'inner_learning_rate': self.inner_learning_rate,
                    'inner_steps': self.inner_steps,
                    'total_adaptations': len(self.adaptation_history),
                    'total_meta_steps': len(self.meta_optimization_history),
                    'avg_meta_gradient_norm': np.mean([step['meta_gradient_norm'] for step in self.meta_optimization_history]) if self.meta_optimization_history else 0
                }
        
        # Test MAML optimizer
        maml = MAMLOptimizer(meta_learning_rate=0.01, inner_learning_rate=0.01, inner_steps=5)
        model_template = nn.Linear(256, 512)
        
        # Initialize meta-parameters
        maml.initialize_meta_parameters(model_template)
        self.assertIsNotNone(maml.meta_parameters)
        
        # Test meta-learning step
        support_tasks = [
            {'id': 0, 'data': np.random.uniform(0, 1, (10, 5))},
            {'id': 1, 'data': np.random.uniform(0, 1, (10, 5))}
        ]
        query_tasks = [
            {'id': 0, 'data': np.random.uniform(0, 1, (5, 5))},
            {'id': 1, 'data': np.random.uniform(0, 1, (5, 5))}
        ]
        
        updated_meta_parameters = maml.meta_learning_step(support_tasks, query_tasks)
        
        # Verify results
        self.assertIsNotNone(updated_meta_parameters)
        self.assertEqual(len(updated_meta_parameters['weights']), 10)
        self.assertEqual(len(updated_meta_parameters['biases']), 5)
        self.assertEqual(len(maml.adaptation_history), 2)
        self.assertEqual(len(maml.meta_optimization_history), 1)
        
        # Check MAML stats
        stats = maml.get_maml_stats()
        self.assertEqual(stats['meta_learning_rate'], 0.01)
        self.assertEqual(stats['inner_learning_rate'], 0.01)
        self.assertEqual(stats['inner_steps'], 5)
        self.assertEqual(stats['total_adaptations'], 2)
        self.assertEqual(stats['total_meta_steps'], 1)
        self.assertGreater(stats['avg_meta_gradient_norm'], 0)
        
    def test_reptile_optimization(self):
        """Test Reptile optimization"""
        class ReptileOptimizer:
            def __init__(self, meta_learning_rate=0.01, inner_learning_rate=0.01, inner_steps=5):
                self.meta_learning_rate = meta_learning_rate
                self.inner_learning_rate = inner_learning_rate
                self.inner_steps = inner_steps
                self.meta_parameters = None
                self.adaptation_history = []
                self.reptile_history = []
                
            def initialize_meta_parameters(self, model_template):
                """Initialize meta-parameters"""
                self.meta_parameters = {
                    'weights': np.random.uniform(-0.1, 0.1, 10),
                    'biases': np.random.uniform(-0.1, 0.1, 5)
                }
                
            def reptile_step(self, tasks):
                """Execute one Reptile step"""
                # Adapt to each task
                adapted_parameters_list = []
                for task in tasks:
                    adapted_parameters = self._adapt_to_task(task)
                    adapted_parameters_list.append(adapted_parameters)
                    
                # Compute Reptile update
                reptile_update = self._compute_reptile_update(adapted_parameters_list)
                
                # Update meta-parameters
                for param_name, param_value in self.meta_parameters.items():
                    self.meta_parameters[param_name] += self.meta_learning_rate * reptile_update[param_name]
                    
                # Record Reptile step
                self.reptile_history.append({
                    'step': len(self.reptile_history),
                    'reptile_update_norm': np.linalg.norm(reptile_update['weights']),
                    'meta_parameters': self.meta_parameters.copy()
                })
                
                return self.meta_parameters
                
            def _adapt_to_task(self, task):
                """Adapt meta-parameters to specific task"""
                adapted_parameters = self.meta_parameters.copy()
                
                # Simulate inner loop adaptation
                for step in range(self.inner_steps):
                    # Compute task-specific gradient
                    task_gradient = self._compute_task_gradient(adapted_parameters, task)
                    
                    # Update adapted parameters
                    adapted_parameters['weights'] -= self.inner_learning_rate * task_gradient['weights']
                    adapted_parameters['biases'] -= self.inner_learning_rate * task_gradient['biases']
                    
                # Record adaptation
                self.adaptation_history.append({
                    'task_id': task.get('id', len(self.adaptation_history)),
                    'adapted_parameters': adapted_parameters.copy(),
                    'adaptation_steps': self.inner_steps
                })
                
                return adapted_parameters
                
            def _compute_reptile_update(self, adapted_parameters_list):
                """Compute Reptile update"""
                # Reptile update: average of (adapted_parameters - meta_parameters)
                reptile_update = {
                    'weights': np.zeros_like(self.meta_parameters['weights']),
                    'biases': np.zeros_like(self.meta_parameters['biases'])
                }
                
                for adapted_parameters in adapted_parameters_list:
                    reptile_update['weights'] += adapted_parameters['weights'] - self.meta_parameters['weights']
                    reptile_update['biases'] += adapted_parameters['biases'] - self.meta_parameters['biases']
                    
                # Average over tasks
                n_tasks = len(adapted_parameters_list)
                reptile_update['weights'] /= n_tasks
                reptile_update['biases'] /= n_tasks
                
                return reptile_update
                
            def _compute_task_gradient(self, parameters, task):
                """Compute gradient for task"""
                # Simulate gradient computation
                gradient = {
                    'weights': np.random.uniform(-1, 1, 10),
                    'biases': np.random.uniform(-1, 1, 5)
                }
                return gradient
                
            def get_reptile_stats(self):
                """Get Reptile statistics"""
                return {
                    'meta_learning_rate': self.meta_learning_rate,
                    'inner_learning_rate': self.inner_learning_rate,
                    'inner_steps': self.inner_steps,
                    'total_adaptations': len(self.adaptation_history),
                    'total_reptile_steps': len(self.reptile_history),
                    'avg_reptile_update_norm': np.mean([step['reptile_update_norm'] for step in self.reptile_history]) if self.reptile_history else 0
                }
        
        # Test Reptile optimizer
        reptile = ReptileOptimizer(meta_learning_rate=0.01, inner_learning_rate=0.01, inner_steps=5)
        model_template = nn.Linear(256, 512)
        
        # Initialize meta-parameters
        reptile.initialize_meta_parameters(model_template)
        self.assertIsNotNone(reptile.meta_parameters)
        
        # Test Reptile step
        tasks = [
            {'id': 0, 'data': np.random.uniform(0, 1, (10, 5))},
            {'id': 1, 'data': np.random.uniform(0, 1, (10, 5))},
            {'id': 2, 'data': np.random.uniform(0, 1, (10, 5))}
        ]
        
        updated_meta_parameters = reptile.reptile_step(tasks)
        
        # Verify results
        self.assertIsNotNone(updated_meta_parameters)
        self.assertEqual(len(updated_meta_parameters['weights']), 10)
        self.assertEqual(len(updated_meta_parameters['biases']), 5)
        self.assertEqual(len(reptile.adaptation_history), 3)
        self.assertEqual(len(reptile.reptile_history), 1)
        
        # Check Reptile stats
        stats = reptile.get_reptile_stats()
        self.assertEqual(stats['meta_learning_rate'], 0.01)
        self.assertEqual(stats['inner_learning_rate'], 0.01)
        self.assertEqual(stats['inner_steps'], 5)
        self.assertEqual(stats['total_adaptations'], 3)
        self.assertEqual(stats['total_reptile_steps'], 1)
        self.assertGreater(stats['avg_reptile_update_norm'], 0)
        
    def test_few_shot_learning(self):
        """Test few-shot learning optimization"""
        class FewShotLearningOptimizer:
            def __init__(self, support_shots=5, query_shots=15, adaptation_steps=10):
                self.support_shots = support_shots
                self.query_shots = query_shots
                self.adaptation_steps = adaptation_steps
                self.meta_parameters = None
                self.few_shot_history = []
                self.adaptation_performance = []
                
            def initialize_meta_parameters(self, model_template):
                """Initialize meta-parameters"""
                self.meta_parameters = {
                    'weights': np.random.uniform(-0.1, 0.1, 10),
                    'biases': np.random.uniform(-0.1, 0.1, 5)
                }
                
            def few_shot_adaptation(self, support_data, support_target, query_data, query_target):
                """Execute few-shot learning adaptation"""
                # Start with meta-parameters
                adapted_parameters = self.meta_parameters.copy()
                
                # Adapt to support set
                for step in range(self.adaptation_steps):
                    # Compute support set loss
                    support_loss = self._compute_loss(adapted_parameters, support_data, support_target)
                    
                    # Compute gradient
                    gradient = self._compute_gradient(adapted_parameters, support_loss)
                    
                    # Update parameters
                    adapted_parameters['weights'] -= 0.01 * gradient['weights']
                    adapted_parameters['biases'] -= 0.01 * gradient['biases']
                    
                # Evaluate on query set
                query_loss = self._compute_loss(adapted_parameters, query_data, query_target)
                query_accuracy = self._compute_accuracy(adapted_parameters, query_data, query_target)
                
                # Record few-shot learning result
                self.few_shot_history.append({
                    'support_shots': self.support_shots,
                    'query_shots': self.query_shots,
                    'adaptation_steps': self.adaptation_steps,
                    'final_support_loss': support_loss,
                    'query_loss': query_loss,
                    'query_accuracy': query_accuracy,
                    'adapted_parameters': adapted_parameters.copy()
                })
                
                return adapted_parameters, query_loss, query_accuracy
                
            def _compute_loss(self, parameters, data, target):
                """Compute loss for given parameters and data"""
                # Simulate loss computation
                loss = np.random.uniform(0, 1)
                return loss
                
            def _compute_gradient(self, parameters, loss):
                """Compute gradient for given parameters and loss"""
                # Simulate gradient computation
                gradient = {
                    'weights': np.random.uniform(-1, 1, 10),
                    'biases': np.random.uniform(-1, 1, 5)
                }
                return gradient
                
            def _compute_accuracy(self, parameters, data, target):
                """Compute accuracy for given parameters and data"""
                # Simulate accuracy computation
                accuracy = np.random.uniform(0.5, 1.0)
                return accuracy
                
            def get_few_shot_stats(self):
                """Get few-shot learning statistics"""
                if not self.few_shot_history:
                    return {}
                    
                return {
                    'support_shots': self.support_shots,
                    'query_shots': self.query_shots,
                    'adaptation_steps': self.adaptation_steps,
                    'total_adaptations': len(self.few_shot_history),
                    'avg_query_accuracy': np.mean([result['query_accuracy'] for result in self.few_shot_history]),
                    'avg_query_loss': np.mean([result['query_loss'] for result in self.few_shot_history]),
                    'best_accuracy': max([result['query_accuracy'] for result in self.few_shot_history])
                }
        
        # Test few-shot learning optimizer
        few_shot = FewShotLearningOptimizer(support_shots=5, query_shots=15, adaptation_steps=10)
        model_template = nn.Linear(256, 512)
        
        # Initialize meta-parameters
        few_shot.initialize_meta_parameters(model_template)
        self.assertIsNotNone(few_shot.meta_parameters)
        
        # Test few-shot adaptation
        support_data = np.random.uniform(0, 1, (5, 10))
        support_target = np.random.randint(0, 2, 5)
        query_data = np.random.uniform(0, 1, (15, 10))
        query_target = np.random.randint(0, 2, 15)
        
        adapted_parameters, query_loss, query_accuracy = few_shot.few_shot_adaptation(
            support_data, support_target, query_data, query_target
        )
        
        # Verify results
        self.assertIsNotNone(adapted_parameters)
        self.assertGreater(query_accuracy, 0)
        self.assertGreater(query_loss, 0)
        self.assertEqual(len(few_shot.few_shot_history), 1)
        
        # Check few-shot stats
        stats = few_shot.get_few_shot_stats()
        self.assertEqual(stats['support_shots'], 5)
        self.assertEqual(stats['query_shots'], 15)
        self.assertEqual(stats['adaptation_steps'], 10)
        self.assertEqual(stats['total_adaptations'], 1)
        self.assertGreater(stats['avg_query_accuracy'], 0)
        self.assertGreater(stats['avg_query_loss'], 0)
        self.assertGreater(stats['best_accuracy'], 0)

class TestAdaptiveOptimization(unittest.TestCase):
    """Test suite for adaptive optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate optimization"""
        class AdaptiveLearningRateOptimizer:
            def __init__(self, initial_lr=0.001, adaptation_rate=0.1):
                self.initial_lr = initial_lr
                self.adaptation_rate = adaptation_rate
                self.current_lr = initial_lr
                self.lr_history = []
                self.performance_history = []
                
            def adapt_learning_rate(self, current_performance, previous_performance):
                """Adapt learning rate based on performance"""
                # Compute performance change
                performance_change = current_performance - previous_performance
                
                # Adapt learning rate
                if performance_change > 0:
                    # Performance improved, increase learning rate
                    self.current_lr *= (1 + self.adaptation_rate)
                else:
                    # Performance decreased, decrease learning rate
                    self.current_lr *= (1 - self.adaptation_rate)
                    
                # Record learning rate and performance
                self.lr_history.append(self.current_lr)
                self.performance_history.append(current_performance)
                
                return self.current_lr
                
            def get_adaptive_stats(self):
                """Get adaptive optimization statistics"""
                if not self.lr_history:
                    return {}
                    
                return {
                    'initial_lr': self.initial_lr,
                    'current_lr': self.current_lr,
                    'lr_changes': len([lr for i, lr in enumerate(self.lr_history) 
                                     if i > 0 and lr != self.lr_history[i-1]]),
                    'avg_performance': np.mean(self.performance_history),
                    'performance_improvement': self.performance_history[-1] - self.performance_history[0] if len(self.performance_history) > 1 else 0
                }
        
        # Test adaptive learning rate optimizer
        adaptive_lr = AdaptiveLearningRateOptimizer(initial_lr=0.001, adaptation_rate=0.1)
        
        # Test learning rate adaptation
        performances = [0.5, 0.6, 0.7, 0.65, 0.8, 0.75, 0.9]
        for i in range(1, len(performances)):
            lr = adaptive_lr.adapt_learning_rate(performances[i], performances[i-1])
            self.assertGreater(lr, 0)
            
        # Check adaptive stats
        stats = adaptive_lr.get_adaptive_stats()
        self.assertEqual(stats['initial_lr'], 0.001)
        self.assertGreater(stats['current_lr'], 0)
        self.assertGreaterEqual(stats['lr_changes'], 0)
        self.assertGreater(stats['avg_performance'], 0)
        self.assertGreaterEqual(stats['performance_improvement'], 0)
        
    def test_adaptive_optimization_strategy(self):
        """Test adaptive optimization strategy"""
        class AdaptiveOptimizationStrategy:
            def __init__(self, strategies):
                self.strategies = strategies
                self.current_strategy = None
                self.strategy_history = []
                self.performance_history = []
                
            def select_strategy(self, current_performance, context):
                """Select optimization strategy based on performance and context"""
                # Select strategy based on performance and context
                if current_performance > 0.8:
                    # High performance, use aggressive strategy
                    self.current_strategy = 'aggressive'
                elif current_performance > 0.5:
                    # Medium performance, use balanced strategy
                    self.current_strategy = 'balanced'
                else:
                    # Low performance, use conservative strategy
                    self.current_strategy = 'conservative'
                    
                # Record strategy selection
                self.strategy_history.append({
                    'strategy': self.current_strategy,
                    'performance': current_performance,
                    'context': context
                })
                
                return self.current_strategy
                
            def get_strategy_stats(self):
                """Get strategy selection statistics"""
                if not self.strategy_history:
                    return {}
                    
                strategies_used = [entry['strategy'] for entry in self.strategy_history]
                strategy_counts = {strategy: strategies_used.count(strategy) for strategy in set(strategies_used)}
                
                return {
                    'total_selections': len(self.strategy_history),
                    'current_strategy': self.current_strategy,
                    'strategy_counts': strategy_counts,
                    'avg_performance': np.mean([entry['performance'] for entry in self.strategy_history])
                }
        
        # Test adaptive optimization strategy
        strategies = ['aggressive', 'balanced', 'conservative']
        adaptive_strategy = AdaptiveOptimizationStrategy(strategies)
        
        # Test strategy selection
        performances = [0.3, 0.6, 0.9, 0.7, 0.4, 0.8]
        contexts = ['initial', 'middle', 'final', 'middle', 'initial', 'final']
        
        for performance, context in zip(performances, contexts):
            strategy = adaptive_strategy.select_strategy(performance, context)
            self.assertIn(strategy, strategies)
            
        # Check strategy stats
        stats = adaptive_strategy.get_strategy_stats()
        self.assertEqual(stats['total_selections'], 6)
        self.assertIn('strategy_counts', stats)
        self.assertGreater(stats['avg_performance'], 0)

if __name__ == '__main__':
    unittest.main()





