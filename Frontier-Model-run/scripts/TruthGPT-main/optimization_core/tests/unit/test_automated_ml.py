"""
Unit tests for automated machine learning (AutoML)
Tests automated model selection, feature engineering, and pipeline optimization
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

class TestAutomatedModelSelection(unittest.TestCase):
    """Test suite for automated model selection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_automated_model_selection(self):
        """Test automated model selection"""
        class AutomatedModelSelector:
            def __init__(self, model_candidates):
                self.model_candidates = model_candidates
                self.evaluation_results = []
                self.best_model = None
                self.best_score = float('inf')
                
            def evaluate_model(self, model, data, target):
                """Evaluate model performance"""
                # Simulate model evaluation
                score = np.random.uniform(0, 1)
                
                result = {
                    'model_name': model['name'],
                    'model_config': model['config'],
                    'score': score,
                    'timestamp': len(self.evaluation_results)
                }
                
                self.evaluation_results.append(result)
                
                # Update best model
                if score < self.best_score:
                    self.best_score = score
                    self.best_model = model.copy()
                    
                return score
                
            def select_best_model(self, data, target):
                """Select best model from candidates"""
                for model in self.model_candidates:
                    score = self.evaluate_model(model, data, target)
                    
                return self.best_model, self.best_score
                
            def get_selection_stats(self):
                """Get model selection statistics"""
                if not self.evaluation_results:
                    return {}
                    
                scores = [result['score'] for result in self.evaluation_results]
                model_names = [result['model_name'] for result in self.evaluation_results]
                
                return {
                    'total_models': len(self.evaluation_results),
                    'best_score': self.best_score,
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'model_rankings': self._rank_models(),
                    'improvement': max(scores) - self.best_score
                }
                
            def _rank_models(self):
                """Rank models by performance"""
                sorted_results = sorted(self.evaluation_results, key=lambda x: x['score'])
                return [(result['model_name'], result['score']) for result in sorted_results]
        
        # Test automated model selection
        model_candidates = [
            {'name': 'linear', 'config': {'input_size': 256, 'output_size': 512}},
            {'name': 'mlp', 'config': {'input_size': 256, 'hidden_size': 512, 'output_size': 512}},
            {'name': 'transformer', 'config': {'d_model': 256, 'n_heads': 8, 'n_layers': 6}},
            {'name': 'cnn', 'config': {'input_channels': 1, 'output_channels': 64, 'kernel_size': 3}}
        ]
        
        selector = AutomatedModelSelector(model_candidates)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test model selection
        best_model, best_score = selector.select_best_model(data, target)
        
        # Verify results
        self.assertIsNotNone(best_model)
        self.assertGreater(best_score, 0)
        self.assertIn('name', best_model)
        self.assertIn('config', best_model)
        
        # Check selection stats
        stats = selector.get_selection_stats()
        self.assertEqual(stats['total_models'], 4)
        self.assertGreater(stats['best_score'], 0)
        self.assertEqual(len(stats['model_rankings']), 4)
        
    def test_ensemble_model_selection(self):
        """Test ensemble model selection"""
        class EnsembleModelSelector:
            def __init__(self, base_models, ensemble_strategies):
                self.base_models = base_models
                self.ensemble_strategies = ensemble_strategies
                self.ensemble_results = []
                self.best_ensemble = None
                self.best_score = float('inf')
                
            def create_ensemble(self, models, strategy):
                """Create ensemble from base models"""
                ensemble = {
                    'models': models,
                    'strategy': strategy,
                    'weights': self._compute_weights(models, strategy)
                }
                return ensemble
                
            def _compute_weights(self, models, strategy):
                """Compute ensemble weights"""
                if strategy == 'uniform':
                    return [1.0 / len(models)] * len(models)
                elif strategy == 'performance_based':
                    # Simulate performance-based weights
                    weights = np.random.uniform(0, 1, len(models))
                    return weights / weights.sum()
                else:
                    return [1.0 / len(models)] * len(models)
                    
            def evaluate_ensemble(self, ensemble, data, target):
                """Evaluate ensemble performance"""
                # Simulate ensemble evaluation
                score = np.random.uniform(0, 1)
                
                result = {
                    'ensemble': ensemble,
                    'score': score,
                    'timestamp': len(self.ensemble_results)
                }
                
                self.ensemble_results.append(result)
                
                # Update best ensemble
                if score < self.best_score:
                    self.best_score = score
                    self.best_ensemble = ensemble.copy()
                    
                return score
                
            def select_best_ensemble(self, data, target):
                """Select best ensemble configuration"""
                for strategy in self.ensemble_strategies:
                    for model_combination in itertools.combinations(self.base_models, 2):
                        ensemble = self.create_ensemble(list(model_combination), strategy)
                        score = self.evaluate_ensemble(ensemble, data, target)
                        
                return self.best_ensemble, self.best_score
                
            def get_ensemble_stats(self):
                """Get ensemble selection statistics"""
                if not self.ensemble_results:
                    return {}
                    
                scores = [result['score'] for result in self.ensemble_results]
                strategies = [result['ensemble']['strategy'] for result in self.ensemble_results]
                
                return {
                    'total_ensembles': len(self.ensemble_results),
                    'best_score': self.best_score,
                    'average_score': np.mean(scores),
                    'strategy_performance': self._analyze_strategy_performance(),
                    'improvement': max(scores) - self.best_score
                }
                
            def _analyze_strategy_performance(self):
                """Analyze performance by strategy"""
                strategy_scores = {}
                for result in self.ensemble_results:
                    strategy = result['ensemble']['strategy']
                    if strategy not in strategy_scores:
                        strategy_scores[strategy] = []
                    strategy_scores[strategy].append(result['score'])
                    
                return {strategy: np.mean(scores) for strategy, scores in strategy_scores.items()}
        
        # Test ensemble model selection
        base_models = ['linear', 'mlp', 'transformer', 'cnn']
        ensemble_strategies = ['uniform', 'performance_based']
        
        ensemble_selector = EnsembleModelSelector(base_models, ensemble_strategies)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test ensemble selection
        best_ensemble, best_score = ensemble_selector.select_best_ensemble(data, target)
        
        # Verify results
        self.assertIsNotNone(best_ensemble)
        self.assertGreater(best_score, 0)
        self.assertIn('models', best_ensemble)
        self.assertIn('strategy', best_ensemble)
        self.assertIn('weights', best_ensemble)
        
        # Check ensemble stats
        stats = ensemble_selector.get_ensemble_stats()
        self.assertGreater(stats['total_ensembles'], 0)
        self.assertGreater(stats['best_score'], 0)
        self.assertGreater(len(stats['strategy_performance']), 0)

class TestAutomatedFeatureEngineering(unittest.TestCase):
    """Test suite for automated feature engineering"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_feature_engineering_pipeline(self):
        """Test automated feature engineering pipeline"""
        class FeatureEngineeringPipeline:
            def __init__(self, feature_engineering_steps):
                self.feature_engineering_steps = feature_engineering_steps
                self.engineered_features = []
                self.feature_importance = {}
                
            def apply_feature_engineering(self, data):
                """Apply feature engineering steps"""
                current_data = data
                
                for step_name, step_func in self.feature_engineering_steps.items():
                    # Apply feature engineering step
                    engineered_data = step_func(current_data)
                    
                    # Record engineered features
                    self.engineered_features.append({
                        'step': step_name,
                        'input_shape': current_data.shape,
                        'output_shape': engineered_data.shape,
                        'feature_count': engineered_data.shape[-1]
                    })
                    
                    current_data = engineered_data
                    
                return current_data
                
            def compute_feature_importance(self, features, target):
                """Compute feature importance"""
                # Simulate feature importance computation
                n_features = features.shape[-1]
                importance = np.random.uniform(0, 1, n_features)
                importance = importance / importance.sum()  # Normalize
                
                self.feature_importance = {
                    'importance_scores': importance,
                    'top_features': np.argsort(importance)[-5:],  # Top 5 features
                    'total_features': n_features
                }
                
                return importance
                
            def select_top_features(self, features, target, top_k=10):
                """Select top-k most important features"""
                importance = self.compute_feature_importance(features, target)
                top_indices = np.argsort(importance)[-top_k:]
                return features[:, :, top_indices]
                
            def get_feature_engineering_stats(self):
                """Get feature engineering statistics"""
                if not self.engineered_features:
                    return {}
                    
                return {
                    'total_steps': len(self.engineered_features),
                    'final_feature_count': self.engineered_features[-1]['feature_count'],
                    'feature_importance': self.feature_importance,
                    'step_details': self.engineered_features
                }
        
        # Test feature engineering pipeline
        def normalize_features(data):
            """Normalize features"""
            return (data - data.mean()) / (data.std() + 1e-8)
            
        def add_polynomial_features(data):
            """Add polynomial features"""
            return torch.cat([data, data**2], dim=-1)
            
        def add_interaction_features(data):
            """Add interaction features"""
            # Simple interaction: multiply first two features
            if data.shape[-1] >= 2:
                interaction = data[:, :, 0:1] * data[:, :, 1:2]
                return torch.cat([data, interaction], dim=-1)
            return data
        
        feature_steps = {
            'normalize': normalize_features,
            'polynomial': add_polynomial_features,
            'interaction': add_interaction_features
        }
        
        pipeline = FeatureEngineeringPipeline(feature_steps)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test feature engineering
        engineered_data = pipeline.apply_feature_engineering(data)
        
        # Verify results
        self.assertGreater(engineered_data.shape[-1], data.shape[-1])
        self.assertEqual(engineered_data.shape[0], data.shape[0])
        self.assertEqual(engineered_data.shape[1], data.shape[1])
        
        # Test feature selection
        top_features = pipeline.select_top_features(engineered_data, target, top_k=10)
        self.assertEqual(top_features.shape[-1], 10)
        
        # Check feature engineering stats
        stats = pipeline.get_feature_engineering_stats()
        self.assertEqual(stats['total_steps'], 3)
        self.assertGreater(stats['final_feature_count'], 256)
        self.assertIn('feature_importance', stats)
        
    def test_automated_feature_selection(self):
        """Test automated feature selection"""
        class AutomatedFeatureSelector:
            def __init__(self, selection_methods):
                self.selection_methods = selection_methods
                self.selection_results = []
                self.selected_features = None
                
            def select_features(self, features, target, method='correlation'):
                """Select features using specified method"""
                if method == 'correlation':
                    selected_indices = self._correlation_selection(features, target)
                elif method == 'mutual_information':
                    selected_indices = self._mutual_information_selection(features, target)
                elif method == 'recursive_elimination':
                    selected_indices = self._recursive_elimination_selection(features, target)
                else:
                    selected_indices = self._random_selection(features, target)
                    
                selected_features = features[:, :, selected_indices]
                
                result = {
                    'method': method,
                    'selected_indices': selected_indices,
                    'feature_count': selected_features.shape[-1],
                    'timestamp': len(self.selection_results)
                }
                
                self.selection_results.append(result)
                self.selected_features = selected_features
                
                return selected_features
                
            def _correlation_selection(self, features, target, threshold=0.1):
                """Select features based on correlation"""
                # Simulate correlation-based selection
                n_features = features.shape[-1]
                correlations = np.random.uniform(0, 1, n_features)
                selected_indices = np.where(correlations > threshold)[0]
                return selected_indices
                
            def _mutual_information_selection(self, features, target, top_k=20):
                """Select features based on mutual information"""
                # Simulate mutual information-based selection
                n_features = features.shape[-1]
                mi_scores = np.random.uniform(0, 1, n_features)
                selected_indices = np.argsort(mi_scores)[-top_k:]
                return selected_indices
                
            def _recursive_elimination_selection(self, features, target, n_features=15):
                """Select features using recursive elimination"""
                # Simulate recursive elimination
                n_total = features.shape[-1]
                selected_indices = np.random.choice(n_total, n_features, replace=False)
                return selected_indices
                
            def _random_selection(self, features, target, n_features=10):
                """Random feature selection"""
                n_total = features.shape[-1]
                selected_indices = np.random.choice(n_total, n_features, replace=False)
                return selected_indices
                
            def get_selection_stats(self):
                """Get feature selection statistics"""
                if not self.selection_results:
                    return {}
                    
                return {
                    'total_selections': len(self.selection_results),
                    'methods_used': [result['method'] for result in self.selection_results],
                    'feature_counts': [result['feature_count'] for result in self.selection_results]
                }
        
        # Test automated feature selection
        selection_methods = ['correlation', 'mutual_information', 'recursive_elimination', 'random']
        
        selector = AutomatedFeatureSelector(selection_methods)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test different selection methods
        for method in selection_methods:
            selected_features = selector.select_features(data, target, method=method)
            self.assertLessEqual(selected_features.shape[-1], data.shape[-1])
            self.assertEqual(selected_features.shape[0], data.shape[0])
            self.assertEqual(selected_features.shape[1], data.shape[1])
            
        # Check selection stats
        stats = selector.get_selection_stats()
        self.assertEqual(stats['total_selections'], 4)
        self.assertEqual(len(stats['methods_used']), 4)
        self.assertEqual(len(stats['feature_counts']), 4)

class TestAutomatedPipelineOptimization(unittest.TestCase):
    """Test suite for automated pipeline optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_automated_pipeline_optimization(self):
        """Test automated pipeline optimization"""
        class AutomatedPipelineOptimizer:
            def __init__(self, pipeline_components):
                self.pipeline_components = pipeline_components
                self.optimization_history = []
                self.best_pipeline = None
                self.best_score = float('inf')
                
            def optimize_pipeline(self, data, target, optimization_steps=5):
                """Optimize complete pipeline"""
                for step in range(optimization_steps):
                    # Generate pipeline configuration
                    pipeline_config = self._generate_pipeline_config()
                    
                    # Evaluate pipeline
                    score = self._evaluate_pipeline(pipeline_config, data, target)
                    
                    # Record optimization step
                    result = {
                        'step': step,
                        'pipeline_config': pipeline_config,
                        'score': score,
                        'timestamp': len(self.optimization_history)
                    }
                    
                    self.optimization_history.append(result)
                    
                    # Update best pipeline
                    if score < self.best_score:
                        self.best_score = score
                        self.best_pipeline = pipeline_config.copy()
                        
                return self.best_pipeline, self.best_score
                
            def _generate_pipeline_config(self):
                """Generate random pipeline configuration"""
                config = {}
                for component_name, options in self.pipeline_components.items():
                    config[component_name] = np.random.choice(options)
                return config
                
            def _evaluate_pipeline(self, config, data, target):
                """Evaluate pipeline configuration"""
                # Simulate pipeline evaluation
                score = np.random.uniform(0, 1)
                return score
                
            def get_optimization_stats(self):
                """Get optimization statistics"""
                if not self.optimization_history:
                    return {}
                    
                scores = [result['score'] for result in self.optimization_history]
                
                return {
                    'total_steps': len(self.optimization_history),
                    'best_score': self.best_score,
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'improvement': max(scores) - self.best_score,
                    'convergence': self._check_convergence()
                }
                
            def _check_convergence(self):
                """Check if optimization has converged"""
                if len(self.optimization_history) < 3:
                    return False
                    
                recent_scores = [result['score'] for result in self.optimization_history[-3:]]
                return np.std(recent_scores) < 0.01
        
        # Test automated pipeline optimization
        pipeline_components = {
            'preprocessing': ['normalize', 'standardize', 'minmax'],
            'feature_engineering': ['polynomial', 'interaction', 'pca'],
            'model': ['linear', 'mlp', 'transformer'],
            'optimization': ['adam', 'sgd', 'rmsprop']
        }
        
        optimizer = AutomatedPipelineOptimizer(pipeline_components)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test pipeline optimization
        best_pipeline, best_score = optimizer.optimize_pipeline(data, target, optimization_steps=5)
        
        # Verify results
        self.assertIsNotNone(best_pipeline)
        self.assertGreater(best_score, 0)
        self.assertIn('preprocessing', best_pipeline)
        self.assertIn('feature_engineering', best_pipeline)
        self.assertIn('model', best_pipeline)
        self.assertIn('optimization', best_pipeline)
        
        # Check optimization stats
        stats = optimizer.get_optimization_stats()
        self.assertEqual(stats['total_steps'], 5)
        self.assertGreater(stats['best_score'], 0)
        self.assertGreaterEqual(stats['improvement'], 0)
        
    def test_automated_hyperparameter_tuning(self):
        """Test automated hyperparameter tuning"""
        class AutomatedHyperparameterTuner:
            def __init__(self, hyperparameter_space):
                self.hyperparameter_space = hyperparameter_space
                self.tuning_history = []
                self.best_hyperparameters = None
                self.best_score = float('inf')
                
            def tune_hyperparameters(self, model, data, target, tuning_steps=10):
                """Tune hyperparameters automatically"""
                for step in range(tuning_steps):
                    # Sample hyperparameters
                    hyperparameters = self._sample_hyperparameters()
                    
                    # Evaluate hyperparameters
                    score = self._evaluate_hyperparameters(hyperparameters, model, data, target)
                    
                    # Record tuning step
                    result = {
                        'step': step,
                        'hyperparameters': hyperparameters,
                        'score': score,
                        'timestamp': len(self.tuning_history)
                    }
                    
                    self.tuning_history.append(result)
                    
                    # Update best hyperparameters
                    if score < self.best_score:
                        self.best_score = score
                        self.best_hyperparameters = hyperparameters.copy()
                        
                return self.best_hyperparameters, self.best_score
                
            def _sample_hyperparameters(self):
                """Sample hyperparameters from space"""
                hyperparameters = {}
                for param_name, param_config in self.hyperparameter_space.items():
                    if param_config['type'] == 'uniform':
                        hyperparameters[param_name] = np.random.uniform(
                            param_config['low'], param_config['high']
                        )
                    elif param_config['type'] == 'choice':
                        hyperparameters[param_name] = np.random.choice(param_config['choices'])
                    elif param_config['type'] == 'int_uniform':
                        hyperparameters[param_name] = np.random.randint(
                            param_config['low'], param_config['high'] + 1
                        )
                return hyperparameters
                
            def _evaluate_hyperparameters(self, hyperparameters, model, data, target):
                """Evaluate hyperparameters"""
                # Simulate hyperparameter evaluation
                score = np.random.uniform(0, 1)
                return score
                
            def get_tuning_stats(self):
                """Get hyperparameter tuning statistics"""
                if not self.tuning_history:
                    return {}
                    
                scores = [result['score'] for result in self.tuning_history]
                
                return {
                    'total_steps': len(self.tuning_history),
                    'best_score': self.best_score,
                    'average_score': np.mean(scores),
                    'score_std': np.std(scores),
                    'improvement': max(scores) - self.best_score
                }
        
        # Test automated hyperparameter tuning
        hyperparameter_space = {
            'learning_rate': {'type': 'uniform', 'low': 0.0001, 'high': 0.1},
            'batch_size': {'type': 'int_uniform', 'low': 16, 'high': 128},
            'dropout': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
            'optimizer': {'type': 'choice', 'choices': ['adam', 'sgd', 'rmsprop']}
        }
        
        tuner = AutomatedHyperparameterTuner(hyperparameter_space)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test hyperparameter tuning
        best_hyperparameters, best_score = tuner.tune_hyperparameters(model, data, target, tuning_steps=5)
        
        # Verify results
        self.assertIsNotNone(best_hyperparameters)
        self.assertGreater(best_score, 0)
        self.assertIn('learning_rate', best_hyperparameters)
        self.assertIn('batch_size', best_hyperparameters)
        self.assertIn('dropout', best_hyperparameters)
        self.assertIn('optimizer', best_hyperparameters)
        
        # Check tuning stats
        stats = tuner.get_tuning_stats()
        self.assertEqual(stats['total_steps'], 5)
        self.assertGreater(stats['best_score'], 0)
        self.assertGreaterEqual(stats['improvement'], 0)

if __name__ == '__main__':
    unittest.main()





