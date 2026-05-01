"""
Unit tests for AI-powered optimization
Tests AI-driven optimization, machine learning optimization, and intelligent optimization systems
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestAIOptimization(unittest.TestCase):
    """Test suite for AI-powered optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_ai_driven_optimization(self):
        """Test AI-driven optimization system"""
        class AIDrivenOptimizer:
            def __init__(self, ai_models):
                self.ai_models = ai_models
                self.optimization_history = []
                self.ai_predictions = []
                self.performance_metrics = {}
                
            def ai_optimize(self, data, target, optimization_context):
                """Execute AI-driven optimization"""
                # AI analysis of optimization context
                context_analysis = self._ai_analyze_context(data, target, optimization_context)
                
                # AI prediction of optimal strategy
                strategy_prediction = self._ai_predict_strategy(context_analysis)
                
                # AI-guided optimization execution
                optimization_result = self._ai_execute_optimization(strategy_prediction, data, target)
                
                # AI performance evaluation
                performance_evaluation = self._ai_evaluate_performance(optimization_result)
                
                # Record AI optimization
                self.optimization_history.append({
                    'context_analysis': context_analysis,
                    'strategy_prediction': strategy_prediction,
                    'optimization_result': optimization_result,
                    'performance_evaluation': performance_evaluation,
                    'timestamp': time.time()
                })
                
                return {
                    'optimization_result': optimization_result,
                    'ai_strategy': strategy_prediction,
                    'ai_confidence': performance_evaluation['confidence'],
                    'performance_score': performance_evaluation['score']
                }
                
            def _ai_analyze_context(self, data, target, optimization_context):
                """AI analysis of optimization context"""
                # Simulate AI context analysis
                context_analysis = {
                    'data_complexity': self._ai_assess_complexity(data),
                    'target_difficulty': self._ai_assess_difficulty(target),
                    'optimization_goal': optimization_context.get('goal', 'minimize_loss'),
                    'constraints': optimization_context.get('constraints', {}),
                    'ai_insights': self._generate_ai_insights(data, target),
                    'recommendation_confidence': np.random.uniform(0.7, 0.95)
                }
                return context_analysis
                
            def _ai_assess_complexity(self, data):
                """AI assessment of data complexity"""
                # Simulate AI complexity assessment
                complexity_features = [
                    data.shape[0],  # batch size
                    data.shape[1],  # sequence length
                    data.shape[2],  # feature dimension
                    np.std(data.numpy()) if hasattr(data, 'numpy') else np.std(data.detach().numpy())
                ]
                
                # AI complexity score
                complexity_score = np.mean(complexity_features) / 100.0
                return min(complexity_score, 1.0)
                
            def _ai_assess_difficulty(self, target):
                """AI assessment of target difficulty"""
                # Simulate AI difficulty assessment
                difficulty_features = [
                    target.shape[0],  # batch size
                    target.shape[1],  # sequence length
                    target.shape[2],  # target dimension
                    np.std(target.numpy()) if hasattr(target, 'numpy') else np.std(target.detach().numpy())
                ]
                
                # AI difficulty score
                difficulty_score = np.mean(difficulty_features) / 100.0
                return min(difficulty_score, 1.0)
                
            def _generate_ai_insights(self, data, target):
                """Generate AI insights"""
                insights = []
                
                # Data insights
                if data.shape[0] > 100:
                    insights.append("Large batch size detected - consider batch optimization")
                if data.shape[1] > 1000:
                    insights.append("Long sequences detected - consider sequence optimization")
                if data.shape[2] > 500:
                    insights.append("High-dimensional features detected - consider dimensionality reduction")
                    
                # Target insights
                if target.shape[2] > 100:
                    insights.append("High-dimensional targets detected - consider target optimization")
                    
                return insights
                
            def _ai_predict_strategy(self, context_analysis):
                """AI prediction of optimal strategy"""
                # Simulate AI strategy prediction
                if context_analysis['data_complexity'] > 0.8:
                    if context_analysis['target_difficulty'] > 0.8:
                        strategy = 'advanced_ai_optimization'
                    else:
                        strategy = 'balanced_ai_optimization'
                else:
                    strategy = 'simple_ai_optimization'
                    
                # Record AI prediction
                self.ai_predictions.append({
                    'strategy': strategy,
                    'confidence': context_analysis['recommendation_confidence'],
                    'context_score': (context_analysis['data_complexity'] + context_analysis['target_difficulty']) / 2
                })
                
                return {
                    'strategy': strategy,
                    'confidence': context_analysis['recommendation_confidence'],
                    'ai_reasoning': f"AI selected {strategy} based on complexity analysis"
                }
                
            def _ai_execute_optimization(self, strategy_prediction, data, target):
                """AI-guided optimization execution"""
                strategy = strategy_prediction['strategy']
                
                if strategy == 'advanced_ai_optimization':
                    return self._execute_advanced_ai_optimization(data, target)
                elif strategy == 'balanced_ai_optimization':
                    return self._execute_balanced_ai_optimization(data, target)
                else:
                    return self._execute_simple_ai_optimization(data, target)
                    
            def _execute_advanced_ai_optimization(self, data, target):
                """Execute advanced AI optimization"""
                return {
                    'success': True,
                    'performance': np.random.uniform(0.85, 0.95),
                    'execution_time': np.random.uniform(5, 15),
                    'ai_techniques_used': ['deep_learning', 'reinforcement_learning', 'meta_learning']
                }
                
            def _execute_balanced_ai_optimization(self, data, target):
                """Execute balanced AI optimization"""
                return {
                    'success': True,
                    'performance': np.random.uniform(0.75, 0.85),
                    'execution_time': np.random.uniform(2, 8),
                    'ai_techniques_used': ['machine_learning', 'optimization_ai']
                }
                
            def _execute_simple_ai_optimization(self, data, target):
                """Execute simple AI optimization"""
                return {
                    'success': True,
                    'performance': np.random.uniform(0.65, 0.75),
                    'execution_time': np.random.uniform(1, 3),
                    'ai_techniques_used': ['basic_ai', 'rule_based']
                }
                
            def _ai_evaluate_performance(self, optimization_result):
                """AI evaluation of optimization performance"""
                # Simulate AI performance evaluation
                performance_score = optimization_result['performance']
                execution_efficiency = 1.0 / (optimization_result['execution_time'] + 1e-8)
                
                # AI confidence in evaluation
                confidence = np.random.uniform(0.8, 0.95)
                
                # AI performance insights
                insights = []
                if performance_score > 0.9:
                    insights.append("Excellent performance achieved")
                elif performance_score > 0.8:
                    insights.append("Good performance achieved")
                else:
                    insights.append("Performance could be improved")
                    
                return {
                    'score': performance_score,
                    'efficiency': execution_efficiency,
                    'confidence': confidence,
                    'insights': insights
                }
                
            def get_ai_optimization_stats(self):
                """Get AI optimization statistics"""
                return {
                    'total_optimizations': len(self.optimization_history),
                    'ai_predictions': len(self.ai_predictions),
                    'avg_ai_confidence': np.mean([pred['confidence'] for pred in self.ai_predictions]) if self.ai_predictions else 0,
                    'avg_performance_score': np.mean([opt['performance_evaluation']['score'] for opt in self.optimization_history]) if self.optimization_history else 0,
                    'ai_effectiveness': self._calculate_ai_effectiveness()
                }
                
            def _calculate_ai_effectiveness(self):
                """Calculate AI effectiveness"""
                if not self.optimization_history:
                    return 0
                    
                performance_scores = [opt['performance_evaluation']['score'] for opt in self.optimization_history]
                confidence_scores = [opt['performance_evaluation']['confidence'] for opt in self.optimization_history]
                
                # AI effectiveness = performance * confidence
                effectiveness = np.mean([score * conf for score, conf in zip(performance_scores, confidence_scores)])
                return effectiveness
        
        # Test AI-driven optimizer
        ai_models = {
            'context_analyzer': {'type': 'deep_learning'},
            'strategy_predictor': {'type': 'reinforcement_learning'},
            'performance_evaluator': {'type': 'machine_learning'}
        }
        
        ai_optimizer = AIDrivenOptimizer(ai_models)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        optimization_context = {'goal': 'minimize_loss', 'constraints': {}}
        
        # Test AI optimization
        result = ai_optimizer.ai_optimize(data, target, optimization_context)
        
        # Verify results
        self.assertIn('optimization_result', result)
        self.assertIn('ai_strategy', result)
        self.assertIn('ai_confidence', result)
        self.assertIn('performance_score', result)
        self.assertGreater(result['ai_confidence'], 0)
        self.assertGreater(result['performance_score'], 0)
        
        # Check AI optimization stats
        stats = ai_optimizer.get_ai_optimization_stats()
        self.assertEqual(stats['total_optimizations'], 1)
        self.assertEqual(stats['ai_predictions'], 1)
        self.assertGreater(stats['avg_ai_confidence'], 0)
        self.assertGreater(stats['avg_performance_score'], 0)
        self.assertGreater(stats['ai_effectiveness'], 0)
        
    def test_machine_learning_optimization(self):
        """Test machine learning optimization"""
        class MachineLearningOptimizer:
            def __init__(self, ml_models):
                self.ml_models = ml_models
                self.optimization_data = []
                self.ml_predictions = []
                self.learning_history = []
                
            def ml_optimize(self, data, target, optimization_problem):
                """Execute machine learning optimization"""
                # Collect optimization data
                optimization_data = self._collect_optimization_data(data, target, optimization_problem)
                self.optimization_data.append(optimization_data)
                
                # ML prediction of optimal parameters
                ml_prediction = self._ml_predict_parameters(optimization_data)
                self.ml_predictions.append(ml_prediction)
                
                # ML-guided optimization execution
                optimization_result = self._ml_execute_optimization(ml_prediction, data, target)
                
                # ML learning from results
                learning_result = self._ml_learn_from_results(optimization_result, ml_prediction)
                self.learning_history.append(learning_result)
                
                return {
                    'optimization_result': optimization_result,
                    'ml_prediction': ml_prediction,
                    'learning_result': learning_result,
                    'ml_confidence': ml_prediction['confidence']
                }
                
            def _collect_optimization_data(self, data, target, optimization_problem):
                """Collect data for ML optimization"""
                data_features = {
                    'data_shape': data.shape,
                    'target_shape': target.shape,
                    'data_mean': np.mean(data.numpy()) if hasattr(data, 'numpy') else np.mean(data.detach().numpy()),
                    'data_std': np.std(data.numpy()) if hasattr(data, 'numpy') else np.std(data.detach().numpy()),
                    'target_mean': np.mean(target.numpy()) if hasattr(target, 'numpy') else np.mean(target.detach().numpy()),
                    'target_std': np.std(target.numpy()) if hasattr(target, 'numpy') else np.std(target.detach().numpy()),
                    'optimization_goal': optimization_problem.get('goal', 'minimize_loss'),
                    'constraints': optimization_problem.get('constraints', {})
                }
                
                return data_features
                
            def _ml_predict_parameters(self, optimization_data):
                """ML prediction of optimal parameters"""
                # Simulate ML parameter prediction
                predicted_parameters = {
                    'learning_rate': np.random.uniform(0.0001, 0.01),
                    'batch_size': np.random.choice([16, 32, 64, 128]),
                    'optimizer': np.random.choice(['adam', 'sgd', 'rmsprop']),
                    'regularization': np.random.uniform(0.001, 0.1),
                    'momentum': np.random.uniform(0.5, 0.99)
                }
                
                # ML confidence in prediction
                confidence = np.random.uniform(0.7, 0.95)
                
                return {
                    'parameters': predicted_parameters,
                    'confidence': confidence,
                    'ml_model_used': 'neural_network',
                    'prediction_features': list(optimization_data.keys())
                }
                
            def _ml_execute_optimization(self, ml_prediction, data, target):
                """ML-guided optimization execution"""
                parameters = ml_prediction['parameters']
                
                # Simulate optimization with ML-predicted parameters
                optimization_result = {
                    'success': True,
                    'performance': np.random.uniform(0.7, 0.95),
                    'execution_time': np.random.uniform(1, 10),
                    'parameters_used': parameters,
                    'convergence_rate': np.random.uniform(0.5, 1.0),
                    'final_loss': np.random.uniform(0.01, 0.5)
                }
                
                return optimization_result
                
            def _ml_learn_from_results(self, optimization_result, ml_prediction):
                """ML learning from optimization results"""
                # Simulate ML learning
                learning_result = {
                    'prediction_accuracy': np.random.uniform(0.6, 0.9),
                    'performance_achieved': optimization_result['performance'],
                    'learning_insights': self._generate_learning_insights(optimization_result, ml_prediction),
                    'model_update': True,
                    'learning_confidence': np.random.uniform(0.7, 0.95)
                }
                
                return learning_result
                
            def _generate_learning_insights(self, optimization_result, ml_prediction):
                """Generate learning insights"""
                insights = []
                
                if optimization_result['performance'] > 0.9:
                    insights.append("ML prediction led to excellent performance")
                elif optimization_result['performance'] > 0.8:
                    insights.append("ML prediction led to good performance")
                else:
                    insights.append("ML prediction could be improved")
                    
                if ml_prediction['confidence'] > 0.9:
                    insights.append("High confidence prediction was accurate")
                else:
                    insights.append("Lower confidence prediction needs improvement")
                    
                return insights
                
            def get_ml_optimization_stats(self):
                """Get ML optimization statistics"""
                return {
                    'total_optimizations': len(self.optimization_data),
                    'ml_predictions': len(self.ml_predictions),
                    'learning_cycles': len(self.learning_history),
                    'avg_prediction_confidence': np.mean([pred['confidence'] for pred in self.ml_predictions]) if self.ml_predictions else 0,
                    'avg_performance': np.mean([opt['performance'] for opt in [self._ml_execute_optimization(pred, None, None) for pred in self.ml_predictions]]) if self.ml_predictions else 0,
                    'learning_effectiveness': self._calculate_learning_effectiveness()
                }
                
            def _calculate_learning_effectiveness(self):
                """Calculate learning effectiveness"""
                if not self.learning_history:
                    return 0
                    
                prediction_accuracies = [learning['prediction_accuracy'] for learning in self.learning_history]
                learning_confidences = [learning['learning_confidence'] for learning in self.learning_history]
                
                # Learning effectiveness = accuracy * confidence
                effectiveness = np.mean([acc * conf for acc, conf in zip(prediction_accuracies, learning_confidences)])
                return effectiveness
        
        # Test machine learning optimizer
        ml_models = {
            'parameter_predictor': {'type': 'neural_network'},
            'performance_predictor': {'type': 'regression'},
            'learning_engine': {'type': 'reinforcement_learning'}
        }
        
        ml_optimizer = MachineLearningOptimizer(ml_models)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        optimization_problem = {'goal': 'minimize_loss', 'constraints': {}}
        
        # Test ML optimization
        result = ml_optimizer.ml_optimize(data, target, optimization_problem)
        
        # Verify results
        self.assertIn('optimization_result', result)
        self.assertIn('ml_prediction', result)
        self.assertIn('learning_result', result)
        self.assertIn('ml_confidence', result)
        self.assertGreater(result['ml_confidence'], 0)
        
        # Check ML optimization stats
        stats = ml_optimizer.get_ml_optimization_stats()
        self.assertEqual(stats['total_optimizations'], 1)
        self.assertEqual(stats['ml_predictions'], 1)
        self.assertEqual(stats['learning_cycles'], 1)
        self.assertGreater(stats['avg_prediction_confidence'], 0)
        self.assertGreater(stats['avg_performance'], 0)
        self.assertGreater(stats['learning_effectiveness'], 0)
        
    def test_intelligent_optimization_system(self):
        """Test intelligent optimization system"""
        class IntelligentOptimizationSystem:
            def __init__(self, intelligence_components):
                self.intelligence_components = intelligence_components
                self.intelligence_history = []
                self.adaptation_cycles = 0
                self.intelligence_metrics = {}
                
            def intelligent_optimize(self, data, target, optimization_context):
                """Execute intelligent optimization"""
                # Intelligence analysis
                intelligence_analysis = self._analyze_with_intelligence(data, target, optimization_context)
                
                # Intelligent strategy selection
                intelligent_strategy = self._select_intelligent_strategy(intelligence_analysis)
                
                # Intelligent optimization execution
                optimization_result = self._execute_intelligent_optimization(intelligent_strategy, data, target)
                
                # Intelligence adaptation
                adaptation_result = self._adapt_intelligence(optimization_result, intelligence_analysis)
                
                # Record intelligence optimization
                self.intelligence_history.append({
                    'intelligence_analysis': intelligence_analysis,
                    'intelligent_strategy': intelligent_strategy,
                    'optimization_result': optimization_result,
                    'adaptation_result': adaptation_result,
                    'timestamp': time.time()
                })
                
                return {
                    'optimization_result': optimization_result,
                    'intelligence_strategy': intelligent_strategy,
                    'adaptation_result': adaptation_result,
                    'intelligence_score': self._calculate_intelligence_score(optimization_result, intelligence_analysis)
                }
                
            def _analyze_with_intelligence(self, data, target, optimization_context):
                """Intelligence analysis of optimization problem"""
                analysis = {
                    'problem_complexity': self._assess_problem_complexity(data, target),
                    'optimization_difficulty': self._assess_optimization_difficulty(optimization_context),
                    'intelligence_requirements': self._determine_intelligence_requirements(data, target),
                    'intelligence_insights': self._generate_intelligence_insights(data, target, optimization_context),
                    'intelligence_confidence': np.random.uniform(0.8, 0.95)
                }
                return analysis
                
            def _assess_problem_complexity(self, data, target):
                """Assess problem complexity with intelligence"""
                # Simulate intelligent complexity assessment
                complexity_score = np.random.uniform(0.3, 1.0)
                return complexity_score
                
            def _assess_optimization_difficulty(self, optimization_context):
                """Assess optimization difficulty with intelligence"""
                # Simulate intelligent difficulty assessment
                difficulty_score = np.random.uniform(0.3, 1.0)
                return difficulty_score
                
            def _determine_intelligence_requirements(self, data, target):
                """Determine intelligence requirements"""
                requirements = []
                
                if data.shape[0] > 1000:
                    requirements.append('high_intelligence')
                if data.shape[1] > 1000:
                    requirements.append('sequence_intelligence')
                if data.shape[2] > 500:
                    requirements.append('feature_intelligence')
                    
                return requirements
                
            def _generate_intelligence_insights(self, data, target, optimization_context):
                """Generate intelligence insights"""
                insights = []
                
                # Data insights
                if data.shape[0] > 100:
                    insights.append("Large dataset detected - intelligent batch processing recommended")
                if data.shape[1] > 1000:
                    insights.append("Long sequences detected - intelligent sequence optimization recommended")
                if data.shape[2] > 500:
                    insights.append("High-dimensional features detected - intelligent dimensionality reduction recommended")
                    
                # Target insights
                if target.shape[2] > 100:
                    insights.append("High-dimensional targets detected - intelligent target optimization recommended")
                    
                # Context insights
                if optimization_context.get('goal') == 'minimize_loss':
                    insights.append("Loss minimization goal - intelligent loss optimization recommended")
                    
                return insights
                
            def _select_intelligent_strategy(self, intelligence_analysis):
                """Select intelligent optimization strategy"""
                # Simulate intelligent strategy selection
                if intelligence_analysis['problem_complexity'] > 0.8:
                    if intelligence_analysis['optimization_difficulty'] > 0.8:
                        strategy = 'advanced_intelligent_optimization'
                    else:
                        strategy = 'balanced_intelligent_optimization'
                else:
                    strategy = 'simple_intelligent_optimization'
                    
                return {
                    'strategy': strategy,
                    'intelligence_level': 'high' if intelligence_analysis['problem_complexity'] > 0.8 else 'medium' if intelligence_analysis['problem_complexity'] > 0.5 else 'low',
                    'confidence': intelligence_analysis['intelligence_confidence']
                }
                
            def _execute_intelligent_optimization(self, intelligent_strategy, data, target):
                """Execute intelligent optimization"""
                strategy = intelligent_strategy['strategy']
                
                if strategy == 'advanced_intelligent_optimization':
                    return self._execute_advanced_intelligent_optimization(data, target)
                elif strategy == 'balanced_intelligent_optimization':
                    return self._execute_balanced_intelligent_optimization(data, target)
                else:
                    return self._execute_simple_intelligent_optimization(data, target)
                    
            def _execute_advanced_intelligent_optimization(self, data, target):
                """Execute advanced intelligent optimization"""
                return {
                    'success': True,
                    'performance': np.random.uniform(0.9, 0.98),
                    'execution_time': np.random.uniform(10, 20),
                    'intelligence_techniques': ['deep_learning', 'reinforcement_learning', 'meta_learning', 'neural_architecture_search']
                }
                
            def _execute_balanced_intelligent_optimization(self, data, target):
                """Execute balanced intelligent optimization"""
                return {
                    'success': True,
                    'performance': np.random.uniform(0.8, 0.9),
                    'execution_time': np.random.uniform(5, 15),
                    'intelligence_techniques': ['machine_learning', 'optimization_ai', 'intelligent_search']
                }
                
            def _execute_simple_intelligent_optimization(self, data, target):
                """Execute simple intelligent optimization"""
                return {
                    'success': True,
                    'performance': np.random.uniform(0.7, 0.8),
                    'execution_time': np.random.uniform(2, 8),
                    'intelligence_techniques': ['basic_ai', 'rule_based_intelligence']
                }
                
            def _adapt_intelligence(self, optimization_result, intelligence_analysis):
                """Adapt intelligence based on results"""
                # Simulate intelligence adaptation
                adaptation_result = {
                    'intelligence_improvement': np.random.uniform(0.1, 0.3),
                    'adaptation_confidence': np.random.uniform(0.7, 0.95),
                    'intelligence_insights': self._generate_adaptation_insights(optimization_result, intelligence_analysis)
                }
                
                self.adaptation_cycles += 1
                return adaptation_result
                
            def _generate_adaptation_insights(self, optimization_result, intelligence_analysis):
                """Generate adaptation insights"""
                insights = []
                
                if optimization_result['performance'] > 0.9:
                    insights.append("Intelligence adaptation led to excellent performance")
                elif optimization_result['performance'] > 0.8:
                    insights.append("Intelligence adaptation led to good performance")
                else:
                    insights.append("Intelligence adaptation needs improvement")
                    
                return insights
                
            def _calculate_intelligence_score(self, optimization_result, intelligence_analysis):
                """Calculate intelligence score"""
                performance_score = optimization_result['performance']
                intelligence_confidence = intelligence_analysis['intelligence_confidence']
                
                # Intelligence score = performance * confidence
                intelligence_score = performance_score * intelligence_confidence
                return intelligence_score
                
            def get_intelligence_stats(self):
                """Get intelligence statistics"""
                return {
                    'total_optimizations': len(self.intelligence_history),
                    'adaptation_cycles': self.adaptation_cycles,
                    'avg_intelligence_score': np.mean([opt['intelligence_score'] for opt in self.intelligence_history]) if self.intelligence_history else 0,
                    'intelligence_effectiveness': self._calculate_intelligence_effectiveness()
                }
                
            def _calculate_intelligence_effectiveness(self):
                """Calculate intelligence effectiveness"""
                if not self.intelligence_history:
                    return 0
                    
                intelligence_scores = [opt['intelligence_score'] for opt in self.intelligence_history]
                return np.mean(intelligence_scores)
        
        # Test intelligent optimization system
        intelligence_components = {
            'intelligence_analyzer': {'type': 'deep_learning'},
            'strategy_selector': {'type': 'reinforcement_learning'},
            'adaptation_engine': {'type': 'meta_learning'}
        }
        
        intelligent_system = IntelligentOptimizationSystem(intelligence_components)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        optimization_context = {'goal': 'minimize_loss', 'constraints': {}}
        
        # Test intelligent optimization
        result = intelligent_system.intelligent_optimize(data, target, optimization_context)
        
        # Verify results
        self.assertIn('optimization_result', result)
        self.assertIn('intelligence_strategy', result)
        self.assertIn('adaptation_result', result)
        self.assertIn('intelligence_score', result)
        self.assertGreater(result['intelligence_score'], 0)
        
        # Check intelligence stats
        stats = intelligent_system.get_intelligence_stats()
        self.assertEqual(stats['total_optimizations'], 1)
        self.assertEqual(stats['adaptation_cycles'], 1)
        self.assertGreater(stats['avg_intelligence_score'], 0)
        self.assertGreater(stats['intelligence_effectiveness'], 0)

if __name__ == '__main__':
    unittest.main()





