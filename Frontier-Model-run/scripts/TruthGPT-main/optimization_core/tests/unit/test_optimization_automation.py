"""
Unit tests for optimization automation
Tests automated optimization pipelines, intelligent optimization, and self-optimizing systems
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

class TestOptimizationAutomation(unittest.TestCase):
    """Test suite for optimization automation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_automated_optimization_pipeline(self):
        """Test automated optimization pipeline"""
        class AutomatedOptimizationPipeline:
            def __init__(self, pipeline_stages):
                self.pipeline_stages = pipeline_stages
                self.pipeline_history = []
                self.automation_rules = {}
                self.performance_metrics = {}
                
            def execute_automated_pipeline(self, data, target, optimization_goal):
                """Execute automated optimization pipeline"""
                current_data = data
                current_target = target
                pipeline_results = {}
                
                for stage_name, stage_func in self.pipeline_stages.items():
                    # Execute stage
                    stage_result = stage_func(current_data, current_target, optimization_goal)
                    
                    # Record stage result
                    pipeline_results[stage_name] = stage_result
                    self.pipeline_history.append({
                        'stage': stage_name,
                        'result': stage_result,
                        'timestamp': time.time()
                    })
                    
                    # Update data for next stage
                    if 'output_data' in stage_result:
                        current_data = stage_result['output_data']
                    if 'output_target' in stage_result:
                        current_target = stage_result['output_target']
                        
                # Evaluate pipeline performance
                pipeline_performance = self._evaluate_pipeline_performance(pipeline_results, optimization_goal)
                
                return {
                    'pipeline_results': pipeline_results,
                    'pipeline_performance': pipeline_performance,
                    'automation_success': pipeline_performance['success_rate'] > 0.8
                }
                
            def _evaluate_pipeline_performance(self, pipeline_results, optimization_goal):
                """Evaluate pipeline performance"""
                performance_metrics = {
                    'total_stages': len(pipeline_results),
                    'successful_stages': sum(1 for result in pipeline_results.values() if result.get('success', False)),
                    'success_rate': 0,
                    'overall_performance': 0,
                    'goal_achievement': 0
                }
                
                # Calculate success rate
                performance_metrics['success_rate'] = performance_metrics['successful_stages'] / performance_metrics['total_stages']
                
                # Calculate overall performance
                stage_performances = [result.get('performance', 0) for result in pipeline_results.values()]
                performance_metrics['overall_performance'] = np.mean(stage_performances)
                
                # Calculate goal achievement
                if 'target_performance' in optimization_goal:
                    target_performance = optimization_goal['target_performance']
                    performance_metrics['goal_achievement'] = min(performance_metrics['overall_performance'] / target_performance, 1.0)
                    
                return performance_metrics
                
            def get_pipeline_stats(self):
                """Get pipeline statistics"""
                return {
                    'total_stages': len(self.pipeline_stages),
                    'total_executions': len(self.pipeline_history),
                    'pipeline_history': self.pipeline_history,
                    'automation_rules': self.automation_rules
                }
        
        # Test automated optimization pipeline
        def preprocessing_stage(data, target, goal):
            """Preprocessing stage"""
            return {
                'output_data': data * 0.9,
                'output_target': target,
                'success': True,
                'performance': 0.8
            }
            
        def optimization_stage(data, target, goal):
            """Optimization stage"""
            return {
                'output_data': data,
                'output_target': target,
                'success': True,
                'performance': 0.9
            }
            
        def validation_stage(data, target, goal):
            """Validation stage"""
            return {
                'output_data': data,
                'output_target': target,
                'success': True,
                'performance': 0.85
            }
        
        pipeline_stages = {
            'preprocessing': preprocessing_stage,
            'optimization': optimization_stage,
            'validation': validation_stage
        }
        
        pipeline = AutomatedOptimizationPipeline(pipeline_stages)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        optimization_goal = {'target_performance': 0.8}
        
        # Test automated pipeline
        result = pipeline.execute_automated_pipeline(data, target, optimization_goal)
        
        # Verify results
        self.assertIn('pipeline_results', result)
        self.assertIn('pipeline_performance', result)
        self.assertIn('automation_success', result)
        self.assertEqual(len(result['pipeline_results']), 3)
        self.assertTrue(result['automation_success'])
        
        # Check pipeline stats
        stats = pipeline.get_pipeline_stats()
        self.assertEqual(stats['total_stages'], 3)
        self.assertEqual(stats['total_executions'], 3)
        self.assertEqual(len(stats['pipeline_history']), 3)
        
    def test_intelligent_optimization_system(self):
        """Test intelligent optimization system"""
        class IntelligentOptimizationSystem:
            def __init__(self, optimization_strategies):
                self.optimization_strategies = optimization_strategies
                self.performance_history = []
                self.strategy_selection_history = []
                self.adaptation_rules = {}
                
            def intelligent_optimization(self, data, target, optimization_context):
                """Execute intelligent optimization"""
                # Analyze optimization context
                context_analysis = self._analyze_optimization_context(data, target, optimization_context)
                
                # Select optimal strategy
                selected_strategy = self._select_optimization_strategy(context_analysis)
                
                # Execute optimization with selected strategy
                optimization_result = self._execute_optimization_strategy(selected_strategy, data, target)
                
                # Adapt strategy based on results
                adapted_strategy = self._adapt_strategy(selected_strategy, optimization_result)
                
                # Record optimization
                self.performance_history.append({
                    'context_analysis': context_analysis,
                    'selected_strategy': selected_strategy,
                    'optimization_result': optimization_result,
                    'adapted_strategy': adapted_strategy,
                    'timestamp': time.time()
                })
                
                return {
                    'optimization_result': optimization_result,
                    'strategy_used': selected_strategy,
                    'adaptation_applied': adapted_strategy != selected_strategy,
                    'intelligence_score': self._calculate_intelligence_score(optimization_result)
                }
                
            def _analyze_optimization_context(self, data, target, optimization_context):
                """Analyze optimization context"""
                context_analysis = {
                    'data_complexity': self._assess_data_complexity(data),
                    'target_complexity': self._assess_target_complexity(target),
                    'optimization_goal': optimization_context.get('goal', 'minimize_loss'),
                    'constraints': optimization_context.get('constraints', {}),
                    'performance_requirements': optimization_context.get('performance_requirements', {}),
                    'context_score': 0
                }
                
                # Calculate context score
                context_analysis['context_score'] = (
                    context_analysis['data_complexity'] + 
                    context_analysis['target_complexity']
                ) / 2
                
                return context_analysis
                
            def _assess_data_complexity(self, data):
                """Assess data complexity"""
                # Simulate data complexity assessment
                complexity = np.random.uniform(0.3, 1.0)
                return complexity
                
            def _assess_target_complexity(self, target):
                """Assess target complexity"""
                # Simulate target complexity assessment
                complexity = np.random.uniform(0.3, 1.0)
                return complexity
                
            def _select_optimization_strategy(self, context_analysis):
                """Select optimal optimization strategy"""
                # Select strategy based on context analysis
                if context_analysis['context_score'] > 0.8:
                    strategy = 'advanced_optimization'
                elif context_analysis['context_score'] > 0.5:
                    strategy = 'balanced_optimization'
                else:
                    strategy = 'simple_optimization'
                    
                # Record strategy selection
                self.strategy_selection_history.append({
                    'context_score': context_analysis['context_score'],
                    'selected_strategy': strategy,
                    'timestamp': time.time()
                })
                
                return strategy
                
            def _execute_optimization_strategy(self, strategy, data, target):
                """Execute optimization strategy"""
                if strategy in self.optimization_strategies:
                    return self.optimization_strategies[strategy](data, target)
                else:
                    # Default optimization
                    return {
                        'success': True,
                        'performance': np.random.uniform(0.7, 0.95),
                        'execution_time': np.random.uniform(1, 10)
                    }
                    
            def _adapt_strategy(self, current_strategy, optimization_result):
                """Adapt strategy based on results"""
                # Simple adaptation logic
                if optimization_result['performance'] < 0.7:
                    # Poor performance, try different strategy
                    if current_strategy == 'simple_optimization':
                        return 'balanced_optimization'
                    elif current_strategy == 'balanced_optimization':
                        return 'advanced_optimization'
                    else:
                        return 'simple_optimization'
                else:
                    # Good performance, keep current strategy
                    return current_strategy
                    
            def _calculate_intelligence_score(self, optimization_result):
                """Calculate intelligence score"""
                # Score based on performance and efficiency
                performance_score = optimization_result['performance']
                efficiency_score = 1.0 / (optimization_result['execution_time'] + 1e-8)
                intelligence_score = (performance_score + efficiency_score) / 2
                return intelligence_score
                
            def get_intelligence_stats(self):
                """Get intelligence statistics"""
                return {
                    'total_optimizations': len(self.performance_history),
                    'strategy_selections': len(self.strategy_selection_history),
                    'avg_intelligence_score': np.mean([opt['intelligence_score'] for opt in self.performance_history]) if self.performance_history else 0,
                    'strategy_adaptation_rate': sum(1 for opt in self.performance_history if opt['adaptation_applied']) / len(self.performance_history) if self.performance_history else 0
                }
        
        # Test intelligent optimization system
        optimization_strategies = {
            'simple_optimization': lambda data, target: {'success': True, 'performance': 0.7, 'execution_time': 1.0},
            'balanced_optimization': lambda data, target: {'success': True, 'performance': 0.8, 'execution_time': 2.0},
            'advanced_optimization': lambda data, target: {'success': True, 'performance': 0.9, 'execution_time': 5.0}
        }
        
        intelligent_system = IntelligentOptimizationSystem(optimization_strategies)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        optimization_context = {'goal': 'minimize_loss', 'constraints': {}}
        
        # Test intelligent optimization
        result = intelligent_system.intelligent_optimization(data, target, optimization_context)
        
        # Verify results
        self.assertIn('optimization_result', result)
        self.assertIn('strategy_used', result)
        self.assertIn('adaptation_applied', result)
        self.assertIn('intelligence_score', result)
        self.assertGreater(result['intelligence_score'], 0)
        
        # Check intelligence stats
        stats = intelligent_system.get_intelligence_stats()
        self.assertEqual(stats['total_optimizations'], 1)
        self.assertEqual(stats['strategy_selections'], 1)
        self.assertGreater(stats['avg_intelligence_score'], 0)
        self.assertGreaterEqual(stats['strategy_adaptation_rate'], 0)
        
    def test_self_optimizing_system(self):
        """Test self-optimizing system"""
        class SelfOptimizingSystem:
            def __init__(self, optimization_components):
                self.optimization_components = optimization_components
                self.self_optimization_history = []
                self.performance_evolution = []
                self.adaptation_cycles = 0
                
            def self_optimize(self, data, target, optimization_goal):
                """Execute self-optimization"""
                # Initial optimization
                current_performance = self._evaluate_current_performance(data, target)
                self.performance_evolution.append(current_performance)
                
                # Self-optimization loop
                max_cycles = 5
                for cycle in range(max_cycles):
                    # Analyze current performance
                    performance_analysis = self._analyze_performance(current_performance)
                    
                    # Identify optimization opportunities
                    opportunities = self._identify_optimization_opportunities(performance_analysis)
                    
                    # Apply self-optimizations
                    optimization_applied = self._apply_self_optimizations(opportunities)
                    
                    # Evaluate improved performance
                    new_performance = self._evaluate_current_performance(data, target)
                    
                    # Record self-optimization cycle
                    self.self_optimization_history.append({
                        'cycle': cycle,
                        'performance_analysis': performance_analysis,
                        'opportunities': opportunities,
                        'optimization_applied': optimization_applied,
                        'performance_improvement': new_performance - current_performance,
                        'timestamp': time.time()
                    })
                    
                    # Check if optimization goal is achieved
                    if self._check_goal_achievement(new_performance, optimization_goal):
                        break
                        
                    # Update performance
                    current_performance = new_performance
                    self.performance_evolution.append(current_performance)
                    
                self.adaptation_cycles = cycle + 1
                
                return {
                    'final_performance': current_performance,
                    'adaptation_cycles': self.adaptation_cycles,
                    'performance_improvement': self.performance_evolution[-1] - self.performance_evolution[0],
                    'goal_achieved': self._check_goal_achievement(current_performance, optimization_goal)
                }
                
            def _evaluate_current_performance(self, data, target):
                """Evaluate current performance"""
                # Simulate performance evaluation
                performance = np.random.uniform(0.5, 0.95)
                return performance
                
            def _analyze_performance(self, performance):
                """Analyze current performance"""
                analysis = {
                    'performance_level': 'high' if performance > 0.8 else 'medium' if performance > 0.6 else 'low',
                    'improvement_potential': 1.0 - performance,
                    'optimization_priority': 'high' if performance < 0.7 else 'medium' if performance < 0.8 else 'low'
                }
                return analysis
                
            def _identify_optimization_opportunities(self, performance_analysis):
                """Identify optimization opportunities"""
                opportunities = []
                
                if performance_analysis['optimization_priority'] == 'high':
                    opportunities.extend([
                        'learning_rate_adjustment',
                        'gradient_clipping',
                        'regularization'
                    ])
                elif performance_analysis['optimization_priority'] == 'medium':
                    opportunities.extend([
                        'learning_rate_adjustment',
                        'momentum_optimization'
                    ])
                else:
                    opportunities.append('fine_tuning')
                    
                return opportunities
                
            def _apply_self_optimizations(self, opportunities):
                """Apply self-optimizations"""
                applied_optimizations = []
                
                for opportunity in opportunities:
                    if opportunity == 'learning_rate_adjustment':
                        applied_optimizations.append('learning_rate_reduced')
                    elif opportunity == 'gradient_clipping':
                        applied_optimizations.append('gradient_clipping_applied')
                    elif opportunity == 'regularization':
                        applied_optimizations.append('regularization_added')
                    elif opportunity == 'momentum_optimization':
                        applied_optimizations.append('momentum_optimized')
                    elif opportunity == 'fine_tuning':
                        applied_optimizations.append('fine_tuning_applied')
                        
                return applied_optimizations
                
            def _check_goal_achievement(self, performance, optimization_goal):
                """Check if optimization goal is achieved"""
                target_performance = optimization_goal.get('target_performance', 0.8)
                return performance >= target_performance
                
            def get_self_optimization_stats(self):
                """Get self-optimization statistics"""
                return {
                    'total_cycles': self.adaptation_cycles,
                    'performance_evolution': self.performance_evolution,
                    'total_optimizations': len(self.self_optimization_history),
                    'avg_performance_improvement': np.mean([opt['performance_improvement'] for opt in self.self_optimization_history]) if self.self_optimization_history else 0,
                    'optimization_effectiveness': self._calculate_optimization_effectiveness()
                }
                
            def _calculate_optimization_effectiveness(self):
                """Calculate optimization effectiveness"""
                if len(self.performance_evolution) < 2:
                    return 0
                    
                initial_performance = self.performance_evolution[0]
                final_performance = self.performance_evolution[-1]
                improvement = final_performance - initial_performance
                
                return improvement / initial_performance if initial_performance > 0 else 0
        
        # Test self-optimizing system
        optimization_components = {
            'learning_rate_optimizer': {'type': 'adaptive'},
            'gradient_optimizer': {'type': 'clipping'},
            'regularization_optimizer': {'type': 'l2'}
        }
        
        self_optimizing_system = SelfOptimizingSystem(optimization_components)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        optimization_goal = {'target_performance': 0.8}
        
        # Test self-optimization
        result = self_optimizing_system.self_optimize(data, target, optimization_goal)
        
        # Verify results
        self.assertIn('final_performance', result)
        self.assertIn('adaptation_cycles', result)
        self.assertIn('performance_improvement', result)
        self.assertIn('goal_achieved', result)
        self.assertGreater(result['adaptation_cycles'], 0)
        
        # Check self-optimization stats
        stats = self_optimizing_system.get_self_optimization_stats()
        self.assertEqual(stats['total_cycles'], result['adaptation_cycles'])
        self.assertGreater(len(stats['performance_evolution']), 0)
        self.assertGreater(stats['total_optimizations'], 0)
        self.assertGreaterEqual(stats['avg_performance_improvement'], 0)
        self.assertGreaterEqual(stats['optimization_effectiveness'], 0)

if __name__ == '__main__':
    unittest.main()





