"""
Unit tests for optimization analytics and insights
Tests analytics frameworks, performance analysis, and optimization insights
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

class TestOptimizationAnalytics(unittest.TestCase):
    """Test suite for optimization analytics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_optimization_analytics_engine(self):
        """Test optimization analytics engine"""
        class OptimizationAnalyticsEngine:
            def __init__(self):
                self.optimization_data = []
                self.analytics_metrics = {}
                self.insights = []
                self.performance_trends = []
                
            def collect_optimization_data(self, optimization_result):
                """Collect optimization data"""
                data_point = {
                    'timestamp': time.time(),
                    'iteration': optimization_result.get('iteration', 0),
                    'loss': optimization_result.get('loss', 0),
                    'gradient_norm': optimization_result.get('gradient_norm', 0),
                    'learning_rate': optimization_result.get('learning_rate', 0),
                    'accuracy': optimization_result.get('accuracy', 0),
                    'convergence': optimization_result.get('converged', False),
                    'execution_time': optimization_result.get('execution_time', 0)
                }
                
                self.optimization_data.append(data_point)
                
            def analyze_optimization_performance(self):
                """Analyze optimization performance"""
                if not self.optimization_data:
                    return {}
                    
                # Extract metrics
                losses = [point['loss'] for point in self.optimization_data]
                gradient_norms = [point['gradient_norm'] for point in self.optimization_data]
                learning_rates = [point['learning_rate'] for point in self.optimization_data]
                accuracies = [point['accuracy'] for point in self.optimization_data]
                execution_times = [point['execution_time'] for point in self.optimization_data]
                
                # Compute analytics metrics
                self.analytics_metrics = {
                    'total_iterations': len(self.optimization_data),
                    'convergence_rate': sum(1 for point in self.optimization_data if point['convergence']) / len(self.optimization_data),
                    'loss_analysis': {
                        'initial_loss': losses[0],
                        'final_loss': losses[-1],
                        'min_loss': min(losses),
                        'max_loss': max(losses),
                        'avg_loss': np.mean(losses),
                        'loss_std': np.std(losses),
                        'loss_improvement': losses[0] - losses[-1]
                    },
                    'gradient_analysis': {
                        'avg_gradient_norm': np.mean(gradient_norms),
                        'max_gradient_norm': max(gradient_norms),
                        'gradient_explosion': any(norm > 10.0 for norm in gradient_norms),
                        'gradient_vanishing': any(norm < 1e-6 for norm in gradient_norms)
                    },
                    'learning_rate_analysis': {
                        'initial_lr': learning_rates[0],
                        'final_lr': learning_rates[-1],
                        'lr_reductions': len([lr for i, lr in enumerate(learning_rates) 
                                            if i > 0 and lr < learning_rates[i-1]]),
                        'lr_ratio': learning_rates[-1] / learning_rates[0] if learning_rates[0] > 0 else 0
                    },
                    'accuracy_analysis': {
                        'initial_accuracy': accuracies[0],
                        'final_accuracy': accuracies[-1],
                        'max_accuracy': max(accuracies),
                        'avg_accuracy': np.mean(accuracies),
                        'accuracy_improvement': accuracies[-1] - accuracies[0]
                    },
                    'performance_analysis': {
                        'total_execution_time': sum(execution_times),
                        'avg_execution_time': np.mean(execution_times),
                        'execution_efficiency': len(self.optimization_data) / sum(execution_times) if sum(execution_times) > 0 else 0
                    }
                }
                
                return self.analytics_metrics
                
            def generate_optimization_insights(self):
                """Generate optimization insights"""
                if not self.analytics_metrics:
                    return []
                    
                insights = []
                
                # Loss insights
                if self.analytics_metrics['loss_analysis']['loss_improvement'] > 0.5:
                    insights.append({
                        'type': 'positive',
                        'category': 'loss',
                        'message': 'Significant loss improvement achieved',
                        'value': self.analytics_metrics['loss_analysis']['loss_improvement']
                    })
                elif self.analytics_metrics['loss_analysis']['loss_improvement'] < 0:
                    insights.append({
                        'type': 'negative',
                        'category': 'loss',
                        'message': 'Loss increased during optimization',
                        'value': self.analytics_metrics['loss_analysis']['loss_improvement']
                    })
                    
                # Gradient insights
                if self.analytics_metrics['gradient_analysis']['gradient_explosion']:
                    insights.append({
                        'type': 'warning',
                        'category': 'gradient',
                        'message': 'Gradient explosion detected',
                        'value': self.analytics_metrics['gradient_analysis']['max_gradient_norm']
                    })
                    
                if self.analytics_metrics['gradient_analysis']['gradient_vanishing']:
                    insights.append({
                        'type': 'warning',
                        'category': 'gradient',
                        'message': 'Gradient vanishing detected',
                        'value': min([point['gradient_norm'] for point in self.optimization_data])
                    })
                    
                # Learning rate insights
                if self.analytics_metrics['learning_rate_analysis']['lr_ratio'] < 0.1:
                    insights.append({
                        'type': 'info',
                        'category': 'learning_rate',
                        'message': 'Learning rate significantly reduced',
                        'value': self.analytics_metrics['learning_rate_analysis']['lr_ratio']
                    })
                    
                # Accuracy insights
                if self.analytics_metrics['accuracy_analysis']['accuracy_improvement'] > 0.1:
                    insights.append({
                        'type': 'positive',
                        'category': 'accuracy',
                        'message': 'Significant accuracy improvement achieved',
                        'value': self.analytics_metrics['accuracy_analysis']['accuracy_improvement']
                    })
                    
                # Performance insights
                if self.analytics_metrics['performance_analysis']['execution_efficiency'] > 10:
                    insights.append({
                        'type': 'positive',
                        'category': 'performance',
                        'message': 'High execution efficiency achieved',
                        'value': self.analytics_metrics['performance_analysis']['execution_efficiency']
                    })
                    
                self.insights = insights
                return insights
                
            def get_analytics_summary(self):
                """Get analytics summary"""
                return {
                    'total_data_points': len(self.optimization_data),
                    'analytics_metrics': self.analytics_metrics,
                    'insights_count': len(self.insights),
                    'insights': self.insights,
                    'optimization_quality': self._assess_optimization_quality()
                }
                
            def _assess_optimization_quality(self):
                """Assess overall optimization quality"""
                if not self.analytics_metrics:
                    return 'unknown'
                    
                quality_score = 0
                
                # Loss improvement
                if self.analytics_metrics['loss_analysis']['loss_improvement'] > 0:
                    quality_score += 1
                    
                # Accuracy improvement
                if self.analytics_metrics['accuracy_analysis']['accuracy_improvement'] > 0:
                    quality_score += 1
                    
                # No gradient issues
                if not self.analytics_metrics['gradient_analysis']['gradient_explosion']:
                    quality_score += 1
                    
                # Good convergence
                if self.analytics_metrics['convergence_rate'] > 0.8:
                    quality_score += 1
                    
                if quality_score >= 3:
                    return 'excellent'
                elif quality_score >= 2:
                    return 'good'
                elif quality_score >= 1:
                    return 'fair'
                else:
                    return 'poor'
        
        # Test optimization analytics engine
        analytics_engine = OptimizationAnalyticsEngine()
        
        # Collect optimization data
        for i in range(10):
            optimization_result = {
                'iteration': i,
                'loss': 1.0 - i * 0.1 + np.random.normal(0, 0.05),
                'gradient_norm': np.random.uniform(0.1, 5.0),
                'learning_rate': 0.001 * (0.95 ** i),
                'accuracy': 0.5 + i * 0.05 + np.random.normal(0, 0.02),
                'converged': i > 7,
                'execution_time': np.random.uniform(0.1, 1.0)
            }
            analytics_engine.collect_optimization_data(optimization_result)
            
        # Analyze optimization performance
        metrics = analytics_engine.analyze_optimization_performance()
        
        # Verify metrics
        self.assertIn('total_iterations', metrics)
        self.assertIn('convergence_rate', metrics)
        self.assertIn('loss_analysis', metrics)
        self.assertIn('gradient_analysis', metrics)
        self.assertIn('learning_rate_analysis', metrics)
        self.assertIn('accuracy_analysis', metrics)
        self.assertIn('performance_analysis', metrics)
        
        # Generate insights
        insights = analytics_engine.generate_optimization_insights()
        self.assertIsInstance(insights, list)
        
        # Check analytics summary
        summary = analytics_engine.get_analytics_summary()
        self.assertEqual(summary['total_data_points'], 10)
        self.assertIn('analytics_metrics', summary)
        self.assertIn('insights_count', summary)
        self.assertIn('insights', summary)
        self.assertIn('optimization_quality', summary)
        
    def test_optimization_benchmarking_analytics(self):
        """Test optimization benchmarking analytics"""
        class OptimizationBenchmarkingAnalytics:
            def __init__(self):
                self.benchmark_results = {}
                self.comparison_metrics = {}
                self.ranking_analysis = {}
                
            def analyze_benchmark_results(self, benchmark_results):
                """Analyze benchmark results"""
                self.benchmark_results = benchmark_results
                
                # Extract performance metrics
                algorithms = list(benchmark_results.keys())
                performance_metrics = {}
                
                for algorithm in algorithms:
                    algorithm_results = benchmark_results[algorithm]
                    performance_metrics[algorithm] = {
                        'avg_performance': np.mean([r['performance'] for r in algorithm_results]),
                        'std_performance': np.std([r['performance'] for r in algorithm_results]),
                        'best_performance': max([r['performance'] for r in algorithm_results]),
                        'worst_performance': min([r['performance'] for r in algorithm_results]),
                        'success_rate': np.mean([r['success'] for r in algorithm_results]),
                        'avg_execution_time': np.mean([r['execution_time'] for r in algorithm_results])
                    }
                    
                # Compute comparison metrics
                self.comparison_metrics = {
                    'performance_rankings': self._rank_algorithms(performance_metrics, 'avg_performance'),
                    'speed_rankings': self._rank_algorithms(performance_metrics, 'avg_execution_time'),
                    'stability_rankings': self._rank_algorithms(performance_metrics, 'std_performance', reverse=True),
                    'success_rate_rankings': self._rank_algorithms(performance_metrics, 'success_rate')
                }
                
                # Generate ranking analysis
                self.ranking_analysis = self._analyze_rankings()
                
                return {
                    'performance_metrics': performance_metrics,
                    'comparison_metrics': self.comparison_metrics,
                    'ranking_analysis': self.ranking_analysis
                }
                
            def _rank_algorithms(self, performance_metrics, metric, reverse=False):
                """Rank algorithms by metric"""
                sorted_algorithms = sorted(performance_metrics.items(), 
                                         key=lambda x: x[1][metric], reverse=reverse)
                return [(algorithm, metrics[metric]) for algorithm, metrics in sorted_algorithms]
                
            def _analyze_rankings(self):
                """Analyze algorithm rankings"""
                rankings = {}
                
                # Overall performance ranking
                overall_scores = {}
                for algorithm in self.benchmark_results.keys():
                    score = 0
                    for ranking_type, ranking in self.comparison_metrics.items():
                        for i, (alg, _) in enumerate(ranking):
                            if alg == algorithm:
                                score += len(ranking) - i
                                break
                    overall_scores[algorithm] = score
                    
                rankings['overall_ranking'] = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Consistency analysis
                rankings['consistency'] = self._analyze_consistency()
                
                # Performance gaps
                rankings['performance_gaps'] = self._analyze_performance_gaps()
                
                return rankings
                
            def _analyze_consistency(self):
                """Analyze algorithm consistency"""
                consistency_scores = {}
                for algorithm in self.benchmark_results.keys():
                    results = self.benchmark_results[algorithm]
                    performance_values = [r['performance'] for r in results]
                    consistency_scores[algorithm] = 1.0 / (1.0 + np.std(performance_values))
                    
                return sorted(consistency_scores.items(), key=lambda x: x[1], reverse=True)
                
            def _analyze_performance_gaps(self):
                """Analyze performance gaps between algorithms"""
                performance_gaps = {}
                algorithms = list(self.benchmark_results.keys())
                
                for i, alg1 in enumerate(algorithms):
                    for j, alg2 in enumerate(algorithms):
                        if i != j:
                            gap_key = f"{alg1}_vs_{alg2}"
                            performance_gaps[gap_key] = {
                                'gap': abs(self.comparison_metrics['performance_rankings'][i][1] - 
                                         self.comparison_metrics['performance_rankings'][j][1]),
                                'better_algorithm': alg1 if self.comparison_metrics['performance_rankings'][i][1] > 
                                                 self.comparison_metrics['performance_rankings'][j][1] else alg2
                            }
                            
                return performance_gaps
                
            def get_benchmarking_summary(self):
                """Get benchmarking summary"""
                return {
                    'total_algorithms': len(self.benchmark_results),
                    'comparison_metrics': self.comparison_metrics,
                    'ranking_analysis': self.ranking_analysis,
                    'best_overall_algorithm': self.ranking_analysis['overall_ranking'][0][0],
                    'most_consistent_algorithm': self.ranking_analysis['consistency'][0][0]
                }
        
        # Test optimization benchmarking analytics
        analytics = OptimizationBenchmarkingAnalytics()
        
        # Create benchmark results
        benchmark_results = {
            'algorithm_a': [
                {'performance': 0.8, 'success': True, 'execution_time': 1.0},
                {'performance': 0.85, 'success': True, 'execution_time': 1.1},
                {'performance': 0.82, 'success': True, 'execution_time': 0.9}
            ],
            'algorithm_b': [
                {'performance': 0.75, 'success': True, 'execution_time': 0.8},
                {'performance': 0.78, 'success': True, 'execution_time': 0.85},
                {'performance': 0.77, 'success': True, 'execution_time': 0.82}
            ],
            'algorithm_c': [
                {'performance': 0.9, 'success': True, 'execution_time': 2.0},
                {'performance': 0.88, 'success': True, 'execution_time': 2.1},
                {'performance': 0.92, 'success': True, 'execution_time': 1.9}
            ]
        }
        
        # Analyze benchmark results
        analysis = analytics.analyze_benchmark_results(benchmark_results)
        
        # Verify analysis
        self.assertIn('performance_metrics', analysis)
        self.assertIn('comparison_metrics', analysis)
        self.assertIn('ranking_analysis', analysis)
        
        # Check benchmarking summary
        summary = analytics.get_benchmarking_summary()
        self.assertEqual(summary['total_algorithms'], 3)
        self.assertIn('comparison_metrics', summary)
        self.assertIn('ranking_analysis', summary)
        self.assertIn('best_overall_algorithm', summary)
        self.assertIn('most_consistent_algorithm', summary)

class TestOptimizationInsights(unittest.TestCase):
    """Test suite for optimization insights"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_optimization_insights_generator(self):
        """Test optimization insights generator"""
        class OptimizationInsightsGenerator:
            def __init__(self):
                self.insights_history = []
                self.insight_patterns = {}
                self.recommendations = []
                
            def generate_insights(self, optimization_data):
                """Generate optimization insights"""
                insights = []
                
                # Loss trend insights
                if len(optimization_data) > 1:
                    loss_trend = self._analyze_loss_trend(optimization_data)
                    if loss_trend['trend'] == 'improving':
                        insights.append({
                            'type': 'positive',
                            'category': 'loss_trend',
                            'message': 'Loss is consistently improving',
                            'confidence': loss_trend['confidence']
                        })
                    elif loss_trend['trend'] == 'plateauing':
                        insights.append({
                            'type': 'warning',
                            'category': 'loss_trend',
                            'message': 'Loss has plateaued, consider adjusting learning rate',
                            'confidence': loss_trend['confidence']
                        })
                        
                # Gradient insights
                gradient_insights = self._analyze_gradient_patterns(optimization_data)
                insights.extend(gradient_insights)
                
                # Learning rate insights
                lr_insights = self._analyze_learning_rate_patterns(optimization_data)
                insights.extend(lr_insights)
                
                # Convergence insights
                convergence_insights = self._analyze_convergence_patterns(optimization_data)
                insights.extend(convergence_insights)
                
                # Record insights
                self.insights_history.append({
                    'timestamp': time.time(),
                    'insights': insights,
                    'data_points': len(optimization_data)
                })
                
                return insights
                
            def _analyze_loss_trend(self, optimization_data):
                """Analyze loss trend"""
                losses = [point['loss'] for point in optimization_data]
                
                if len(losses) < 3:
                    return {'trend': 'insufficient_data', 'confidence': 0}
                    
                # Calculate trend
                recent_losses = losses[-5:] if len(losses) >= 5 else losses
                trend_slope = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                
                if trend_slope < -0.01:
                    return {'trend': 'improving', 'confidence': abs(trend_slope)}
                elif trend_slope > 0.01:
                    return {'trend': 'worsening', 'confidence': abs(trend_slope)}
                else:
                    return {'trend': 'plateauing', 'confidence': 1.0 - abs(trend_slope)}
                    
            def _analyze_gradient_patterns(self, optimization_data):
                """Analyze gradient patterns"""
                insights = []
                gradient_norms = [point['gradient_norm'] for point in optimization_data]
                
                # Check for gradient explosion
                if any(norm > 10.0 for norm in gradient_norms):
                    insights.append({
                        'type': 'critical',
                        'category': 'gradient',
                        'message': 'Gradient explosion detected',
                        'confidence': 1.0
                    })
                    
                # Check for gradient vanishing
                if any(norm < 1e-6 for norm in gradient_norms):
                    insights.append({
                        'type': 'warning',
                        'category': 'gradient',
                        'message': 'Gradient vanishing detected',
                        'confidence': 1.0
                    })
                    
                # Check gradient stability
                if len(gradient_norms) > 5:
                    gradient_std = np.std(gradient_norms)
                    if gradient_std > 5.0:
                        insights.append({
                            'type': 'warning',
                            'category': 'gradient',
                            'message': 'High gradient variance detected',
                            'confidence': min(gradient_std / 10.0, 1.0)
                        })
                        
                return insights
                
            def _analyze_learning_rate_patterns(self, optimization_data):
                """Analyze learning rate patterns"""
                insights = []
                learning_rates = [point['learning_rate'] for point in optimization_data]
                
                # Check for learning rate reduction
                if len(learning_rates) > 1:
                    lr_ratio = learning_rates[-1] / learning_rates[0]
                    if lr_ratio < 0.1:
                        insights.append({
                            'type': 'info',
                            'category': 'learning_rate',
                            'message': 'Learning rate significantly reduced',
                            'confidence': 1.0 - lr_ratio
                        })
                        
                # Check for learning rate stability
                if len(learning_rates) > 3:
                    lr_std = np.std(learning_rates)
                    if lr_std < 0.001:
                        insights.append({
                            'type': 'info',
                            'category': 'learning_rate',
                            'message': 'Learning rate is stable',
                            'confidence': 1.0 - lr_std
                        })
                        
                return insights
                
            def _analyze_convergence_patterns(self, optimization_data):
                """Analyze convergence patterns"""
                insights = []
                converged_count = sum(1 for point in optimization_data if point.get('converged', False))
                total_points = len(optimization_data)
                
                if total_points > 0:
                    convergence_rate = converged_count / total_points
                    if convergence_rate > 0.8:
                        insights.append({
                            'type': 'positive',
                            'category': 'convergence',
                            'message': 'High convergence rate achieved',
                            'confidence': convergence_rate
                        })
                    elif convergence_rate < 0.2:
                        insights.append({
                            'type': 'warning',
                            'category': 'convergence',
                            'message': 'Low convergence rate',
                            'confidence': 1.0 - convergence_rate
                        })
                        
                return insights
                
            def get_insights_summary(self):
                """Get insights summary"""
                if not self.insights_history:
                    return {}
                    
                all_insights = []
                for entry in self.insights_history:
                    all_insights.extend(entry['insights'])
                    
                insight_types = {}
                insight_categories = {}
                
                for insight in all_insights:
                    insight_type = insight['type']
                    insight_category = insight['category']
                    
                    if insight_type not in insight_types:
                        insight_types[insight_type] = 0
                    insight_types[insight_type] += 1
                    
                    if insight_category not in insight_categories:
                        insight_categories[insight_category] = 0
                    insight_categories[insight_category] += 1
                    
                return {
                    'total_insights': len(all_insights),
                    'insight_types': insight_types,
                    'insight_categories': insight_categories,
                    'avg_confidence': np.mean([insight['confidence'] for insight in all_insights]) if all_insights else 0
                }
        
        # Test optimization insights generator
        insights_generator = OptimizationInsightsGenerator()
        
        # Generate test optimization data
        optimization_data = []
        for i in range(10):
            optimization_data.append({
                'loss': 1.0 - i * 0.1 + np.random.normal(0, 0.05),
                'gradient_norm': np.random.uniform(0.1, 5.0),
                'learning_rate': 0.001 * (0.95 ** i),
                'converged': i > 7
            })
            
        # Generate insights
        insights = insights_generator.generate_insights(optimization_data)
        
        # Verify insights
        self.assertIsInstance(insights, list)
        for insight in insights:
            self.assertIn('type', insight)
            self.assertIn('category', insight)
            self.assertIn('message', insight)
            self.assertIn('confidence', insight)
            
        # Check insights summary
        summary = insights_generator.get_insights_summary()
        self.assertIn('total_insights', summary)
        self.assertIn('insight_types', summary)
        self.assertIn('insight_categories', summary)
        self.assertIn('avg_confidence', summary)

if __name__ == '__main__':
    unittest.main()





