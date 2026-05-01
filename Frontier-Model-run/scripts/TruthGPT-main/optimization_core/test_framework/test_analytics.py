"""
Test Analytics Framework
Comprehensive analytics and reporting for test execution
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics
import json
from datetime import datetime

from .test_metrics import TestMetricsCollector

@dataclass
class AnalyticsData:
    """Analytics data structure."""
    test_results: List[Any] = field(default_factory=list)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    optimization_metrics: Dict[str, Any] = field(default_factory=dict)
    efficiency_metrics: Dict[str, Any] = field(default_factory=dict)
    scalability_metrics: Dict[str, Any] = field(default_factory=dict)
    trend_data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class TestAnalytics:
    """Comprehensive test analytics and reporting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = TestMetricsCollector()
        self.analytics_history = []
        self.trend_data = defaultdict(list)
        
    def generate_comprehensive_report(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive test report with analytics."""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_skipped = 0
        total_timeouts = 0
        total_time = 0
        total_memory = 0
        category_stats = {}
        priority_stats = {}
        tag_stats = {}
        optimization_stats = {}
        quality_stats = {}
        reliability_stats = {}
        performance_stats = {}
        efficiency_stats = {}
        scalability_stats = {}
        
        for result in test_results:
            if result['success'] and result['result']:
                test_result = result['result']
                total_tests += len(test_result.test_results)
                total_passed += len([r for r in test_result.test_results if r.status == 'PASS'])
                total_failed += len([r for r in test_result.test_results if r.status == 'FAIL'])
                total_errors += len([r for r in test_result.test_results if r.status == 'ERROR'])
                total_skipped += len([r for r in test_result.test_results if r.status == 'SKIP'])
                total_timeouts += len([r for r in test_result.test_results if r.status == 'TIMEOUT'])
                total_time += result['execution_time']
                
                # Category statistics
                category = result.get('category', 'unknown')
                if category not in category_stats:
                    category_stats[category] = {
                        'tests': 0, 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0, 'timeouts': 0
                    }
                
                category_stats[category]['tests'] += len(test_result.test_results)
                category_stats[category]['passed'] += len([r for r in test_result.test_results if r.status == 'PASS'])
                category_stats[category]['failed'] += len([r for r in test_result.test_results if r.status == 'FAIL'])
                category_stats[category]['errors'] += len([r for r in test_result.test_results if r.status == 'ERROR'])
                category_stats[category]['skipped'] += len([r for r in test_result.test_results if r.status == 'SKIP'])
                category_stats[category]['timeouts'] += len([r for r in test_result.test_results if r.status == 'TIMEOUT'])
                
                # Priority statistics
                priority = result.get('priority', 'unknown')
                if priority not in priority_stats:
                    priority_stats[priority] = {'tests': 0, 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0, 'timeouts': 0}
                
                priority_stats[priority]['tests'] += len(test_result.test_results)
                priority_stats[priority]['passed'] += len([r for r in test_result.test_results if r.status == 'PASS'])
                priority_stats[priority]['failed'] += len([r for r in test_result.test_results if r.status == 'FAIL'])
                priority_stats[priority]['errors'] += len([r for r in test_result.test_results if r.status == 'ERROR'])
                priority_stats[priority]['skipped'] += len([r for r in test_result.test_results if r.status == 'SKIP'])
                priority_stats[priority]['timeouts'] += len([r for r in test_result.test_results if r.status == 'TIMEOUT'])
                
                # Tag statistics
                tags = result.get('tags', [])
                for tag in tags:
                    if tag not in tag_stats:
                        tag_stats[tag] = {'tests': 0, 'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0, 'timeouts': 0}
                    
                    tag_stats[tag]['tests'] += len(test_result.test_results)
                    tag_stats[tag]['passed'] += len([r for r in test_result.test_results if r.status == 'PASS'])
                    tag_stats[tag]['failed'] += len([r for r in test_result.test_results if r.status == 'FAIL'])
                    tag_stats[tag]['errors'] += len([r for r in test_result.test_results if r.status == 'ERROR'])
                    tag_stats[tag]['skipped'] += len([r for r in test_result.test_results if r.status == 'SKIP'])
                    tag_stats[tag]['timeouts'] += len([r for r in test_result.test_results if r.status == 'TIMEOUT'])
                
                # Optimization statistics
                optimization_type = result.get('optimization_type')
                if optimization_type:
                    if optimization_type not in optimization_stats:
                        optimization_stats[optimization_type] = {'tests': 0, 'passed': 0, 'failed': 0, 'errors': 0}
                    
                    optimization_stats[optimization_type]['tests'] += len(test_result.test_results)
                    optimization_stats[optimization_type]['passed'] += len([r for r in test_result.test_results if r.status == 'PASS'])
                    optimization_stats[optimization_type]['failed'] += len([r for r in test_result.test_results if r.status == 'FAIL'])
                    optimization_stats[optimization_type]['errors'] += len([r for r in test_result.test_results if r.status == 'ERROR'])
                
                # Quality statistics
                quality_scores = [r.quality_score for r in test_result.test_results if r.quality_score is not None]
                if quality_scores:
                    if category not in quality_stats:
                        quality_stats[category] = []
                    quality_stats[category].extend(quality_scores)
                
                # Reliability statistics
                reliability_scores = [r.reliability_score for r in test_result.test_results if r.reliability_score is not None]
                if reliability_scores:
                    if category not in reliability_stats:
                        reliability_stats[category] = []
                    reliability_stats[category].extend(reliability_scores)
                
                # Performance statistics
                performance_scores = [r.performance_score for r in test_result.test_results if r.performance_score is not None]
                if performance_scores:
                    if category not in performance_stats:
                        performance_stats[category] = []
                    performance_stats[category].extend(performance_scores)
                
                # Efficiency statistics
                efficiency_scores = [r.efficiency_score for r in test_result.test_results if r.efficiency_score is not None]
                if efficiency_scores:
                    if category not in efficiency_stats:
                        efficiency_stats[category] = []
                    efficiency_stats[category].extend(efficiency_scores)
                
                # Scalability statistics
                scalability_scores = [r.scalability_score for r in test_result.test_results if r.scalability_score is not None]
                if scalability_scores:
                    if category not in scalability_stats:
                        scalability_stats[category] = []
                    scalability_stats[category].extend(scalability_scores)
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Generate metrics summary
        metrics_summary = self.metrics_collector.generate_metrics_summary(test_results)
        
        # Generate performance recommendations
        recommendations = self.metrics_collector.get_performance_recommendations(test_results)
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'errors': total_errors,
                'skipped': total_skipped,
                'timeouts': total_timeouts,
                'success_rate': success_rate,
                'total_execution_time': total_time,
                'total_memory_usage': total_memory
            },
            'category_stats': category_stats,
            'priority_stats': priority_stats,
            'tag_stats': tag_stats,
            'optimization_stats': optimization_stats,
            'quality_stats': {k: {'avg': statistics.mean(v), 'min': min(v), 'max': max(v), 'count': len(v)} for k, v in quality_stats.items()},
            'reliability_stats': {k: {'avg': statistics.mean(v), 'min': min(v), 'max': max(v), 'count': len(v)} for k, v in reliability_stats.items()},
            'performance_stats': {k: {'avg': statistics.mean(v), 'min': min(v), 'max': max(v), 'count': len(v)} for k, v in performance_stats.items()},
            'efficiency_stats': {k: {'avg': statistics.mean(v), 'min': min(v), 'max': max(v), 'count': len(v)} for k, v in efficiency_stats.items()},
            'scalability_stats': {k: {'avg': statistics.mean(v), 'min': min(v), 'max': max(v), 'count': len(v)} for k, v in scalability_stats.items()},
            'detailed_results': test_results,
            'metrics_summary': metrics_summary,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': '3.9.7',
                'platform': 'linux',
                'cpu_count': 32,
                'memory_gb': 64.0,
                'execution_mode': 'ultra_intelligent',
                'max_workers': 32
            }
        }
        
        return report
    
    def analyze_trends(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in test execution."""
        trends = {
            'execution_time_trend': [],
            'memory_usage_trend': [],
            'success_rate_trend': [],
            'quality_trend': [],
            'reliability_trend': [],
            'performance_trend': [],
            'efficiency_trend': [],
            'scalability_trend': []
        }
        
        for result in test_results:
            if result['success'] and result['result']:
                test_result = result['result']
                
                # Execution time trend
                trends['execution_time_trend'].append(result['execution_time'])
                
                # Memory usage trend
                if hasattr(test_result, 'metrics'):
                    trends['memory_usage_trend'].append(test_result.metrics.memory_usage)
                
                # Success rate trend
                if hasattr(test_result, 'test_results'):
                    passed = len([r for r in test_result.test_results if r.status == 'PASS'])
                    total = len(test_result.test_results)
                    success_rate = (passed / total * 100) if total > 0 else 0
                    trends['success_rate_trend'].append(success_rate)
                
                # Quality trends
                if hasattr(test_result, 'test_results'):
                    quality_scores = [r.quality_score for r in test_result.test_results if r.quality_score is not None]
                    if quality_scores:
                        trends['quality_trend'].append(statistics.mean(quality_scores))
                
                # Reliability trends
                if hasattr(test_result, 'test_results'):
                    reliability_scores = [r.reliability_score for r in test_result.test_results if r.reliability_score is not None]
                    if reliability_scores:
                        trends['reliability_trend'].append(statistics.mean(reliability_scores))
                
                # Performance trends
                if hasattr(test_result, 'test_results'):
                    performance_scores = [r.performance_score for r in test_result.test_results if r.performance_score is not None]
                    if performance_scores:
                        trends['performance_trend'].append(statistics.mean(performance_scores))
                
                # Efficiency trends
                if hasattr(test_result, 'test_results'):
                    efficiency_scores = [r.efficiency_score for r in test_result.test_results if r.efficiency_score is not None]
                    if efficiency_scores:
                        trends['efficiency_trend'].append(statistics.mean(efficiency_scores))
                
                # Scalability trends
                if hasattr(test_result, 'test_results'):
                    scalability_scores = [r.scalability_score for r in test_result.test_results if r.scalability_score is not None]
                    if scalability_scores:
                        trends['scalability_trend'].append(statistics.mean(scalability_scores))
        
        return trends
    
    def generate_insights(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from test execution."""
        insights = {
            'performance_insights': [],
            'quality_insights': [],
            'reliability_insights': [],
            'optimization_insights': [],
            'efficiency_insights': [],
            'scalability_insights': [],
            'recommendations': []
        }
        
        # Analyze performance patterns
        performance = self.metrics_collector.analyze_performance_patterns(test_results)
        
        # Performance insights
        if performance.slow_tests:
            insights['performance_insights'].append(f"Found {len(performance.slow_tests)} slow tests that may need optimization")
        
        if performance.memory_leaks:
            insights['performance_insights'].append(f"Detected {len(performance.memory_leaks)} potential memory leaks")
        
        # Quality insights
        quality_scores = []
        for result in test_results:
            if result['success'] and result['result']:
                test_result = result['result']
                if hasattr(test_result, 'test_results'):
                    quality_scores.extend([r.quality_score for r in test_result.test_results if r.quality_score is not None])
        
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            if avg_quality < 0.8:
                insights['quality_insights'].append(f"Average quality score is {avg_quality:.2f}, consider improving test quality")
            else:
                insights['quality_insights'].append(f"Good quality score: {avg_quality:.2f}")
        
        # Reliability insights
        reliability_scores = []
        for result in test_results:
            if result['success'] and result['result']:
                test_result = result['result']
                if hasattr(test_result, 'test_results'):
                    reliability_scores.extend([r.reliability_score for r in test_result.test_results if r.reliability_score is not None])
        
        if reliability_scores:
            avg_reliability = statistics.mean(reliability_scores)
            if avg_reliability < 0.9:
                insights['reliability_insights'].append(f"Average reliability score is {avg_reliability:.2f}, consider improving test reliability")
            else:
                insights['reliability_insights'].append(f"Good reliability score: {avg_reliability:.2f}")
        
        # Optimization insights
        optimization_scores = []
        for result in test_results:
            if result['success'] and result['result']:
                test_result = result['result']
                if hasattr(test_result, 'test_results'):
                    optimization_scores.extend([r.optimization_score for r in test_result.test_results if r.optimization_score is not None])
        
        if optimization_scores:
            avg_optimization = statistics.mean(optimization_scores)
            if avg_optimization < 0.8:
                insights['optimization_insights'].append(f"Average optimization score is {avg_optimization:.2f}, consider improving optimization")
            else:
                insights['optimization_insights'].append(f"Good optimization score: {avg_optimization:.2f}")
        
        # Efficiency insights
        efficiency_scores = []
        for result in test_results:
            if result['success'] and result['result']:
                test_result = result['result']
                if hasattr(test_result, 'test_results'):
                    efficiency_scores.extend([r.efficiency_score for r in test_result.test_results if r.efficiency_score is not None])
        
        if efficiency_scores:
            avg_efficiency = statistics.mean(efficiency_scores)
            if avg_efficiency < 0.8:
                insights['efficiency_insights'].append(f"Average efficiency score is {avg_efficiency:.2f}, consider improving efficiency")
            else:
                insights['efficiency_insights'].append(f"Good efficiency score: {avg_efficiency:.2f}")
        
        # Scalability insights
        scalability_scores = []
        for result in test_results:
            if result['success'] and result['result']:
                test_result = result['result']
                if hasattr(test_result, 'test_results'):
                    scalability_scores.extend([r.scalability_score for r in test_result.test_results if r.scalability_score is not None])
        
        if scalability_scores:
            avg_scalability = statistics.mean(scalability_scores)
            if avg_scalability < 0.8:
                insights['scalability_insights'].append(f"Average scalability score is {avg_scalability:.2f}, consider improving scalability")
            else:
                insights['scalability_insights'].append(f"Good scalability score: {avg_scalability:.2f}")
        
        # Generate recommendations
        recommendations = self.metrics_collector.get_performance_recommendations(test_results)
        insights['recommendations'] = recommendations
        
        return insights
    
    def export_analytics_data(self, test_results: List[Dict[str, Any]], output_file: str):
        """Export analytics data to file."""
        try:
            analytics_data = AnalyticsData()
            analytics_data.test_results = test_results
            analytics_data.execution_metrics = self.metrics_collector.generate_metrics_summary(test_results)
            analytics_data.trend_data = self.analyze_trends(test_results)
            analytics_data.recommendations = self.metrics_collector.get_performance_recommendations(test_results)
            
            with open(output_file, 'w') as f:
                json.dump(analytics_data.__dict__, f, indent=2, default=str)
            
            self.logger.info(f"Analytics data exported to {output_file}")
        except Exception as e:
            self.logger.error(f"Error exporting analytics data: {e}")
    
    def get_historical_analytics(self) -> Dict[str, Any]:
        """Get historical analytics data."""
        if not self.analytics_history:
            return {}
        
        # Analyze historical trends
        historical_trends = {
            'execution_time_history': [],
            'success_rate_history': [],
            'quality_history': [],
            'reliability_history': [],
            'performance_history': [],
            'efficiency_history': [],
            'scalability_history': []
        }
        
        for historical_data in self.analytics_history:
            if 'execution_time' in historical_data:
                historical_trends['execution_time_history'].append(historical_data['execution_time'])
            if 'success_rate' in historical_data:
                historical_trends['success_rate_history'].append(historical_data['success_rate'])
            if 'quality_score' in historical_data:
                historical_trends['quality_history'].append(historical_data['quality_score'])
            if 'reliability_score' in historical_data:
                historical_trends['reliability_history'].append(historical_data['reliability_score'])
            if 'performance_score' in historical_data:
                historical_trends['performance_history'].append(historical_data['performance_score'])
            if 'efficiency_score' in historical_data:
                historical_trends['efficiency_history'].append(historical_data['efficiency_score'])
            if 'scalability_score' in historical_data:
                historical_trends['scalability_history'].append(historical_data['scalability_score'])
        
        return historical_trends











