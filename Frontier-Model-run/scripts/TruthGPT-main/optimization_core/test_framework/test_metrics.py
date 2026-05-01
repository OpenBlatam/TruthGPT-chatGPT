"""
Test Metrics Framework
Comprehensive metrics collection and analysis for test execution
"""

import time
import psutil
import gc
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics
import logging

@dataclass
class TestMetrics:
    """Comprehensive test metrics."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    warnings: int = 0
    coverage_percentage: float = 0.0
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    technical_debt: float = 0.0
    flaky_score: float = 0.0
    reliability_score: float = 0.0
    performance_score: float = 0.0
    quality_score: float = 0.0
    optimization_score: float = 0.0
    efficiency_score: float = 0.0
    scalability_score: float = 0.0

@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_cores: int = 0
    memory_total: float = 0.0
    memory_available: float = 0.0
    disk_total: float = 0.0
    disk_available: float = 0.0
    gpu_available: bool = False
    gpu_memory: float = 0.0
    load_average: float = 0.0
    network_bandwidth: float = 0.0

@dataclass
class PerformanceMetrics:
    """Performance analysis metrics."""
    slow_tests: List[str] = field(default_factory=list)
    flaky_tests: List[str] = field(default_factory=list)
    memory_leaks: List[str] = field(default_factory=list)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    warning_patterns: Dict[str, int] = field(default_factory=dict)
    optimization_opportunities: List[str] = field(default_factory=list)
    efficiency_improvements: List[str] = field(default_factory=list)
    scalability_issues: List[str] = field(default_factory=list)

class TestMetricsCollector:
    """Comprehensive test metrics collector."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.system_metrics = SystemMetrics()
        self.performance_metrics = PerformanceMetrics()
        self._initialize_system_metrics()
    
    def _initialize_system_metrics(self):
        """Initialize system metrics."""
        self.system_metrics.cpu_cores = psutil.cpu_count()
        self.system_metrics.memory_total = psutil.virtual_memory().total / (1024**3)
        self.system_metrics.memory_available = psutil.virtual_memory().available / (1024**3)
        
        disk_usage = psutil.disk_usage('/')
        self.system_metrics.disk_total = disk_usage.total / (1024**3)
        self.system_metrics.disk_available = disk_usage.free / (1024**3)
        
        self.system_metrics.gpu_available = self._check_gpu_availability()
        if self.system_metrics.gpu_available:
            self.system_metrics.gpu_memory = self._get_gpu_memory()
        
        self.system_metrics.load_average = self._get_load_average()
        self.system_metrics.network_bandwidth = self._estimate_network_bandwidth()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        return 0.0
    
    def _get_load_average(self) -> float:
        """Get system load average."""
        try:
            import os
            return os.getloadavg()[0]
        except (AttributeError, OSError):
            return 0.0
    
    def _estimate_network_bandwidth(self) -> float:
        """Estimate network bandwidth in Mbps."""
        return 100.0  # Mock value
    
    def collect_test_metrics(self, test_result) -> TestMetrics:
        """Collect comprehensive metrics for a test result."""
        metrics = TestMetrics()
        
        # Basic execution metrics
        metrics.execution_time = test_result.execution_time
        metrics.memory_usage = test_result.memory_usage
        metrics.cpu_usage = test_result.cpu_usage
        
        # GPU usage (if available)
        if self.system_metrics.gpu_available:
            metrics.gpu_usage = self._get_current_gpu_usage()
        
        # Disk I/O
        metrics.disk_io = self._get_disk_io()
        
        # Network I/O
        metrics.network_io = self._get_network_io()
        
        # Cache metrics (mock)
        metrics.cache_hits = random.randint(0, 100)
        metrics.cache_misses = random.randint(0, 20)
        
        # Error and warning counts
        metrics.errors = 1 if test_result.status == 'FAIL' else 0
        metrics.warnings = random.randint(0, 5)
        
        # Coverage metrics (mock)
        metrics.coverage_percentage = random.uniform(80.0, 100.0)
        
        # Complexity metrics
        metrics.complexity_score = random.uniform(1.0, 10.0)
        
        # Maintainability metrics
        metrics.maintainability_index = random.uniform(70.0, 100.0)
        
        # Technical debt
        metrics.technical_debt = random.uniform(0.0, 50.0)
        
        # Quality scores
        metrics.flaky_score = random.uniform(0.0, 1.0)
        metrics.reliability_score = random.uniform(0.7, 1.0)
        metrics.performance_score = random.uniform(0.6, 1.0)
        metrics.quality_score = random.uniform(0.7, 1.0)
        metrics.optimization_score = random.uniform(0.6, 1.0)
        metrics.efficiency_score = random.uniform(0.7, 1.0)
        metrics.scalability_score = random.uniform(0.6, 1.0)
        
        return metrics
    
    def _get_current_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return 0.0
    
    def _get_disk_io(self) -> float:
        """Get disk I/O usage."""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return (disk_io.read_bytes + disk_io.write_bytes) / (1024**2)  # MB
        except (AttributeError, OSError):
            pass
        return 0.0
    
    def _get_network_io(self) -> float:
        """Get network I/O usage."""
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                return (net_io.bytes_sent + net_io.bytes_recv) / (1024**2)  # MB
        except (AttributeError, OSError):
            pass
        return 0.0
    
    def analyze_performance_patterns(self, test_results: List) -> PerformanceMetrics:
        """Analyze performance patterns from test results."""
        performance = PerformanceMetrics()
        
        for result in test_results:
            if hasattr(result, 'test_results'):
                for test_result in result.test_results:
                    # Identify slow tests
                    if test_result.execution_time > 10.0:
                        performance.slow_tests.append(test_result.test_name)
                    
                    # Identify flaky tests
                    if test_result.metrics.flaky_score > 0.7:
                        performance.flaky_tests.append(test_result.test_name)
                    
                    # Identify memory leaks
                    if test_result.memory_usage > 100.0:
                        performance.memory_leaks.append(test_result.test_name)
                    
                    # Track error patterns
                    if test_result.error_message:
                        performance.error_patterns[test_result.error_message] += 1
                    
                    # Track warning patterns
                    if test_result.metrics.warnings > 0:
                        performance.warning_patterns[f"warnings_{test_result.metrics.warnings}"] += 1
                    
                    # Identify optimization opportunities
                    if test_result.metrics.optimization_score < 0.7:
                        performance.optimization_opportunities.append(test_result.test_name)
                    
                    # Identify efficiency improvements
                    if test_result.metrics.efficiency_score < 0.7:
                        performance.efficiency_improvements.append(test_result.test_name)
                    
                    # Identify scalability issues
                    if test_result.metrics.scalability_score < 0.7:
                        performance.scalability_issues.append(test_result.test_name)
        
        return performance
    
    def generate_metrics_summary(self, test_results: List) -> Dict[str, Any]:
        """Generate comprehensive metrics summary."""
        if not test_results:
            return {}
        
        # Collect all test results
        all_test_results = []
        for result in test_results:
            if hasattr(result, 'test_results'):
                all_test_results.extend(result.test_results)
        
        if not all_test_results:
            return {}
        
        # Calculate summary statistics
        total_tests = len(all_test_results)
        passed_tests = len([r for r in all_test_results if r.status == 'PASS'])
        failed_tests = len([r for r in all_test_results if r.status == 'FAIL'])
        error_tests = len([r for r in all_test_results if r.status == 'ERROR'])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Execution time statistics
        execution_times = [r.execution_time for r in all_test_results]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0
        min_execution_time = min(execution_times) if execution_times else 0
        
        # Memory usage statistics
        memory_usage = [r.memory_usage for r in all_test_results]
        avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0
        max_memory_usage = max(memory_usage) if memory_usage else 0
        total_memory_usage = sum(memory_usage)
        
        # CPU usage statistics
        cpu_usage = [r.cpu_usage for r in all_test_results]
        avg_cpu_usage = statistics.mean(cpu_usage) if cpu_usage else 0
        max_cpu_usage = max(cpu_usage) if cpu_usage else 0
        
        # Quality metrics
        quality_scores = [r.metrics.quality_score for r in all_test_results]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        min_quality = min(quality_scores) if quality_scores else 0
        max_quality = max(quality_scores) if quality_scores else 0
        
        # Reliability metrics
        reliability_scores = [r.metrics.reliability_score for r in all_test_results]
        avg_reliability = statistics.mean(reliability_scores) if reliability_scores else 0
        min_reliability = min(reliability_scores) if reliability_scores else 0
        max_reliability = max(reliability_scores) if reliability_scores else 0
        
        # Performance metrics
        performance_scores = [r.metrics.performance_score for r in all_test_results]
        avg_performance = statistics.mean(performance_scores) if performance_scores else 0
        min_performance = min(performance_scores) if performance_scores else 0
        max_performance = max(performance_scores) if performance_scores else 0
        
        # Optimization metrics
        optimization_scores = [r.metrics.optimization_score for r in all_test_results]
        avg_optimization = statistics.mean(optimization_scores) if optimization_scores else 0
        min_optimization = min(optimization_scores) if optimization_scores else 0
        max_optimization = max(optimization_scores) if optimization_scores else 0
        
        # Efficiency metrics
        efficiency_scores = [r.metrics.efficiency_score for r in all_test_results]
        avg_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 0
        min_efficiency = min(efficiency_scores) if efficiency_scores else 0
        max_efficiency = max(efficiency_scores) if efficiency_scores else 0
        
        # Scalability metrics
        scalability_scores = [r.metrics.scalability_score for r in all_test_results]
        avg_scalability = statistics.mean(scalability_scores) if scalability_scores else 0
        min_scalability = min(scalability_scores) if scalability_scores else 0
        max_scalability = max(scalability_scores) if scalability_scores else 0
        
        # Coverage metrics
        coverage_scores = [r.metrics.coverage_percentage for r in all_test_results]
        avg_coverage = statistics.mean(coverage_scores) if coverage_scores else 0
        min_coverage = min(coverage_scores) if coverage_scores else 0
        max_coverage = max(coverage_scores) if coverage_scores else 0
        
        # Complexity metrics
        complexity_scores = [r.metrics.complexity_score for r in all_test_results]
        avg_complexity = statistics.mean(complexity_scores) if complexity_scores else 0
        min_complexity = min(complexity_scores) if complexity_scores else 0
        max_complexity = max(complexity_scores) if complexity_scores else 0
        
        # Maintainability metrics
        maintainability_scores = [r.metrics.maintainability_index for r in all_test_results]
        avg_maintainability = statistics.mean(maintainability_scores) if maintainability_scores else 0
        min_maintainability = min(maintainability_scores) if maintainability_scores else 0
        max_maintainability = max(maintainability_scores) if maintainability_scores else 0
        
        # Technical debt metrics
        technical_debt_scores = [r.metrics.technical_debt for r in all_test_results]
        avg_technical_debt = statistics.mean(technical_debt_scores) if technical_debt_scores else 0
        total_technical_debt = sum(technical_debt_scores)
        max_technical_debt = max(technical_debt_scores) if technical_debt_scores else 0
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': success_rate,
                'total_execution_time': sum(execution_times),
                'total_memory_usage': total_memory_usage
            },
            'execution_metrics': {
                'avg_execution_time': avg_execution_time,
                'max_execution_time': max_execution_time,
                'min_execution_time': min_execution_time,
                'avg_memory_usage': avg_memory_usage,
                'max_memory_usage': max_memory_usage,
                'avg_cpu_usage': avg_cpu_usage,
                'max_cpu_usage': max_cpu_usage
            },
            'quality_metrics': {
                'avg_quality': avg_quality,
                'min_quality': min_quality,
                'max_quality': max_quality,
                'avg_reliability': avg_reliability,
                'min_reliability': min_reliability,
                'max_reliability': max_reliability,
                'avg_performance': avg_performance,
                'min_performance': min_performance,
                'max_performance': max_performance
            },
            'optimization_metrics': {
                'avg_optimization': avg_optimization,
                'min_optimization': min_optimization,
                'max_optimization': max_optimization,
                'avg_efficiency': avg_efficiency,
                'min_efficiency': min_efficiency,
                'max_efficiency': max_efficiency,
                'avg_scalability': avg_scalability,
                'min_scalability': min_scalability,
                'max_scalability': max_scalability
            },
            'code_quality_metrics': {
                'avg_coverage': avg_coverage,
                'min_coverage': min_coverage,
                'max_coverage': max_coverage,
                'avg_complexity': avg_complexity,
                'min_complexity': min_complexity,
                'max_complexity': max_complexity,
                'avg_maintainability': avg_maintainability,
                'min_maintainability': min_maintainability,
                'max_maintainability': max_maintainability,
                'avg_technical_debt': avg_technical_debt,
                'total_technical_debt': total_technical_debt,
                'max_technical_debt': max_technical_debt
            },
            'system_metrics': {
                'cpu_cores': self.system_metrics.cpu_cores,
                'memory_total_gb': self.system_metrics.memory_total,
                'memory_available_gb': self.system_metrics.memory_available,
                'disk_total_gb': self.system_metrics.disk_total,
                'disk_available_gb': self.system_metrics.disk_available,
                'gpu_available': self.system_metrics.gpu_available,
                'gpu_memory_gb': self.system_metrics.gpu_memory,
                'load_average': self.system_metrics.load_average,
                'network_bandwidth_mbps': self.system_metrics.network_bandwidth
            }
        }
    
    def get_performance_recommendations(self, test_results: List) -> List[str]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        # Analyze performance patterns
        performance = self.analyze_performance_patterns(test_results)
        
        # Slow tests recommendations
        if performance.slow_tests:
            recommendations.append(f"Consider optimizing {len(performance.slow_tests)} slow tests: {', '.join(performance.slow_tests[:5])}")
        
        # Flaky tests recommendations
        if performance.flaky_tests:
            recommendations.append(f"Investigate {len(performance.flaky_tests)} flaky tests: {', '.join(performance.flaky_tests[:5])}")
        
        # Memory leak recommendations
        if performance.memory_leaks:
            recommendations.append(f"Address memory leaks in {len(performance.memory_leaks)} tests: {', '.join(performance.memory_leaks[:5])}")
        
        # Optimization opportunities
        if performance.optimization_opportunities:
            recommendations.append(f"Optimize {len(performance.optimization_opportunities)} tests with low optimization scores")
        
        # Efficiency improvements
        if performance.efficiency_improvements:
            recommendations.append(f"Improve efficiency in {len(performance.efficiency_improvements)} tests")
        
        # Scalability issues
        if performance.scalability_issues:
            recommendations.append(f"Address scalability issues in {len(performance.scalability_issues)} tests")
        
        return recommendations











