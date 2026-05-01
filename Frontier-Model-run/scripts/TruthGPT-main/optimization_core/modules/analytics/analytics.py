"""
TruthGPT Advanced Analytics Module
Advanced analytics and insights for TruthGPT models
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTAnalyticsConfig:
    """Configuration for TruthGPT analytics."""
    # Analytics settings
    enable_performance_analytics: bool = True
    enable_model_analytics: bool = True
    enable_training_analytics: bool = True
    enable_inference_analytics: bool = True
    
    # Visualization settings
    enable_plots: bool = True
    plot_style: str = "seaborn"  # seaborn, matplotlib, plotly
    plot_format: str = "png"  # png, svg, pdf, html
    plot_dpi: int = 300
    
    # Export settings
    enable_export: bool = True
    export_format: str = "json"  # json, csv, excel, html
    export_path: str = "./analytics_results"
    
    # Advanced analytics
    enable_correlation_analysis: bool = True
    enable_trend_analysis: bool = True
    enable_anomaly_detection: bool = True
    enable_clustering: bool = True
    
    # Performance metrics
    enable_throughput_analysis: bool = True
    enable_memory_analysis: bool = True
    enable_latency_analysis: bool = True
    enable_accuracy_analysis: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_performance_analytics': self.enable_performance_analytics,
            'enable_model_analytics': self.enable_model_analytics,
            'enable_training_analytics': self.enable_training_analytics,
            'enable_inference_analytics': self.enable_inference_analytics,
            'enable_plots': self.enable_plots,
            'plot_style': self.plot_style,
            'plot_format': self.plot_format,
            'plot_dpi': self.plot_dpi,
            'enable_export': self.enable_export,
            'export_format': self.export_format,
            'export_path': self.export_path,
            'enable_correlation_analysis': self.enable_correlation_analysis,
            'enable_trend_analysis': self.enable_trend_analysis,
            'enable_anomaly_detection': self.enable_anomaly_detection,
            'enable_clustering': self.enable_clustering,
            'enable_throughput_analysis': self.enable_throughput_analysis,
            'enable_memory_analysis': self.enable_memory_analysis,
            'enable_latency_analysis': self.enable_latency_analysis,
            'enable_accuracy_analysis': self.enable_accuracy_analysis
        }

class TruthGPTPerformanceAnalytics:
    """Advanced performance analytics for TruthGPT."""
    
    def __init__(self, config: TruthGPTAnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Analytics state
        self.performance_data = []
        self.performance_metrics = {}
        
        # Setup plotting
        if config.plot_style == "seaborn":
            sns.set_style("whitegrid")
        elif config.plot_style == "matplotlib":
            plt.style.use("default")
    
    def analyze_throughput(self, throughput_data: List[float]) -> Dict[str, Any]:
        """Analyze throughput performance."""
        if not throughput_data:
            return {}
        
        throughput_array = np.array(throughput_data)
        
        analysis = {
            'mean_throughput': np.mean(throughput_array),
            'median_throughput': np.median(throughput_array),
            'std_throughput': np.std(throughput_array),
            'min_throughput': np.min(throughput_array),
            'max_throughput': np.max(throughput_array),
            'percentile_25': np.percentile(throughput_array, 25),
            'percentile_75': np.percentile(throughput_array, 75),
            'percentile_90': np.percentile(throughput_array, 90),
            'percentile_95': np.percentile(throughput_array, 95),
            'percentile_99': np.percentile(throughput_array, 99)
        }
        
        self.logger.info(f"Throughput analysis completed - Mean: {analysis['mean_throughput']:.2f}")
        return analysis
    
    def analyze_memory_usage(self, memory_data: List[float]) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not memory_data:
            return {}
        
        memory_array = np.array(memory_data)
        
        analysis = {
            'mean_memory': np.mean(memory_array),
            'median_memory': np.median(memory_array),
            'std_memory': np.std(memory_array),
            'min_memory': np.min(memory_array),
            'max_memory': np.max(memory_array),
            'memory_growth_rate': self._calculate_growth_rate(memory_array),
            'memory_peak_usage': np.max(memory_array),
            'memory_efficiency': np.mean(memory_array) / np.max(memory_array) if np.max(memory_array) > 0 else 0
        }
        
        self.logger.info(f"Memory analysis completed - Mean: {analysis['mean_memory']:.2f} MB")
        return analysis
    
    def analyze_latency(self, latency_data: List[float]) -> Dict[str, Any]:
        """Analyze latency performance."""
        if not latency_data:
            return {}
        
        latency_array = np.array(latency_data)
        
        analysis = {
            'mean_latency': np.mean(latency_array),
            'median_latency': np.median(latency_array),
            'std_latency': np.std(latency_array),
            'min_latency': np.min(latency_array),
            'max_latency': np.max(latency_array),
            'p95_latency': np.percentile(latency_array, 95),
            'p99_latency': np.percentile(latency_array, 99),
            'latency_variance': np.var(latency_array),
            'latency_stability': 1.0 - (np.std(latency_array) / np.mean(latency_array)) if np.mean(latency_array) > 0 else 0
        }
        
        self.logger.info(f"Latency analysis completed - Mean: {analysis['mean_latency']:.2f} ms")
        return analysis
    
    def analyze_accuracy(self, accuracy_data: List[float]) -> Dict[str, Any]:
        """Analyze accuracy performance."""
        if not accuracy_data:
            return {}
        
        accuracy_array = np.array(accuracy_data)
        
        analysis = {
            'mean_accuracy': np.mean(accuracy_array),
            'median_accuracy': np.median(accuracy_array),
            'std_accuracy': np.std(accuracy_array),
            'min_accuracy': np.min(accuracy_array),
            'max_accuracy': np.max(accuracy_array),
            'accuracy_trend': self._calculate_trend(accuracy_array),
            'accuracy_consistency': 1.0 - (np.std(accuracy_array) / np.mean(accuracy_array)) if np.mean(accuracy_array) > 0 else 0,
            'accuracy_improvement': accuracy_array[-1] - accuracy_array[0] if len(accuracy_array) > 1 else 0
        }
        
        self.logger.info(f"Accuracy analysis completed - Mean: {analysis['mean_accuracy']:.4f}")
        return analysis
    
    def _calculate_growth_rate(self, data: np.ndarray) -> float:
        """Calculate growth rate of data."""
        if len(data) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        return slope
    
    def _calculate_trend(self, data: np.ndarray) -> str:
        """Calculate trend of data."""
        if len(data) < 2:
            return "stable"
        
        # Simple trend analysis
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        if second_mean > first_mean * 1.05:
            return "increasing"
        elif second_mean < first_mean * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def create_performance_report(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Create comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'performance_metrics': {}
        }
        
        # Analyze throughput
        if 'throughput' in performance_data:
            report['performance_metrics']['throughput'] = self.analyze_throughput(performance_data['throughput'])
        
        # Analyze memory usage
        if 'memory' in performance_data:
            report['performance_metrics']['memory'] = self.analyze_memory_usage(performance_data['memory'])
        
        # Analyze latency
        if 'latency' in performance_data:
            report['performance_metrics']['latency'] = self.analyze_latency(performance_data['latency'])
        
        # Analyze accuracy
        if 'accuracy' in performance_data:
            report['performance_metrics']['accuracy'] = self.analyze_accuracy(performance_data['accuracy'])
        
        self.logger.info("Performance report created")
        return report

class TruthGPTModelAnalytics:
    """Advanced model analytics for TruthGPT."""
    
    def __init__(self, config: TruthGPTAnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Model analytics state
        self.model_data = {}
        self.model_metrics = {}
    
    def analyze_model_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture."""
        analysis = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            'num_layers': len(list(model.modules())),
            'num_parameters_per_layer': {}
        }
        
        # Analyze parameters per layer
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0:
                layer_params = sum(p.numel() for p in module.parameters())
                analysis['num_parameters_per_layer'][name] = layer_params
        
        self.logger.info(f"Model architecture analysis completed - {analysis['total_parameters']:,} parameters")
        return analysis
    
    def analyze_model_complexity(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model complexity."""
        # Calculate FLOPs (simplified)
        total_flops = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Simplified FLOP calculation for linear layers
                total_flops += module.in_features * module.out_features
        
        analysis = {
            'total_flops': total_flops,
            'flops_per_parameter': total_flops / sum(p.numel() for p in model.parameters()) if sum(p.numel() for p in model.parameters()) > 0 else 0,
            'model_depth': self._calculate_model_depth(model),
            'model_width': self._calculate_model_width(model),
            'complexity_score': self._calculate_complexity_score(model)
        }
        
        self.logger.info(f"Model complexity analysis completed - {analysis['total_flops']:,} FLOPs")
        return analysis
    
    def analyze_model_weights(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model weights distribution."""
        all_weights = []
        weight_stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                all_weights.extend(weights)
                
                weight_stats[name] = {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights),
                    'zero_ratio': np.sum(weights == 0) / len(weights)
                }
        
        all_weights = np.array(all_weights)
        
        analysis = {
            'global_weight_stats': {
                'mean': np.mean(all_weights),
                'std': np.std(all_weights),
                'min': np.min(all_weights),
                'max': np.max(all_weights),
                'zero_ratio': np.sum(all_weights == 0) / len(all_weights)
            },
            'layer_weight_stats': weight_stats,
            'weight_distribution': {
                'percentile_25': np.percentile(all_weights, 25),
                'percentile_50': np.percentile(all_weights, 50),
                'percentile_75': np.percentile(all_weights, 75),
                'percentile_90': np.percentile(all_weights, 90),
                'percentile_95': np.percentile(all_weights, 95),
                'percentile_99': np.percentile(all_weights, 99)
            }
        }
        
        self.logger.info("Model weights analysis completed")
        return analysis
    
    def _calculate_model_depth(self, model: nn.Module) -> int:
        """Calculate model depth."""
        depth = 0
        for module in model.modules():
            if len(list(module.children())) == 0:  # Leaf module
                depth += 1
        return depth
    
    def _calculate_model_width(self, model: nn.Module) -> int:
        """Calculate model width (max parameters in a layer)."""
        max_params = 0
        for module in model.modules():
            if len(list(module.parameters())) > 0:
                layer_params = sum(p.numel() for p in module.parameters())
                max_params = max(max_params, layer_params)
        return max_params
    
    def _calculate_complexity_score(self, model: nn.Module) -> float:
        """Calculate model complexity score."""
        total_params = sum(p.numel() for p in model.parameters())
        model_depth = self._calculate_model_depth(model)
        model_width = self._calculate_model_width(model)
        
        # Simple complexity score
        complexity_score = (total_params * model_depth * model_width) / (1000000 * 100 * 1000)
        return min(complexity_score, 1.0)  # Cap at 1.0

class TruthGPTTrainingAnalytics:
    """Advanced training analytics for TruthGPT."""
    
    def __init__(self, config: TruthGPTAnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Training analytics state
        self.training_data = []
        self.training_metrics = {}
    
    def analyze_training_curves(self, training_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze training curves."""
        analysis = {}
        
        for metric_name, metric_values in training_data.items():
            if not metric_values:
                continue
            
            values = np.array(metric_values)
            
            analysis[metric_name] = {
                'initial_value': values[0] if len(values) > 0 else 0,
                'final_value': values[-1] if len(values) > 0 else 0,
                'best_value': np.min(values) if 'loss' in metric_name.lower() else np.max(values),
                'worst_value': np.max(values) if 'loss' in metric_name.lower() else np.min(values),
                'improvement': values[-1] - values[0] if len(values) > 0 else 0,
                'stability': 1.0 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0,
                'trend': self._calculate_trend(values)
            }
        
        self.logger.info("Training curves analysis completed")
        return analysis
    
    def analyze_convergence(self, loss_data: List[float]) -> Dict[str, Any]:
        """Analyze training convergence."""
        if not loss_data:
            return {}
        
        loss_array = np.array(loss_data)
        
        # Calculate convergence metrics
        convergence_analysis = {
            'convergence_epoch': self._find_convergence_epoch(loss_array),
            'convergence_loss': loss_array[self._find_convergence_epoch(loss_array)] if self._find_convergence_epoch(loss_array) < len(loss_array) else loss_array[-1],
            'final_loss': loss_array[-1],
            'loss_reduction': loss_array[0] - loss_array[-1],
            'convergence_speed': self._calculate_convergence_speed(loss_array),
            'stability_epochs': self._calculate_stability_epochs(loss_array)
        }
        
        self.logger.info(f"Convergence analysis completed - Convergence at epoch {convergence_analysis['convergence_epoch']}")
        return convergence_analysis
    
    def analyze_learning_rate_impact(self, lr_data: List[float], loss_data: List[float]) -> Dict[str, Any]:
        """Analyze learning rate impact on training."""
        if not lr_data or not loss_data or len(lr_data) != len(loss_data):
            return {}
        
        lr_array = np.array(lr_data)
        loss_array = np.array(loss_data)
        
        # Calculate correlation
        correlation = np.corrcoef(lr_array, loss_array)[0, 1]
        
        analysis = {
            'lr_loss_correlation': correlation,
            'lr_impact_score': abs(correlation),
            'optimal_lr_range': self._find_optimal_lr_range(lr_array, loss_array),
            'lr_sensitivity': self._calculate_lr_sensitivity(lr_array, loss_array)
        }
        
        self.logger.info(f"Learning rate impact analysis completed - Correlation: {correlation:.4f}")
        return analysis
    
    def _find_convergence_epoch(self, loss_array: np.ndarray) -> int:
        """Find convergence epoch."""
        if len(loss_array) < 10:
            return len(loss_array) - 1
        
        # Simple convergence detection
        window_size = min(10, len(loss_array) // 4)
        for i in range(window_size, len(loss_array)):
            window = loss_array[i-window_size:i]
            if np.std(window) < np.mean(window) * 0.01:  # 1% variation
                return i
        
        return len(loss_array) - 1
    
    def _calculate_convergence_speed(self, loss_array: np.ndarray) -> float:
        """Calculate convergence speed."""
        if len(loss_array) < 2:
            return 0.0
        
        # Calculate how quickly loss decreases
        initial_loss = loss_array[0]
        final_loss = loss_array[-1]
        epochs = len(loss_array)
        
        if initial_loss == final_loss:
            return 0.0
        
        speed = (initial_loss - final_loss) / epochs
        return speed
    
    def _calculate_stability_epochs(self, loss_array: np.ndarray) -> int:
        """Calculate number of stable epochs."""
        if len(loss_array) < 5:
            return 0
        
        # Count consecutive epochs with minimal change
        stable_epochs = 0
        threshold = np.mean(loss_array) * 0.01  # 1% of mean loss
        
        for i in range(1, len(loss_array)):
            if abs(loss_array[i] - loss_array[i-1]) < threshold:
                stable_epochs += 1
            else:
                stable_epochs = 0
        
        return stable_epochs
    
    def _find_optimal_lr_range(self, lr_array: np.ndarray, loss_array: np.ndarray) -> Tuple[float, float]:
        """Find optimal learning rate range."""
        if len(lr_array) < 2:
            return (0.0, 0.0)
        
        # Find LR range with lowest loss
        min_loss_idx = np.argmin(loss_array)
        optimal_lr = lr_array[min_loss_idx]
        
        # Find range around optimal LR
        lr_std = np.std(lr_array)
        lr_range = (max(0, optimal_lr - lr_std), optimal_lr + lr_std)
        
        return lr_range
    
    def _calculate_lr_sensitivity(self, lr_array: np.ndarray, loss_array: np.ndarray) -> float:
        """Calculate learning rate sensitivity."""
        if len(lr_array) < 2:
            return 0.0
        
        # Calculate how much loss changes with LR changes
        lr_changes = np.diff(lr_array)
        loss_changes = np.diff(loss_array)
        
        if np.sum(np.abs(lr_changes)) == 0:
            return 0.0
        
        sensitivity = np.sum(np.abs(loss_changes)) / np.sum(np.abs(lr_changes))
        return sensitivity
    
    def _calculate_trend(self, data: np.ndarray) -> str:
        """Calculate trend of data."""
        if len(data) < 2:
            return "stable"
        
        # Simple trend analysis
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        if second_mean > first_mean * 1.05:
            return "increasing"
        elif second_mean < first_mean * 0.95:
            return "decreasing"
        else:
            return "stable"

class TruthGPTAnalyticsManager:
    """Advanced analytics manager for TruthGPT."""
    
    def __init__(self, config: TruthGPTAnalyticsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Analytics components
        self.performance_analytics = TruthGPTPerformanceAnalytics(config)
        self.model_analytics = TruthGPTModelAnalytics(config)
        self.training_analytics = TruthGPTTrainingAnalytics(config)
        
        # Analytics state
        self.analytics_results = {}
        self.analytics_history = []
    
    def analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """Comprehensive model analysis."""
        self.logger.info("🔍 Starting comprehensive TruthGPT model analysis")
        
        analysis = {
            'timestamp': time.time(),
            'architecture': self.model_analytics.analyze_model_architecture(model),
            'complexity': self.model_analytics.analyze_model_complexity(model),
            'weights': self.model_analytics.analyze_model_weights(model)
        }
        
        self.analytics_results['model_analysis'] = analysis
        self.logger.info("✅ Model analysis completed")
        
        return analysis
    
    def analyze_training(self, training_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Comprehensive training analysis."""
        self.logger.info("🔍 Starting comprehensive TruthGPT training analysis")
        
        analysis = {
            'timestamp': time.time(),
            'training_curves': self.training_analytics.analyze_training_curves(training_data),
            'convergence': self.training_analytics.analyze_convergence(training_data.get('loss', [])),
            'learning_rate_impact': self.training_analytics.analyze_learning_rate_impact(
                training_data.get('learning_rate', []),
                training_data.get('loss', [])
            )
        }
        
        self.analytics_results['training_analysis'] = analysis
        self.logger.info("✅ Training analysis completed")
        
        return analysis
    
    def analyze_performance(self, performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Comprehensive performance analysis."""
        self.logger.info("🔍 Starting comprehensive TruthGPT performance analysis")
        
        analysis = self.performance_analytics.create_performance_report(performance_data)
        
        self.analytics_results['performance_analysis'] = analysis
        self.logger.info("✅ Performance analysis completed")
        
        return analysis
    
    def create_comprehensive_report(self, model: nn.Module, 
                                  training_data: Dict[str, List[float]],
                                  performance_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Create comprehensive analytics report."""
        self.logger.info("📊 Creating comprehensive TruthGPT analytics report")
        
        report = {
            'timestamp': time.time(),
            'model_analysis': self.analyze_model(model),
            'training_analysis': self.analyze_training(training_data),
            'performance_analysis': self.analyze_performance(performance_data),
            'summary': self._create_summary()
        }
        
        # Export report if enabled
        if self.config.enable_export:
            self._export_report(report)
        
        self.analytics_history.append(report)
        self.logger.info("✅ Comprehensive analytics report created")
        
        return report
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create analytics summary."""
        summary = {
            'total_analyses': len(self.analytics_history),
            'analytics_timestamp': time.time(),
            'key_insights': []
        }
        
        # Add key insights based on available data
        if 'model_analysis' in self.analytics_results:
            model_analysis = self.analytics_results['model_analysis']
            summary['key_insights'].append(f"Model has {model_analysis['architecture']['total_parameters']:,} parameters")
        
        if 'training_analysis' in self.analytics_results:
            training_analysis = self.analytics_results['training_analysis']
            if 'convergence' in training_analysis:
                summary['key_insights'].append(f"Training converged at epoch {training_analysis['convergence']['convergence_epoch']}")
        
        if 'performance_analysis' in self.analytics_results:
            performance_analysis = self.analytics_results['performance_analysis']
            if 'performance_metrics' in performance_analysis:
                summary['key_insights'].append("Performance metrics analyzed")
        
        return summary
    
    def _export_report(self, report: Dict[str, Any]) -> None:
        """Export analytics report."""
        if not self.config.enable_export:
            return
        
        export_path = Path(self.config.export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        if self.config.export_format == "json":
            filepath = export_path / f"analytics_report_{timestamp}.json"
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        elif self.config.export_format == "csv":
            # Convert to CSV format
            filepath = export_path / f"analytics_report_{timestamp}.csv"
            # Simplified CSV export
            with open(filepath, 'w') as f:
                f.write("metric,value\n")
                f.write(f"timestamp,{report['timestamp']}\n")
        elif self.config.export_format == "html":
            filepath = export_path / f"analytics_report_{timestamp}.html"
            # Simplified HTML export
            with open(filepath, 'w') as f:
                f.write(f"<html><body><h1>TruthGPT Analytics Report</h1><pre>{json.dumps(report, indent=2)}</pre></body></html>")
        
        self.logger.info(f"Analytics report exported to {filepath}")
    
    def get_analytics_history(self) -> List[Dict[str, Any]]:
        """Get analytics history."""
        return self.analytics_history.copy()
    
    def get_latest_analytics(self) -> Optional[Dict[str, Any]]:
        """Get latest analytics results."""
        return self.analytics_results.copy() if self.analytics_results else None

# Factory functions
def create_truthgpt_analytics_manager(config: TruthGPTAnalyticsConfig) -> TruthGPTAnalyticsManager:
    """Create TruthGPT analytics manager."""
    return TruthGPTAnalyticsManager(config)

def analyze_truthgpt_model(model: nn.Module, config: TruthGPTAnalyticsConfig) -> Dict[str, Any]:
    """Quick analyze TruthGPT model."""
    manager = create_truthgpt_analytics_manager(config)
    return manager.analyze_model(model)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT analytics
    print("🚀 TruthGPT Advanced Analytics Demo")
    print("=" * 50)
    
    # Create analytics configuration
    config = TruthGPTAnalyticsConfig(
        enable_performance_analytics=True,
        enable_model_analytics=True,
        enable_training_analytics=True,
        enable_export=True,
        export_format="json"
    )
    
    # Create analytics manager
    manager = create_truthgpt_analytics_manager(config)
    
    # Create sample model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 10000)
        
        def forward(self, x):
            return self.linear(x)
    
    model = TruthGPTModel()
    
    # Analyze model
    model_analysis = manager.analyze_model(model)
    print(f"Model analysis: {model_analysis['architecture']['total_parameters']:,} parameters")
    
    # Create sample training data
    training_data = {
        'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
        'accuracy': [0.5, 0.6, 0.7, 0.8, 0.9],
        'learning_rate': [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
    }
    
    # Analyze training
    training_analysis = manager.analyze_training(training_data)
    print(f"Training analysis: {training_analysis['convergence']['convergence_epoch']} epochs to converge")
    
    print("✅ TruthGPT analytics demo completed!")



