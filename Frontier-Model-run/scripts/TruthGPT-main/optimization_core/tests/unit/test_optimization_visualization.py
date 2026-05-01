"""
Unit tests for optimization visualization and monitoring
Tests visualization tools, real-time monitoring, and optimization dashboards
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestOptimizationVisualization(unittest.TestCase):
    """Test suite for optimization visualization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_loss_curve_visualization(self):
        """Test loss curve visualization"""
        class LossCurveVisualizer:
            def __init__(self):
                self.loss_history = []
                self.visualization_data = {}
                
            def update_loss(self, loss):
                """Update loss history"""
                self.loss_history.append(loss)
                
            def create_loss_curve(self, title="Loss Curve"):
                """Create loss curve visualization"""
                if not self.loss_history:
                    return None
                    
                # Create visualization data
                self.visualization_data = {
                    'x': list(range(len(self.loss_history))),
                    'y': self.loss_history,
                    'title': title,
                    'xlabel': 'Iteration',
                    'ylabel': 'Loss',
                    'type': 'line'
                }
                
                return self.visualization_data
                
            def get_loss_stats(self):
                """Get loss statistics"""
                if not self.loss_history:
                    return {}
                    
                return {
                    'total_iterations': len(self.loss_history),
                    'initial_loss': self.loss_history[0],
                    'final_loss': self.loss_history[-1],
                    'min_loss': min(self.loss_history),
                    'max_loss': max(self.loss_history),
                    'loss_improvement': self.loss_history[0] - self.loss_history[-1],
                    'convergence_rate': self._calculate_convergence_rate()
                }
                
            def _calculate_convergence_rate(self):
                """Calculate convergence rate"""
                if len(self.loss_history) < 2:
                    return 0
                    
                initial_loss = self.loss_history[0]
                final_loss = self.loss_history[-1]
                
                if initial_loss == 0:
                    return 0
                    
                return (initial_loss - final_loss) / initial_loss
        
        # Test loss curve visualizer
        visualizer = LossCurveVisualizer()
        
        # Test loss updates
        losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        for loss in losses:
            visualizer.update_loss(loss)
            
        # Test loss curve creation
        curve_data = visualizer.create_loss_curve("Test Loss Curve")
        
        # Verify results
        self.assertIsNotNone(curve_data)
        self.assertEqual(len(curve_data['x']), 10)
        self.assertEqual(len(curve_data['y']), 10)
        self.assertEqual(curve_data['title'], "Test Loss Curve")
        self.assertEqual(curve_data['type'], "line")
        
        # Check loss stats
        stats = visualizer.get_loss_stats()
        self.assertEqual(stats['total_iterations'], 10)
        self.assertEqual(stats['initial_loss'], 1.0)
        self.assertEqual(stats['final_loss'], 0.005)
        self.assertEqual(stats['min_loss'], 0.005)
        self.assertEqual(stats['max_loss'], 1.0)
        self.assertGreater(stats['loss_improvement'], 0)
        self.assertGreater(stats['convergence_rate'], 0)
        
    def test_gradient_visualization(self):
        """Test gradient visualization"""
        class GradientVisualizer:
            def __init__(self):
                self.gradient_history = []
                self.gradient_norms = []
                self.visualization_data = {}
                
            def update_gradients(self, model):
                """Update gradient information"""
                grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        grad_norms.append(grad_norm)
                        
                if grad_norms:
                    avg_grad_norm = np.mean(grad_norms)
                    max_grad_norm = np.max(grad_norms)
                    
                    self.gradient_history.append({
                        'avg_grad_norm': avg_grad_norm,
                        'max_grad_norm': max_grad_norm,
                        'grad_norms': grad_norms
                    })
                    
                    self.gradient_norms.append(avg_grad_norm)
                    
            def create_gradient_plot(self, title="Gradient Norms"):
                """Create gradient visualization"""
                if not self.gradient_history:
                    return None
                    
                # Create visualization data
                self.visualization_data = {
                    'x': list(range(len(self.gradient_norms))),
                    'y': self.gradient_norms,
                    'title': title,
                    'xlabel': 'Iteration',
                    'ylabel': 'Gradient Norm',
                    'type': 'line'
                }
                
                return self.visualization_data
                
            def get_gradient_stats(self):
                """Get gradient statistics"""
                if not self.gradient_history:
                    return {}
                    
                return {
                    'total_iterations': len(self.gradient_history),
                    'avg_grad_norm': np.mean(self.gradient_norms),
                    'max_grad_norm': np.max(self.gradient_norms),
                    'min_grad_norm': np.min(self.gradient_norms),
                    'grad_norm_std': np.std(self.gradient_norms),
                    'gradient_explosion': np.any(np.array(self.gradient_norms) > 10.0),
                    'gradient_vanishing': np.any(np.array(self.gradient_norms) < 1e-6)
                }
        
        # Test gradient visualizer
        visualizer = GradientVisualizer()
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test gradient updates
        for _ in range(5):
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            visualizer.update_gradients(model)
            
        # Test gradient plot creation
        plot_data = visualizer.create_gradient_plot("Test Gradient Norms")
        
        # Verify results
        self.assertIsNotNone(plot_data)
        self.assertEqual(len(plot_data['x']), 5)
        self.assertEqual(len(plot_data['y']), 5)
        self.assertEqual(plot_data['title'], "Test Gradient Norms")
        self.assertEqual(plot_data['type'], "line")
        
        # Check gradient stats
        stats = visualizer.get_gradient_stats()
        self.assertEqual(stats['total_iterations'], 5)
        self.assertGreater(stats['avg_grad_norm'], 0)
        self.assertGreater(stats['max_grad_norm'], 0)
        self.assertGreater(stats['min_grad_norm'], 0)
        self.assertGreaterEqual(stats['grad_norm_std'], 0)
        self.assertIsInstance(stats['gradient_explosion'], bool)
        self.assertIsInstance(stats['gradient_vanishing'], bool)
        
    def test_learning_rate_visualization(self):
        """Test learning rate visualization"""
        class LearningRateVisualizer:
            def __init__(self):
                self.lr_history = []
                self.visualization_data = {}
                
            def update_learning_rate(self, lr):
                """Update learning rate history"""
                self.lr_history.append(lr)
                
            def create_lr_plot(self, title="Learning Rate Schedule"):
                """Create learning rate visualization"""
                if not self.lr_history:
                    return None
                    
                # Create visualization data
                self.visualization_data = {
                    'x': list(range(len(self.lr_history))),
                    'y': self.lr_history,
                    'title': title,
                    'xlabel': 'Iteration',
                    'ylabel': 'Learning Rate',
                    'type': 'line'
                }
                
                return self.visualization_data
                
            def get_lr_stats(self):
                """Get learning rate statistics"""
                if not self.lr_history:
                    return {}
                    
                return {
                    'total_iterations': len(self.lr_history),
                    'initial_lr': self.lr_history[0],
                    'final_lr': self.lr_history[-1],
                    'min_lr': min(self.lr_history),
                    'max_lr': max(self.lr_history),
                    'lr_reductions': len([lr for i, lr in enumerate(self.lr_history) 
                                        if i > 0 and lr < self.lr_history[i-1]]),
                    'lr_ratio': self.lr_history[-1] / self.lr_history[0] if self.lr_history[0] > 0 else 0
                }
        
        # Test learning rate visualizer
        visualizer = LearningRateVisualizer()
        
        # Test learning rate updates
        lrs = [0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0001, 0.0001, 0.00005, 0.00005, 0.00001]
        for lr in lrs:
            visualizer.update_learning_rate(lr)
            
        # Test learning rate plot creation
        plot_data = visualizer.create_lr_plot("Test Learning Rate Schedule")
        
        # Verify results
        self.assertIsNotNone(plot_data)
        self.assertEqual(len(plot_data['x']), 10)
        self.assertEqual(len(plot_data['y']), 10)
        self.assertEqual(plot_data['title'], "Test Learning Rate Schedule")
        self.assertEqual(plot_data['type'], "line")
        
        # Check learning rate stats
        stats = visualizer.get_lr_stats()
        self.assertEqual(stats['total_iterations'], 10)
        self.assertEqual(stats['initial_lr'], 0.001)
        self.assertEqual(stats['final_lr'], 0.00001)
        self.assertEqual(stats['min_lr'], 0.00001)
        self.assertEqual(stats['max_lr'], 0.001)
        self.assertGreater(stats['lr_reductions'], 0)
        self.assertLess(stats['lr_ratio'], 1.0)

class TestOptimizationDashboard(unittest.TestCase):
    """Test suite for optimization dashboard"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_optimization_dashboard(self):
        """Test optimization dashboard"""
        class OptimizationDashboard:
            def __init__(self):
                self.dashboard_data = {}
                self.visualization_components = {}
                self.dashboard_history = []
                
            def update_dashboard(self, optimization_data):
                """Update dashboard with optimization data"""
                # Process optimization data
                processed_data = self._process_optimization_data(optimization_data)
                
                # Update dashboard data
                self.dashboard_data.update(processed_data)
                
                # Generate visualizations
                self._generate_visualizations(processed_data)
                
                # Record dashboard update
                self.dashboard_history.append({
                    'data': processed_data,
                    'timestamp': len(self.dashboard_history)
                })
                
            def _process_optimization_data(self, data):
                """Process optimization data for dashboard"""
                processed = {}
                
                # Process metrics
                if 'metrics' in data:
                    processed['metrics'] = data['metrics']
                    
                # Process performance
                if 'performance' in data:
                    processed['performance'] = data['performance']
                    
                # Process convergence
                if 'convergence' in data:
                    processed['convergence'] = data['convergence']
                    
                return processed
                
            def _generate_visualizations(self, data):
                """Generate visualization components"""
                # Simulate visualization generation
                self.visualization_components = {
                    'loss_curve': {
                        'type': 'line',
                        'data': np.random.uniform(0, 1, 10),
                        'title': 'Loss Curve'
                    },
                    'gradient_norms': {
                        'type': 'line',
                        'data': np.random.uniform(0, 5, 10),
                        'title': 'Gradient Norms'
                    },
                    'learning_rate': {
                        'type': 'line',
                        'data': np.random.uniform(0.001, 0.01, 10),
                        'title': 'Learning Rate'
                    },
                    'performance_metrics': {
                        'type': 'bar',
                        'data': {
                            'accuracy': np.random.uniform(0, 1),
                            'precision': np.random.uniform(0, 1),
                            'recall': np.random.uniform(0, 1)
                        },
                        'title': 'Performance Metrics'
                    }
                }
                
            def get_dashboard_summary(self):
                """Get dashboard summary"""
                return {
                    'total_updates': len(self.dashboard_history),
                    'dashboard_data_keys': list(self.dashboard_data.keys()),
                    'visualization_components': list(self.visualization_components.keys()),
                    'latest_update': self.dashboard_history[-1] if self.dashboard_history else None
                }
                
            def export_dashboard_data(self):
                """Export dashboard data"""
                return {
                    'dashboard_data': self.dashboard_data,
                    'visualization_components': self.visualization_components,
                    'dashboard_history': self.dashboard_history
                }
        
        # Test optimization dashboard
        dashboard = OptimizationDashboard()
        
        # Test dashboard updates
        for i in range(3):
            optimization_data = {
                'metrics': {
                    'loss': np.random.uniform(0, 1),
                    'grad_norm': np.random.uniform(0, 5)
                },
                'performance': {
                    'accuracy': np.random.uniform(0, 1),
                    'speed': np.random.uniform(0, 1)
                },
                'convergence': {
                    'converged': np.random.uniform(0, 1) > 0.5
                }
            }
            dashboard.update_dashboard(optimization_data)
            
        # Check dashboard summary
        summary = dashboard.get_dashboard_summary()
        self.assertEqual(summary['total_updates'], 3)
        self.assertGreater(len(summary['dashboard_data_keys']), 0)
        self.assertGreater(len(summary['visualization_components']), 0)
        self.assertIsNotNone(summary['latest_update'])
        
        # Test dashboard data export
        export_data = dashboard.export_dashboard_data()
        self.assertIn('dashboard_data', export_data)
        self.assertIn('visualization_components', export_data)
        self.assertIn('dashboard_history', export_data)
        
    def test_real_time_monitoring(self):
        """Test real-time monitoring"""
        class RealTimeMonitor:
            def __init__(self):
                self.monitoring_data = {}
                self.alerts = []
                self.monitoring_active = False
                
            def start_monitoring(self):
                """Start real-time monitoring"""
                self.monitoring_active = True
                self.monitoring_data = {}
                self.alerts = []
                
            def stop_monitoring(self):
                """Stop real-time monitoring"""
                self.monitoring_active = False
                
            def update_monitoring(self, metrics):
                """Update monitoring data"""
                if not self.monitoring_active:
                    return
                    
                # Update monitoring data
                self.monitoring_data.update(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
            def _check_alerts(self, metrics):
                """Check for monitoring alerts"""
                # Check for performance degradation
                if 'loss' in metrics and metrics['loss'] > 1.0:
                    self.alerts.append({
                        'type': 'performance_degradation',
                        'message': 'Loss is too high',
                        'value': metrics['loss'],
                        'timestamp': len(self.monitoring_data)
                    })
                    
                # Check for gradient explosion
                if 'grad_norm' in metrics and metrics['grad_norm'] > 10.0:
                    self.alerts.append({
                        'type': 'gradient_explosion',
                        'message': 'Gradient norm is too high',
                        'value': metrics['grad_norm'],
                        'timestamp': len(self.monitoring_data)
                    })
                    
                # Check for convergence
                if 'loss' in metrics and metrics['loss'] < 0.01:
                    self.alerts.append({
                        'type': 'convergence',
                        'message': 'Optimization has converged',
                        'value': metrics['loss'],
                        'timestamp': len(self.monitoring_data)
                    })
                    
            def get_monitoring_status(self):
                """Get monitoring status"""
                return {
                    'monitoring_active': self.monitoring_active,
                    'total_alerts': len(self.alerts),
                    'alert_types': list(set(alert['type'] for alert in self.alerts)),
                    'monitoring_data': self.monitoring_data
                }
        
        # Test real-time monitoring
        monitor = RealTimeMonitor()
        
        # Test monitoring start
        monitor.start_monitoring()
        self.assertTrue(monitor.monitoring_active)
        
        # Test monitoring updates
        for i in range(5):
            metrics = {
                'loss': np.random.uniform(0, 2),
                'grad_norm': np.random.uniform(0, 15),
                'learning_rate': 0.001
            }
            monitor.update_monitoring(metrics)
            
        # Test monitoring stop
        monitor.stop_monitoring()
        self.assertFalse(monitor.monitoring_active)
        
        # Check monitoring status
        status = monitor.get_monitoring_status()
        self.assertFalse(status['monitoring_active'])
        self.assertGreaterEqual(status['total_alerts'], 0)
        self.assertGreaterEqual(len(status['alert_types']), 0)
        self.assertIn('monitoring_data', status)

class TestVisualizationUtilities(unittest.TestCase):
    """Test suite for visualization utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_plot_generator(self):
        """Test plot generator utility"""
        class PlotGenerator:
            def __init__(self):
                self.plot_templates = {}
                self.generated_plots = []
                
            def create_line_plot(self, x_data, y_data, title="Line Plot"):
                """Create line plot"""
                plot_data = {
                    'type': 'line',
                    'x': x_data,
                    'y': y_data,
                    'title': title,
                    'xlabel': 'X',
                    'ylabel': 'Y'
                }
                
                self.generated_plots.append(plot_data)
                return plot_data
                
            def create_bar_plot(self, categories, values, title="Bar Plot"):
                """Create bar plot"""
                plot_data = {
                    'type': 'bar',
                    'categories': categories,
                    'values': values,
                    'title': title,
                    'xlabel': 'Categories',
                    'ylabel': 'Values'
                }
                
                self.generated_plots.append(plot_data)
                return plot_data
                
            def create_scatter_plot(self, x_data, y_data, title="Scatter Plot"):
                """Create scatter plot"""
                plot_data = {
                    'type': 'scatter',
                    'x': x_data,
                    'y': y_data,
                    'title': title,
                    'xlabel': 'X',
                    'ylabel': 'Y'
                }
                
                self.generated_plots.append(plot_data)
                return plot_data
                
            def create_heatmap(self, data, title="Heatmap"):
                """Create heatmap"""
                plot_data = {
                    'type': 'heatmap',
                    'data': data,
                    'title': title,
                    'xlabel': 'X',
                    'ylabel': 'Y'
                }
                
                self.generated_plots.append(plot_data)
                return plot_data
                
            def get_plot_stats(self):
                """Get plot generation statistics"""
                return {
                    'total_plots': len(self.generated_plots),
                    'plot_types': list(set(plot['type'] for plot in self.generated_plots)),
                    'plot_titles': [plot['title'] for plot in self.generated_plots]
                }
        
        # Test plot generator
        generator = PlotGenerator()
        
        # Test line plot
        x_data = list(range(10))
        y_data = [i**2 for i in x_data]
        line_plot = generator.create_line_plot(x_data, y_data, "Quadratic Function")
        
        # Test bar plot
        categories = ['A', 'B', 'C', 'D']
        values = [1, 2, 3, 4]
        bar_plot = generator.create_bar_plot(categories, values, "Category Values")
        
        # Test scatter plot
        x_scatter = np.random.uniform(0, 10, 20)
        y_scatter = np.random.uniform(0, 10, 20)
        scatter_plot = generator.create_scatter_plot(x_scatter, y_scatter, "Random Scatter")
        
        # Test heatmap
        heatmap_data = np.random.uniform(0, 1, (5, 5))
        heatmap_plot = generator.create_heatmap(heatmap_data, "Random Heatmap")
        
        # Verify results
        self.assertEqual(line_plot['type'], 'line')
        self.assertEqual(len(line_plot['x']), 10)
        self.assertEqual(len(line_plot['y']), 10)
        self.assertEqual(line_plot['title'], "Quadratic Function")
        
        self.assertEqual(bar_plot['type'], 'bar')
        self.assertEqual(len(bar_plot['categories']), 4)
        self.assertEqual(len(bar_plot['values']), 4)
        self.assertEqual(bar_plot['title'], "Category Values")
        
        self.assertEqual(scatter_plot['type'], 'scatter')
        self.assertEqual(len(scatter_plot['x']), 20)
        self.assertEqual(len(scatter_plot['y']), 20)
        self.assertEqual(scatter_plot['title'], "Random Scatter")
        
        self.assertEqual(heatmap_plot['type'], 'heatmap')
        self.assertEqual(heatmap_plot['data'].shape, (5, 5))
        self.assertEqual(heatmap_plot['title'], "Random Heatmap")
        
        # Check plot stats
        stats = generator.get_plot_stats()
        self.assertEqual(stats['total_plots'], 4)
        self.assertEqual(len(stats['plot_types']), 4)
        self.assertEqual(len(stats['plot_titles']), 4)
        
    def test_visualization_export(self):
        """Test visualization export functionality"""
        class VisualizationExporter:
            def __init__(self):
                self.export_formats = ['png', 'svg', 'pdf', 'html']
                self.exported_visualizations = []
                
            def export_visualization(self, plot_data, format='png', filename=None):
                """Export visualization"""
                if filename is None:
                    filename = f"plot_{len(self.exported_visualizations)}"
                    
                export_info = {
                    'plot_data': plot_data,
                    'format': format,
                    'filename': filename,
                    'exported': True
                }
                
                self.exported_visualizations.append(export_info)
                return export_info
                
            def export_dashboard(self, dashboard_data, format='html'):
                """Export complete dashboard"""
                export_info = {
                    'dashboard_data': dashboard_data,
                    'format': format,
                    'exported': True
                }
                
                self.exported_visualizations.append(export_info)
                return export_info
                
            def get_export_stats(self):
                """Get export statistics"""
                return {
                    'total_exports': len(self.exported_visualizations),
                    'export_formats': list(set(export['format'] for export in self.exported_visualizations)),
                    'exported_files': [export['filename'] for export in self.exported_visualizations if 'filename' in export]
                }
        
        # Test visualization exporter
        exporter = VisualizationExporter()
        
        # Test single plot export
        plot_data = {
            'type': 'line',
            'x': list(range(10)),
            'y': [i**2 for i in range(10)],
            'title': 'Test Plot'
        }
        
        export_info = exporter.export_visualization(plot_data, format='png', filename='test_plot')
        
        # Test dashboard export
        dashboard_data = {
            'plots': [plot_data],
            'title': 'Test Dashboard'
        }
        
        dashboard_export = exporter.export_dashboard(dashboard_data, format='html')
        
        # Verify results
        self.assertEqual(export_info['format'], 'png')
        self.assertEqual(export_info['filename'], 'test_plot')
        self.assertTrue(export_info['exported'])
        
        self.assertEqual(dashboard_export['format'], 'html')
        self.assertTrue(dashboard_export['exported'])
        
        # Check export stats
        stats = exporter.get_export_stats()
        self.assertEqual(stats['total_exports'], 2)
        self.assertEqual(len(stats['export_formats']), 2)
        self.assertEqual(len(stats['exported_files']), 1)

if __name__ == '__main__':
    unittest.main()





