"""
Unit tests for optimizer core functionality
Tests core optimization algorithms, advanced techniques, and optimization strategies
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestAdvancedOptimizers(unittest.TestCase):
    """Test suite for advanced optimizers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_adaptive_moment_estimation(self):
        """Test Adam optimizer with adaptive learning rates"""
        class AdaptiveMomentEstimation:
            def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
                self.learning_rate = learning_rate
                self.beta1 = beta1
                self.beta2 = beta2
                self.epsilon = epsilon
                self.m = {}  # First moment estimates
                self.v = {}  # Second moment estimates
                self.t = 0   # Time step
                
            def step(self, model, loss):
                """Perform Adam optimization step"""
                self.t += 1
                
                # Compute gradients
                gradients = self._compute_gradients(model, loss)
                
                # Update parameters
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.data
                        
                        # Initialize moments
                        if name not in self.m:
                            self.m[name] = torch.zeros_like(grad)
                            self.v[name] = torch.zeros_like(grad)
                            
                        # Update biased first moment estimate
                        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                        
                        # Update biased second raw moment estimate
                        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                        
                        # Compute bias-corrected first moment estimate
                        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                        
                        # Compute bias-corrected second raw moment estimate
                        v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                        
                        # Update parameters
                        param.data -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
                        
                return {'optimized': True, 'step': self.t}
                
            def _compute_gradients(self, model, loss):
                """Compute gradients"""
                # Simulate gradient computation
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.data
                return gradients
                
            def get_optimizer_stats(self):
                """Get optimizer statistics"""
                return {
                    'time_step': self.t,
                    'learning_rate': self.learning_rate,
                    'beta1': self.beta1,
                    'beta2': self.beta2,
                    'parameters_tracked': len(self.m)
                }
        
        # Test Adam optimizer
        adam = AdaptiveMomentEstimation(learning_rate=0.001)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test optimization step
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        result = adam.step(model, loss)
        self.assertTrue(result['optimized'])
        self.assertEqual(result['step'], 1)
        
        # Check optimizer stats
        stats = adam.get_optimizer_stats()
        self.assertEqual(stats['time_step'], 1)
        self.assertEqual(stats['learning_rate'], 0.001)
        self.assertGreater(stats['parameters_tracked'], 0)
        
    def test_adaptive_gradient_algorithm(self):
        """Test AdaGrad optimizer"""
        class AdaptiveGradientAlgorithm:
            def __init__(self, learning_rate=0.01, epsilon=1e-8):
                self.learning_rate = learning_rate
                self.epsilon = epsilon
                self.squared_gradients = {}
                
            def step(self, model, loss):
                """Perform AdaGrad optimization step"""
                # Compute gradients
                gradients = self._compute_gradients(model, loss)
                
                # Update parameters
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.data
                        
                        # Initialize squared gradients
                        if name not in self.squared_gradients:
                            self.squared_gradients[name] = torch.zeros_like(grad)
                            
                        # Accumulate squared gradients
                        self.squared_gradients[name] += grad ** 2
                        
                        # Update parameters
                        param.data -= self.learning_rate * grad / (torch.sqrt(self.squared_gradients[name]) + self.epsilon)
                        
                return {'optimized': True}
                
            def _compute_gradients(self, model, loss):
                """Compute gradients"""
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.data
                return gradients
                
            def get_optimizer_stats(self):
                """Get optimizer statistics"""
                return {
                    'learning_rate': self.learning_rate,
                    'epsilon': self.epsilon,
                    'parameters_tracked': len(self.squared_gradients)
                }
        
        # Test AdaGrad optimizer
        adagrad = AdaptiveGradientAlgorithm(learning_rate=0.01)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test optimization step
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        result = adagrad.step(model, loss)
        self.assertTrue(result['optimized'])
        
        # Check optimizer stats
        stats = adagrad.get_optimizer_stats()
        self.assertEqual(stats['learning_rate'], 0.01)
        self.assertGreater(stats['parameters_tracked'], 0)
        
    def test_rmsprop_optimizer(self):
        """Test RMSprop optimizer"""
        class RMSpropOptimizer:
            def __init__(self, learning_rate=0.01, alpha=0.99, epsilon=1e-8):
                self.learning_rate = learning_rate
                self.alpha = alpha
                self.epsilon = epsilon
                self.squared_gradients = {}
                
            def step(self, model, loss):
                """Perform RMSprop optimization step"""
                # Compute gradients
                gradients = self._compute_gradients(model, loss)
                
                # Update parameters
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.data
                        
                        # Initialize squared gradients
                        if name not in self.squared_gradients:
                            self.squared_gradients[name] = torch.zeros_like(grad)
                            
                        # Update squared gradients with exponential moving average
                        self.squared_gradients[name] = (self.alpha * self.squared_gradients[name] + 
                                                       (1 - self.alpha) * (grad ** 2))
                        
                        # Update parameters
                        param.data -= self.learning_rate * grad / (torch.sqrt(self.squared_gradients[name]) + self.epsilon)
                        
                return {'optimized': True}
                
            def _compute_gradients(self, model, loss):
                """Compute gradients"""
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.data
                return gradients
                
            def get_optimizer_stats(self):
                """Get optimizer statistics"""
                return {
                    'learning_rate': self.learning_rate,
                    'alpha': self.alpha,
                    'epsilon': self.epsilon,
                    'parameters_tracked': len(self.squared_gradients)
                }
        
        # Test RMSprop optimizer
        rmsprop = RMSpropOptimizer(learning_rate=0.01)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test optimization step
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        result = rmsprop.step(model, loss)
        self.assertTrue(result['optimized'])
        
        # Check optimizer stats
        stats = rmsprop.get_optimizer_stats()
        self.assertEqual(stats['learning_rate'], 0.01)
        self.assertEqual(stats['alpha'], 0.99)
        self.assertGreater(stats['parameters_tracked'], 0)

class TestOptimizationStrategies(unittest.TestCase):
    """Test suite for optimization strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling strategies"""
        class LearningRateScheduler:
            def __init__(self, initial_lr=0.001, strategy='exponential'):
                self.initial_lr = initial_lr
                self.current_lr = initial_lr
                self.strategy = strategy
                self.step_count = 0
                self.lr_history = []
                
            def step(self):
                """Update learning rate"""
                self.step_count += 1
                
                if self.strategy == 'exponential':
                    self.current_lr = self.initial_lr * (0.95 ** self.step_count)
                elif self.strategy == 'cosine':
                    self.current_lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * self.step_count / 100))
                elif self.strategy == 'step':
                    self.current_lr = self.initial_lr * (0.1 ** (self.step_count // 30))
                elif self.strategy == 'plateau':
                    # Simulate plateau detection
                    if self.step_count % 10 == 0:
                        self.current_lr *= 0.5
                        
                self.lr_history.append(self.current_lr)
                return self.current_lr
                
            def get_lr_stats(self):
                """Get learning rate statistics"""
                return {
                    'current_lr': self.current_lr,
                    'initial_lr': self.initial_lr,
                    'strategy': self.strategy,
                    'step_count': self.step_count,
                    'lr_reductions': len([lr for i, lr in enumerate(self.lr_history) 
                                        if i > 0 and lr < self.lr_history[i-1]])
                }
        
        # Test different learning rate strategies
        strategies = ['exponential', 'cosine', 'step', 'plateau']
        
        for strategy in strategies:
            scheduler = LearningRateScheduler(initial_lr=0.001, strategy=strategy)
            
            # Test learning rate updates
            for _ in range(10):
                lr = scheduler.step()
                self.assertGreater(lr, 0)
                
            # Check scheduler stats
            stats = scheduler.get_lr_stats()
            self.assertEqual(stats['strategy'], strategy)
            self.assertEqual(stats['step_count'], 10)
            self.assertGreaterEqual(stats['lr_reductions'], 0)
            
    def test_gradient_clipping(self):
        """Test gradient clipping strategies"""
        class GradientClipper:
            def __init__(self, max_norm=1.0, strategy='norm'):
                self.max_norm = max_norm
                self.strategy = strategy
                self.clipping_history = []
                
            def clip_gradients(self, model):
                """Clip gradients"""
                if self.strategy == 'norm':
                    return self._clip_by_norm(model)
                elif self.strategy == 'value':
                    return self._clip_by_value(model)
                elif self.strategy == 'adaptive':
                    return self._adaptive_clipping(model)
                else:
                    return self._no_clipping(model)
                    
            def _clip_by_norm(self, model):
                """Clip gradients by norm"""
                total_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                clip_coef = min(1, self.max_norm / (total_norm + 1e-6))
                
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coef)
                        
                self.clipping_history.append({
                    'total_norm': total_norm,
                    'clip_coef': clip_coef,
                    'clipped': clip_coef < 1
                })
                
                return clip_coef < 1
                
            def _clip_by_value(self, model):
                """Clip gradients by value"""
                clipped = False
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-self.max_norm, self.max_norm)
                        if torch.any(torch.abs(param.grad.data) >= self.max_norm):
                            clipped = True
                            
                self.clipping_history.append({
                    'max_norm': self.max_norm,
                    'clipped': clipped
                })
                
                return clipped
                
            def _adaptive_clipping(self, model):
                """Adaptive gradient clipping"""
                # Compute gradient statistics
                grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.data.norm(2).item())
                        
                if not grad_norms:
                    return False
                    
                mean_norm = np.mean(grad_norms)
                std_norm = np.std(grad_norms)
                adaptive_threshold = mean_norm + 2 * std_norm
                
                clipped = False
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        if param_norm > adaptive_threshold:
                            clip_coef = adaptive_threshold / param_norm
                            param.grad.data.mul_(clip_coef)
                            clipped = True
                            
                self.clipping_history.append({
                    'mean_norm': mean_norm,
                    'std_norm': std_norm,
                    'adaptive_threshold': adaptive_threshold,
                    'clipped': clipped
                })
                
                return clipped
                
            def _no_clipping(self, model):
                """No gradient clipping"""
                self.clipping_history.append({
                    'clipped': False
                })
                return False
                
            def get_clipping_stats(self):
                """Get clipping statistics"""
                if not self.clipping_history:
                    return {}
                    
                clipped_count = sum(1 for entry in self.clipping_history if entry['clipped'])
                total_count = len(self.clipping_history)
                
                return {
                    'strategy': self.strategy,
                    'max_norm': self.max_norm,
                    'total_clips': total_count,
                    'clipped_count': clipped_count,
                    'clipping_rate': clipped_count / total_count if total_count > 0 else 0
                }
        
        # Test different gradient clipping strategies
        strategies = ['norm', 'value', 'adaptive', 'none']
        
        for strategy in strategies:
            clipper = GradientClipper(max_norm=1.0, strategy=strategy)
            model = nn.Linear(256, 512)
            data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
            target = torch.randn_like(data)
            
            # Test gradient clipping
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            clipped = clipper.clip_gradients(model)
            self.assertIsInstance(clipped, bool)
            
            # Check clipping stats
            stats = clipper.get_clipping_stats()
            self.assertEqual(stats['strategy'], strategy)
            self.assertEqual(stats['total_clips'], 1)
            self.assertGreaterEqual(stats['clipping_rate'], 0)
            
    def test_optimization_ensemble(self):
        """Test ensemble of optimizers"""
        class OptimizationEnsemble:
            def __init__(self, optimizers):
                self.optimizers = optimizers
                self.optimization_history = []
                self.best_optimizer = None
                self.best_performance = float('inf')
                
            def optimize(self, model, data, target):
                """Run ensemble optimization"""
                results = {}
                
                for name, optimizer in self.optimizers.items():
                    # Create model copy for each optimizer
                    model_copy = self._copy_model(model)
                    
                    # Run optimization
                    result = self._run_optimization(optimizer, model_copy, data, target)
                    results[name] = result
                    
                    # Update best optimizer
                    if result['performance'] < self.best_performance:
                        self.best_performance = result['performance']
                        self.best_optimizer = name
                        
                # Record ensemble results
                self.optimization_history.append({
                    'results': results,
                    'best_optimizer': self.best_optimizer,
                    'best_performance': self.best_performance
                })
                
                return results
                
            def _copy_model(self, model):
                """Create model copy"""
                # Simple model copying simulation
                return model
                
            def _run_optimization(self, optimizer, model, data, target):
                """Run single optimizer"""
                # Simulate optimization
                performance = np.random.uniform(0, 1)
                
                result = {
                    'optimizer': optimizer,
                    'performance': performance,
                    'converged': performance < 0.5,
                    'iterations': np.random.randint(10, 100)
                }
                
                return result
                
            def get_ensemble_stats(self):
                """Get ensemble statistics"""
                if not self.optimization_history:
                    return {}
                    
                return {
                    'total_optimizations': len(self.optimization_history),
                    'best_optimizer': self.best_optimizer,
                    'best_performance': self.best_performance,
                    'optimizer_count': len(self.optimizers)
                }
        
        # Test optimization ensemble
        optimizers = {
            'adam': {'learning_rate': 0.001},
            'sgd': {'learning_rate': 0.01},
            'rmsprop': {'learning_rate': 0.005}
        }
        
        ensemble = OptimizationEnsemble(optimizers)
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test ensemble optimization
        results = ensemble.optimize(model, data, target)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn('adam', results)
        self.assertIn('sgd', results)
        self.assertIn('rmsprop', results)
        
        for name, result in results.items():
            self.assertIn('performance', result)
            self.assertIn('converged', result)
            self.assertIn('iterations', result)
            
        # Check ensemble stats
        stats = ensemble.get_ensemble_stats()
        self.assertEqual(stats['total_optimizations'], 1)
        self.assertIsNotNone(stats['best_optimizer'])
        self.assertGreater(stats['best_performance'], 0)
        self.assertEqual(stats['optimizer_count'], 3)

class TestOptimizationMetrics(unittest.TestCase):
    """Test suite for optimization metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_convergence_detection(self):
        """Test convergence detection algorithms"""
        class ConvergenceDetector:
            def __init__(self, tolerance=1e-6, patience=5):
                self.tolerance = tolerance
                self.patience = patience
                self.loss_history = []
                self.convergence_history = []
                
            def check_convergence(self, loss):
                """Check if optimization has converged"""
                self.loss_history.append(loss)
                
                if len(self.loss_history) < self.patience:
                    return False
                    
                # Check if loss has stabilized
                recent_losses = self.loss_history[-self.patience:]
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)
                
                # Check relative change
                if len(self.loss_history) > self.patience:
                    previous_losses = self.loss_history[-2*self.patience:-self.patience]
                    previous_mean = np.mean(previous_losses)
                    relative_change = abs(loss_mean - previous_mean) / abs(previous_mean)
                    
                    converged = (loss_std < self.tolerance) or (relative_change < self.tolerance)
                else:
                    converged = loss_std < self.tolerance
                    
                self.convergence_history.append({
                    'loss': loss,
                    'loss_std': loss_std,
                    'converged': converged
                })
                
                return converged
                
            def get_convergence_stats(self):
                """Get convergence statistics"""
                if not self.convergence_history:
                    return {}
                    
                converged_count = sum(1 for entry in self.convergence_history if entry['converged'])
                total_checks = len(self.convergence_history)
                
                return {
                    'total_checks': total_checks,
                    'converged_count': converged_count,
                    'convergence_rate': converged_count / total_checks if total_checks > 0 else 0,
                    'tolerance': self.tolerance,
                    'patience': self.patience
                }
        
        # Test convergence detection
        detector = ConvergenceDetector(tolerance=1e-6, patience=5)
        
        # Test with converging losses
        converging_losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        for loss in converging_losses:
            converged = detector.check_convergence(loss)
            if loss < 0.01:  # Should converge near the end
                self.assertTrue(converged)
            else:
                self.assertFalse(converged)
                
        # Check convergence stats
        stats = detector.get_convergence_stats()
        self.assertEqual(stats['total_checks'], 10)
        self.assertGreater(stats['converged_count'], 0)
        self.assertGreater(stats['convergence_rate'], 0)
        
    def test_optimization_metrics(self):
        """Test optimization metrics calculation"""
        class OptimizationMetrics:
            def __init__(self):
                self.metrics_history = []
                self.current_metrics = {}
                
            def calculate_metrics(self, model, data, target, loss):
                """Calculate optimization metrics"""
                # Model complexity metrics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Performance metrics
                with torch.no_grad():
                    output = model(data)
                    mse = nn.MSELoss()(output, target).item()
                    mae = nn.L1Loss()(output, target).item()
                    
                # Gradient metrics
                grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.data.norm(2).item())
                        
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
                max_grad_norm = np.max(grad_norms) if grad_norms else 0
                
                # Learning rate metrics
                lr = 0.001  # Simulate learning rate
                
                metrics = {
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'mse': mse,
                    'mae': mae,
                    'avg_grad_norm': avg_grad_norm,
                    'max_grad_norm': max_grad_norm,
                    'learning_rate': lr,
                    'loss': loss.item() if hasattr(loss, 'item') else loss
                }
                
                self.current_metrics = metrics
                self.metrics_history.append(metrics.copy())
                
                return metrics
                
            def get_metrics_summary(self):
                """Get metrics summary"""
                if not self.metrics_history:
                    return {}
                    
                # Calculate summary statistics
                mse_values = [m['mse'] for m in self.metrics_history]
                mae_values = [m['mae'] for m in self.metrics_history]
                grad_norms = [m['avg_grad_norm'] for m in self.metrics_history]
                
                return {
                    'total_iterations': len(self.metrics_history),
                    'final_mse': mse_values[-1],
                    'final_mae': mae_values[-1],
                    'mse_improvement': mse_values[0] - mse_values[-1] if len(mse_values) > 1 else 0,
                    'mae_improvement': mae_values[0] - mae_values[-1] if len(mae_values) > 1 else 0,
                    'avg_grad_norm': np.mean(grad_norms),
                    'max_grad_norm': np.max(grad_norms),
                    'total_params': self.metrics_history[-1]['total_params'],
                    'trainable_params': self.metrics_history[-1]['trainable_params']
                }
        
        # Test optimization metrics
        metrics_calculator = OptimizationMetrics()
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test metrics calculation
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        metrics = metrics_calculator.calculate_metrics(model, data, target, loss)
        
        # Verify metrics
        self.assertIn('total_params', metrics)
        self.assertIn('trainable_params', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('avg_grad_norm', metrics)
        self.assertIn('max_grad_norm', metrics)
        self.assertIn('learning_rate', metrics)
        self.assertIn('loss', metrics)
        
        # Check metrics summary
        summary = metrics_calculator.get_metrics_summary()
        self.assertEqual(summary['total_iterations'], 1)
        self.assertGreater(summary['total_params'], 0)
        self.assertGreater(summary['trainable_params'], 0)
        self.assertGreater(summary['final_mse'], 0)
        self.assertGreater(summary['final_mae'], 0)

if __name__ == '__main__':
    unittest.main()
