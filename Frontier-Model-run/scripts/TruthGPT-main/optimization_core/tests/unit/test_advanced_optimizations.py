"""
Unit tests for advanced optimization techniques
Tests advanced optimization algorithms, meta-learning, and adaptive techniques
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

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
        
    def test_meta_learning_optimizer(self):
        """Test meta-learning optimizer"""
        class MetaLearningOptimizer:
            def __init__(self, base_learning_rate=0.001, meta_learning_rate=0.01):
                self.base_lr = base_learning_rate
                self.meta_lr = meta_learning_rate
                self.meta_parameters = {}
                self.adaptation_history = []
                
            def adapt_to_task(self, task_data, task_target, adaptation_steps=5):
                """Adapt to a specific task using meta-learning"""
                # Initialize task-specific parameters
                task_params = self._initialize_task_params(task_data)
                
                # Meta-learning adaptation
                for step in range(adaptation_steps):
                    # Forward pass
                    output = self._forward_pass(task_data, task_params)
                    loss = nn.MSELoss()(output, task_target)
                    
                    # Compute gradients
                    gradients = self._compute_gradients(loss, task_params)
                    
                    # Meta-learning update
                    task_params = self._meta_update(task_params, gradients)
                    
                    self.adaptation_history.append({
                        'step': step,
                        'loss': loss.item(),
                        'task_params': task_params.copy()
                    })
                
                return task_params
                
            def _initialize_task_params(self, data):
                """Initialize task-specific parameters"""
                return {
                    'learning_rate': self.base_lr,
                    'momentum': 0.9,
                    'weight_decay': 1e-4
                }
                
            def _forward_pass(self, data, params):
                """Forward pass with task-specific parameters"""
                # Simulate model forward pass
                return torch.randn_like(data)
                
            def _compute_gradients(self, loss, params):
                """Compute gradients for meta-learning"""
                # Simulate gradient computation
                return {
                    'learning_rate': torch.tensor(0.001),
                    'momentum': torch.tensor(0.01),
                    'weight_decay': torch.tensor(0.0001)
                }
                
            def _meta_update(self, params, gradients):
                """Meta-learning parameter update"""
                updated_params = {}
                for key, value in params.items():
                    if key in gradients:
                        updated_params[key] = value + self.meta_lr * gradients[key].item()
                    else:
                        updated_params[key] = value
                return updated_params
                
            def get_adaptation_stats(self):
                """Get adaptation statistics"""
                if not self.adaptation_history:
                    return {}
                    
                losses = [entry['loss'] for entry in self.adaptation_history]
                return {
                    'total_steps': len(self.adaptation_history),
                    'initial_loss': losses[0],
                    'final_loss': losses[-1],
                    'improvement': losses[0] - losses[-1],
                    'convergence_rate': (losses[0] - losses[-1]) / len(losses)
                }
        
        # Test meta-learning optimizer
        meta_optimizer = MetaLearningOptimizer()
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test adaptation
        adapted_params = meta_optimizer.adapt_to_task(data, target)
        
        # Verify results
        self.assertIsInstance(adapted_params, dict)
        self.assertIn('learning_rate', adapted_params)
        self.assertIn('momentum', adapted_params)
        self.assertIn('weight_decay', adapted_params)
        
        # Check adaptation stats
        stats = meta_optimizer.get_adaptation_stats()
        self.assertGreater(stats['total_steps'], 0)
        self.assertGreaterEqual(stats['improvement'], 0)
        
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate optimization"""
        class AdaptiveLearningRate:
            def __init__(self, initial_lr=0.001, patience=5, factor=0.5):
                self.initial_lr = initial_lr
                self.current_lr = initial_lr
                self.patience = patience
                self.factor = factor
                self.best_loss = float('inf')
                self.wait = 0
                self.lr_history = []
                
            def update_learning_rate(self, current_loss):
                """Update learning rate based on loss"""
                self.lr_history.append(self.current_lr)
                
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    
                if self.wait >= self.patience:
                    self.current_lr *= self.factor
                    self.wait = 0
                    
                return self.current_lr
                
            def get_lr_stats(self):
                """Get learning rate statistics"""
                return {
                    'current_lr': self.current_lr,
                    'initial_lr': self.initial_lr,
                    'lr_reductions': len([lr for i, lr in enumerate(self.lr_history) 
                                        if i > 0 and lr < self.lr_history[i-1]]),
                    'final_lr': self.lr_history[-1] if self.lr_history else self.initial_lr
                }
        
        # Test adaptive learning rate
        adaptive_lr = AdaptiveLearningRate()
        
        # Simulate training with varying losses
        losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
        
        for loss in losses:
            lr = adaptive_lr.update_learning_rate(loss)
            self.assertGreater(lr, 0)
            
        # Check stats
        stats = adaptive_lr.get_lr_stats()
        self.assertGreater(stats['current_lr'], 0)
        self.assertGreaterEqual(stats['lr_reductions'], 0)
        
    def test_evolutionary_optimization(self):
        """Test evolutionary optimization"""
        class EvolutionaryOptimizer:
            def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.8):
                self.population_size = population_size
                self.mutation_rate = mutation_rate
                self.crossover_rate = crossover_rate
                self.population = []
                self.fitness_history = []
                
            def initialize_population(self, param_ranges):
                """Initialize population with random parameters"""
                self.population = []
                for _ in range(self.population_size):
                    individual = {}
                    for param, (min_val, max_val) in param_ranges.items():
                        individual[param] = np.random.uniform(min_val, max_val)
                    self.population.append(individual)
                    
            def evaluate_fitness(self, individual, data, target):
                """Evaluate fitness of an individual"""
                # Simulate fitness evaluation
                fitness = np.random.uniform(0, 1)
                return fitness
                
            def evolve_generation(self, data, target):
                """Evolve one generation"""
                # Evaluate fitness
                fitness_scores = []
                for individual in self.population:
                    fitness = self.evaluate_fitness(individual, data, target)
                    fitness_scores.append(fitness)
                    
                # Record best fitness
                best_fitness = max(fitness_scores)
                self.fitness_history.append(best_fitness)
                
                # Selection, crossover, and mutation
                new_population = []
                for _ in range(self.population_size):
                    parent1 = self._tournament_selection(fitness_scores)
                    parent2 = self._tournament_selection(fitness_scores)
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_population.append(child)
                    
                self.population = new_population
                return best_fitness
                
            def _tournament_selection(self, fitness_scores, tournament_size=3):
                """Tournament selection"""
                tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                return self.population[winner_index].copy()
                
            def _crossover(self, parent1, parent2):
                """Crossover operation"""
                child = {}
                for key in parent1:
                    if np.random.random() < self.crossover_rate:
                        child[key] = (parent1[key] + parent2[key]) / 2
                    else:
                        child[key] = parent1[key]
                return child
                
            def _mutate(self, individual):
                """Mutation operation"""
                mutated = individual.copy()
                for key in mutated:
                    if np.random.random() < self.mutation_rate:
                        # Add random noise
                        noise = np.random.normal(0, 0.1)
                        mutated[key] += noise
                return mutated
                
            def get_evolution_stats(self):
                """Get evolution statistics"""
                if not self.fitness_history:
                    return {}
                    
                return {
                    'generations': len(self.fitness_history),
                    'best_fitness': max(self.fitness_history),
                    'final_fitness': self.fitness_history[-1],
                    'improvement': self.fitness_history[-1] - self.fitness_history[0],
                    'convergence': self._check_convergence()
                }
                
            def _check_convergence(self):
                """Check if evolution has converged"""
                if len(self.fitness_history) < 10:
                    return False
                    
                recent_fitness = self.fitness_history[-10:]
                return np.std(recent_fitness) < 0.01
        
        # Test evolutionary optimization
        evo_optimizer = EvolutionaryOptimizer()
        
        # Initialize population
        param_ranges = {
            'learning_rate': (0.0001, 0.01),
            'momentum': (0.5, 0.99),
            'weight_decay': (1e-6, 1e-3)
        }
        evo_optimizer.initialize_population(param_ranges)
        
        # Test evolution
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        for generation in range(5):
            best_fitness = evo_optimizer.evolve_generation(data, target)
            self.assertGreater(best_fitness, 0)
            
        # Check evolution stats
        stats = evo_optimizer.get_evolution_stats()
        self.assertGreater(stats['generations'], 0)
        self.assertGreater(stats['best_fitness'], 0)

class TestAdvancedOptimizationTechniques(unittest.TestCase):
    """Test suite for advanced optimization techniques"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_gradient_accumulation(self):
        """Test gradient accumulation optimization"""
        class GradientAccumulationOptimizer:
            def __init__(self, accumulation_steps=4):
                self.accumulation_steps = accumulation_steps
                self.step_count = 0
                self.accumulated_gradients = {}
                
            def accumulate_gradients(self, model, loss):
                """Accumulate gradients over multiple steps"""
                # Simulate gradient accumulation
                self.step_count += 1
                
                # Accumulate gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if name not in self.accumulated_gradients:
                            self.accumulated_gradients[name] = torch.zeros_like(param.grad)
                        self.accumulated_gradients[name] += param.grad
                
                # Apply gradients when accumulation is complete
                if self.step_count % self.accumulation_steps == 0:
                    self._apply_accumulated_gradients(model)
                    self.accumulated_gradients = {}
                    
                return self.step_count % self.accumulation_steps == 0
                
            def _apply_accumulated_gradients(self, model):
                """Apply accumulated gradients"""
                for name, param in model.named_parameters():
                    if name in self.accumulated_gradients:
                        param.grad = self.accumulated_gradients[name] / self.accumulation_steps
                        
            def get_accumulation_stats(self):
                """Get accumulation statistics"""
                return {
                    'step_count': self.step_count,
                    'accumulation_steps': self.accumulation_steps,
                    'accumulated_params': len(self.accumulated_gradients)
                }
        
        # Test gradient accumulation
        acc_optimizer = GradientAccumulationOptimizer()
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test accumulation
        for step in range(8):
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            applied = acc_optimizer.accumulate_gradients(model, loss)
            if step % 4 == 3:  # Every 4 steps
                self.assertTrue(applied)
            else:
                self.assertFalse(applied)
                
        # Check stats
        stats = acc_optimizer.get_accumulation_stats()
        self.assertEqual(stats['step_count'], 8)
        self.assertEqual(stats['accumulation_steps'], 4)
        
    def test_mixed_precision_training(self):
        """Test mixed precision training"""
        class MixedPrecisionOptimizer:
            def __init__(self, loss_scale=2**16):
                self.loss_scale = loss_scale
                self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
                self.training_stats = {
                    'fp16_operations': 0,
                    'fp32_operations': 0,
                    'loss_scaling_applied': 0
                }
                
            def forward_pass(self, model, data, target):
                """Mixed precision forward pass"""
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = nn.MSELoss()(output, target)
                        scaled_loss = loss * self.loss_scale
                        self.training_stats['fp16_operations'] += 1
                        self.training_stats['loss_scaling_applied'] += 1
                        return scaled_loss, output
                else:
                    # Fallback to FP32
                    output = model(data)
                    loss = nn.MSELoss()(output, target)
                    self.training_stats['fp32_operations'] += 1
                    return loss, output
                    
            def backward_pass(self, scaled_loss, model):
                """Mixed precision backward pass"""
                if self.scaler is not None:
                    scaled_loss.backward()
                    self.scaler.step(model.parameters())
                    self.scaler.update()
                else:
                    scaled_loss.backward()
                    
            def get_precision_stats(self):
                """Get precision statistics"""
                total_operations = self.training_stats['fp16_operations'] + self.training_stats['fp32_operations']
                fp16_ratio = self.training_stats['fp16_operations'] / total_operations if total_operations > 0 else 0
                
                return {
                    'fp16_operations': self.training_stats['fp16_operations'],
                    'fp32_operations': self.training_stats['fp32_operations'],
                    'fp16_ratio': fp16_ratio,
                    'loss_scaling_applied': self.training_stats['loss_scaling_applied']
                }
        
        # Test mixed precision training
        mp_optimizer = MixedPrecisionOptimizer()
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test forward pass
        scaled_loss, output = mp_optimizer.forward_pass(model, data, target)
        self.assertIsInstance(scaled_loss, torch.Tensor)
        self.assertEqual(output.shape, target.shape)
        
        # Test backward pass
        mp_optimizer.backward_pass(scaled_loss, model)
        
        # Check precision stats
        stats = mp_optimizer.get_precision_stats()
        self.assertGreaterEqual(stats['fp16_operations'], 0)
        self.assertGreaterEqual(stats['fp32_operations'], 0)
        self.assertGreaterEqual(stats['fp16_ratio'], 0)
        
    def test_adaptive_optimization_scheduler(self):
        """Test adaptive optimization scheduler"""
        class AdaptiveOptimizationScheduler:
            def __init__(self):
                self.optimization_history = []
                self.current_optimizer = None
                self.optimizer_performance = {}
                
            def add_optimizer(self, name, optimizer, performance_threshold=0.1):
                """Add optimizer to scheduler"""
                self.optimizer_performance[name] = {
                    'optimizer': optimizer,
                    'threshold': performance_threshold,
                    'performance_history': [],
                    'usage_count': 0
                }
                
            def select_optimizer(self, current_performance):
                """Select best optimizer based on performance"""
                best_optimizer = None
                best_score = float('inf')
                
                for name, opt_info in self.optimizer_performance.items():
                    if not opt_info['performance_history'] or current_performance < opt_info['threshold']:
                        # Calculate score based on performance and usage
                        usage_penalty = opt_info['usage_count'] * 0.1
                        score = current_performance + usage_penalty
                        
                        if score < best_score:
                            best_score = score
                            best_optimizer = name
                            
                return best_optimizer
                
            def optimize(self, model, data, target):
                """Run optimization with adaptive scheduler"""
                output = model(data)
                loss = nn.MSELoss()(output, target)
                current_performance = loss.item()
                
                # Select optimizer
                selected_optimizer = self.select_optimizer(current_performance)
                
                if selected_optimizer and selected_optimizer in self.optimizer_performance:
                    optimizer_info = self.optimizer_performance[selected_optimizer]
                    optimizer = optimizer_info['optimizer']
                    
                    # Run optimization
                    result = optimizer.step(loss)
                    
                    # Update performance history
                    optimizer_info['performance_history'].append(current_performance)
                    optimizer_info['usage_count'] += 1
                    
                    # Record optimization
                    self.optimization_history.append({
                        'optimizer': selected_optimizer,
                        'performance': current_performance,
                        'result': result
                    })
                    
                    return result
                    
                return None
                
            def get_scheduler_stats(self):
                """Get scheduler statistics"""
                total_optimizations = len(self.optimization_history)
                optimizer_usage = {name: info['usage_count'] 
                                 for name, info in self.optimizer_performance.items()}
                
                return {
                    'total_optimizations': total_optimizations,
                    'optimizer_usage': optimizer_usage,
                    'average_performance': sum(opt['performance'] for opt in self.optimization_history) / 
                                         total_optimizations if total_optimizations > 0 else 0
                }
        
        # Test adaptive optimization scheduler
        scheduler = AdaptiveOptimizationScheduler()
        
        # Add optimizers
        scheduler.add_optimizer("adam", MockOptimizer(learning_rate=0.001), 0.1)
        scheduler.add_optimizer("sgd", MockOptimizer(learning_rate=0.01), 0.5)
        scheduler.add_optimizer("rmsprop", MockOptimizer(learning_rate=0.0005), 0.2)
        
        # Test optimization
        model = nn.Linear(256, 512)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Run multiple optimizations
        for i in range(5):
            result = scheduler.optimize(model, data, target)
            self.assertIsNotNone(result)
            
        # Check scheduler stats
        stats = scheduler.get_scheduler_stats()
        self.assertGreater(stats['total_optimizations'], 0)
        self.assertGreater(len(stats['optimizer_usage']), 0)

class TestNeuralArchitectureSearch(unittest.TestCase):
    """Test suite for neural architecture search"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_architecture_search(self):
        """Test neural architecture search"""
        class ArchitectureSearch:
            def __init__(self, search_space):
                self.search_space = search_space
                self.architecture_history = []
                self.performance_history = []
                
            def generate_architecture(self):
                """Generate random architecture from search space"""
                architecture = {}
                for component, options in self.search_space.items():
                    architecture[component] = np.random.choice(options)
                return architecture
                
            def evaluate_architecture(self, architecture, data, target):
                """Evaluate architecture performance"""
                # Simulate architecture evaluation
                performance = np.random.uniform(0, 1)
                
                self.architecture_history.append(architecture)
                self.performance_history.append(performance)
                
                return performance
                
            def search_architectures(self, data, target, num_trials=10):
                """Search for best architecture"""
                best_architecture = None
                best_performance = 0
                
                for trial in range(num_trials):
                    architecture = self.generate_architecture()
                    performance = self.evaluate_architecture(architecture, data, target)
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_architecture = architecture
                        
                return best_architecture, best_performance
                
            def get_search_stats(self):
                """Get search statistics"""
                if not self.performance_history:
                    return {}
                    
                return {
                    'total_trials': len(self.performance_history),
                    'best_performance': max(self.performance_history),
                    'average_performance': np.mean(self.performance_history),
                    'performance_std': np.std(self.performance_history),
                    'improvement': max(self.performance_history) - min(self.performance_history)
                }
        
        # Test architecture search
        search_space = {
            'hidden_size': [256, 512, 1024, 2048],
            'num_layers': [2, 4, 6, 8],
            'activation': ['relu', 'gelu', 'swish'],
            'dropout': [0.1, 0.2, 0.3, 0.5]
        }
        
        arch_search = ArchitectureSearch(search_space)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test architecture search
        best_arch, best_perf = arch_search.search_architectures(data, target, num_trials=5)
        
        # Verify results
        self.assertIsNotNone(best_arch)
        self.assertGreater(best_perf, 0)
        self.assertIn('hidden_size', best_arch)
        self.assertIn('num_layers', best_arch)
        
        # Check search stats
        stats = arch_search.get_search_stats()
        self.assertEqual(stats['total_trials'], 5)
        self.assertGreater(stats['best_performance'], 0)
        self.assertGreaterEqual(stats['improvement'], 0)

# Mock optimizer for testing
class MockOptimizer:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.step_count = 0
        
    def step(self, loss):
        self.step_count += 1
        return {'optimized': True, 'step': self.step_count}

if __name__ == '__main__':
    unittest.main()





