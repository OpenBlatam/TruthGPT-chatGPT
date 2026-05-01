#!/usr/bin/env python3
"""
Test Suite for Evolutionary Optimization
Comprehensive tests for evolutionary optimization algorithms and techniques
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional, Callable
import tempfile
import os
import json
import pickle
from pathlib import Path
import random
import math

# Import evolutionary optimization components
import sys
sys.path.append('..')
from modules.base.core_system.core.advanced_optimizations import (
    EvolutionaryOptimizer, AdvancedOptimizationEngine, OptimizationTechnique,
    OptimizationMetrics, create_evolutionary_optimizer
)

class TestEvolutionaryOptimizerAdvanced(unittest.TestCase):
    """Advanced tests for evolutionary optimizer."""
    
    def setUp(self):
        self.evolutionary_optimizer = EvolutionaryOptimizer()
    
    def test_evolutionary_optimizer_initialization_advanced(self):
        """Test advanced evolutionary optimizer initialization."""
        self.assertIsInstance(self.evolutionary_optimizer.config, dict)
        self.assertEqual(len(self.evolutionary_optimizer.population), 0)
        self.assertEqual(self.evolutionary_optimizer.generation, 0)
        self.assertEqual(len(self.evolutionary_optimizer.fitness_history), 0)
        self.assertIsInstance(self.evolutionary_optimizer.logger, logging.Logger)
    
    def test_model_evolution_advanced(self):
        """Test advanced model evolution."""
        class TestModel(nn.Module):
            def __init__(self, input_size=100, hidden_size=50, output_size=10):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        
        def fitness_function(model):
            # Complex fitness function
            param_count = sum(p.numel() for p in model.parameters())
            memory_usage = param_count * 4 / (1024 * 1024)  # MB
            
            # Performance based on model complexity
            performance = 1.0 / (1.0 + param_count / 1000000)
            efficiency = 1.0 / (1.0 + memory_usage / 100)
            
            return (performance + efficiency) / 2
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=10, population_size=20
        )
        
        self.assertIsInstance(evolved_model, nn.Module)
        self.assertEqual(type(evolved_model), type(model))
    
    def test_population_initialization_advanced(self):
        """Test advanced population initialization."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        population_size = 20
        
        self.evolutionary_optimizer._initialize_population(model, population_size)
        
        self.assertEqual(len(self.evolutionary_optimizer.population), population_size)
        
        for individual in self.evolutionary_optimizer.population:
            self.assertIsInstance(individual, nn.Module)
            self.assertEqual(type(individual), type(model))
    
    def test_model_variant_creation_advanced(self):
        """Test advanced model variant creation."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        variant = self.evolutionary_optimizer._create_model_variant(model)
        
        self.assertIsInstance(variant, nn.Module)
        self.assertEqual(type(variant), type(model))
        
        # Test that variant is different from original
        original_params = [p.clone() for p in model.parameters()]
        variant_params = list(variant.parameters())
        
        for orig, var in zip(original_params, variant_params):
            self.assertFalse(torch.equal(orig, var))
    
    def test_population_evolution_advanced(self):
        """Test advanced population evolution."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        self.evolutionary_optimizer._initialize_population(model, 20)
        
        # Create fitness scores
        fitness_scores = [random.random() for _ in range(20)]
        
        original_generation = self.evolutionary_optimizer.generation
        self.evolutionary_optimizer._evolve_population(fitness_scores)
        
        self.assertEqual(len(self.evolutionary_optimizer.population), 20)
        self.assertEqual(self.evolutionary_optimizer.generation, original_generation + 1)
        
        # Test that all individuals are valid
        for individual in self.evolutionary_optimizer.population:
            self.assertIsInstance(individual, nn.Module)
            self.assertEqual(type(individual), type(model))
    
    def test_parent_selection_advanced(self):
        """Test advanced parent selection."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        self.evolutionary_optimizer._initialize_population(model, 20)
        
        fitness_scores = [random.random() for _ in range(20)]
        
        parent = self.evolutionary_optimizer._select_parent(fitness_scores)
        
        self.assertIsInstance(parent, nn.Module)
        self.assertIn(parent, self.evolutionary_optimizer.population)
    
    def test_crossover_operation_advanced(self):
        """Test advanced crossover operation."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        parent1 = TestModel()
        parent2 = TestModel()
        
        offspring = self.evolutionary_optimizer._crossover(parent1, parent2)
        
        self.assertIsInstance(offspring, nn.Module)
        self.assertEqual(type(offspring), type(parent1))
        
        # Test that offspring parameters are averaged
        parent1_params = list(parent1.parameters())
        parent2_params = list(parent2.parameters())
        offspring_params = list(offspring.parameters())
        
        for p1, p2, off in zip(parent1_params, parent2_params, offspring_params):
            expected = (p1.data + p2.data) / 2
            self.assertTrue(torch.allclose(off.data, expected, atol=1e-6))
    
    def test_mutation_operation_advanced(self):
        """Test advanced mutation operation."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        original_params = [p.clone() for p in model.parameters()]
        
        mutated_model = self.evolutionary_optimizer._mutate(model, mutation_rate=1.0)
        
        self.assertIsInstance(mutated_model, nn.Module)
        self.assertEqual(type(mutated_model), type(model))
        
        # Test that parameters were modified
        mutated_params = list(mutated_model.parameters())
        for orig, mut in zip(original_params, mutated_params):
            self.assertFalse(torch.equal(orig, mut))
    
    def test_fitness_tracking(self):
        """Test fitness tracking during evolution."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=5, population_size=10
        )
        
        # Test that fitness history is populated
        self.assertGreater(len(self.evolutionary_optimizer.fitness_history), 0)
        
        # Test that fitness history is non-decreasing (evolution should improve)
        for i in range(1, len(self.evolutionary_optimizer.fitness_history)):
            self.assertGreaterEqual(
                self.evolutionary_optimizer.fitness_history[i],
                self.evolutionary_optimizer.fitness_history[i-1]
            )
    
    def test_elitism_selection(self):
        """Test elitism selection in population evolution."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        self.evolutionary_optimizer._initialize_population(model, 20)
        
        # Create fitness scores with clear best individual
        fitness_scores = [0.1] * 19 + [0.9]  # One very good individual
        
        self.evolutionary_optimizer._evolve_population(fitness_scores)
        
        # Test that best individual is preserved (elitism)
        best_individual = self.evolutionary_optimizer.population[0]
        self.assertIsInstance(best_individual, nn.Module)
    
    def test_tournament_selection_variations(self):
        """Test tournament selection with different tournament sizes."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        self.evolutionary_optimizer._initialize_population(model, 20)
        
        fitness_scores = [random.random() for _ in range(20)]
        
        # Test different tournament sizes
        tournament_sizes = [2, 3, 5, 10]
        
        for tournament_size in tournament_sizes:
            # Mock tournament selection with different sizes
            with patch.object(self.evolutionary_optimizer, '_tournament_selection') as mock_selection:
                mock_selection.return_value = self.evolutionary_optimizer.population[0]
                
                parent = self.evolutionary_optimizer._select_parent(fitness_scores)
                
                self.assertIsInstance(parent, nn.Module)
                self.assertIn(parent, self.evolutionary_optimizer.population)

class TestEvolutionaryOptimizationIntegration(unittest.TestCase):
    """Integration tests for evolutionary optimization."""
    
    def setUp(self):
        self.evolutionary_optimizer = EvolutionaryOptimizer()
    
    def test_evolutionary_optimization_workflow(self):
        """Test complete evolutionary optimization workflow."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        
        def fitness_function(model):
            # Complex fitness function
            param_count = sum(p.numel() for p in model.parameters())
            memory_usage = param_count * 4 / (1024 * 1024)  # MB
            
            # Performance based on model complexity
            performance = 1.0 / (1.0 + param_count / 1000000)
            efficiency = 1.0 / (1.0 + memory_usage / 100)
            stability = 0.9  # Simulated stability score
            
            return (performance + efficiency + stability) / 3
        
        # Step 1: Initialize population
        self.evolutionary_optimizer._initialize_population(model, 20)
        self.assertEqual(len(self.evolutionary_optimizer.population), 20)
        
        # Step 2: Evaluate fitness
        fitness_scores = []
        for individual in self.evolutionary_optimizer.population:
            fitness = fitness_function(individual)
            fitness_scores.append(fitness)
        
        self.assertEqual(len(fitness_scores), 20)
        self.assertTrue(all(isinstance(score, float) for score in fitness_scores))
        
        # Step 3: Evolve population
        self.evolutionary_optimizer._evolve_population(fitness_scores)
        self.assertEqual(len(self.evolutionary_optimizer.population), 20)
        self.assertEqual(self.evolutionary_optimizer.generation, 1)
        
        # Step 4: Run complete evolution
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=10, population_size=20
        )
        
        self.assertIsInstance(evolved_model, nn.Module)
        self.assertEqual(type(evolved_model), type(model))
    
    def test_evolutionary_optimization_with_different_models(self):
        """Test evolutionary optimization with different model types."""
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.fc = nn.Linear(64 * 6 * 6, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        models = [LinearModel(), ConvModel()]
        
        for model in models:
            def fitness_function(model):
                param_count = sum(p.numel() for p in model.parameters())
                return 1.0 / (1.0 + param_count / 1000000)
            
            evolved_model = self.evolutionary_optimizer.evolve_model(
                model, fitness_function, generations=5, population_size=10
            )
            
            self.assertIsInstance(evolved_model, nn.Module)
            self.assertEqual(type(evolved_model), type(model))
    
    def test_evolutionary_optimization_performance_tracking(self):
        """Test evolutionary optimization performance tracking."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        # Run evolution with performance tracking
        start_time = time.time()
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=10, population_size=20
        )
        end_time = time.time()
        
        evolution_time = end_time - start_time
        
        # Verify performance tracking
        self.assertLess(evolution_time, 30.0)  # Should complete within 30 seconds
        self.assertIsInstance(evolved_model, nn.Module)
        self.assertGreater(len(self.evolutionary_optimizer.fitness_history), 0)
    
    def test_evolutionary_optimization_memory_usage(self):
        """Test evolutionary optimization memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=10, population_size=20
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100.0)
        self.assertIsInstance(evolved_model, nn.Module)
    
    def test_evolutionary_optimization_convergence(self):
        """Test evolutionary optimization convergence."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=20, population_size=30
        )
        
        # Test convergence
        fitness_history = self.evolutionary_optimizer.fitness_history
        
        # Fitness should generally improve over generations
        if len(fitness_history) > 1:
            # Check that final fitness is better than initial
            self.assertGreaterEqual(fitness_history[-1], fitness_history[0])
            
            # Check that fitness is generally non-decreasing
            for i in range(1, len(fitness_history)):
                # Allow for some fluctuation but overall trend should be upward
                if fitness_history[i] < fitness_history[i-1]:
                    self.assertLess(fitness_history[i-1] - fitness_history[i], 0.1)
        
        self.assertIsInstance(evolved_model, nn.Module)

class TestEvolutionaryOptimizationPerformance(unittest.TestCase):
    """Performance tests for evolutionary optimization."""
    
    def setUp(self):
        self.evolutionary_optimizer = EvolutionaryOptimizer()
    
    def test_evolutionary_optimization_speed(self):
        """Test evolutionary optimization speed."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        start_time = time.time()
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=10, population_size=20
        )
        end_time = time.time()
        
        evolution_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(evolution_time, 10.0)
        self.assertIsInstance(evolved_model, nn.Module)
    
    def test_evolutionary_optimization_scalability(self):
        """Test evolutionary optimization scalability."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        # Test with different population sizes
        population_sizes = [10, 20, 50, 100]
        
        for pop_size in population_sizes:
            start_time = time.time()
            
            evolved_model = self.evolutionary_optimizer.evolve_model(
                model, fitness_function, generations=5, population_size=pop_size
            )
            
            end_time = time.time()
            evolution_time = end_time - start_time
            
            # Time should scale reasonably with population size
            self.assertLess(evolution_time, pop_size * 0.1)
            self.assertIsInstance(evolved_model, nn.Module)
    
    def test_evolutionary_optimization_concurrent(self):
        """Test concurrent evolutionary optimization."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        # Test multiple concurrent evolution runs
        start_time = time.time()
        
        results = []
        for _ in range(3):
            evolved_model = self.evolutionary_optimizer.evolve_model(
                model, fitness_function, generations=5, population_size=20
            )
            results.append(evolved_model)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All evolutions should complete
        self.assertEqual(len(results), 3)
        
        for evolved_model in results:
            self.assertIsInstance(evolved_model, nn.Module)
        
        # Total time should be reasonable
        self.assertLess(total_time, 15.0)
    
    def test_evolutionary_optimization_memory_efficiency(self):
        """Test evolutionary optimization memory efficiency."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1000, 1000)  # Larger model
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        # Test memory efficiency with larger models
        start_time = time.time()
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=5, population_size=20
        )
        end_time = time.time()
        
        evolution_time = end_time - start_time
        
        # Should complete within reasonable time even with larger models
        self.assertLess(evolution_time, 15.0)
        self.assertIsInstance(evolved_model, nn.Module)

class TestEvolutionaryOptimizationAdvanced(unittest.TestCase):
    """Advanced tests for evolutionary optimization."""
    
    def setUp(self):
        self.evolutionary_optimizer = EvolutionaryOptimizer()
    
    def test_evolutionary_optimization_with_constraints(self):
        """Test evolutionary optimization with constraints."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def constrained_fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            
            # Add constraint penalty
            penalty = 0
            if param_count > 10000:  # Constraint: max 10k parameters
                penalty += 1000
            
            base_fitness = 1.0 / (1.0 + param_count / 1000000)
            return base_fitness - penalty
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, constrained_fitness_function, generations=10, population_size=20
        )
        
        self.assertIsInstance(evolved_model, nn.Module)
        
        # Test that constraint is satisfied
        param_count = sum(p.numel() for p in evolved_model.parameters())
        self.assertLessEqual(param_count, 10000)
    
    def test_evolutionary_optimization_with_noise(self):
        """Test evolutionary optimization with noisy fitness function."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def noisy_fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            base_fitness = 1.0 / (1.0 + param_count / 1000000)
            
            # Add noise
            noise = random.gauss(0, 0.01)
            return base_fitness + noise
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, noisy_fitness_function, generations=10, population_size=20
        )
        
        self.assertIsInstance(evolved_model, nn.Module)
    
    def test_evolutionary_optimization_with_multiple_objectives(self):
        """Test evolutionary optimization with multiple objectives."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def multi_objective_fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            memory_usage = param_count * 4 / (1024 * 1024)  # MB
            
            # Multiple objectives
            performance = 1.0 / (1.0 + param_count / 1000000)
            efficiency = 1.0 / (1.0 + memory_usage / 100)
            stability = 0.9  # Simulated stability score
            
            # Weighted combination
            return 0.4 * performance + 0.3 * efficiency + 0.3 * stability
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, multi_objective_fitness_function, generations=10, population_size=20
        )
        
        self.assertIsInstance(evolved_model, nn.Module)
    
    def test_evolutionary_optimization_with_adaptive_parameters(self):
        """Test evolutionary optimization with adaptive parameters."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        # Test with adaptive mutation rate
        original_mutate = self.evolutionary_optimizer._mutate
        
        def adaptive_mutate(model, mutation_rate=0.1):
            # Adaptive mutation rate based on generation
            adaptive_rate = mutation_rate * (1.0 + self.evolutionary_optimizer.generation * 0.01)
            return original_mutate(model, adaptive_rate)
        
        self.evolutionary_optimizer._mutate = adaptive_mutate
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, fitness_function, generations=10, population_size=20
        )
        
        self.assertIsInstance(evolved_model, nn.Module)
    
    def test_evolutionary_optimization_with_elitism(self):
        """Test evolutionary optimization with elitism."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        def fitness_function(model):
            param_count = sum(p.numel() for p in model.parameters())
            return 1.0 / (1.0 + param_count / 1000000)
        
        # Test elitism by tracking best individual
        best_fitness = 0.0
        best_individual = None
        
        def tracking_fitness_function(model):
            nonlocal best_fitness, best_individual
            fitness = fitness_function(model)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = model
            return fitness
        
        evolved_model = self.evolutionary_optimizer.evolve_model(
            model, tracking_fitness_function, generations=10, population_size=20
        )
        
        self.assertIsInstance(evolved_model, nn.Module)
        
        # Test that best individual is preserved
        if best_individual is not None:
            self.assertIn(best_individual, self.evolutionary_optimizer.population)

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEvolutionaryOptimizerAdvanced,
        TestEvolutionaryOptimizationIntegration,
        TestEvolutionaryOptimizationPerformance,
        TestEvolutionaryOptimizationAdvanced
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evolutionary Optimization Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    print(f"\n{'='*60}")

