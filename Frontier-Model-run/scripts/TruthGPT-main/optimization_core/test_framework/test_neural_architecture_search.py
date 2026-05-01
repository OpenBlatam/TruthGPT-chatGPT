#!/usr/bin/env python3
"""
Test Suite for Neural Architecture Search (NAS)
Comprehensive tests for Neural Architecture Search functionality and algorithms
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import json
import pickle
from pathlib import Path
import random
import math

# Import NAS components
import sys
sys.path.append('..')
from bulk.ultra_advanced_optimizer import (
    NeuralArchitectureSearch, NeuralArchitecture, UltraAdvancedOptimizer
)
from modules.base.core_system.core.advanced_optimizations import (
    NeuralArchitectureSearch as AdvancedNAS, AdvancedOptimizationEngine,
    OptimizationTechnique, OptimizationMetrics
)

class TestNeuralArchitectureAdvanced(unittest.TestCase):
    """Advanced tests for neural architecture representation."""
    
    def setUp(self):
        self.architecture = NeuralArchitecture(
            layers=[{'type': 'linear', 'size': 128}, {'type': 'conv2d', 'size': 64}],
            connections=[(0, 1), (1, 2)],
            activation_functions=['relu', 'gelu'],
            dropout_rates=[0.1, 0.2],
            batch_norm=[True, False],
            performance_score=0.85,
            complexity_score=0.3
        )
    
    def test_neural_architecture_creation_advanced(self):
        """Test advanced neural architecture creation."""
        self.assertEqual(len(self.architecture.layers), 2)
        self.assertEqual(len(self.architecture.connections), 2)
        self.assertEqual(len(self.architecture.activation_functions), 2)
        self.assertEqual(len(self.architecture.dropout_rates), 2)
        self.assertEqual(len(self.architecture.batch_norm), 2)
        self.assertEqual(self.architecture.performance_score, 0.85)
        self.assertEqual(self.architecture.complexity_score, 0.3)
    
    def test_neural_architecture_validation(self):
        """Test neural architecture validation."""
        # Test layer validation
        for layer in self.architecture.layers:
            self.assertIn('type', layer)
            self.assertIn('size', layer)
            self.assertIn(layer['type'], ['linear', 'conv1d', 'conv2d', 'lstm', 'gru'])
            self.assertGreater(layer['size'], 0)
        
        # Test connection validation
        for connection in self.architecture.connections:
            self.assertIsInstance(connection, tuple)
            self.assertEqual(len(connection), 2)
            self.assertLess(connection[0], connection[1])
            self.assertGreaterEqual(connection[0], 0)
            self.assertLess(connection[1], len(self.architecture.layers))
        
        # Test activation function validation
        valid_activations = ['relu', 'sigmoid', 'tanh', 'gelu', 'swish', 'mish']
        for activation in self.architecture.activation_functions:
            self.assertIn(activation, valid_activations)
        
        # Test dropout rate validation
        for dropout_rate in self.architecture.dropout_rates:
            self.assertGreaterEqual(dropout_rate, 0.0)
            self.assertLessEqual(dropout_rate, 1.0)
        
        # Test batch norm validation
        for batch_norm in self.architecture.batch_norm:
            self.assertIsInstance(batch_norm, bool)
    
    def test_neural_architecture_serialization(self):
        """Test neural architecture serialization."""
        # Test JSON serialization
        arch_dict = {
            'layers': self.architecture.layers,
            'connections': self.architecture.connections,
            'activation_functions': self.architecture.activation_functions,
            'dropout_rates': self.architecture.dropout_rates,
            'batch_norm': self.architecture.batch_norm,
            'performance_score': self.architecture.performance_score,
            'complexity_score': self.architecture.complexity_score
        }
        
        json_str = json.dumps(arch_dict)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized['layers'], self.architecture.layers)
        self.assertEqual(deserialized['connections'], self.architecture.connections)
        self.assertEqual(deserialized['activation_functions'], self.architecture.activation_functions)
        self.assertEqual(deserialized['dropout_rates'], self.architecture.dropout_rates)
        self.assertEqual(deserialized['batch_norm'], self.architecture.batch_norm)
        self.assertEqual(deserialized['performance_score'], self.architecture.performance_score)
        self.assertEqual(deserialized['complexity_score'], self.architecture.complexity_score)
    
    def test_neural_architecture_complexity_calculation(self):
        """Test neural architecture complexity calculation."""
        # Calculate complexity manually
        n_layers = len(self.architecture.layers)
        n_connections = len(self.architecture.connections)
        n_parameters = sum(layer.get('size', 0) for layer in self.architecture.layers)
        
        expected_complexity = n_layers * 0.1 + n_connections * 0.05 + n_parameters * 0.001
        
        # Test complexity calculation
        self.assertGreater(self.architecture.complexity_score, 0)
        self.assertIsInstance(self.architecture.complexity_score, float)
    
    def test_neural_architecture_performance_estimation(self):
        """Test neural architecture performance estimation."""
        # Test performance score validation
        self.assertGreaterEqual(self.architecture.performance_score, 0.0)
        self.assertLessEqual(self.architecture.performance_score, 1.0)
        self.assertIsInstance(self.architecture.performance_score, float)
    
    def test_neural_architecture_mutation(self):
        """Test neural architecture mutation."""
        # Create a copy for mutation
        mutated_arch = NeuralArchitecture(
            layers=self.architecture.layers.copy(),
            connections=self.architecture.connections.copy(),
            activation_functions=self.architecture.activation_functions.copy(),
            dropout_rates=self.architecture.dropout_rates.copy(),
            batch_norm=self.architecture.batch_norm.copy(),
            performance_score=self.architecture.performance_score,
            complexity_score=self.architecture.complexity_score
        )
        
        # Mutate layers
        mutated_arch.layers.append({'type': 'linear', 'size': 256})
        
        # Mutate connections
        mutated_arch.connections.append((2, 3))
        
        # Mutate activations
        mutated_arch.activation_functions.append('swish')
        
        # Mutate dropout rates
        mutated_arch.dropout_rates.append(0.3)
        
        # Mutate batch norm
        mutated_arch.batch_norm.append(True)
        
        # Verify mutations
        self.assertEqual(len(mutated_arch.layers), 3)
        self.assertEqual(len(mutated_arch.connections), 3)
        self.assertEqual(len(mutated_arch.activation_functions), 3)
        self.assertEqual(len(mutated_arch.dropout_rates), 3)
        self.assertEqual(len(mutated_arch.batch_norm), 3)
    
    def test_neural_architecture_crossover(self):
        """Test neural architecture crossover."""
        # Create second architecture
        arch2 = NeuralArchitecture(
            layers=[{'type': 'conv2d', 'size': 32}, {'type': 'linear', 'size': 16}],
            connections=[(0, 1)],
            activation_functions=['tanh'],
            dropout_rates=[0.3],
            batch_norm=[False],
            performance_score=0.75,
            complexity_score=0.2
        )
        
        # Perform crossover
        child_arch = NeuralArchitecture(
            layers=self.architecture.layers[:1] + arch2.layers[1:],
            connections=list(set(self.architecture.connections + arch2.connections)),
            activation_functions=self.architecture.activation_functions[:1] + arch2.activation_functions[1:],
            dropout_rates=self.architecture.dropout_rates[:1] + arch2.dropout_rates[1:],
            batch_norm=self.architecture.batch_norm[:1] + arch2.batch_norm[1:],
            performance_score=0.0,
            complexity_score=0.0
        )
        
        # Verify crossover
        self.assertEqual(len(child_arch.layers), 2)
        self.assertEqual(len(child_arch.connections), 3)  # Union of connections
        self.assertEqual(len(child_arch.activation_functions), 2)
        self.assertEqual(len(child_arch.dropout_rates), 2)
        self.assertEqual(len(child_arch.batch_norm), 2)

class TestNeuralArchitectureSearchAdvanced(unittest.TestCase):
    """Advanced tests for Neural Architecture Search."""
    
    def setUp(self):
        self.nas = NeuralArchitectureSearch({})
    
    def test_nas_initialization_advanced(self):
        """Test advanced NAS initialization."""
        self.assertIsInstance(self.nas.config, dict)
        self.assertIsInstance(self.nas.search_space, dict)
        self.assertEqual(len(self.nas.architecture_history), 0)
        self.assertIsNone(self.nas.best_architecture)
        self.assertEqual(self.nas.best_performance, 0.0)
        self.assertIsInstance(self.nas.logger, logging.Logger)
    
    def test_search_space_creation_advanced(self):
        """Test advanced search space creation."""
        search_space = self.nas._create_search_space()
        
        # Test all required keys
        required_keys = [
            'layer_types', 'activation_functions', 'normalization_layers',
            'dropout_rates', 'hidden_sizes', 'num_layers', 'attention_heads', 'kernel_sizes'
        ]
        
        for key in required_keys:
            self.assertIn(key, search_space)
            self.assertIsInstance(search_space[key], list)
            self.assertGreater(len(search_space[key]), 0)
        
        # Test layer types
        valid_layer_types = ['linear', 'conv2d', 'lstm', 'transformer', 'attention']
        for layer_type in search_space['layer_types']:
            self.assertIn(layer_type, valid_layer_types)
        
        # Test activation functions
        valid_activations = ['relu', 'gelu', 'swish', 'mish', 'leaky_relu']
        for activation in search_space['activation_functions']:
            self.assertIn(activation, valid_activations)
        
        # Test normalization layers
        valid_norms = ['batch_norm', 'layer_norm', 'group_norm', 'instance_norm']
        for norm in search_space['normalization_layers']:
            self.assertIn(norm, valid_norms)
        
        # Test dropout rates
        for dropout_rate in search_space['dropout_rates']:
            self.assertGreaterEqual(dropout_rate, 0.0)
            self.assertLessEqual(dropout_rate, 1.0)
        
        # Test hidden sizes
        for hidden_size in search_space['hidden_sizes']:
            self.assertGreater(hidden_size, 0)
            self.assertIsInstance(hidden_size, int)
        
        # Test number of layers
        for num_layers in search_space['num_layers']:
            self.assertGreater(num_layers, 0)
            self.assertIsInstance(num_layers, int)
        
        # Test attention heads
        for attention_heads in search_space['attention_heads']:
            self.assertGreater(attention_heads, 0)
            self.assertIsInstance(attention_heads, int)
        
        # Test kernel sizes
        for kernel_size in search_space['kernel_sizes']:
            self.assertGreater(kernel_size, 0)
            self.assertIsInstance(kernel_size, int)
    
    def test_architecture_search_advanced(self):
        """Test advanced architecture search."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(784, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dataset_info = {'input_size': 784, 'output_size': 10}
        
        result = self.nas.search_optimal_architecture(
            model, dataset_info, max_iterations=10
        )
        
        self.assertIn('best_architecture', result)
        self.assertIn('best_performance', result)
        self.assertIn('search_history', result)
        self.assertIsInstance(result['best_architecture'], dict)
        self.assertIsInstance(result['best_performance'], float)
        self.assertIsInstance(result['search_history'], list)
        
        # Test that best architecture is valid
        best_arch = result['best_architecture']
        self.assertIn('hidden_sizes', best_arch)
        self.assertIn('num_layers', best_arch)
        self.assertIn('activation_functions', best_arch)
        self.assertIn('dropout_rates', best_arch)
        self.assertIn('normalization_layers', best_arch)
    
    def test_population_initialization_advanced(self):
        """Test advanced population initialization."""
        population = self.nas._initialize_population(20)
        
        self.assertEqual(len(population), 20)
        
        for individual in population:
            self.assertIsInstance(individual, dict)
            
            # Test that all search space keys are present
            for key in self.nas.search_space.keys():
                self.assertIn(key, individual)
                self.assertIn(individual[key], self.nas.search_space[key])
            
            # Test individual validity
            self.assertIn(individual['hidden_sizes'], self.nas.search_space['hidden_sizes'])
            self.assertIn(individual['num_layers'], self.nas.search_space['num_layers'])
            self.assertIn(individual['activation_functions'], self.nas.search_space['activation_functions'])
            self.assertIn(individual['dropout_rates'], self.nas.search_space['dropout_rates'])
            self.assertIn(individual['normalization_layers'], self.nas.search_space['normalization_layers'])
    
    def test_architecture_evaluation_advanced(self):
        """Test advanced architecture evaluation."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(784, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dataset_info = {'input_size': 784, 'output_size': 10}
        architecture = {
            'hidden_sizes': 128,
            'num_layers': 3,
            'activation_functions': 'relu',
            'dropout_rates': 0.2,
            'normalization_layers': 'batch_norm'
        }
        
        fitness = self.nas._evaluate_architecture(architecture, model, dataset_info)
        
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
    
    def test_model_building_advanced(self):
        """Test advanced model building from architecture."""
        dataset_info = {'input_size': 784, 'output_size': 10}
        architecture = {
            'hidden_sizes': 128,
            'num_layers': 3,
            'activation_functions': 'relu',
            'dropout_rates': 0.2,
            'normalization_layers': 'batch_norm'
        }
        
        model = self.nas._build_model_from_architecture(architecture, dataset_info)
        
        self.assertIsInstance(model, nn.Module)
        self.assertGreater(len(list(model.modules())), 1)
        
        # Test model forward pass
        test_input = torch.randn(1, 784)
        with torch.no_grad():
            output = model(test_input)
            self.assertEqual(output.shape, (1, 10))
    
    def test_performance_score_calculation_advanced(self):
        """Test advanced performance score calculation."""
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
        dataset_info = {'input_size': 100, 'output_size': 10}
        
        score = self.nas._calculate_performance_score(model, dataset_info)
        
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_efficiency_score_calculation_advanced(self):
        """Test advanced efficiency score calculation."""
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
        score = self.nas._calculate_efficiency_score(model)
        
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_stability_score_calculation_advanced(self):
        """Test advanced stability score calculation."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(784, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        score = self.nas._calculate_stability_score(model)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_flops_estimation_advanced(self):
        """Test advanced FLOPs estimation."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                self.conv1 = nn.Conv2d(3, 32, 3)
                self.conv2 = nn.Conv2d(32, 64, 3)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = self.linear2(x)
                return x
        
        model = TestModel()
        flops = self.nas._estimate_flops(model)
        
        self.assertIsInstance(flops, int)
        self.assertGreater(flops, 0)
    
    def test_population_evolution_advanced(self):
        """Test advanced population evolution."""
        population = self.nas._initialize_population(20)
        fitness_scores = [random.random() for _ in range(20)]
        
        new_population = self.nas._evolve_population(population, fitness_scores)
        
        self.assertEqual(len(new_population), len(population))
        self.assertTrue(all(isinstance(individual, dict) for individual in new_population))
        
        # Test that all individuals are valid
        for individual in new_population:
            for key in self.nas.search_space.keys():
                self.assertIn(key, individual)
                self.assertIn(individual[key], self.nas.search_space[key])
    
    def test_tournament_selection_advanced(self):
        """Test advanced tournament selection."""
        population = self.nas._initialize_population(20)
        fitness_scores = [random.random() for _ in range(20)]
        
        selected = self.nas._tournament_selection(population, fitness_scores)
        
        self.assertIsInstance(selected, dict)
        self.assertIn(selected, population)
        
        # Test that selected individual is valid
        for key in self.nas.search_space.keys():
            self.assertIn(key, selected)
            self.assertIn(selected[key], self.nas.search_space[key])
    
    def test_crossover_operation_advanced(self):
        """Test advanced crossover operation."""
        parent1 = self.nas._initialize_population(1)[0]
        parent2 = self.nas._initialize_population(1)[0]
        
        child = self.nas._crossover(parent1, parent2)
        
        self.assertIsInstance(child, dict)
        
        # Test that child has all required keys
        for key in self.nas.search_space.keys():
            self.assertIn(key, child)
            self.assertIn(child[key], self.nas.search_space[key])
    
    def test_mutation_operation_advanced(self):
        """Test advanced mutation operation."""
        individual = self.nas._initialize_population(1)[0]
        
        mutated = self.nas._mutate(individual, mutation_rate=0.5)
        
        self.assertIsInstance(mutated, dict)
        
        # Test that mutated individual has all required keys
        for key in self.nas.search_space.keys():
            self.assertIn(key, mutated)
            self.assertIn(mutated[key], self.nas.search_space[key])

class TestNeuralArchitectureSearchIntegration(unittest.TestCase):
    """Integration tests for Neural Architecture Search."""
    
    def setUp(self):
        self.nas = NeuralArchitectureSearch({})
    
    def test_nas_workflow_complete(self):
        """Test complete NAS workflow."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(784, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dataset_info = {'input_size': 784, 'output_size': 10}
        
        # Step 1: Initialize population
        population = self.nas._initialize_population(10)
        self.assertEqual(len(population), 10)
        
        # Step 2: Evaluate population
        fitness_scores = []
        for individual in population:
            fitness = self.nas._evaluate_architecture(individual, model, dataset_info)
            fitness_scores.append(fitness)
        
        self.assertEqual(len(fitness_scores), 10)
        self.assertTrue(all(isinstance(score, float) for score in fitness_scores))
        
        # Step 3: Evolve population
        new_population = self.nas._evolve_population(population, fitness_scores)
        self.assertEqual(len(new_population), 10)
        
        # Step 4: Run complete search
        result = self.nas.search_optimal_architecture(
            model, dataset_info, max_iterations=5
        )
        
        self.assertIn('best_architecture', result)
        self.assertIn('best_performance', result)
        self.assertIn('search_history', result)
    
    def test_nas_with_different_models(self):
        """Test NAS with different model types."""
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
        dataset_infos = [
            {'input_size': 100, 'output_size': 10},
            {'input_size': 3 * 32 * 32, 'output_size': 10}
        ]
        
        for model, dataset_info in zip(models, dataset_infos):
            result = self.nas.search_optimal_architecture(
                model, dataset_info, max_iterations=5
            )
            
            self.assertIn('best_architecture', result)
            self.assertIn('best_performance', result)
            self.assertIn('search_history', result)
    
    def test_nas_performance_tracking(self):
        """Test NAS performance tracking."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(784, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dataset_info = {'input_size': 784, 'output_size': 10}
        
        # Run NAS with performance tracking
        start_time = time.time()
        result = self.nas.search_optimal_architecture(
            model, dataset_info, max_iterations=10
        )
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # Verify performance tracking
        self.assertLess(search_time, 30.0)  # Should complete within 30 seconds
        self.assertIn('best_architecture', result)
        self.assertIn('best_performance', result)
        self.assertIn('search_history', result)
        
        # Test that search history is populated
        self.assertGreater(len(result['search_history']), 0)
    
    def test_nas_memory_usage(self):
        """Test NAS memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(784, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dataset_info = {'input_size': 784, 'output_size': 10}
        
        result = self.nas.search_optimal_architecture(
            model, dataset_info, max_iterations=10
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100.0)
        self.assertIn('best_architecture', result)
        self.assertIn('best_performance', result)
        self.assertIn('search_history', result)

class TestNeuralArchitectureSearchPerformance(unittest.TestCase):
    """Performance tests for Neural Architecture Search."""
    
    def setUp(self):
        self.nas = NeuralArchitectureSearch({})
    
    def test_nas_speed_benchmark(self):
        """Test NAS speed benchmark."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dataset_info = {'input_size': 100, 'output_size': 10}
        
        start_time = time.time()
        result = self.nas.search_optimal_architecture(
            model, dataset_info, max_iterations=20
        )
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(search_time, 10.0)
        self.assertIn('best_architecture', result)
        self.assertIn('best_performance', result)
        self.assertIn('search_history', result)
    
    def test_nas_scalability(self):
        """Test NAS scalability with different population sizes."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dataset_info = {'input_size': 100, 'output_size': 10}
        
        population_sizes = [5, 10, 20, 50]
        
        for pop_size in population_sizes:
            start_time = time.time()
            
            # Initialize population
            population = self.nas._initialize_population(pop_size)
            self.assertEqual(len(population), pop_size)
            
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = self.nas._evaluate_architecture(individual, model, dataset_info)
                fitness_scores.append(fitness)
            
            # Evolve population
            new_population = self.nas._evolve_population(population, fitness_scores)
            self.assertEqual(len(new_population), pop_size)
            
            end_time = time.time()
            evolution_time = end_time - start_time
            
            # Time should scale reasonably with population size
            self.assertLess(evolution_time, pop_size * 0.1)
    
    def test_nas_concurrent_execution(self):
        """Test concurrent NAS execution."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        dataset_info = {'input_size': 100, 'output_size': 10}
        
        # Test multiple concurrent NAS runs
        start_time = time.time()
        
        results = []
        for _ in range(3):
            result = self.nas.search_optimal_architecture(
                model, dataset_info, max_iterations=5
            )
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All searches should complete
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIn('best_architecture', result)
            self.assertIn('best_performance', result)
            self.assertIn('search_history', result)
        
        # Total time should be reasonable
        self.assertLess(total_time, 15.0)

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestNeuralArchitectureAdvanced,
        TestNeuralArchitectureSearchAdvanced,
        TestNeuralArchitectureSearchIntegration,
        TestNeuralArchitectureSearchPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Neural Architecture Search Test Results")
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

