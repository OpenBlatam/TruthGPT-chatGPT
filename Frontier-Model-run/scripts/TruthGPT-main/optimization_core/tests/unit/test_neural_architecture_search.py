"""
Unit tests for neural architecture search (NAS)
Tests automated architecture discovery, search strategies, and architecture optimization
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import itertools

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestNeuralArchitectureSearch(unittest.TestCase):
    """Test suite for neural architecture search"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_architecture_search_space(self):
        """Test architecture search space definition"""
        class ArchitectureSearchSpace:
            def __init__(self):
                self.search_space = {
                    'layers': [2, 4, 6, 8, 12, 16],
                    'hidden_sizes': [128, 256, 512, 1024, 2048],
                    'activations': ['relu', 'gelu', 'swish', 'mish'],
                    'dropout_rates': [0.1, 0.2, 0.3, 0.4, 0.5],
                    'attention_heads': [4, 8, 16, 32],
                    'attention_types': ['self', 'cross', 'multi_head']
                }
                self.architecture_history = []
                
            def generate_architecture(self):
                """Generate random architecture from search space"""
                architecture = {
                    'layers': np.random.choice(self.search_space['layers']),
                    'hidden_size': np.random.choice(self.search_space['hidden_sizes']),
                    'activation': np.random.choice(self.search_space['activations']),
                    'dropout_rate': np.random.choice(self.search_space['dropout_rates']),
                    'attention_heads': np.random.choice(self.search_space['attention_heads']),
                    'attention_type': np.random.choice(self.search_space['attention_types'])
                }
                return architecture
                
            def evaluate_architecture(self, architecture, data, target):
                """Evaluate architecture performance"""
                # Simulate architecture evaluation
                performance = np.random.uniform(0, 1)
                
                result = {
                    'architecture': architecture,
                    'performance': performance,
                    'timestamp': len(self.architecture_history)
                }
                
                self.architecture_history.append(result)
                return performance
                
            def get_search_space_stats(self):
                """Get search space statistics"""
                total_combinations = 1
                for param_name, values in self.search_space.items():
                    total_combinations *= len(values)
                    
                return {
                    'total_combinations': total_combinations,
                    'parameters': list(self.search_space.keys()),
                    'parameter_counts': {param: len(values) for param, values in self.search_space.items()},
                    'evaluated_architectures': len(self.architecture_history)
                }
        
        # Test architecture search space
        search_space = ArchitectureSearchSpace()
        
        # Test architecture generation
        architecture = search_space.generate_architecture()
        self.assertIn('layers', architecture)
        self.assertIn('hidden_size', architecture)
        self.assertIn('activation', architecture)
        self.assertIn('dropout_rate', architecture)
        self.assertIn('attention_heads', architecture)
        self.assertIn('attention_type', architecture)
        
        # Test architecture evaluation
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        performance = search_space.evaluate_architecture(architecture, data, target)
        self.assertGreater(performance, 0)
        
        # Check search space stats
        stats = search_space.get_search_space_stats()
        self.assertGreater(stats['total_combinations'], 0)
        self.assertGreater(len(stats['parameters']), 0)
        self.assertEqual(stats['evaluated_architectures'], 1)
        
    def test_evolutionary_architecture_search(self):
        """Test evolutionary architecture search"""
        class EvolutionaryArchitectureSearch:
            def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.8):
                self.population_size = population_size
                self.mutation_rate = mutation_rate
                self.crossover_rate = crossover_rate
                self.population = []
                self.fitness_history = []
                self.best_architecture = None
                self.best_fitness = float('-inf')
                
            def initialize_population(self, search_space):
                """Initialize population with random architectures"""
                self.population = []
                for _ in range(self.population_size):
                    architecture = self._generate_random_architecture(search_space)
                    self.population.append(architecture)
                    
            def _generate_random_architecture(self, search_space):
                """Generate random architecture"""
                architecture = {}
                for param_name, values in search_space.items():
                    architecture[param_name] = np.random.choice(values)
                return architecture
                
            def evaluate_population(self, data, target):
                """Evaluate population fitness"""
                fitness_scores = []
                for individual in self.population:
                    # Simulate fitness evaluation
                    fitness = np.random.uniform(0, 1)
                    fitness_scores.append(fitness)
                    
                    # Update best architecture
                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_architecture = individual.copy()
                        
                # Record generation stats
                self.fitness_history.append({
                    'generation': len(self.fitness_history),
                    'best_fitness': max(fitness_scores),
                    'average_fitness': np.mean(fitness_scores),
                    'fitness_std': np.std(fitness_scores)
                })
                
                return fitness_scores
                
            def selection(self, fitness_scores):
                """Tournament selection"""
                tournament_size = 3
                selected = []
                
                for _ in range(self.population_size):
                    # Tournament selection
                    tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
                    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                    winner_index = tournament_indices[np.argmax(tournament_fitness)]
                    selected.append(self.population[winner_index].copy())
                    
                return selected
                
            def crossover(self, parent1, parent2):
                """Crossover operation"""
                child = {}
                for param_name in parent1:
                    if np.random.random() < self.crossover_rate:
                        # Random choice between parents
                        child[param_name] = np.random.choice([parent1[param_name], parent2[param_name]])
                    else:
                        child[param_name] = parent1[param_name]
                return child
                
            def mutation(self, individual, search_space):
                """Mutation operation"""
                mutated = individual.copy()
                for param_name in individual:
                    if np.random.random() < self.mutation_rate:
                        # Random mutation
                        mutated[param_name] = np.random.choice(search_space[param_name])
                return mutated
                
            def evolve_generation(self, data, target, search_space):
                """Evolve one generation"""
                # Evaluate population
                fitness_scores = self.evaluate_population(data, target)
                
                # Selection
                selected = self.selection(fitness_scores)
                
                # Create new generation
                new_population = []
                for i in range(0, self.population_size, 2):
                    parent1 = selected[i]
                    parent2 = selected[i + 1] if i + 1 < len(selected) else selected[i]
                    
                    # Crossover
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                    
                    # Mutation
                    child1 = self.mutation(child1, search_space)
                    child2 = self.mutation(child2, search_space)
                    
                    new_population.extend([child1, child2])
                    
                self.population = new_population[:self.population_size]
                return max(fitness_scores)
                
            def search(self, search_space, data, target, generations=10):
                """Run evolutionary architecture search"""
                self.initialize_population(search_space)
                
                for generation in range(generations):
                    best_fitness = self.evolve_generation(data, target, search_space)
                    
                return self.best_architecture, self.best_fitness
                
            def get_evolution_stats(self):
                """Get evolution statistics"""
                if not self.fitness_history:
                    return {}
                    
                return {
                    'total_generations': len(self.fitness_history),
                    'best_fitness': self.best_fitness,
                    'final_fitness': self.fitness_history[-1]['best_fitness'],
                    'average_fitness': np.mean([gen['average_fitness'] for gen in self.fitness_history]),
                    'fitness_improvement': self.fitness_history[-1]['best_fitness'] - self.fitness_history[0]['best_fitness']
                }
        
        # Test evolutionary architecture search
        search_space = {
            'layers': [2, 4, 6, 8],
            'hidden_sizes': [128, 256, 512],
            'activations': ['relu', 'gelu'],
            'dropout_rates': [0.1, 0.2, 0.3]
        }
        
        evo_search = EvolutionaryArchitectureSearch(population_size=10)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test evolutionary search
        best_architecture, best_fitness = evo_search.search(search_space, data, target, generations=5)
        
        # Verify results
        self.assertIsNotNone(best_architecture)
        self.assertGreater(best_fitness, 0)
        self.assertIn('layers', best_architecture)
        self.assertIn('hidden_sizes', best_architecture)
        self.assertIn('activations', best_architecture)
        self.assertIn('dropout_rates', best_architecture)
        
        # Check evolution stats
        stats = evo_search.get_evolution_stats()
        self.assertEqual(stats['total_generations'], 5)
        self.assertGreater(stats['best_fitness'], 0)
        self.assertGreaterEqual(stats['fitness_improvement'], 0)

class TestReinforcementLearningNAS(unittest.TestCase):
    """Test suite for reinforcement learning based NAS"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_rl_nas_controller(self):
        """Test RL-based NAS controller"""
        class RLNASController:
            def __init__(self, action_space_size, state_size):
                self.action_space_size = action_space_size
                self.state_size = state_size
                self.policy_network = self._initialize_policy_network()
                self.experience_buffer = []
                self.reward_history = []
                
            def _initialize_policy_network(self):
                """Initialize policy network"""
                return {
                    'weights': np.random.normal(0, 0.1, (self.state_size, self.action_space_size)),
                    'biases': np.random.normal(0, 0.1, self.action_space_size)
                }
                
            def get_action(self, state):
                """Get action from policy"""
                # Policy network forward pass
                logits = np.dot(state, self.policy_network['weights']) + self.policy_network['biases']
                probabilities = self._softmax(logits)
                
                # Sample action
                action = np.random.choice(self.action_space_size, p=probabilities)
                return action, probabilities
                
            def _softmax(self, x):
                """Softmax function"""
                exp_x = np.exp(x - np.max(x))
                return exp_x / np.sum(exp_x)
                
            def update_policy(self, states, actions, rewards):
                """Update policy network"""
                # Simple policy gradient update
                for state, action, reward in zip(states, actions, rewards):
                    # Compute gradient
                    _, probabilities = self.get_action(state)
                    gradient = np.zeros_like(self.policy_network['weights'])
                    
                    for i in range(self.action_space_size):
                        if i == action:
                            gradient[:, i] = state * (1 - probabilities[i])
                        else:
                            gradient[:, i] = -state * probabilities[i]
                    
                    # Update weights
                    learning_rate = 0.01
                    self.policy_network['weights'] += learning_rate * reward * gradient
                    
            def store_experience(self, state, action, reward):
                """Store experience"""
                self.experience_buffer.append({
                    'state': state,
                    'action': action,
                    'reward': reward
                })
                
            def get_controller_stats(self):
                """Get controller statistics"""
                return {
                    'total_experiences': len(self.experience_buffer),
                    'average_reward': np.mean([exp['reward'] for exp in self.experience_buffer]) if self.experience_buffer else 0,
                    'policy_entropy': self._calculate_policy_entropy()
                }
                
            def _calculate_policy_entropy(self):
                """Calculate policy entropy"""
                if not self.experience_buffer:
                    return 0
                    
                # Sample recent states
                recent_states = [exp['state'] for exp in self.experience_buffer[-10:]]
                if not recent_states:
                    return 0
                    
                entropies = []
                for state in recent_states:
                    _, probabilities = self.get_action(state)
                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                    entropies.append(entropy)
                    
                return np.mean(entropies)
        
        # Test RL NAS controller
        controller = RLNASController(action_space_size=10, state_size=5)
        
        # Test action selection
        state = np.random.uniform(0, 1, 5)
        action, probabilities = controller.get_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 10)
        self.assertEqual(len(probabilities), 10)
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)
        
        # Test experience storage
        for _ in range(5):
            state = np.random.uniform(0, 1, 5)
            action = np.random.randint(0, 10)
            reward = np.random.uniform(0, 1)
            controller.store_experience(state, action, reward)
            
        # Test policy update
        states = [np.random.uniform(0, 1, 5) for _ in range(3)]
        actions = [np.random.randint(0, 10) for _ in range(3)]
        rewards = [np.random.uniform(0, 1) for _ in range(3)]
        controller.update_policy(states, actions, rewards)
        
        # Check controller stats
        stats = controller.get_controller_stats()
        self.assertEqual(stats['total_experiences'], 5)
        self.assertGreaterEqual(stats['average_reward'], 0)
        self.assertGreaterEqual(stats['policy_entropy'], 0)
        
    def test_rl_nas_environment(self):
        """Test RL NAS environment"""
        class RLNASEnvironment:
            def __init__(self, search_space):
                self.search_space = search_space
                self.current_architecture = None
                self.architecture_history = []
                self.performance_history = []
                
            def reset(self):
                """Reset environment"""
                self.current_architecture = {}
                self.architecture_history = []
                self.performance_history = []
                return self._get_state()
                
            def step(self, action):
                """Take step in environment"""
                # Update architecture based on action
                self._update_architecture(action)
                
                # Evaluate architecture
                performance = self._evaluate_architecture()
                
                # Get reward
                reward = self._calculate_reward(performance)
                
                # Get next state
                next_state = self._get_state()
                
                # Check if done
                done = self._is_done()
                
                return next_state, reward, done, {'performance': performance}
                
            def _update_architecture(self, action):
                """Update architecture based on action"""
                # Map action to architecture parameter
                param_names = list(self.search_space.keys())
                param_idx = action % len(param_names)
                param_name = param_names[param_idx]
                
                # Update parameter
                if param_name not in self.current_architecture:
                    self.current_architecture[param_name] = np.random.choice(self.search_space[param_name])
                else:
                    # Try to improve parameter
                    current_value = self.current_architecture[param_name]
                    param_values = self.search_space[param_name]
                    current_idx = list(param_values).index(current_value)
                    
                    # Move to next value
                    next_idx = (current_idx + 1) % len(param_values)
                    self.current_architecture[param_name] = param_values[next_idx]
                    
            def _evaluate_architecture(self):
                """Evaluate current architecture"""
                # Simulate architecture evaluation
                performance = np.random.uniform(0, 1)
                
                self.architecture_history.append(self.current_architecture.copy())
                self.performance_history.append(performance)
                
                return performance
                
            def _calculate_reward(self, performance):
                """Calculate reward"""
                if not self.performance_history:
                    return performance
                    
                # Reward improvement
                previous_performance = self.performance_history[-2] if len(self.performance_history) > 1 else 0
                improvement = performance - previous_performance
                
                # Reward based on improvement
                reward = improvement * 10  # Scale reward
                
                return reward
                
            def _get_state(self):
                """Get current state"""
                if not self.current_architecture:
                    return np.zeros(len(self.search_space))
                    
                state = []
                for param_name in self.search_space:
                    if param_name in self.current_architecture:
                        # Normalize parameter value
                        param_value = self.current_architecture[param_name]
                        param_values = self.search_space[param_name]
                        normalized_value = list(param_values).index(param_value) / len(param_values)
                        state.append(normalized_value)
                    else:
                        state.append(0)
                        
                return np.array(state)
                
            def _is_done(self):
                """Check if episode is done"""
                return len(self.architecture_history) >= 10
                
            def get_environment_stats(self):
                """Get environment statistics"""
                if not self.performance_history:
                    return {}
                    
                return {
                    'total_architectures': len(self.architecture_history),
                    'best_performance': max(self.performance_history),
                    'average_performance': np.mean(self.performance_history),
                    'performance_improvement': self.performance_history[-1] - self.performance_history[0] if len(self.performance_history) > 1 else 0
                }
        
        # Test RL NAS environment
        search_space = {
            'layers': [2, 4, 6, 8],
            'hidden_sizes': [128, 256, 512],
            'activations': ['relu', 'gelu']
        }
        
        env = RLNASEnvironment(search_space)
        
        # Test environment reset
        initial_state = env.reset()
        self.assertEqual(len(initial_state), 3)
        
        # Test environment step
        for _ in range(5):
            action = np.random.randint(0, 10)
            next_state, reward, done, info = env.step(action)
            
            self.assertEqual(len(next_state), 3)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
            self.assertIn('performance', info)
            
        # Check environment stats
        stats = env.get_environment_stats()
        self.assertGreater(stats['total_architectures'], 0)
        self.assertGreater(stats['best_performance'], 0)
        self.assertGreaterEqual(stats['performance_improvement'], 0)

class TestDifferentiableArchitectureSearch(unittest.TestCase):
    """Test suite for differentiable architecture search"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_differentiable_nas(self):
        """Test differentiable architecture search"""
        class DifferentiableNAS:
            def __init__(self, architecture_candidates):
                self.architecture_candidates = architecture_candidates
                self.architecture_weights = self._initialize_architecture_weights()
                self.performance_history = []
                
            def _initialize_architecture_weights(self):
                """Initialize architecture weights"""
                weights = {}
                for candidate_name in self.architecture_candidates:
                    weights[candidate_name] = np.random.uniform(0, 1)
                return weights
                
            def forward(self, x):
                """Forward pass with architecture weights"""
                # Compute architecture probabilities
                probabilities = self._compute_architecture_probabilities()
                
                # Weighted combination of architectures
                output = np.zeros_like(x)
                for i, (candidate_name, prob) in enumerate(probabilities.items()):
                    # Simulate architecture forward pass
                    architecture_output = self._simulate_architecture_forward(x, candidate_name)
                    output += prob * architecture_output
                    
                return output
                
            def _compute_architecture_probabilities(self):
                """Compute architecture probabilities"""
                # Softmax over architecture weights
                weights = list(self.architecture_weights.values())
                exp_weights = np.exp(weights - np.max(weights))
                probabilities = exp_weights / np.sum(exp_weights)
                
                prob_dict = {}
                for i, candidate_name in enumerate(self.architecture_candidates):
                    prob_dict[candidate_name] = probabilities[i]
                    
                return prob_dict
                
            def _simulate_architecture_forward(self, x, architecture_name):
                """Simulate architecture forward pass"""
                # Simple simulation
                return x * np.random.uniform(0.5, 1.5)
                
            def backward(self, loss):
                """Backward pass for architecture weights"""
                # Compute gradients for architecture weights
                gradients = {}
                for candidate_name in self.architecture_candidates:
                    # Simulate gradient computation
                    gradients[candidate_name] = np.random.normal(0, 0.1)
                    
                return gradients
                
            def update_architecture_weights(self, gradients, learning_rate=0.01):
                """Update architecture weights"""
                for candidate_name, gradient in gradients.items():
                    self.architecture_weights[candidate_name] -= learning_rate * gradient
                    
            def get_best_architecture(self):
                """Get best architecture based on weights"""
                probabilities = self._compute_architecture_probabilities()
                best_architecture = max(probabilities, key=probabilities.get)
                return best_architecture, probabilities[best_architecture]
                
            def get_nas_stats(self):
                """Get NAS statistics"""
                probabilities = self._compute_architecture_probabilities()
                return {
                    'total_candidates': len(self.architecture_candidates),
                    'architecture_probabilities': probabilities,
                    'best_architecture': max(probabilities, key=probabilities.get),
                    'entropy': -np.sum([p * np.log(p + 1e-8) for p in probabilities.values()])
                }
        
        # Test differentiable NAS
        architecture_candidates = ['linear', 'mlp', 'transformer', 'cnn']
        nas = DifferentiableNAS(architecture_candidates)
        
        # Test forward pass
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        output = nas.forward(data)
        
        self.assertEqual(output.shape, data.shape)
        
        # Test backward pass
        loss = np.random.uniform(0, 1)
        gradients = nas.backward(loss)
        
        self.assertEqual(len(gradients), len(architecture_candidates))
        for candidate_name in architecture_candidates:
            self.assertIn(candidate_name, gradients)
            
        # Test architecture weight update
        nas.update_architecture_weights(gradients)
        
        # Test best architecture selection
        best_arch, best_prob = nas.get_best_architecture()
        self.assertIn(best_arch, architecture_candidates)
        self.assertGreater(best_prob, 0)
        
        # Check NAS stats
        stats = nas.get_nas_stats()
        self.assertEqual(stats['total_candidates'], 4)
        self.assertEqual(len(stats['architecture_probabilities']), 4)
        self.assertIn(stats['best_architecture'], architecture_candidates)
        self.assertGreater(stats['entropy'], 0)

if __name__ == '__main__':
    unittest.main()





