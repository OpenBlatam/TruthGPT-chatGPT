"""
Unit tests for optimization research and experimental techniques
Tests cutting-edge optimization algorithms, research methodologies, and experimental frameworks
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestExperimentalOptimization(unittest.TestCase):
    """Test suite for experimental optimization techniques"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_swarm_optimization(self):
        """Test particle swarm optimization"""
        class ParticleSwarmOptimizer:
            def __init__(self, n_particles=20, w=0.9, c1=2.0, c2=2.0):
                self.n_particles = n_particles
                self.w = w  # Inertia weight
                self.c1 = c1  # Cognitive parameter
                self.c2 = c2  # Social parameter
                self.particles = []
                self.global_best = None
                self.global_best_fitness = float('inf')
                self.optimization_history = []
                
            def initialize_particles(self, problem_dimensions):
                """Initialize particle swarm"""
                self.particles = []
                for _ in range(self.n_particles):
                    particle = {
                        'position': np.random.uniform(-5, 5, problem_dimensions),
                        'velocity': np.random.uniform(-1, 1, problem_dimensions),
                        'best_position': None,
                        'best_fitness': float('inf')
                    }
                    self.particles.append(particle)
                    
            def update_particles(self, fitness_function):
                """Update particle positions and velocities"""
                for particle in self.particles:
                    # Evaluate fitness
                    fitness = fitness_function(particle['position'])
                    
                    # Update personal best
                    if fitness < particle['best_fitness']:
                        particle['best_fitness'] = fitness
                        particle['best_position'] = particle['position'].copy()
                        
                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best = particle['position'].copy()
                        
                # Update velocities and positions
                for particle in self.particles:
                    # Update velocity
                    r1 = np.random.random(particle['position'].shape)
                    r2 = np.random.random(particle['position'].shape)
                    
                    cognitive = self.c1 * r1 * (particle['best_position'] - particle['position'])
                    social = self.c2 * r2 * (self.global_best - particle['position'])
                    
                    particle['velocity'] = (self.w * particle['velocity'] + 
                                         cognitive + social)
                    
                    # Update position
                    particle['position'] += particle['velocity']
                    
                # Record optimization step
                self.optimization_history.append({
                    'global_best_fitness': self.global_best_fitness,
                    'average_fitness': np.mean([p['best_fitness'] for p in self.particles]),
                    'diversity': self._calculate_diversity()
                })
                
            def _calculate_diversity(self):
                """Calculate swarm diversity"""
                positions = np.array([p['position'] for p in self.particles])
                center = np.mean(positions, axis=0)
                distances = [np.linalg.norm(pos - center) for pos in positions]
                return np.mean(distances)
                
            def optimize(self, fitness_function, problem_dimensions, max_iterations=100):
                """Run particle swarm optimization"""
                self.initialize_particles(problem_dimensions)
                
                for iteration in range(max_iterations):
                    self.update_particles(fitness_function)
                    
                return self.global_best, self.global_best_fitness
                
            def get_swarm_stats(self):
                """Get swarm optimization statistics"""
                if not self.optimization_history:
                    return {}
                    
                return {
                    'total_iterations': len(self.optimization_history),
                    'best_fitness': self.global_best_fitness,
                    'final_diversity': self.optimization_history[-1]['diversity'],
                    'convergence_rate': self._calculate_convergence_rate()
                }
                
            def _calculate_convergence_rate(self):
                """Calculate convergence rate"""
                if len(self.optimization_history) < 2:
                    return 0
                    
                initial_fitness = self.optimization_history[0]['global_best_fitness']
                final_fitness = self.optimization_history[-1]['global_best_fitness']
                
                if initial_fitness == 0:
                    return 0
                    
                return (initial_fitness - final_fitness) / initial_fitness
        
        # Test particle swarm optimization
        pso = ParticleSwarmOptimizer(n_particles=10)
        
        # Define test function
        def test_function(x):
            return np.sum(x**2) + np.random.normal(0, 0.1)
        
        # Test optimization
        best_solution, best_fitness = pso.optimize(test_function, problem_dimensions=5, max_iterations=20)
        
        # Verify results
        self.assertIsNotNone(best_solution)
        self.assertEqual(len(best_solution), 5)
        self.assertGreater(best_fitness, 0)
        
        # Check swarm stats
        stats = pso.get_swarm_stats()
        self.assertEqual(stats['total_iterations'], 20)
        self.assertGreater(stats['best_fitness'], 0)
        self.assertGreaterEqual(stats['convergence_rate'], 0)
        
    def test_genetic_algorithm(self):
        """Test genetic algorithm optimization"""
        class GeneticAlgorithm:
            def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.8):
                self.population_size = population_size
                self.mutation_rate = mutation_rate
                self.crossover_rate = crossover_rate
                self.population = []
                self.fitness_history = []
                self.best_individual = None
                self.best_fitness = float('inf')
                
            def initialize_population(self, chromosome_length):
                """Initialize population"""
                self.population = []
                for _ in range(self.population_size):
                    chromosome = np.random.uniform(-5, 5, chromosome_length)
                    self.population.append(chromosome)
                    
            def evaluate_fitness(self, chromosome, fitness_function):
                """Evaluate chromosome fitness"""
                return fitness_function(chromosome)
                
            def selection(self, fitness_scores):
                """Tournament selection"""
                selected = []
                for _ in range(self.population_size):
                    # Tournament selection
                    tournament_size = 3
                    tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
                    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                    winner_index = tournament_indices[np.argmax(tournament_fitness)]
                    selected.append(self.population[winner_index].copy())
                return selected
                
            def crossover(self, parent1, parent2):
                """Single-point crossover"""
                if np.random.random() < self.crossover_rate:
                    crossover_point = np.random.randint(1, len(parent1))
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                    return child1, child2
                else:
                    return parent1.copy(), parent2.copy()
                    
            def mutation(self, chromosome):
                """Gaussian mutation"""
                mutated = chromosome.copy()
                for i in range(len(chromosome)):
                    if np.random.random() < self.mutation_rate:
                        mutated[i] += np.random.normal(0, 0.1)
                return mutated
                
            def evolve_generation(self, fitness_function):
                """Evolve one generation"""
                # Evaluate fitness
                fitness_scores = []
                for chromosome in self.population:
                    fitness = self.evaluate_fitness(chromosome, fitness_function)
                    fitness_scores.append(fitness)
                    
                    # Update best individual
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_individual = chromosome.copy()
                        
                # Record generation stats
                self.fitness_history.append({
                    'generation': len(self.fitness_history),
                    'best_fitness': max(fitness_scores),
                    'average_fitness': np.mean(fitness_scores),
                    'fitness_std': np.std(fitness_scores)
                })
                
                # Selection
                selected = self.selection(fitness_scores)
                
                # Create new generation
                new_population = []
                for i in range(0, self.population_size, 2):
                    parent1 = selected[i]
                    parent2 = selected[i + 1] if i + 1 < len(selected) else selected[i]
                    
                    # Crossover
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    # Mutation
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)
                    
                    new_population.extend([child1, child2])
                    
                self.population = new_population[:self.population_size]
                return max(fitness_scores)
                
            def optimize(self, fitness_function, chromosome_length, generations=100):
                """Run genetic algorithm optimization"""
                self.initialize_population(chromosome_length)
                
                for generation in range(generations):
                    best_fitness = self.evolve_generation(fitness_function)
                    
                return self.best_individual, self.best_fitness
                
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
        
        # Test genetic algorithm
        ga = GeneticAlgorithm(population_size=10)
        
        # Define test function
        def test_function(x):
            return np.sum(x**2) + np.random.normal(0, 0.1)
        
        # Test optimization
        best_solution, best_fitness = ga.optimize(test_function, chromosome_length=5, generations=20)
        
        # Verify results
        self.assertIsNotNone(best_solution)
        self.assertEqual(len(best_solution), 5)
        self.assertGreater(best_fitness, 0)
        
        # Check evolution stats
        stats = ga.get_evolution_stats()
        self.assertEqual(stats['total_generations'], 20)
        self.assertGreater(stats['best_fitness'], 0)
        self.assertGreaterEqual(stats['fitness_improvement'], 0)
        
    def test_differential_evolution(self):
        """Test differential evolution optimization"""
        class DifferentialEvolution:
            def __init__(self, population_size=20, F=0.8, CR=0.9):
                self.population_size = population_size
                self.F = F  # Scaling factor
                self.CR = CR  # Crossover rate
                self.population = []
                self.fitness_history = []
                self.best_individual = None
                self.best_fitness = float('inf')
                
            def initialize_population(self, problem_dimensions, bounds):
                """Initialize population"""
                self.population = []
                for _ in range(self.population_size):
                    individual = np.random.uniform(bounds[0], bounds[1], problem_dimensions)
                    self.population.append(individual)
                    
            def mutation(self, target_index):
                """DE mutation"""
                # Select three random individuals
                candidates = list(range(self.population_size))
                candidates.remove(target_index)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Create mutant vector
                mutant = (self.population[a] + 
                         self.F * (self.population[b] - self.population[c]))
                return mutant
                
            def crossover(self, target, mutant):
                """DE crossover"""
                trial = target.copy()
                crossover_points = np.random.random(len(target)) < self.CR
                trial[crossover_points] = mutant[crossover_points]
                return trial
                
            def selection(self, target, trial, fitness_function):
                """DE selection"""
                target_fitness = fitness_function(target)
                trial_fitness = fitness_function(trial)
                
                if trial_fitness < target_fitness:
                    return trial, trial_fitness
                else:
                    return target, target_fitness
                    
            def evolve_generation(self, fitness_function):
                """Evolve one generation"""
                new_population = []
                fitness_scores = []
                
                for i in range(self.population_size):
                    target = self.population[i]
                    
                    # Mutation
                    mutant = self.mutation(i)
                    
                    # Crossover
                    trial = self.crossover(target, mutant)
                    
                    # Selection
                    selected, fitness = self.selection(target, trial, fitness_function)
                    new_population.append(selected)
                    fitness_scores.append(fitness)
                    
                    # Update best individual
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_individual = selected.copy()
                        
                self.population = new_population
                
                # Record generation stats
                self.fitness_history.append({
                    'generation': len(self.fitness_history),
                    'best_fitness': max(fitness_scores),
                    'average_fitness': np.mean(fitness_scores),
                    'fitness_std': np.std(fitness_scores)
                })
                
                return max(fitness_scores)
                
            def optimize(self, fitness_function, problem_dimensions, bounds, max_iterations=100):
                """Run differential evolution optimization"""
                self.initialize_population(problem_dimensions, bounds)
                
                for iteration in range(max_iterations):
                    best_fitness = self.evolve_generation(fitness_function)
                    
                return self.best_individual, self.best_fitness
                
            def get_evolution_stats(self):
                """Get evolution statistics"""
                if not self.fitness_history:
                    return {}
                    
                return {
                    'total_iterations': len(self.fitness_history),
                    'best_fitness': self.best_fitness,
                    'final_fitness': self.fitness_history[-1]['best_fitness'],
                    'average_fitness': np.mean([gen['average_fitness'] for gen in self.fitness_history]),
                    'fitness_improvement': self.fitness_history[-1]['best_fitness'] - self.fitness_history[0]['best_fitness']
                }
        
        # Test differential evolution
        de = DifferentialEvolution(population_size=10)
        
        # Define test function
        def test_function(x):
            return np.sum(x**2) + np.random.normal(0, 0.1)
        
        # Test optimization
        best_solution, best_fitness = de.optimize(test_function, problem_dimensions=5, bounds=(-5, 5), max_iterations=20)
        
        # Verify results
        self.assertIsNotNone(best_solution)
        self.assertEqual(len(best_solution), 5)
        self.assertGreater(best_fitness, 0)
        
        # Check evolution stats
        stats = de.get_evolution_stats()
        self.assertEqual(stats['total_iterations'], 20)
        self.assertGreater(stats['best_fitness'], 0)
        self.assertGreaterEqual(stats['fitness_improvement'], 0)

class TestResearchMethodologies(unittest.TestCase):
    """Test suite for research methodologies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        
    def test_ablation_study(self):
        """Test ablation study framework"""
        class AblationStudy:
            def __init__(self, base_config, ablation_configs):
                self.base_config = base_config
                self.ablation_configs = ablation_configs
                self.study_results = {}
                self.performance_metrics = {}
                
            def run_ablation_study(self, data, target):
                """Run complete ablation study"""
                # Run base configuration
                base_result = self._run_configuration(self.base_config, data, target)
                self.study_results['base'] = base_result
                
                # Run ablation configurations
                for config_name, config in self.ablation_configs.items():
                    result = self._run_configuration(config, data, target)
                    self.study_results[config_name] = result
                    
                # Analyze results
                self._analyze_results()
                
                return self.study_results
                
            def _run_configuration(self, config, data, target):
                """Run single configuration"""
                # Simulate configuration execution
                result = {
                    'config': config,
                    'performance': np.random.uniform(0, 1),
                    'convergence_time': np.random.uniform(10, 100),
                    'memory_usage': np.random.uniform(100, 1000),
                    'accuracy': np.random.uniform(0.8, 0.99)
                }
                return result
                
            def _analyze_results(self):
                """Analyze ablation study results"""
                base_performance = self.study_results['base']['performance']
                
                for config_name, result in self.study_results.items():
                    if config_name != 'base':
                        performance_change = result['performance'] - base_performance
                        self.performance_metrics[config_name] = {
                            'performance_change': performance_change,
                            'relative_change': performance_change / base_performance,
                            'impact': 'positive' if performance_change > 0 else 'negative'
                        }
                        
            def get_ablation_stats(self):
                """Get ablation study statistics"""
                return {
                    'total_configurations': len(self.study_results),
                    'base_performance': self.study_results['base']['performance'],
                    'performance_metrics': self.performance_metrics,
                    'best_configuration': max(self.study_results.items(), key=lambda x: x[1]['performance'])[0],
                    'worst_configuration': min(self.study_results.items(), key=lambda x: x[1]['performance'])[0]
                }
        
        # Test ablation study
        base_config = {'learning_rate': 0.001, 'batch_size': 32, 'dropout': 0.1}
        ablation_configs = {
            'no_dropout': {'learning_rate': 0.001, 'batch_size': 32, 'dropout': 0.0},
            'high_lr': {'learning_rate': 0.01, 'batch_size': 32, 'dropout': 0.1},
            'large_batch': {'learning_rate': 0.001, 'batch_size': 128, 'dropout': 0.1}
        }
        
        ablation_study = AblationStudy(base_config, ablation_configs)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test ablation study
        results = ablation_study.run_ablation_study(data, target)
        
        # Verify results
        self.assertEqual(len(results), 4)  # base + 3 ablations
        self.assertIn('base', results)
        self.assertIn('no_dropout', results)
        self.assertIn('high_lr', results)
        self.assertIn('large_batch', results)
        
        # Check ablation stats
        stats = ablation_study.get_ablation_stats()
        self.assertEqual(stats['total_configurations'], 4)
        self.assertGreater(stats['base_performance'], 0)
        self.assertEqual(len(stats['performance_metrics']), 3)
        self.assertIn('best_configuration', stats)
        self.assertIn('worst_configuration', stats)
        
    def test_hyperparameter_sensitivity_analysis(self):
        """Test hyperparameter sensitivity analysis"""
        class SensitivityAnalysis:
            def __init__(self, parameter_ranges, sensitivity_metrics):
                self.parameter_ranges = parameter_ranges
                self.sensitivity_metrics = sensitivity_metrics
                self.sensitivity_results = {}
                self.parameter_importance = {}
                
            def run_sensitivity_analysis(self, data, target):
                """Run sensitivity analysis"""
                for param_name, param_range in self.parameter_ranges.items():
                    sensitivity_result = self._analyze_parameter_sensitivity(
                        param_name, param_range, data, target
                    )
                    self.sensitivity_results[param_name] = sensitivity_result
                    
                # Calculate parameter importance
                self._calculate_parameter_importance()
                
                return self.sensitivity_results
                
            def _analyze_parameter_sensitivity(self, param_name, param_range, data, target):
                """Analyze sensitivity of single parameter"""
                # Sample parameter values
                param_values = np.linspace(param_range[0], param_range[1], 10)
                performance_values = []
                
                for param_value in param_values:
                    # Simulate performance evaluation
                    performance = np.random.uniform(0, 1)
                    performance_values.append(performance)
                    
                # Calculate sensitivity metrics
                sensitivity_metrics = {}
                for metric_name, metric_func in self.sensitivity_metrics.items():
                    sensitivity_metrics[metric_name] = metric_func(param_values, performance_values)
                    
                return {
                    'parameter_values': param_values,
                    'performance_values': performance_values,
                    'sensitivity_metrics': sensitivity_metrics
                }
                
            def _calculate_parameter_importance(self):
                """Calculate parameter importance"""
                for param_name, result in self.sensitivity_results.items():
                    # Calculate importance based on performance variance
                    performance_std = np.std(result['performance_values'])
                    self.parameter_importance[param_name] = {
                        'importance_score': performance_std,
                        'sensitivity_level': 'high' if performance_std > 0.1 else 'low'
                    }
                    
            def get_sensitivity_stats(self):
                """Get sensitivity analysis statistics"""
                return {
                    'total_parameters': len(self.parameter_ranges),
                    'sensitivity_results': self.sensitivity_results,
                    'parameter_importance': self.parameter_importance,
                    'most_sensitive': max(self.parameter_importance.items(), key=lambda x: x[1]['importance_score'])[0],
                    'least_sensitive': min(self.parameter_importance.items(), key=lambda x: x[1]['importance_score'])[0]
                }
        
        # Test sensitivity analysis
        parameter_ranges = {
            'learning_rate': (0.0001, 0.1),
            'batch_size': (16, 128),
            'dropout': (0.1, 0.5)
        }
        
        sensitivity_metrics = {
            'variance': lambda x, y: np.var(y),
            'range': lambda x, y: np.max(y) - np.min(y),
            'slope': lambda x, y: np.polyfit(x, y, 1)[0]
        }
        
        sensitivity_analysis = SensitivityAnalysis(parameter_ranges, sensitivity_metrics)
        data = self.test_data.create_mlp_data(batch_size=2, seq_len=64, d_model=256)
        target = torch.randn_like(data)
        
        # Test sensitivity analysis
        results = sensitivity_analysis.run_sensitivity_analysis(data, target)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn('learning_rate', results)
        self.assertIn('batch_size', results)
        self.assertIn('dropout', results)
        
        # Check sensitivity stats
        stats = sensitivity_analysis.get_sensitivity_stats()
        self.assertEqual(stats['total_parameters'], 3)
        self.assertEqual(len(stats['sensitivity_results']), 3)
        self.assertEqual(len(stats['parameter_importance']), 3)
        self.assertIn('most_sensitive', stats)
        self.assertIn('least_sensitive', stats)
        
    def test_optimization_benchmarking(self):
        """Test optimization benchmarking framework"""
        class OptimizationBenchmark:
            def __init__(self, algorithms, test_functions):
                self.algorithms = algorithms
                self.test_functions = test_functions
                self.benchmark_results = {}
                self.performance_rankings = {}
                
            def run_benchmark(self, n_runs=10):
                """Run optimization benchmark"""
                for algorithm_name, algorithm in self.algorithms.items():
                    algorithm_results = {}
                    
                    for function_name, test_function in self.test_functions.items():
                        function_results = []
                        
                        for run in range(n_runs):
                            # Run algorithm on test function
                            result = self._run_algorithm(algorithm, test_function)
                            function_results.append(result)
                            
                        algorithm_results[function_name] = function_results
                        
                    self.benchmark_results[algorithm_name] = algorithm_results
                    
                # Calculate performance rankings
                self._calculate_rankings()
                
                return self.benchmark_results
                
            def _run_algorithm(self, algorithm, test_function):
                """Run single algorithm on test function"""
                # Simulate algorithm execution
                result = {
                    'best_solution': np.random.uniform(-5, 5, 5),
                    'best_fitness': test_function(np.random.uniform(-5, 5, 5)),
                    'convergence_time': np.random.uniform(10, 100),
                    'function_evaluations': np.random.randint(100, 1000),
                    'success': np.random.uniform(0, 1) > 0.2
                }
                return result
                
            def _calculate_rankings(self):
                """Calculate performance rankings"""
                for function_name in self.test_functions:
                    function_rankings = {}
                    
                    for algorithm_name in self.algorithms:
                        results = self.benchmark_results[algorithm_name][function_name]
                        avg_fitness = np.mean([r['best_fitness'] for r in results])
                        success_rate = np.mean([r['success'] for r in results])
                        
                        function_rankings[algorithm_name] = {
                            'avg_fitness': avg_fitness,
                            'success_rate': success_rate,
                            'combined_score': avg_fitness * success_rate
                        }
                        
                    # Sort by combined score
                    sorted_rankings = sorted(function_rankings.items(), 
                                           key=lambda x: x[1]['combined_score'])
                    
                    self.performance_rankings[function_name] = sorted_rankings
                    
            def get_benchmark_stats(self):
                """Get benchmark statistics"""
                return {
                    'total_algorithms': len(self.algorithms),
                    'total_functions': len(self.test_functions),
                    'benchmark_results': self.benchmark_results,
                    'performance_rankings': self.performance_rankings,
                    'overall_winner': self._calculate_overall_winner()
                }
                
            def _calculate_overall_winner(self):
                """Calculate overall winner across all functions"""
                algorithm_scores = {}
                
                for function_name, rankings in self.performance_rankings.items():
                    for rank, (algorithm_name, scores) in enumerate(rankings):
                        if algorithm_name not in algorithm_scores:
                            algorithm_scores[algorithm_name] = 0
                        algorithm_scores[algorithm_name] += len(rankings) - rank
                        
                return max(algorithm_scores.items(), key=lambda x: x[1])[0]
        
        # Test optimization benchmark
        algorithms = {
            'genetic_algorithm': {'type': 'evolutionary'},
            'particle_swarm': {'type': 'swarm'},
            'differential_evolution': {'type': 'evolutionary'}
        }
        
        test_functions = {
            'sphere': lambda x: np.sum(x**2),
            'rosenbrock': lambda x: np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2),
            'rastrigin': lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
        }
        
        benchmark = OptimizationBenchmark(algorithms, test_functions)
        
        # Test benchmark
        results = benchmark.run_benchmark(n_runs=5)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn('genetic_algorithm', results)
        self.assertIn('particle_swarm', results)
        self.assertIn('differential_evolution', results)
        
        # Check benchmark stats
        stats = benchmark.get_benchmark_stats()
        self.assertEqual(stats['total_algorithms'], 3)
        self.assertEqual(stats['total_functions'], 3)
        self.assertEqual(len(stats['performance_rankings']), 3)
        self.assertIn('overall_winner', stats)

if __name__ == '__main__':
    unittest.main()





