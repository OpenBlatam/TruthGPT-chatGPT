"""
TruthGPT Bio-Inspired Computing Module
Advanced bio-inspired algorithms and evolutionary computing for TruthGPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import random
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import math

logger = logging.getLogger(__name__)

class BioAlgorithm(Enum):
    """Bio-inspired algorithms."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM_OPTIMIZATION = "pso"
    ANT_COLONY_OPTIMIZATION = "aco"
    ARTIFICIAL_BEE_COLONY = "abc"
    FIREFLY_ALGORITHM = "firefly"
    BAT_ALGORITHM = "bat"
    CUCKOO_SEARCH = "cuckoo"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    NEURAL_EVOLUTION = "neural_evolution"

class SelectionMethod(Enum):
    """Selection methods for evolutionary algorithms."""
    ROULETTE_WHEEL = "roulette_wheel"
    TOURNAMENT = "tournament"
    RANK_SELECTION = "rank_selection"
    ELITIST = "elitist"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"

class CrossoverMethod(Enum):
    """Crossover methods."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLEND = "blend"

class MutationMethod(Enum):
    """Mutation methods."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    NON_UNIFORM = "non_uniform"

@dataclass
class BioConfig:
    """Configuration for bio-inspired computing."""
    algorithm: BioAlgorithm = BioAlgorithm.GENETIC_ALGORITHM
    population_size: int = 100
    generations: int = 1000
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    elitism_count: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

class Individual:
    """
    Individual in evolutionary algorithm.
    Represents a candidate solution.
    """
    
    def __init__(self, genes: List[float], fitness: float = 0.0):
        """
        Initialize individual.
        
        Args:
            genes: Genetic material
            fitness: Fitness value
        """
        self.genes = genes
        self.fitness = fitness
        self.age = 0
        self.parents = []
        self.children = []
    
    def copy(self) -> 'Individual':
        """Create a copy of the individual."""
        return Individual(self.genes.copy(), self.fitness)
    
    def mutate(self, mutation_rate: float, mutation_method: MutationMethod) -> None:
        """
        Mutate individual.
        
        Args:
            mutation_rate: Mutation rate
            mutation_method: Mutation method
        """
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                if mutation_method == MutationMethod.GAUSSIAN:
                    self.genes[i] += random.gauss(0, 0.1)
                elif mutation_method == MutationMethod.UNIFORM:
                    self.genes[i] += random.uniform(-0.1, 0.1)
                elif mutation_method == MutationMethod.POLYNOMIAL:
                    self.genes[i] += random.gauss(0, 0.1) * (1 - self.age / 1000)
    
    def crossover(self, other: 'Individual', crossover_method: CrossoverMethod) -> Tuple['Individual', 'Individual']:
        """
        Perform crossover with another individual.
        
        Args:
            other: Other individual
            crossover_method: Crossover method
            
        Returns:
            Two offspring individuals
        """
        if len(self.genes) != len(other.genes):
            raise ValueError("Individuals must have same gene length")
        
        child1_genes = []
        child2_genes = []
        
        if crossover_method == CrossoverMethod.SINGLE_POINT:
            point = random.randint(1, len(self.genes) - 1)
            child1_genes = self.genes[:point] + other.genes[point:]
            child2_genes = other.genes[:point] + self.genes[point:]
        
        elif crossover_method == CrossoverMethod.TWO_POINT:
            point1 = random.randint(1, len(self.genes) - 2)
            point2 = random.randint(point1 + 1, len(self.genes) - 1)
            child1_genes = self.genes[:point1] + other.genes[point1:point2] + self.genes[point2:]
            child2_genes = other.genes[:point1] + self.genes[point1:point2] + other.genes[point2:]
        
        elif crossover_method == CrossoverMethod.UNIFORM:
            for i in range(len(self.genes)):
                if random.random() < 0.5:
                    child1_genes.append(self.genes[i])
                    child2_genes.append(other.genes[i])
                else:
                    child1_genes.append(other.genes[i])
                    child2_genes.append(self.genes[i])
        
        elif crossover_method == CrossoverMethod.ARITHMETIC:
            alpha = random.random()
            for i in range(len(self.genes)):
                child1_genes.append(alpha * self.genes[i] + (1 - alpha) * other.genes[i])
                child2_genes.append((1 - alpha) * self.genes[i] + alpha * other.genes[i])
        
        child1 = Individual(child1_genes)
        child2 = Individual(child2_genes)
        
        # Set parent relationships
        child1.parents = [self, other]
        child2.parents = [self, other]
        self.children.extend([child1, child2])
        other.children.extend([child1, child2])
        
        return child1, child2

class GeneticAlgorithm:
    """
    Genetic Algorithm implementation.
    Evolutionary optimization algorithm inspired by natural selection.
    """
    
    def __init__(self, config: BioConfig):
        """
        Initialize genetic algorithm.
        
        Args:
            config: Bio configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.GeneticAlgorithm")
        
        # Population
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        
        # Statistics
        self.ga_stats = {
            'generations': 0,
            'evaluations': 0,
            'best_fitness': float('-inf'),
            'average_fitness': 0.0,
            'diversity': 0.0,
            'convergence_generation': 0
        }
    
    def initialize_population(self, gene_length: int, bounds: Tuple[float, float] = (-1.0, 1.0)) -> None:
        """
        Initialize random population.
        
        Args:
            gene_length: Length of gene vector
            bounds: Bounds for gene values
        """
        self.population = []
        for _ in range(self.config.population_size):
            genes = [random.uniform(bounds[0], bounds[1]) for _ in range(gene_length)]
            individual = Individual(genes)
            self.population.append(individual)
        
        self.logger.info(f"Initialized population of {len(self.population)} individuals")
    
    def evaluate_population(self, fitness_function: Callable[[List[float]], float]) -> None:
        """
        Evaluate population fitness.
        
        Args:
            fitness_function: Fitness function
        """
        for individual in self.population:
            individual.fitness = fitness_function(individual.genes)
            self.ga_stats['evaluations'] += 1
        
        # Update statistics
        fitnesses = [ind.fitness for ind in self.population]
        self.ga_stats['best_fitness'] = max(fitnesses)
        self.ga_stats['average_fitness'] = sum(fitnesses) / len(fitnesses)
        self.ga_stats['diversity'] = np.std(fitnesses)
        
        # Update best individual
        best_idx = np.argmax(fitnesses)
        if self.best_individual is None or self.population[best_idx].fitness > self.best_individual.fitness:
            self.best_individual = self.population[best_idx].copy()
    
    def select_parents(self) -> List[Individual]:
        """
        Select parents for reproduction.
        
        Returns:
            Selected parents
        """
        parents = []
        
        if self.config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            parents = self._roulette_wheel_selection()
        elif self.config.selection_method == SelectionMethod.TOURNAMENT:
            parents = self._tournament_selection()
        elif self.config.selection_method == SelectionMethod.RANK_SELECTION:
            parents = self._rank_selection()
        elif self.config.selection_method == SelectionMethod.ELITIST:
            parents = self._elitist_selection()
        
        return parents
    
    def _roulette_wheel_selection(self) -> List[Individual]:
        """Roulette wheel selection."""
        fitnesses = [ind.fitness for ind in self.population]
        min_fitness = min(fitnesses)
        
        # Shift fitnesses to be positive
        shifted_fitnesses = [f - min_fitness + 1 for f in fitnesses]
        total_fitness = sum(shifted_fitnesses)
        
        parents = []
        for _ in range(self.config.population_size):
            r = random.uniform(0, total_fitness)
            cumulative = 0
            for i, fitness in enumerate(shifted_fitnesses):
                cumulative += fitness
                if cumulative >= r:
                    parents.append(self.population[i])
                    break
        
        return parents
    
    def _tournament_selection(self) -> List[Individual]:
        """Tournament selection."""
        parents = []
        for _ in range(self.config.population_size):
            tournament = random.sample(self.population, self.config.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _rank_selection(self) -> List[Individual]:
        """Rank selection."""
        sorted_population = sorted(self.population, key=lambda x: x.fitness)
        ranks = list(range(1, len(sorted_population) + 1))
        
        parents = []
        for _ in range(self.config.population_size):
            r = random.uniform(0, sum(ranks))
            cumulative = 0
            for i, rank in enumerate(ranks):
                cumulative += rank
                if cumulative >= r:
                    parents.append(sorted_population[i])
                    break
        
        return parents
    
    def _elitist_selection(self) -> List[Individual]:
        """Elitist selection."""
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents = sorted_population[:self.config.population_size]
        return parents
    
    def evolve_generation(self, fitness_function: Callable[[List[float]], float]) -> None:
        """
        Evolve one generation.
        
        Args:
            fitness_function: Fitness function
        """
        # Evaluate current population
        self.evaluate_population(fitness_function)
        
        # Select parents
        parents = self.select_parents()
        
        # Create new population
        new_population = []
        
        # Elitism
        if self.config.elitism_count > 0:
            elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.config.elitism_count]
            new_population.extend([ind.copy() for ind in elite])
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1, parent2 = random.sample(parents, 2)
            
            if random.random() < self.config.crossover_rate:
                child1, child2 = parent1.crossover(parent2, self.config.crossover_method)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutate offspring
            child1.mutate(self.config.mutation_rate, self.config.mutation_method)
            child2.mutate(self.config.mutation_rate, self.config.mutation_method)
            
            new_population.extend([child1, child2])
        
        # Replace population
        self.population = new_population[:self.config.population_size]
        self.generation += 1
        self.ga_stats['generations'] = self.generation
    
    def run_optimization(self, fitness_function: Callable[[List[float]], float], 
                        gene_length: int, bounds: Tuple[float, float] = (-1.0, 1.0)) -> Dict[str, Any]:
        """
        Run complete optimization.
        
        Args:
            fitness_function: Fitness function
            gene_length: Length of gene vector
            bounds: Bounds for gene values
            
        Returns:
            Optimization results
        """
        self.logger.info("Starting genetic algorithm optimization...")
        
        # Initialize population
        self.initialize_population(gene_length, bounds)
        
        # Run evolution
        for generation in range(self.config.generations):
            self.evolve_generation(fitness_function)
            
            # Check convergence
            if generation > 0:
                fitness_improvement = abs(self.ga_stats['best_fitness'] - self.ga_stats['average_fitness'])
                if fitness_improvement < self.config.convergence_threshold:
                    self.ga_stats['convergence_generation'] = generation
                    self.logger.info(f"Converged at generation {generation}")
                    break
            
            if generation % 100 == 0:
                self.logger.info(f"Generation {generation}: Best fitness = {self.ga_stats['best_fitness']:.6f}")
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.ga_stats['best_fitness'],
            'generations': self.ga_stats['generations'],
            'evaluations': self.ga_stats['evaluations'],
            'convergence_generation': self.ga_stats['convergence_generation'],
            'final_diversity': self.ga_stats['diversity']
        }

class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization implementation.
    Swarm intelligence algorithm inspired by bird flocking.
    """
    
    def __init__(self, config: BioConfig):
        """
        Initialize PSO.
        
        Args:
            config: Bio configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ParticleSwarmOptimization")
        
        # Swarm parameters
        self.swarm_size = config.population_size
        self.inertia_weight = 0.9
        self.cognitive_weight = 2.0
        self.social_weight = 2.0
        
        # Particles
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
        # Statistics
        self.pso_stats = {
            'iterations': 0,
            'evaluations': 0,
            'best_fitness': float('-inf'),
            'swarm_diversity': 0.0
        }
    
    def initialize_swarm(self, dimension: int, bounds: Tuple[float, float] = (-1.0, 1.0)) -> None:
        """
        Initialize particle swarm.
        
        Args:
            dimension: Problem dimension
            bounds: Bounds for particle positions
        """
        self.particles = []
        for _ in range(self.swarm_size):
            position = [random.uniform(bounds[0], bounds[1]) for _ in range(dimension)]
            velocity = [random.uniform(-0.1, 0.1) for _ in range(dimension)]
            particle = {
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': float('-inf'),
                'fitness': float('-inf')
            }
            self.particles.append(particle)
        
        self.logger.info(f"Initialized swarm of {len(self.particles)} particles")
    
    def evaluate_swarm(self, fitness_function: Callable[[List[float]], float]) -> None:
        """
        Evaluate swarm fitness.
        
        Args:
            fitness_function: Fitness function
        """
        for particle in self.particles:
            particle['fitness'] = fitness_function(particle['position'])
            self.pso_stats['evaluations'] += 1
            
            # Update personal best
            if particle['fitness'] > particle['best_fitness']:
                particle['best_fitness'] = particle['fitness']
                particle['best_position'] = particle['position'].copy()
            
            # Update global best
            if particle['fitness'] > self.global_best_fitness:
                self.global_best_fitness = particle['fitness']
                self.global_best_position = particle['position'].copy()
        
        # Update statistics
        fitnesses = [p['fitness'] for p in self.particles]
        self.pso_stats['best_fitness'] = max(fitnesses)
        self.pso_stats['swarm_diversity'] = np.std(fitnesses)
    
    def update_particles(self) -> None:
        """Update particle positions and velocities."""
        for particle in self.particles:
            # Update velocity
            for i in range(len(particle['position'])):
                r1, r2 = random.random(), random.random()
                
                cognitive_component = self.cognitive_weight * r1 * (particle['best_position'][i] - particle['position'][i])
                social_component = self.social_weight * r2 * (self.global_best_position[i] - particle['position'][i])
                
                particle['velocity'][i] = (self.inertia_weight * particle['velocity'][i] + 
                                         cognitive_component + social_component)
            
            # Update position
            for i in range(len(particle['position'])):
                particle['position'][i] += particle['velocity'][i]
    
    def run_optimization(self, fitness_function: Callable[[List[float]], float], 
                        dimension: int, bounds: Tuple[float, float] = (-1.0, 1.0)) -> Dict[str, Any]:
        """
        Run PSO optimization.
        
        Args:
            fitness_function: Fitness function
            dimension: Problem dimension
            bounds: Bounds for particle positions
            
        Returns:
            Optimization results
        """
        self.logger.info("Starting PSO optimization...")
        
        # Initialize swarm
        self.initialize_swarm(dimension, bounds)
        
        # Run optimization
        for iteration in range(self.config.generations):
            self.evaluate_swarm(fitness_function)
            self.update_particles()
            
            self.pso_stats['iterations'] = iteration + 1
            
            if iteration % 100 == 0:
                self.logger.info(f"Iteration {iteration}: Best fitness = {self.pso_stats['best_fitness']:.6f}")
        
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'iterations': self.pso_stats['iterations'],
            'evaluations': self.pso_stats['evaluations'],
            'final_diversity': self.pso_stats['swarm_diversity']
        }

class AntColonyOptimization:
    """
    Ant Colony Optimization implementation.
    Swarm intelligence algorithm inspired by ant behavior.
    """
    
    def __init__(self, config: BioConfig):
        """
        Initialize ACO.
        
        Args:
            config: Bio configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AntColonyOptimization")
        
        # ACO parameters
        self.num_ants = config.population_size
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.rho = 0.5    # Evaporation rate
        self.q0 = 0.9     # Exploitation probability
        
        # Pheromone matrix
        self.pheromone_matrix = None
        self.heuristic_matrix = None
        
        # Statistics
        self.aco_stats = {
            'iterations': 0,
            'best_solution': None,
            'best_fitness': float('-inf'),
            'average_fitness': 0.0
        }
    
    def initialize_matrices(self, problem_size: int) -> None:
        """
        Initialize pheromone and heuristic matrices.
        
        Args:
            problem_size: Size of the problem
        """
        self.pheromone_matrix = np.ones((problem_size, problem_size)) * 0.1
        self.heuristic_matrix = np.random.rand(problem_size, problem_size)
        
        # Set diagonal to 0
        np.fill_diagonal(self.pheromone_matrix, 0)
        np.fill_diagonal(self.heuristic_matrix, 0)
    
    def construct_solution(self, start_node: int) -> List[int]:
        """
        Construct solution using ant behavior.
        
        Args:
            start_node: Starting node
            
        Returns:
            Constructed solution
        """
        solution = [start_node]
        current_node = start_node
        unvisited = set(range(len(self.pheromone_matrix)))
        unvisited.remove(start_node)
        
        while unvisited:
            # Calculate probabilities
            probabilities = []
            for next_node in unvisited:
                pheromone = self.pheromone_matrix[current_node][next_node]
                heuristic = self.heuristic_matrix[current_node][next_node]
                probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append((next_node, probability))
            
            # Select next node
            if random.random() < self.q0:
                # Exploitation
                next_node = max(probabilities, key=lambda x: x[1])[0]
            else:
                # Exploration
                total_prob = sum(p[1] for p in probabilities)
                if total_prob > 0:
                    probabilities = [(node, prob / total_prob) for node, prob in probabilities]
                    r = random.random()
                    cumulative = 0
                    for node, prob in probabilities:
                        cumulative += prob
                        if cumulative >= r:
                            next_node = node
                            break
                else:
                    next_node = random.choice(list(unvisited))
            
            solution.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        
        return solution
    
    def update_pheromones(self, solutions: List[List[int]], fitnesses: List[float]) -> None:
        """
        Update pheromone matrix.
        
        Args:
            solutions: Ant solutions
            fitnesses: Solution fitnesses
        """
        # Evaporate pheromones
        self.pheromone_matrix *= (1 - self.rho)
        
        # Deposit pheromones
        for solution, fitness in zip(solutions, fitnesses):
            pheromone_deposit = fitness / max(fitnesses) if max(fitnesses) > 0 else 0.1
            
            for i in range(len(solution) - 1):
                current_node = solution[i]
                next_node = solution[i + 1]
                self.pheromone_matrix[current_node][next_node] += pheromone_deposit
    
    def run_optimization(self, fitness_function: Callable[[List[int]], float], 
                        problem_size: int) -> Dict[str, Any]:
        """
        Run ACO optimization.
        
        Args:
            fitness_function: Fitness function
            problem_size: Size of the problem
            
        Returns:
            Optimization results
        """
        self.logger.info("Starting ACO optimization...")
        
        # Initialize matrices
        self.initialize_matrices(problem_size)
        
        # Run optimization
        for iteration in range(self.config.generations):
            solutions = []
            fitnesses = []
            
            # Construct solutions
            for ant in range(self.num_ants):
                start_node = random.randint(0, problem_size - 1)
                solution = self.construct_solution(start_node)
                fitness = fitness_function(solution)
                
                solutions.append(solution)
                fitnesses.append(fitness)
            
            # Update pheromones
            self.update_pheromones(solutions, fitnesses)
            
            # Update statistics
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.aco_stats['best_fitness']:
                self.aco_stats['best_fitness'] = fitnesses[best_idx]
                self.aco_stats['best_solution'] = solutions[best_idx]
            
            self.aco_stats['iterations'] = iteration + 1
            self.aco_stats['average_fitness'] = np.mean(fitnesses)
            
            if iteration % 100 == 0:
                self.logger.info(f"Iteration {iteration}: Best fitness = {self.aco_stats['best_fitness']:.6f}")
        
        return {
            'best_solution': self.aco_stats['best_solution'],
            'best_fitness': self.aco_stats['best_fitness'],
            'iterations': self.aco_stats['iterations'],
            'average_fitness': self.aco_stats['average_fitness']
        }

class BioInspiredOptimizer:
    """
    Bio-inspired optimizer for TruthGPT.
    Coordinates various bio-inspired algorithms.
    """
    
    def __init__(self, config: BioConfig):
        """
        Initialize bio-inspired optimizer.
        
        Args:
            config: Bio configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BioInspiredOptimizer")
        
        # Algorithm instances
        self.genetic_algorithm = GeneticAlgorithm(config)
        self.particle_swarm = ParticleSwarmOptimization(config)
        self.ant_colony = AntColonyOptimization(config)
        
        # Statistics
        self.optimizer_stats = {
            'algorithms_used': 0,
            'total_evaluations': 0,
            'best_solution': None,
            'best_fitness': float('-inf')
        }
    
    def optimize(self, fitness_function: Callable, problem_dimension: int, 
                bounds: Tuple[float, float] = (-1.0, 1.0)) -> Dict[str, Any]:
        """
        Optimize using bio-inspired algorithms.
        
        Args:
            fitness_function: Fitness function
            problem_dimension: Problem dimension
            bounds: Bounds for optimization
            
        Returns:
            Optimization results
        """
        self.logger.info("Starting bio-inspired optimization...")
        
        results = {}
        
        if self.config.algorithm == BioAlgorithm.GENETIC_ALGORITHM:
            results = self.genetic_algorithm.run_optimization(fitness_function, problem_dimension, bounds)
        elif self.config.algorithm == BioAlgorithm.PARTICLE_SWARM_OPTIMIZATION:
            results = self.particle_swarm.run_optimization(fitness_function, problem_dimension, bounds)
        elif self.config.algorithm == BioAlgorithm.ANT_COLONY_OPTIMIZATION:
            # Convert continuous function to discrete for ACO
            def discrete_fitness(solution):
                # Convert discrete solution to continuous
                continuous_solution = [bounds[0] + (bounds[1] - bounds[0]) * (node / problem_dimension) for node in solution]
                return fitness_function(continuous_solution)
            
            results = self.ant_colony.run_optimization(discrete_fitness, problem_dimension)
        
        # Update statistics
        self.optimizer_stats['algorithms_used'] += 1
        self.optimizer_stats['total_evaluations'] += results.get('evaluations', 0)
        
        if results.get('best_fitness', 0) > self.optimizer_stats['best_fitness']:
            self.optimizer_stats['best_fitness'] = results['best_fitness']
            self.optimizer_stats['best_solution'] = results.get('best_individual', results.get('best_position', results.get('best_solution')))
        
        return results
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            **self.optimizer_stats,
            'algorithm': self.config.algorithm.value,
            'population_size': self.config.population_size,
            'generations': self.config.generations
        }

# Factory functions
def create_bio_inspired_optimizer(config: BioConfig) -> BioInspiredOptimizer:
    """Create bio-inspired optimizer instance."""
    return BioInspiredOptimizer(config)

def create_genetic_algorithm(config: BioConfig) -> GeneticAlgorithm:
    """Create genetic algorithm instance."""
    return GeneticAlgorithm(config)

def create_particle_swarm_optimization(config: BioConfig) -> ParticleSwarmOptimization:
    """Create particle swarm optimization instance."""
    return ParticleSwarmOptimization(config)

def create_ant_colony_optimization(config: BioConfig) -> AntColonyOptimization:
    """Create ant colony optimization instance."""
    return AntColonyOptimization(config)

# Example usage
if __name__ == "__main__":
    # Create bio configuration
    config = BioConfig(
        algorithm=BioAlgorithm.GENETIC_ALGORITHM,
        population_size=50,
        generations=500,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.UNIFORM,
        mutation_method=MutationMethod.GAUSSIAN
    )
    
    # Create bio-inspired optimizer
    optimizer = create_bio_inspired_optimizer(config)
    
    # Define fitness function (example: sphere function)
    def sphere_function(x):
        return -sum(xi**2 for xi in x)  # Negative because we maximize
    
    # Run optimization
    results = optimizer.optimize(sphere_function, dimension=10, bounds=(-5.0, 5.0))
    
    print(f"Optimization results: {results}")
    
    # Get optimizer statistics
    stats = optimizer.get_optimizer_stats()
    print(f"Optimizer stats: {stats}")

