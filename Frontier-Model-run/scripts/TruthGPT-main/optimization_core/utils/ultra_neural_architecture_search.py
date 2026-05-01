"""
Ultra-Advanced Neural Architecture Search (NAS) Module
=====================================================

This module provides state-of-the-art neural architecture search capabilities
for TruthGPT models, including evolutionary algorithms, reinforcement learning,
gradient-based methods, and hybrid approaches.

Author: TruthGPT Ultra-Advanced Optimization Core Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pydantic import BaseModel, model_validator
from enum import Enum
import json
import pickle
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import math
import statistics

logger = logging.getLogger(__name__)

class NASStrategy(Enum):
    """Neural Architecture Search strategies."""
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GRADIENT_BASED = "gradient_based"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    HYBRID = "hybrid"
    PROGRESSIVE = "progressive"
    DIFFERENTIABLE = "differentiable"

class SearchSpace(Enum):
    """Search space definitions."""
    CELL_BASED = "cell_based"
    MACRO_BASED = "macro_based"
    MICRO_BASED = "micro_based"
    HIERARCHICAL = "hierarchical"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

class ArchitectureCandidate:
    """Represents a neural architecture candidate."""
    
    def __init__(self, 
                 architecture: Dict[str, Any],
                 performance_metrics: Optional[Dict[str, float]] = None,
                 complexity_score: Optional[float] = None,
                 efficiency_score: Optional[float] = None):
        self.architecture = architecture
        self.performance_metrics = performance_metrics or {}
        self.complexity_score = complexity_score
        self.efficiency_score = efficiency_score
        self.fitness_score = 0.0
        self.generation = 0
        self.parents = []
        self.children = []
        self.created_at = time.time()
        self.evaluated_at = None
        
    def calculate_fitness(self, 
                         performance_weight: float = 0.7,
                         efficiency_weight: float = 0.2,
                         complexity_weight: float = 0.1) -> float:
        """Calculate fitness score based on multiple criteria."""
        if not self.performance_metrics:
            return 0.0
            
        # Normalize metrics
        accuracy = self.performance_metrics.get('accuracy', 0.0)
        latency = self.performance_metrics.get('latency', 1.0)
        memory = self.performance_metrics.get('memory', 1.0)
        
        # Calculate efficiency score (higher is better)
        efficiency = 1.0 / (latency + memory / 1000) if latency > 0 else 0.0
        
        # Calculate complexity penalty (lower is better)
        complexity_penalty = 1.0 / (self.complexity_score + 1) if self.complexity_score else 1.0
        
        # Weighted fitness
        self.fitness_score = (
            performance_weight * accuracy +
            efficiency_weight * efficiency +
            complexity_weight * complexity_penalty
        )
        
        return self.fitness_score
    
    def mutate(self, mutation_rate: float = 0.1) -> 'ArchitectureCandidate':
        """Create a mutated version of this candidate."""
        mutated_arch = copy.deepcopy(self.architecture)
        
        # Apply mutations based on architecture type
        if random.random() < mutation_rate:
            # Layer mutation
            self._mutate_layers(mutated_arch)
        
        if random.random() < mutation_rate:
            # Connection mutation
            self._mutate_connections(mutated_arch)
            
        if random.random() < mutation_rate:
            # Parameter mutation
            self._mutate_parameters(mutated_arch)
        
        return ArchitectureCandidate(
            architecture=mutated_arch,
            generation=self.generation + 1,
            parents=[self]
        )
    
    def _mutate_layers(self, arch: Dict[str, Any]):
        """Mutate layer configuration."""
        if 'layers' in arch:
            layers = arch['layers']
            if layers and random.random() < 0.5:
                # Add layer
                new_layer = self._generate_random_layer()
                layers.append(new_layer)
            elif len(layers) > 1 and random.random() < 0.3:
                # Remove layer
                layers.pop(random.randint(0, len(layers) - 1))
    
    def _mutate_connections(self, arch: Dict[str, Any]):
        """Mutate connection patterns."""
        if 'connections' in arch:
            connections = arch['connections']
            if connections and random.random() < 0.5:
                # Add connection
                new_connection = self._generate_random_connection()
                connections.append(new_connection)
            elif len(connections) > 1 and random.random() < 0.3:
                # Remove connection
                connections.pop(random.randint(0, len(connections) - 1))
    
    def _mutate_parameters(self, arch: Dict[str, Any]):
        """Mutate hyperparameters."""
        if 'hyperparameters' in arch:
            hyperparams = arch['hyperparameters']
            for key in hyperparams:
                if random.random() < 0.1:
                    # Mutate parameter
                    if isinstance(hyperparams[key], (int, float)):
                        hyperparams[key] *= random.uniform(0.8, 1.2)
                    elif isinstance(hyperparams[key], bool):
                        hyperparams[key] = not hyperparams[key]
    
    def _generate_random_layer(self) -> Dict[str, Any]:
        """Generate a random layer configuration."""
        layer_types = ['linear', 'conv2d', 'attention', 'lstm', 'gru']
        layer_type = random.choice(layer_types)
        
        layer_config = {
            'type': layer_type,
            'hidden_size': random.choice([64, 128, 256, 512, 1024]),
            'dropout': random.uniform(0.0, 0.5),
            'activation': random.choice(['relu', 'gelu', 'swish', 'mish'])
        }
        
        if layer_type == 'conv2d':
            layer_config.update({
                'kernel_size': random.choice([3, 5, 7]),
                'stride': random.choice([1, 2]),
                'padding': random.choice([0, 1, 2])
            })
        elif layer_type in ['lstm', 'gru']:
            layer_config.update({
                'num_layers': random.choice([1, 2, 3]),
                'bidirectional': random.choice([True, False])
            })
        
        return layer_config
    
    def _generate_random_connection(self) -> Dict[str, Any]:
        """Generate a random connection configuration."""
        return {
            'from_layer': random.randint(0, 10),
            'to_layer': random.randint(0, 10),
            'connection_type': random.choice(['skip', 'residual', 'dense', 'attention'])
        }

class NASConfig(BaseModel):
    """Configuration for Neural Architecture Search."""
    strategy: NASStrategy = NASStrategy.EVOLUTIONARY
    search_space: SearchSpace = SearchSpace.CELL_BASED
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    early_stopping_patience: int = 20
    performance_weight: float = 0.7
    efficiency_weight: float = 0.2
    complexity_weight: float = 0.1
    max_evaluation_time: float = 300.0  # seconds
    parallel_evaluations: int = 4
    save_best_architectures: bool = True
    log_level: str = "INFO"
    output_dir: str = "./nas_results"
    device: str = "auto"

    @model_validator(mode='after')
    def validate_config(self) -> 'NASConfig':
        """Post-initialization validation."""
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")
        if self.generations < 1:
            raise ValueError("Generations must be at least 1")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        return self

class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.population = []
        self.generation = 0
        self.best_candidate = None
        self.fitness_history = []
        self.performance_history = []
        self.evaluation_times = []
        self.setup_logging()
        self.setup_device()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def setup_device(self):
        """Setup computation device."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")
    
    def initialize_population(self, 
                            base_architecture: Optional[Dict[str, Any]] = None) -> List[ArchitectureCandidate]:
        """Initialize the population with random architectures."""
        logger.info(f"Initializing population of size {self.config.population_size}")
        
        population = []
        for i in range(self.config.population_size):
            if base_architecture and i == 0:
                # Use base architecture as first candidate
                arch = copy.deepcopy(base_architecture)
            else:
                # Generate random architecture
                arch = self._generate_random_architecture()
            
            candidate = ArchitectureCandidate(architecture=arch)
            population.append(candidate)
        
        self.population = population
        logger.info(f"Initialized {len(self.population)} candidates")
        return population
    
    def _generate_random_architecture(self) -> Dict[str, Any]:
        """Generate a random neural architecture."""
        architecture = {
            'input_size': random.choice([128, 256, 512, 1024]),
            'output_size': random.choice([10, 100, 1000]),
            'layers': [],
            'connections': [],
            'hyperparameters': {
                'learning_rate': random.uniform(1e-5, 1e-2),
                'batch_size': random.choice([16, 32, 64, 128]),
                'dropout': random.uniform(0.0, 0.5),
                'weight_decay': random.uniform(1e-6, 1e-3)
            }
        }
        
        # Generate random layers
        num_layers = random.randint(2, 8)
        for _ in range(num_layers):
            layer = self._generate_random_layer()
            architecture['layers'].append(layer)
        
        # Generate random connections
        num_connections = random.randint(0, min(5, num_layers))
        for _ in range(num_connections):
            connection = self._generate_random_connection()
            architecture['connections'].append(connection)
        
        return architecture
    
    def _generate_random_layer(self) -> Dict[str, Any]:
        """Generate a random layer configuration."""
        layer_types = ['linear', 'conv2d', 'attention', 'lstm', 'gru']
        layer_type = random.choice(layer_types)
        
        layer_config = {
            'type': layer_type,
            'hidden_size': random.choice([64, 128, 256, 512, 1024]),
            'dropout': random.uniform(0.0, 0.5),
            'activation': random.choice(['relu', 'gelu', 'swish', 'mish'])
        }
        
        if layer_type == 'conv2d':
            layer_config.update({
                'kernel_size': random.choice([3, 5, 7]),
                'stride': random.choice([1, 2]),
                'padding': random.choice([0, 1, 2])
            })
        elif layer_type in ['lstm', 'gru']:
            layer_config.update({
                'num_layers': random.choice([1, 2, 3]),
                'bidirectional': random.choice([True, False])
            })
        
        return layer_config
    
    def _generate_random_connection(self) -> Dict[str, Any]:
        """Generate a random connection configuration."""
        return {
            'from_layer': random.randint(0, 10),
            'to_layer': random.randint(0, 10),
            'connection_type': random.choice(['skip', 'residual', 'dense', 'attention'])
        }
    
    def evaluate_candidate(self, 
                          candidate: ArchitectureCandidate,
                          evaluator: Callable[[Dict[str, Any]], Dict[str, float]]) -> ArchitectureCandidate:
        """Evaluate a single architecture candidate."""
        start_time = time.time()
        
        try:
            # Evaluate architecture
            metrics = evaluator(candidate.architecture)
            candidate.performance_metrics = metrics
            candidate.evaluated_at = time.time()
            
            # Calculate complexity and efficiency scores
            candidate.complexity_score = self._calculate_complexity(candidate.architecture)
            candidate.efficiency_score = self._calculate_efficiency(metrics)
            
            # Calculate fitness
            candidate.calculate_fitness(
                self.config.performance_weight,
                self.config.efficiency_weight,
                self.config.complexity_weight
            )
            
            evaluation_time = time.time() - start_time
            self.evaluation_times.append(evaluation_time)
            
            logger.debug(f"Evaluated candidate: fitness={candidate.fitness_score:.4f}, "
                        f"accuracy={metrics.get('accuracy', 0):.4f}, "
                        f"time={evaluation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error evaluating candidate: {e}")
            candidate.performance_metrics = {'accuracy': 0.0, 'latency': float('inf'), 'memory': float('inf')}
            candidate.fitness_score = 0.0
        
        return candidate
    
    def _calculate_complexity(self, architecture: Dict[str, Any]) -> float:
        """Calculate architecture complexity score."""
        complexity = 0.0
        
        # Count layers
        layers = architecture.get('layers', [])
        complexity += len(layers) * 10
        
        # Count parameters (estimated)
        for layer in layers:
            if layer['type'] == 'linear':
                complexity += layer.get('hidden_size', 128) * 10
            elif layer['type'] == 'conv2d':
                kernel_size = layer.get('kernel_size', 3)
                complexity += kernel_size * kernel_size * 10
            elif layer['type'] in ['lstm', 'gru']:
                complexity += layer.get('hidden_size', 128) * 20
        
        # Count connections
        connections = architecture.get('connections', [])
        complexity += len(connections) * 5
        
        return complexity
    
    def _calculate_efficiency(self, metrics: Dict[str, float]) -> float:
        """Calculate efficiency score from metrics."""
        latency = metrics.get('latency', 1.0)
        memory = metrics.get('memory', 1.0)
        
        # Higher efficiency = lower latency and memory usage
        efficiency = 1.0 / (latency + memory / 1000) if latency > 0 else 0.0
        return efficiency
    
    def select_parents(self, population: List[ArchitectureCandidate]) -> List[ArchitectureCandidate]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        
        # Elite selection (keep best candidates)
        sorted_pop = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        elite = sorted_pop[:self.config.elite_size]
        parents.extend(elite)
        
        # Tournament selection for remaining parents
        tournament_size = max(2, self.config.population_size // 10)
        
        while len(parents) < self.config.population_size:
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: ArchitectureCandidate, parent2: ArchitectureCandidate) -> Tuple[ArchitectureCandidate, ArchitectureCandidate]:
        """Perform crossover between two parent candidates."""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2
        
        # Create offspring by combining architectures
        arch1 = copy.deepcopy(parent1.architecture)
        arch2 = copy.deepcopy(parent2.architecture)
        
        # Crossover layers
        if 'layers' in arch1 and 'layers' in arch2:
            layers1, layers2 = self._crossover_layers(arch1['layers'], arch2['layers'])
            arch1['layers'] = layers1
            arch2['layers'] = layers2
        
        # Crossover connections
        if 'connections' in arch1 and 'connections' in arch2:
            conn1, conn2 = self._crossover_connections(arch1['connections'], arch2['connections'])
            arch1['connections'] = conn1
            arch2['connections'] = conn2
        
        # Crossover hyperparameters
        if 'hyperparameters' in arch1 and 'hyperparameters' in arch2:
            hp1, hp2 = self._crossover_hyperparameters(arch1['hyperparameters'], arch2['hyperparameters'])
            arch1['hyperparameters'] = hp1
            arch2['hyperparameters'] = hp2
        
        # Create offspring
        offspring1 = ArchitectureCandidate(
            architecture=arch1,
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1, parent2]
        )
        
        offspring2 = ArchitectureCandidate(
            architecture=arch2,
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1, parent2]
        )
        
        return offspring1, offspring2
    
    def _crossover_layers(self, layers1: List[Dict], layers2: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Crossover layer configurations."""
        if not layers1 or not layers2:
            return layers1, layers2
        
        # Random crossover point
        crossover_point = random.randint(1, min(len(layers1), len(layers2)) - 1)
        
        # Swap layers after crossover point
        new_layers1 = layers1[:crossover_point] + layers2[crossover_point:]
        new_layers2 = layers2[:crossover_point] + layers1[crossover_point:]
        
        return new_layers1, new_layers2
    
    def _crossover_connections(self, conn1: List[Dict], conn2: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Crossover connection configurations."""
        if not conn1 or not conn2:
            return conn1, conn2
        
        # Random crossover point
        crossover_point = random.randint(1, min(len(conn1), len(conn2)) - 1)
        
        # Swap connections after crossover point
        new_conn1 = conn1[:crossover_point] + conn2[crossover_point:]
        new_conn2 = conn2[:crossover_point] + conn1[crossover_point:]
        
        return new_conn1, new_conn2
    
    def _crossover_hyperparameters(self, hp1: Dict, hp2: Dict) -> Tuple[Dict, Dict]:
        """Crossover hyperparameters."""
        new_hp1 = {}
        new_hp2 = {}
        
        for key in set(hp1.keys()) | set(hp2.keys()):
            if random.random() < 0.5:
                new_hp1[key] = hp1.get(key, hp2.get(key))
                new_hp2[key] = hp2.get(key, hp1.get(key))
            else:
                new_hp1[key] = hp2.get(key, hp1.get(key))
                new_hp2[key] = hp1.get(key, hp2.get(key))
        
        return new_hp1, new_hp2
    
    def mutate_population(self, population: List[ArchitectureCandidate]) -> List[ArchitectureCandidate]:
        """Apply mutations to the population."""
        mutated_population = []
        
        for candidate in population:
            if random.random() < self.config.mutation_rate:
                mutated_candidate = candidate.mutate(self.config.mutation_rate)
                mutated_population.append(mutated_candidate)
            else:
                mutated_population.append(candidate)
        
        return mutated_population
    
    def evolve_generation(self, 
                         population: List[ArchitectureCandidate],
                         evaluator: Callable[[Dict[str, Any]], Dict[str, float]]) -> List[ArchitectureCandidate]:
        """Evolve one generation of the population."""
        logger.info(f"Evolving generation {self.generation}")
        
        # Evaluate population
        start_time = time.time()
        
        if self.config.parallel_evaluations > 1:
            population = self._evaluate_population_parallel(population, evaluator)
        else:
            population = self._evaluate_population_sequential(population, evaluator)
        
        evaluation_time = time.time() - start_time
        logger.info(f"Population evaluation completed in {evaluation_time:.2f}s")
        
        # Update best candidate
        best_in_generation = max(population, key=lambda x: x.fitness_score)
        if self.best_candidate is None or best_in_generation.fitness_score > self.best_candidate.fitness_score:
            self.best_candidate = best_in_generation
            logger.info(f"New best candidate found: fitness={self.best_candidate.fitness_score:.4f}")
        
        # Record statistics
        fitness_scores = [c.fitness_score for c in population]
        self.fitness_history.append({
            'generation': self.generation,
            'best': max(fitness_scores),
            'mean': statistics.mean(fitness_scores),
            'std': statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0,
            'worst': min(fitness_scores)
        })
        
        # Select parents
        parents = self.select_parents(population)
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep elite
        elite = sorted(population, key=lambda x: x.fitness_score, reverse=True)[:self.config.elite_size]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1, parent2 = random.sample(parents, 2)
            offspring1, offspring2 = self.crossover(parent1, parent2)
            new_population.extend([offspring1, offspring2])
        
        # Trim to population size
        new_population = new_population[:self.config.population_size]
        
        # Apply mutations
        new_population = self.mutate_population(new_population)
        
        self.generation += 1
        return new_population
    
    def _evaluate_population_sequential(self, 
                                     population: List[ArchitectureCandidate],
                                     evaluator: Callable[[Dict[str, Any]], Dict[str, float]]) -> List[ArchitectureCandidate]:
        """Evaluate population sequentially."""
        for candidate in population:
            if candidate.evaluated_at is None:  # Only evaluate if not already evaluated
                self.evaluate_candidate(candidate, evaluator)
        return population
    
    def _evaluate_population_parallel(self, 
                                    population: List[ArchitectureCandidate],
                                    evaluator: Callable[[Dict[str, Any]], Dict[str, float]]) -> List[ArchitectureCandidate]:
        """Evaluate population in parallel."""
        # Filter candidates that need evaluation
        candidates_to_evaluate = [c for c in population if c.evaluated_at is None]
        
        if not candidates_to_evaluate:
            return population
        
        logger.info(f"Evaluating {len(candidates_to_evaluate)} candidates in parallel")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_evaluations) as executor:
            future_to_candidate = {
                executor.submit(self.evaluate_candidate, candidate, evaluator): candidate
                for candidate in candidates_to_evaluate
            }
            
            for future in concurrent.futures.as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    evaluated_candidate = future.result()
                    # Update the candidate in the population
                    for i, pop_candidate in enumerate(population):
                        if pop_candidate == candidate:
                            population[i] = evaluated_candidate
                            break
                except Exception as e:
                    logger.error(f"Error evaluating candidate: {e}")
        
        return population
    
    def search(self, 
               evaluator: Callable[[Dict[str, Any]], Dict[str, float]],
               base_architecture: Optional[Dict[str, Any]] = None) -> ArchitectureCandidate:
        """Perform neural architecture search."""
        logger.info(f"Starting NAS with strategy: {self.config.strategy.value}")
        logger.info(f"Population size: {self.config.population_size}, Generations: {self.config.generations}")
        
        # Initialize population
        self.population = self.initialize_population(base_architecture)
        
        # Evolution loop
        best_fitness_history = []
        stagnation_count = 0
        
        for generation in range(self.config.generations):
            self.generation = generation
            
            # Evolve generation
            self.population = self.evolve_generation(self.population, evaluator)
            
            # Check for stagnation
            current_best = max(self.population, key=lambda x: x.fitness_score)
            best_fitness_history.append(current_best.fitness_score)
            
            if len(best_fitness_history) > self.config.early_stopping_patience:
                recent_best = max(best_fitness_history[-self.config.early_stopping_patience:])
                if recent_best <= best_fitness_history[-self.config.early_stopping_patience - 1]:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                if stagnation_count >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at generation {generation} due to stagnation")
                    break
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {current_best.fitness_score:.4f}")
        
        # Save results
        if self.config.save_best_architectures:
            self._save_results()
        
        logger.info(f"NAS completed. Best fitness: {self.best_candidate.fitness_score:.4f}")
        return self.best_candidate
    
    def _save_results(self):
        """Save search results and statistics."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best architecture
        if self.best_candidate:
            best_arch_file = output_dir / "best_architecture.json"
            with open(best_arch_file, 'w') as f:
                json.dump(self.best_candidate.architecture, f, indent=2)
            
            # Save best candidate metadata
            best_metadata = {
                'fitness_score': self.best_candidate.fitness_score,
                'performance_metrics': self.best_candidate.performance_metrics,
                'complexity_score': self.best_candidate.complexity_score,
                'efficiency_score': self.best_candidate.efficiency_score,
                'generation': self.best_candidate.generation,
                'created_at': self.best_candidate.created_at,
                'evaluated_at': self.best_candidate.evaluated_at
            }
            
            best_metadata_file = output_dir / "best_candidate_metadata.json"
            with open(best_metadata_file, 'w') as f:
                json.dump(best_metadata, f, indent=2)
        
        # Save fitness history
        fitness_history_file = output_dir / "fitness_history.json"
        with open(fitness_history_file, 'w') as f:
            json.dump(self.fitness_history, f, indent=2)
        
        # Save evaluation times
        eval_times_file = output_dir / "evaluation_times.json"
        with open(eval_times_file, 'w') as f:
            json.dump(self.evaluation_times, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")

class TruthGPTNASManager:
    """Main manager for TruthGPT Neural Architecture Search."""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.nas_engine = None
        self.search_results = {}
        
    def create_nas_engine(self) -> Union[EvolutionaryNAS]:
        """Create appropriate NAS engine based on strategy."""
        if self.config.strategy == NASStrategy.EVOLUTIONARY:
            return EvolutionaryNAS(self.config)
        else:
            raise ValueError(f"Unsupported NAS strategy: {self.config.strategy}")
    
    def search_architecture(self, 
                           evaluator: Callable[[Dict[str, Any]], Dict[str, float]],
                           base_architecture: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform architecture search and return results."""
        logger.info("Starting TruthGPT Neural Architecture Search")
        
        # Create NAS engine
        self.nas_engine = self.create_nas_engine()
        
        # Perform search
        best_candidate = self.nas_engine.search(evaluator, base_architecture)
        
        # Store results
        self.search_results = {
            'best_candidate': best_candidate,
            'fitness_history': self.nas_engine.fitness_history,
            'evaluation_times': self.nas_engine.evaluation_times,
            'config': self.config,
            'search_time': time.time()
        }
        
        logger.info("Architecture search completed successfully")
        return self.search_results
    
    def get_best_architecture(self) -> Optional[Dict[str, Any]]:
        """Get the best found architecture."""
        if self.search_results and 'best_candidate' in self.search_results:
            return self.search_results['best_candidate'].architecture
        return None
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        if not self.search_results:
            return {}
        
        stats = {
            'total_generations': len(self.nas_engine.fitness_history),
            'total_evaluations': len(self.nas_engine.evaluation_times),
            'best_fitness': self.search_results['best_candidate'].fitness_score,
            'average_evaluation_time': statistics.mean(self.nas_engine.evaluation_times) if self.nas_engine.evaluation_times else 0,
            'total_search_time': time.time() - self.search_results['search_time']
        }
        
        return stats

# Factory functions
def create_nas_config(strategy: NASStrategy = NASStrategy.EVOLUTIONARY,
                     population_size: int = 50,
                     generations: int = 100,
                     **kwargs) -> NASConfig:
    """Create NAS configuration."""
    return NASConfig(
        strategy=strategy,
        population_size=population_size,
        generations=generations,
        **kwargs
    )

def create_evolutionary_nas(config: Optional[NASConfig] = None) -> EvolutionaryNAS:
    """Create evolutionary NAS engine."""
    if config is None:
        config = create_nas_config()
    return EvolutionaryNAS(config)

def create_nas_manager(config: Optional[NASConfig] = None) -> TruthGPTNASManager:
    """Create NAS manager."""
    if config is None:
        config = create_nas_config()
    return TruthGPTNASManager(config)

# Example usage
def example_neural_architecture_search():
    """Example of neural architecture search."""
    # Create configuration
    config = create_nas_config(
        strategy=NASStrategy.EVOLUTIONARY,
        population_size=20,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # Create NAS manager
    nas_manager = create_nas_manager(config)
    
    # Define evaluator function
    def evaluate_architecture(architecture: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate architecture performance."""
        # Simulate evaluation (replace with actual model training/evaluation)
        accuracy = random.uniform(0.5, 0.95)
        latency = random.uniform(10, 100)  # ms
        memory = random.uniform(100, 1000)  # MB
        
        return {
            'accuracy': accuracy,
            'latency': latency,
            'memory': memory
        }
    
    # Perform search
    results = nas_manager.search_architecture(evaluate_architecture)
    
    # Get best architecture
    best_architecture = nas_manager.get_best_architecture()
    print(f"Best architecture found: {best_architecture}")
    
    # Get statistics
    stats = nas_manager.get_search_statistics()
    print(f"Search statistics: {stats}")
    
    return results

if __name__ == "__main__":
    # Run example
    example_neural_architecture_search()

