"""
Evolutionary NAS
================

Evolutionary algorithm for Neural Architecture Search.
"""
import torch.nn as nn
import random
import numpy as np
import logging
from typing import Dict, Any, Callable
from .config import NASConfig
from .gene import ArchitectureGene
from .architecture import NeuralArchitecture

logger = logging.getLogger(__name__)

class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search"""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.population = []
        self.generation = 0
        self.best_architecture = None
        self.search_history = []
        
        logger.info("✅ Evolutionary NAS initialized")
    
    def initialize_population(self):
        """Initialize random population"""
        self.population = []
        
        for _ in range(self.config.population_size):
            # Random number of layers
            num_layers = random.randint(self.config.min_layers, self.config.max_layers)
            
            # Generate random genes
            genes = []
            for _ in range(num_layers):
                layer_type = random.choice(self.config.layer_types)
                params = self._generate_random_params(layer_type)
                gene = ArchitectureGene(layer_type, params)
                genes.append(gene)
            
            architecture = NeuralArchitecture(genes)
            self.population.append(architecture)
        
        logger.info(f"✅ Population initialized with {len(self.population)} architectures")
    
    def evaluate_population(self, evaluation_func: Callable[[nn.Module], float]):
        """Evaluate population fitness"""
        for architecture in self.population:
            try:
                # Convert to model
                model = architecture.to_model()
                
                # Evaluate
                fitness = evaluation_func(model)
                architecture.fitness = fitness
                
                # Update best
                if self.best_architecture is None or fitness > self.best_architecture.fitness:
                    self.best_architecture = architecture
                
            except Exception as e:
                logger.warning(f"Architecture evaluation failed: {e}")
                architecture.fitness = 0.0
    
    def evolve_generation(self):
        """Evolve one generation"""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep top performers
        elite_size = max(1, self.config.population_size // 10)
        elite = self.population[:elite_size]
        
        # Generate new population
        new_population = elite.copy()
        
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            child1.mutate(self.config)
            child2.mutate(self.config)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        self.population = new_population[:self.config.population_size]
        self.generation += 1
        
        # Record generation stats
        avg_fitness = np.mean([arch.fitness for arch in self.population])
        max_fitness = max([arch.fitness for arch in self.population])
        
        self.search_history.append({
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'best_fitness': self.best_architecture.fitness if self.best_architecture else 0.0
        })
        
        logger.info(f"Generation {self.generation}: avg_fitness={avg_fitness:.4f}, max_fitness={max_fitness:.4f}")
    
    def _tournament_selection(self, tournament_size: int = 3) -> NeuralArchitecture:
        """Tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _generate_random_params(self, layer_type: str) -> Dict[str, Any]:
        """Generate random parameters for layer type"""
        if layer_type == 'Linear':
            return {
                'in_features': random.randint(32, 512),
                'out_features': random.randint(32, 512)
            }
        elif layer_type == 'Conv2d':
            return {
                'in_channels': random.choice([1, 3, 16, 32]),
                'out_channels': random.choice([16, 32, 64, 128]),
                'kernel_size': random.choice([3, 5, 7]),
                'stride': random.choice([1, 2]),
                'padding': random.choice([0, 1, 2])
            }
        elif layer_type == 'LSTM':
            return {
                'input_size': random.randint(32, 256),
                'hidden_size': random.randint(32, 256),
                'num_layers': random.randint(1, 3)
            }
        elif layer_type == 'MultiheadAttention':
            return {
                'embed_dim': random.choice([64, 128, 256, 512]),
                'num_heads': random.choice([4, 8, 16]),
                'dropout': random.uniform(0.0, 0.3)
            }
        else:
            return {}
    
    def search(self, evaluation_func: Callable[[nn.Module], float]) -> NeuralArchitecture:
        """Perform architecture search"""
        logger.info("🚀 Starting Neural Architecture Search...")
        
        # Initialize population
        self.initialize_population()
        
        # Search loop
        for generation in range(self.config.generations):
            # Evaluate population
            self.evaluate_population(evaluation_func)
            
            # Check early stopping
            if self.best_architecture and self.best_architecture.fitness >= self.config.performance_threshold:
                logger.info(f"✅ Performance threshold reached at generation {generation}")
                break
            
            # Evolve
            self.evolve_generation()
            
            # Check convergence
            if len(self.search_history) >= self.config.early_stopping_patience:
                recent_fitness = [h['max_fitness'] for h in self.search_history[-self.config.early_stopping_patience:]]
                if max(recent_fitness) - min(recent_fitness) < 0.001:
                    logger.info(f"✅ Convergence detected at generation {generation}")
                    break
        
        logger.info(f"✅ NAS completed. Best fitness: {self.best_architecture.fitness:.4f}")
        return self.best_architecture
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        if not self.search_history:
            return {}
        
        return {
            'total_generations': self.generation,
            'best_fitness': self.best_architecture.fitness if self.best_architecture else 0.0,
            'final_avg_fitness': self.search_history[-1]['avg_fitness'] if self.search_history else 0.0,
            'fitness_improvement': self.search_history[-1]['max_fitness'] - self.search_history[0]['max_fitness'] if len(self.search_history) > 1 else 0.0,
            'search_history': self.search_history,
            'best_architecture_complexity': self.best_architecture.complexity if self.best_architecture else 0.0
        }

