"""
Evolutionary Optimizer
======================

Hyperparameter optimization using evolutionary algorithms.
"""
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Callable
from .config import HpoConfig, HpoAlgorithm

logger = logging.getLogger(__name__)

class EvolutionaryOptimizer:
    """Evolutionary algorithm implementation"""
    
    def __init__(self, config: HpoConfig):
        self.config = config
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.training_history = []
        logger.info("✅ Evolutionary Optimizer initialized")
    
    def create_individual(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Create individual (parameter set)"""
        individual = {}
        for param_name, param_range in search_space.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    individual[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    individual[param_name] = np.random.uniform(param_range[0], param_range[1])
            elif isinstance(param_range, list):
                individual[param_name] = np.random.choice(param_range)
            else:
                individual[param_name] = param_range
        
        return individual
    
    def initialize_population(self, search_space: Dict[str, Any]):
        """Initialize population"""
        self.population = []
        self.fitness_scores = []
        
        for _ in range(self.config.population_size):
            individual = self.create_individual(search_space)
            self.population.append(individual)
            self.fitness_scores.append(-np.inf)
    
    def evaluate_population(self, objective_function: Callable):
        """Evaluate population fitness"""
        for i, individual in enumerate(self.population):
            if self.fitness_scores[i] == -np.inf:  # Not evaluated yet
                fitness = objective_function(individual)
                self.fitness_scores[i] = fitness
                
                # Update best individual
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
    
    def selection(self) -> List[Dict[str, Any]]:
        """Selection operator"""
        # Tournament selection
        tournament_size = 3
        selected = []
        
        for _ in range(self.config.population_size):
            tournament_indices = np.random.choice(
                len(self.population), tournament_size, replace=False
            )
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx].copy())
        
        return selected
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operator"""
        if np.random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Uniform crossover
        for param_name in parent1.keys():
            if np.random.random() < 0.5:
                child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def mutation(self, individual: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operator"""
        mutated = individual.copy()
        
        for param_name, param_range in search_space.items():
            if np.random.random() < self.config.mutation_rate:
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        mutated[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                    else:
                        mutated[param_name] = np.random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    mutated[param_name] = np.random.choice(param_range)
        
        return mutated
    
    def optimize(self, objective_function: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters using evolutionary algorithm"""
        logger.info("🧬 Optimizing hyperparameters using evolutionary algorithm")
        
        # Initialize population
        self.initialize_population(search_space)
        
        # Evolution loop
        for generation in range(self.config.n_generations):
            # Evaluate population
            self.evaluate_population(objective_function)
            
            # Selection
            selected = self.selection()
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[i]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1, search_space)
                child2 = self.mutation(child2, search_space)
                
                new_population.extend([child1, child2])
            
            # Update population
            self.population = new_population[:self.config.population_size]
            self.fitness_scores = [-np.inf] * len(self.population)
            
            # Store generation history
            self.training_history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean([f for f in self.fitness_scores if f != -np.inf]),
                'best_individual': self.best_individual
            })
            
            if generation % 5 == 0:
                logger.info(f"   Generation {generation}: Best fitness = {self.best_fitness:.4f}")
        
        optimization_result = {
            'algorithm': HpoAlgorithm.EVOLUTIONARY_ALGORITHM.value,
            'n_generations': self.config.n_generations,
            'population_size': self.config.population_size,
            'best_params': self.best_individual,
            'best_score': self.best_fitness,
            'training_history': self.training_history,
            'status': 'success'
        }
        
        return optimization_result

