
from .enums import (
    EvolutionaryAlgorithm,
    SelectionMethod,
    CrossoverMethod,
    MutationMethod
)
from .config import EvolutionaryConfig
from .individual import Individual
from .population import Population
from .optimizer import EvolutionaryOptimizer

import numpy as np

# Factory functions
def create_evolutionary_config(**kwargs) -> EvolutionaryConfig:
    """Create evolutionary configuration"""
    return EvolutionaryConfig(**kwargs)

def create_individual(genes: np.ndarray, fitness: float = None) -> Individual:
    """Create individual"""
    return Individual(genes, fitness)

def create_population(config: EvolutionaryConfig) -> Population:
    """Create population"""
    return Population(config)

def create_evolutionary_optimizer(config: EvolutionaryConfig) -> EvolutionaryOptimizer:
    """Create evolutionary optimizer"""
    return EvolutionaryOptimizer(config)

# Example usage
def example_evolutionary_computing():
    """Example of evolutionary computing system"""
    # Create configuration
    config = create_evolutionary_config(
        evolutionary_algorithm=EvolutionaryAlgorithm.GENETIC_ALGORITHM,
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.SINGLE_POINT,
        mutation_method=MutationMethod.GAUSSIAN,
        population_size=100,
        elite_size=10,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.1,
        mutation_strength=0.1,
        max_generations=1000,
        convergence_threshold=1e-6,
        stagnation_limit=50,
        enable_multi_objective=False,
        n_objectives=2,
        pareto_front_size=20,
        enable_adaptive_parameters=True,
        enable_diversity_maintenance=True,
        enable_local_search=False,
        enable_hybrid_evolution=False
    )
    
    # Create evolutionary optimizer
    evolutionary_optimizer = create_evolutionary_optimizer(config)
    
    # Define fitness function
    def fitness_function(genes):
        # Simulate fitness function (e.g., neural network hyperparameter optimization)
        return -np.sum(genes**2) + np.random.normal(0, 0.1)
    
    # Define bounds
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    gene_length = len(bounds)
    
    # Optimize
    optimization_results = evolutionary_optimizer.optimize(fitness_function, gene_length, bounds)
    
    # Generate report
    optimization_report = evolutionary_optimizer.generate_optimization_report(optimization_results)
    
    print(f"✅ Evolutionary Computing Example Complete!")
    print(f"🚀 Evolutionary Computing Statistics:")
    print(f"   Evolutionary Algorithm: {config.evolutionary_algorithm.value}")
    
    print(f"\n📋 Evolutionary Computing Report:")
    print(optimization_report)
    
    return evolutionary_optimizer

__all__ = [
    'SelectionMethod',
    'CrossoverMethod',
    'MutationMethod',
    'EvolutionaryAlgorithm',
    'EvolutionaryConfig',
    'Individual',
    'Population',
    'EvolutionaryOptimizer',
    'create_evolutionary_config',
    'create_individual',
    'create_population',
    'create_evolutionary_optimizer',
    'example_evolutionary_computing'
]

