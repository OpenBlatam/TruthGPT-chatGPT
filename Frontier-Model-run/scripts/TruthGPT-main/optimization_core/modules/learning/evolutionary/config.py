"""
Evolutionary Configuration
==========================

Configuration class for evolutionary computing systems, parameters, and algorithms.
"""
from dataclasses import dataclass
from typing import Optional

from optimization_core.modules.learning.evolutionary.enums import (
    CrossoverMethod,
    EvolutionaryAlgorithm,
    MutationMethod,
    SelectionMethod,
)


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary computing system.

    Attributes:
        evolutionary_algorithm: The core algorithm to use (e.g., GA, ES).
        selection_method: Method for selecting parents from the population.
        crossover_method: Method for recombining parent genes into offspring.
        mutation_method: Method for introducing random variations in genes.
        population_size: Total number of individuals in the population.
        elite_size: Number of top individuals to preserve across generations.
        tournament_size: Number of participants in tournament selection.
        crossover_rate: Probability of performing crossover (0.0 to 1.0).
        mutation_rate: Probability of performing mutation (0.0 to 1.0).
        mutation_strength: Intensity of mutations (e.g., standard deviation for Gaussian).
        max_generations: Maximum number of generations to evolve.
        convergence_threshold: Fitness improvement threshold for early stopping.
        stagnation_limit: Number of generations with no improvement before stopping.
        enable_multi_objective: Whether to use multi-objective optimization (Pareto).
        n_objectives: Number of objectives to optimize (if multi-objective enabled).
        pareto_front_size: Maximum size of the preserved Pareto front.
        enable_adaptive_parameters: Whether to dynamically adjust parameters during evolution.
        enable_diversity_maintenance: Whether to explicitly encourage population diversity.
        enable_local_search: Whether to perform local search refinement on individuals.
        enable_hybrid_evolution: Whether to combine with other optimization techniques.
    """

    # Basic settings
    evolutionary_algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm.GENETIC_ALGORITHM
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN

    # Population settings
    population_size: int = 100
    elite_size: int = 10
    tournament_size: int = 3

    # Genetic operators
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1

    # Evolution settings
    max_generations: int = 1000
    convergence_threshold: float = 1e-6
    stagnation_limit: int = 50

    # Multi-objective settings
    enable_multi_objective: bool = False
    n_objectives: int = 2
    pareto_front_size: int = 20

    # Advanced features
    enable_adaptive_parameters: bool = True
    enable_diversity_maintenance: bool = True
    enable_local_search: bool = False
    enable_hybrid_evolution: bool = False

    def __post_init__(self) -> None:
        """Validate evolutionary configuration parameters.

        Raises:
            ValueError: If any configuration parameter is out of valid range.
        """
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.elite_size < 0:
            raise ValueError("Elite size must be non-negative")
        if self.tournament_size <= 0:
            raise ValueError("Tournament size must be positive")
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1")
        if self.mutation_strength <= 0:
            raise ValueError("Mutation strength must be positive")
        if self.max_generations <= 0:
            raise ValueError("Maximum generations must be positive")
        if self.convergence_threshold <= 0:
            raise ValueError("Convergence threshold must be positive")
        if self.stagnation_limit <= 0:
            raise ValueError("Stagnation limit must be positive")
        if self.n_objectives <= 0:
            raise ValueError("Number of objectives must be positive")
        if self.pareto_front_size <= 0:
            raise ValueError("Pareto front size must be positive")

