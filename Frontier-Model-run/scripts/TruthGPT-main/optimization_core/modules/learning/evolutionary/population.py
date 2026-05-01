"""
Evolutionary Population
=======================

Population management and genetic operators for evolutionary algorithms.
"""
import logging
import random
from typing import Callable, List, Optional, Tuple

import numpy as np

from optimization_core.modules.learning.evolutionary.config import EvolutionaryConfig
from optimization_core.modules.learning.evolutionary.enums import (
    CrossoverMethod,
    SelectionMethod,
)
from optimization_core.modules.learning.evolutionary.individual import Individual

logger = logging.getLogger(__name__)


class Population:
    """Manager for a collection of individuals in an evolutionary algorithm.

    Handles initialization, fitness evaluation, selection, crossover, and mutation
    orchestration for a population of individuals.

    Attributes:
        config: Configuration parameters for the population and algorithms.
        individuals: List of currently active individuals.
        generation: Current generation counter.
        best_fitness_history: Record of the best fitness in each generation.
        average_fitness_history: Record of the mean fitness in each generation.
        diversity_history: Record of population diversity over time.
    """

    def __init__(self, config: EvolutionaryConfig) -> None:
        """Initialize the population with configuration.

        Args:
            config: The evolutionary configuration to use.
        """
        self.config: EvolutionaryConfig = config
        self.individuals: List[Individual] = []
        self.generation: int = 0
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        # logger.info("✅ Population initialized")

    def initialize(self, gene_length: int, bounds: Optional[List[Tuple[float, float]]] = None) -> None:
        """Initialize population with random individuals.

        Args:
            gene_length: Number of genes (parameters) per individual.
            bounds: Optional list of (min, max) limits for each gene.
        """
        logger.info(f"🏗️ Initializing population with {self.config.population_size} individuals")

        self.individuals = []

        for _ in range(self.config.population_size):
            if bounds:
                genes = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(gene_length)])
            else:
                genes = np.random.randn(gene_length)

            individual = Individual(genes)
            self.individuals.append(individual)

        logger.info("✅ Population initialized")

    def evaluate_fitness(self, fitness_function: Callable[[np.ndarray], float]) -> None:
        """Evaluate fitness for all individuals and update histories.

        Args:
            fitness_function: Callback that takes genes and returns a fitness score.
        """
        # logger.info("📊 Evaluating fitness for all individuals")

        for individual in self.individuals:
            if individual.fitness is None:
                individual.fitness = fitness_function(individual.genes)

        # Sort by fitness (descending: higher is better)
        self.individuals.sort(key=lambda x: x.fitness if x.fitness is not None else -float("inf"), reverse=True)

        # Store fitness history
        best_fitness = self.individuals[0].fitness or 0.0
        # Filter out None fitness for average calculation
        valid_fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        average_fitness = float(np.mean(valid_fitnesses)) if valid_fitnesses else 0.0

        self.best_fitness_history.append(best_fitness)
        self.average_fitness_history.append(average_fitness)

        # logger.info(f"   Best fitness: {best_fitness:.4f}, Average fitness: {average_fitness:.4f}")

    def select_parents(self) -> List[Individual]:
        """Select individuals for reproduction based on the configured method.

        Returns:
            A list of selected individuals (copies) to be used as parents.
        """
        # logger.debug(f"👥 Selecting parents using {self.config.selection_method.value}")

        if self.config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection()
        elif self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.config.selection_method == SelectionMethod.RANK:
            return self._rank_selection()
        elif self.config.selection_method == SelectionMethod.ELITIST:
            return self._elitist_selection()
        elif self.config.selection_method == SelectionMethod.STOCHASTIC_UNIVERSAL:
            return self._stochastic_universal_selection()
        elif self.config.selection_method == SelectionMethod.TRUNCATION:
            return self._truncation_selection()
        else:
            return self._tournament_selection()

    def _roulette_wheel_selection(self) -> List[Individual]:
        """Fitness proportionate selection.

        Returns:
            Selected parent individuals.
        """
        fitness_values = [ind.fitness or 0.0 for ind in self.individuals]
        min_fitness = min(fitness_values)

        # Shift fitness values to be positive
        shifted_fitness = [f - min_fitness + 1e-8 for f in fitness_values]

        # Select with replacement
        selected = random.choices(self.individuals, weights=shifted_fitness, k=self.config.population_size)
        return [ind.copy() for ind in selected]

    def _tournament_selection(self) -> List[Individual]:
        """Select best from random sub-groups.

        Returns:
            Selected parent individuals.
        """
        parents = []
        for _ in range(self.config.population_size):
            tournament = random.sample(self.individuals, self.config.tournament_size)
            # Handle potential None fitness by treating as -inf
            winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else -float("inf"))
            parents.append(winner.copy())
        return parents

    def _rank_selection(self) -> List[Individual]:
        """Selection based on rank rather than absolute fitness.

        Returns:
            Selected parent individuals.
        """
        # Individuals are already sorted by evaluate_fitness (descending)
        # We assign rank weights: Best gets N, Worst gets 1
        n = len(self.individuals)
        ranks = list(range(n, 0, -1))  # [N, N-1, ..., 1]

        selected = random.choices(self.individuals, weights=ranks, k=self.config.population_size)
        return [ind.copy() for ind in selected]

    def _elitist_selection(self) -> List[Individual]:
        """Preserve top individuals and fill rest via tournament.

        Returns:
            Selected parent individuals including elites.
        """
        elite = self.individuals[: self.config.elite_size]
        # We need (Pop - Elite) more individuals
        needed = self.config.population_size - self.config.elite_size

        # Use tournament for the rest
        remaining_parents = []
        for _ in range(needed):
            tournament = random.sample(self.individuals, self.config.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else -float("inf"))
            remaining_parents.append(winner.copy())

        parents = [ind.copy() for ind in elite] + remaining_parents
        return parents

    def _stochastic_universal_selection(self) -> List[Individual]:
        """Stochastic universal selection (SUS) for minimal bias.

        Returns:
            Selected parent individuals.
        """
        fitness_values = [ind.fitness or 0.0 for ind in self.individuals]
        min_fitness = min(fitness_values)
        shifted_fitness = [f - min_fitness + 1e-8 for f in fitness_values]
        total_fitness = sum(shifted_fitness)
        interval = total_fitness / self.config.population_size

        parents = []
        start = random.uniform(0, interval)
        for i in range(self.config.population_size):
            r = start + i * interval
            cumulative = 0
            for individual, fitness in zip(self.individuals, shifted_fitness):
                cumulative += fitness
                if cumulative >= r:
                    parents.append(individual.copy())
                    break
        return parents

    def _truncation_selection(self) -> List[Individual]:
        """Select top half of the population.

        Returns:
            Selected parent individuals.
        """
        # Ensure we have enough individuals
        cutoff = self.config.population_size // 2
        top_individuals = self.individuals[:cutoff]

        parents = []
        # Duplicate to fill population
        while len(parents) < self.config.population_size:
            for ind in top_individuals:
                if len(parents) < self.config.population_size:
                    parents.append(ind.copy())
                else:
                    break
        return parents

    def crossover(
        self, parents: List[Individual], bounds: Optional[List[Tuple[float, float]]] = None
    ) -> List[Individual]:
        """Combine parent pairs to create offspring.

        Args:
            parents: List of individuals selected for reproduction.
            bounds: Optional gene constraints.

        Returns:
            List of new offspring individuals.
        """
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                if random.random() < self.config.crossover_rate:
                    if self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
                        child1, child2 = self._single_point_crossover(parent1, parent2)
                    elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
                        child1, child2 = self._two_point_crossover(parent1, parent2)
                    elif self.config.crossover_method == CrossoverMethod.UNIFORM:
                        child1, child2 = self._uniform_crossover(parent1, parent2)
                    elif self.config.crossover_method == CrossoverMethod.ARITHMETIC:
                        child1, child2 = self._arithmetic_crossover(parent1, parent2)
                    elif self.config.crossover_method == CrossoverMethod.BLEND:
                        child1, child2 = self._blend_crossover(parent1, parent2)
                    elif self.config.crossover_method == CrossoverMethod.SIMULATED_BINARY:
                        child1, child2 = self._simulated_binary_crossover(parent1, parent2)
                    else:
                        child1, child2 = self._single_point_crossover(parent1, parent2)

                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1.copy(), parent2.copy()])
            else:
                offspring.append(parents[i].copy())

        return offspring

    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single point crossover logic."""
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        child1_genes = np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]])
        child2_genes = np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]])
        return Individual(child1_genes), Individual(child2_genes)

    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two point crossover logic."""
        if len(parent1.genes) < 3:
            return self._single_point_crossover(parent1, parent2)

        point1 = random.randint(1, len(parent1.genes) - 2)
        point2 = random.randint(point1 + 1, len(parent1.genes) - 1)
        child1_genes = np.concatenate(
            [parent1.genes[:point1], parent2.genes[point1:point2], parent1.genes[point2:]]
        )
        child2_genes = np.concatenate(
            [parent2.genes[:point1], parent1.genes[point1:point2], parent2.genes[point2:]]
        )
        return Individual(child1_genes), Individual(child2_genes)

    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover logic."""
        mask = np.random.random(len(parent1.genes)) < 0.5
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)
        return Individual(child1_genes), Individual(child2_genes)

    def _arithmetic_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Linear combination crossover."""
        alpha = random.random()
        child1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        child2_genes = (1 - alpha) * parent1.genes + alpha * parent2.genes
        return Individual(child1_genes), Individual(child2_genes)

    def _blend_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Blend crossover (BLX-alpha)."""
        alpha = 0.5
        child1_genes = np.zeros_like(parent1.genes)
        child2_genes = np.zeros_like(parent2.genes)
        for i in range(len(parent1.genes)):
            d = abs(parent1.genes[i] - parent2.genes[i])
            low = min(parent1.genes[i], parent2.genes[i]) - alpha * d
            high = max(parent1.genes[i], parent2.genes[i]) + alpha * d
            child1_genes[i] = random.uniform(low, high)
            child2_genes[i] = random.uniform(low, high)
        return Individual(child1_genes), Individual(child2_genes)

    def _simulated_binary_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Simulated binary crossover (SBX)."""
        eta = 20
        child1_genes = np.zeros_like(parent1.genes)
        child2_genes = np.zeros_like(parent2.genes)

        for i in range(len(parent1.genes)):
            if random.random() < 0.5 and abs(parent1.genes[i] - parent2.genes[i]) > 1e-14:
                y1, y2 = min(parent1.genes[i], parent2.genes[i]), max(parent1.genes[i], parent2.genes[i])
                beta = 1.0 + (2.0 * (y1 - 0) / (y2 - y1))
                alpha = 2.0 - beta ** (-eta - 1)

                u = random.random()
                if u <= (1.0 / alpha):
                    beta_q = (alpha * u) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - alpha * u)) ** (1.0 / (eta + 1))

                c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                child1_genes[i] = c1
                child2_genes[i] = c2
            else:
                child1_genes[i] = parent1.genes[i]
                child2_genes[i] = parent2.genes[i]
        return Individual(child1_genes), Individual(child2_genes)

    def mutate_offspring(self, offspring: List[Individual], bounds: Optional[List[Tuple[float, float]]] = None) -> None:
        """Apply mutation to the next generation candidates.

        Args:
            offspring: List of individuals produced by crossover.
            bounds: Optional gene constraints.
        """
        for individual in offspring:
            individual.mutate(
                self.config.mutation_method,
                self.config.mutation_rate,
                self.config.mutation_strength,
                bounds,
            )

    def replace_population(self, offspring: List[Individual]) -> None:
        """Update current population with new offspring.

        Args:
            offspring: New individuals to form the next generation.
        """
        if self.config.elite_size > 0:
            elite = self.individuals[: self.config.elite_size]
            self.individuals = elite + offspring[: self.config.population_size - self.config.elite_size]
        else:
            self.individuals = offspring[: self.config.population_size]

        self.generation += 1
        for individual in self.individuals:
            individual.age += 1

    def calculate_diversity(self) -> float:
        """Calculate and record the population's genetic diversity.

        Returns:
            Mean Euclidean distance between individuals.
        """
        if len(self.individuals) < 2:
            return 0.0

        # Optimize diversity calculation (sample if population is large)
        if len(self.individuals) > 100:
            sample = random.sample(self.individuals, 100)
        else:
            sample = self.individuals

        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                distance = np.linalg.norm(sample[i].genes - sample[j].genes)
                distances.append(float(distance))

        diversity = float(np.mean(distances)) if distances else 0.0
        self.diversity_history.append(diversity)
        return diversity

    def check_convergence(self) -> bool:
        """Check if population fitness has stabilized.

        Returns:
            True if improvement is below threshold or diversity is nearly zero.
        """
        if len(self.best_fitness_history) < 10:
            return False

        recent_improvement = abs(self.best_fitness_history[-1] - self.best_fitness_history[-10])
        if recent_improvement < self.config.convergence_threshold:
            return True

        if len(self.diversity_history) > 0 and self.diversity_history[-1] < 1e-6:
            return True

        return False

    def check_stagnation(self) -> bool:
        """Check if population has stalled for multiple generations.

        Returns:
            True if best fitness hasn't improved beyond a limit.
        """
        if len(self.best_fitness_history) < self.config.stagnation_limit:
            return False

        # Check if max fitness in window hasn't improved over the start of window
        window = self.best_fitness_history[-self.config.stagnation_limit :]
        if max(window) <= window[0] + 1e-9:
            return True
        return False

