"""
Evolutionary Individual
=======================

Individual representation for evolutionary algorithms, managing genes and mutations.
"""
import logging
import random
from typing import List, Optional, Tuple

import numpy as np

from optimization_core.modules.learning.evolutionary.enums import MutationMethod

logger = logging.getLogger(__name__)


class Individual:
    """Individual representation in an evolutionary algorithm.

    An individual contains a set of genes (parameters) and a fitness score representing
    its quality relative to the optimization target.

    Attributes:
        genes: Numpy array containing the genetic material (parameters).
        fitness: Scalar fitness score (higher is typically better).
        objectives: List of objective values for multi-objective optimization.
        age: Number of generations this individual has survived.
    """

    def __init__(self, genes: np.ndarray, fitness: Optional[float] = None) -> None:
        """Initialize an individual with genes and optional fitness.

        Args:
            genes: The genetic material for this individual.
            fitness: Initial fitness score, if known.
        """
        self.genes: np.ndarray = genes.copy()
        self.fitness: Optional[float] = fitness
        self.objectives: List[float] = []
        self.age: int = 0
        # logger.debug("✅ Individual created") # Reduced spam

    def copy(self) -> "Individual":
        """Create a deep copy of the individual.

        Returns:
            A new Individual instance with same genes, fitness, and state.
        """
        new_individual = Individual(self.genes, self.fitness)
        new_individual.objectives = self.objectives.copy()
        new_individual.age = self.age
        return new_individual

    def mutate(
        self,
        mutation_method: MutationMethod,
        mutation_rate: float,
        mutation_strength: float,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Apply mutation to the individual's genes.

        Args:
            mutation_method: The strategy for mutation (e.g., GAUSSIAN, UNIFORM).
            mutation_rate: Probability of mutating the individual.
            mutation_strength: Scaling factor for mutation changes.
            bounds: Optional list of (min, max) constraints for each gene.
        """
        if random.random() < mutation_rate:
            if mutation_method == MutationMethod.GAUSSIAN:
                self._gaussian_mutation(mutation_strength)
            elif mutation_method == MutationMethod.UNIFORM:
                self._uniform_mutation(mutation_strength, bounds)
            elif mutation_method == MutationMethod.POLYNOMIAL:
                self._polynomial_mutation(mutation_strength, bounds)
            elif mutation_method == MutationMethod.NON_UNIFORM:
                self._non_uniform_mutation(mutation_strength, bounds)
            elif mutation_method == MutationMethod.BOUNDARY:
                self._boundary_mutation(bounds)
            elif mutation_method == MutationMethod.CREEP:
                self._creep_mutation(mutation_strength)
            else:
                self._gaussian_mutation(mutation_strength)

            # Apply bounds if provided
            if bounds:
                for i, (low, high) in enumerate(bounds):
                    self.genes[i] = np.clip(self.genes[i], low, high)

    def _gaussian_mutation(self, mutation_strength: float) -> None:
        """Apply Gaussian noise mutation.

        Args:
            mutation_strength: Standard deviation of the Gaussian distribution.
        """
        noise = np.random.normal(0, mutation_strength, self.genes.shape)
        self.genes += noise

    def _uniform_mutation(self, mutation_strength: float, bounds: Optional[List[Tuple[float, float]]]) -> None:
        """Apply uniform random mutation.

        Args:
            mutation_strength: Range for uniform noise if bounds are not provided.
            bounds: List of (min, max) constraints.
        """
        for i in range(len(self.genes)):
            if random.random() < 0.1:  # 10% chance per gene
                if bounds:
                    low, high = bounds[i]
                    self.genes[i] = random.uniform(low, high)
                else:
                    self.genes[i] += random.uniform(-mutation_strength, mutation_strength)

    def _polynomial_mutation(self, mutation_strength: float, bounds: Optional[List[Tuple[float, float]]]) -> None:
        """Apply polynomial mutation.

        Args:
            mutation_strength: Distribution index for polynomial mutation.
            bounds: List of (min, max) constraints.
        """
        for i in range(len(self.genes)):
            if random.random() < 0.1:  # 10% chance per gene
                if bounds:
                    low, high = bounds[i]
                    delta = random.uniform(-1, 1)
                    self.genes[i] = low + (high - low) * (0.5 + delta * mutation_strength)

    def _non_uniform_mutation(self, mutation_strength: float, bounds: Optional[List[Tuple[float, float]]]) -> None:
        """Apply non-uniform mutation.

        Args:
            mutation_strength: Scaling factor for mutation.
            bounds: List of (min, max) constraints.
        """
        for i in range(len(self.genes)):
            if random.random() < 0.1:  # 10% chance per gene
                if bounds:
                    low, high = bounds[i]
                    delta = random.uniform(-1, 1)
                    self.genes[i] = low + (high - low) * (0.5 + delta * mutation_strength)

    def _boundary_mutation(self, bounds: Optional[List[Tuple[float, float]]]) -> None:
        """Apply boundary mutation (setting gene to one of the bounds).

        Args:
            bounds: List of (min, max) constraints.
        """
        if bounds:
            for i in range(len(self.genes)):
                if random.random() < 0.1:  # 10% chance per gene
                    low, high = bounds[i]
                    self.genes[i] = random.choice([low, high])

    def _creep_mutation(self, mutation_strength: float) -> None:
        """Apply creep mutation (small random changes).

        Args:
            mutation_strength: Range of the creep change.
        """
        for i in range(len(self.genes)):
            if random.random() < 0.1:  # 10% chance per gene
                self.genes[i] += random.uniform(-mutation_strength, mutation_strength)

