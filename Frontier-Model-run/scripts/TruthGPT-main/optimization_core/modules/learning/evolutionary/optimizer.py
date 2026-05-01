"""
Evolutionary Optimizer
======================

Main optimizer class for evolutionary algorithms.
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from optimization_core.modules.learning.evolutionary.config import EvolutionaryConfig
from optimization_core.modules.learning.evolutionary.population import Population

logger = logging.getLogger(__name__)


class EvolutionaryOptimizer:
    """Main evolutionary optimizer orchestration system.

    Coordinates the evolution process, tracking metrics, handling stopping criteria,
    and generating optimization reports.

    Attributes:
        config: Configuration parameters for the optimizer and population.
        population: The manager for the collection of individuals.
        optimization_history: Detailed record of all previous optimization runs.
    """

    def __init__(self, config: EvolutionaryConfig) -> None:
        """Initialize the evolutionary optimizer.

        Args:
            config: The evolutionary configuration to use.
        """
        self.config: EvolutionaryConfig = config
        self.population: Population = Population(config)
        self.optimization_history: List[Dict[str, Any]] = []
        logger.info("✅ Evolutionary Optimizer initialized")

    def optimize(
        self,
        fitness_function: Callable[[Any], float],
        gene_length: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Optimize using evolutionary algorithm logic.

        Args:
            fitness_function: Function that takes genes and returns a scalar fitness score.
            gene_length: Number of genes (parameters) per individual.
            bounds: Optional list of (min, max) limits for each gene.

        Returns:
            Dictionary containing optimization results, stats, and the best solution.
        """
        logger.info(f"🚀 Optimizing using {self.config.evolutionary_algorithm.value}")

        start_time = time.time()

        # Initialize
        self.population.initialize(gene_length, bounds)
        self.population.evaluate_fitness(fitness_function)

        generations_data: List[Dict[str, Any]] = []

        # Evolution loop
        for generation in range(self.config.max_generations):
            # Run one generation
            self._run_generation(fitness_function, bounds)

            # Collect metrics
            gen_data = self._collect_generation_metrics(generation)
            generations_data.append(gen_data)

            # Log progress
            if generation % 10 == 0:
                self._log_progress(generation, gen_data)

            # Check stopping criteria
            if self.population.check_convergence():
                logger.info("✅ Population converged")
                break

            if self.population.check_stagnation():
                logger.info("⚠️ Population stagnated")
                break

        # Finalize results
        end_time = time.time()
        results = {
            "start_time": start_time,
            "end_time": end_time,
            "total_duration": end_time - start_time,
            "config": self.config,
            "generations": generations_data,
            "best_solution": self.population.individuals[0].genes.copy(),
            "best_fitness": self.population.individuals[0].fitness,
            "final_generation": self.population.generation,
        }

        self.optimization_history.append(results)
        logger.info("✅ Evolutionary optimization completed")
        return results

    def _run_generation(
        self, fitness_function: Callable[[Any], float], bounds: Optional[List[Tuple[float, float]]] = None
    ) -> None:
        """Execute a single generation step.

        Args:
            fitness_function: Function to evaluate fitness of new offspring.
            bounds: Optional gene constraints.
        """
        parents = self.population.select_parents()
        offspring = self.population.crossover(parents, bounds)
        self.population.mutate_offspring(offspring, bounds)

        # Evaluate new candidates
        for individual in offspring:
            if individual.fitness is None:
                individual.fitness = fitness_function(individual.genes)

        self.population.replace_population(offspring)
        self.population.calculate_diversity()

    def _collect_generation_metrics(self, generation: int) -> Dict[str, Any]:
        """Collect metrics for the current generation.

        Args:
            generation: The index of the current generation.

        Returns:
            Dictionary with fitness and diversity metrics.
        """
        return {
            "generation": generation,
            "best_fitness": self.population.best_fitness_history[-1],
            "average_fitness": self.population.average_fitness_history[-1],
            "diversity": self.population.diversity_history[-1] if self.population.diversity_history else 0.0,
            "best_individual": self.population.individuals[0].genes.copy(),
        }

    def _log_progress(self, generation: int, data: Dict[str, Any]) -> None:
        """Log progress for the current generation.

        Args:
            generation: The index of the current generation.
            data: The metrics for this generation.
        """
        logger.info(
            f"   Gen {generation}: Best={data['best_fitness']:.4f}, "
            f"Avg={data['average_fitness']:.4f}, Div={data['diversity']:.4f}"
        )

    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate a readable optimization report.

        Args:
            results: The metrics returned by the optimize method.

        Returns:
            Formatted string representation of the optimization run.
        """
        report = [
            "=" * 50,
            "EVOLUTIONARY COMPUTING REPORT",
            "=" * 50,
            "\nEVOLUTIONARY COMPUTING CONFIGURATION:",
            "-" * 35,
            f"Evolutionary Algorithm: {self.config.evolutionary_algorithm.value}",
            f"Selection Method: {self.config.selection_method.value}",
            f"Crossover Method: {self.config.crossover_method.value}",
            f"Mutation Method: {self.config.mutation_method.value}",
            f"Population Size: {self.config.population_size}",
            f"Max Generations: {self.config.max_generations}",
            f"Elite Size: {self.config.elite_size}",
            f"Mutation Rate: {self.config.mutation_rate}",
            "\nEVOLUTIONARY COMPUTING RESULTS:",
            "-" * 32,
            f"Total Duration: {results.get('total_duration', 0):.2f} seconds",
            f"Final Generation: {results.get('final_generation', 0)}",
            f"Best Fitness: {results.get('best_fitness', 0):.4f}",
        ]
        return "\n".join(report)

    def visualize_optimization_results(self, save_path: Optional[str] = None) -> None:
        """Visualize optimization results using matplotlib.

        Args:
            save_path: Optional path to save the generated plot.
        """
        if not self.optimization_history:
            logger.warning("No optimization history to visualize")
            return

        try:
            fig, axes = plt.subplots(1, 1, figsize=(10, 5))
            durations = [r.get("total_duration", 0) for r in self.optimization_history]
            axes.plot(durations, "b-", linewidth=2)
            axes.set_title("Duration Over Time")
            axes.set_xlabel("Run")
            axes.set_ylabel("Seconds")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

