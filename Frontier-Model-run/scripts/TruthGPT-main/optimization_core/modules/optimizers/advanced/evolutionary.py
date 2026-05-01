"""
Enterprise TruthGPT Neural Evolutionary Optimization Engine
Advanced neural evolution with genetic algorithms, neuroevolution, and adaptive optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import random
import math
import copy
import json

class EvolutionStrategy(Enum):
    """Evolution strategy enum."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM_OPTIMIZATION = "particle_swarm_optimization"
    NEUROEVOLUTION = "neuroevolution"
    COEVOLUTION = "coevolution"
    MULTI_OBJECTIVE_EVOLUTION = "multi_objective_evolution"

class SelectionMethod(Enum):
    """Selection method enum."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_SELECTION = "rank_selection"
    ELITISM = "elitism"
    STOCHASTIC_UNIVERSAL_SAMPLING = "stochastic_universal_sampling"

class MutationType(Enum):
    """Mutation type enum."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"
    NEURAL_MUTATION = "neural_mutation"

@dataclass
class NeuralEvolutionConfig:
    """Neural evolution configuration."""
    strategy: EvolutionStrategy = EvolutionStrategy.GENETIC_ALGORITHM
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    mutation_type: MutationType = MutationType.GAUSSIAN
    population_size: int = 100
    generations: int = 1000
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 10
    tournament_size: int = 3
    mutation_strength: float = 0.1
    learning_rate: float = 1e-4
    fitness_threshold: float = 0.95
    convergence_threshold: float = 0.001
    max_stagnation: int = 50
    use_adaptive_parameters: bool = True
    use_neural_architecture_search: bool = True
    use_multi_objective: bool = False

@dataclass
class NeuralIndividual:
    """Neural individual representation."""
    id: str
    genotype: Dict[str, Any]
    phenotype: Optional[nn.Module] = None
    fitness: float = 0.0
    fitness_history: List[float] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"individual_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"

@dataclass
class EvolutionResult:
    """Evolution result."""
    best_individual: NeuralIndividual
    population_fitness: List[float]
    generation: int
    convergence_achieved: bool
    optimization_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class NeuralArchitectureGenerator:
    """Neural architecture generator."""
    
    def __init__(self, config: NeuralEvolutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Architecture building blocks
        self.layer_types = ["linear", "conv2d", "conv1d", "lstm", "gru", "attention", "transformer"]
        self.activation_functions = ["relu", "gelu", "swish", "tanh", "sigmoid", "leaky_relu"]
        self.optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
        
    def generate_random_architecture(self) -> Dict[str, Any]:
        """Generate random neural architecture."""
        architecture = {
            "layers": [],
            "optimizer": random.choice(self.optimizers),
            "learning_rate": random.uniform(1e-5, 1e-2),
            "batch_size": random.choice([16, 32, 64, 128]),
            "dropout_rate": random.uniform(0.0, 0.5),
            "weight_decay": random.uniform(0.0, 1e-3)
        }
        
        # Generate layers
        num_layers = random.randint(3, 10)
        for i in range(num_layers):
            layer = self._generate_random_layer(i)
            architecture["layers"].append(layer)
        
        return architecture
    
    def _generate_random_layer(self, layer_index: int) -> Dict[str, Any]:
        """Generate random layer."""
        layer_type = random.choice(self.layer_types)
        
        layer = {
            "type": layer_type,
            "activation": random.choice(self.activation_functions),
            "dropout": random.uniform(0.0, 0.3)
        }
        
        if layer_type == "linear":
            layer["input_size"] = random.randint(64, 1024)
            layer["output_size"] = random.randint(64, 1024)
        elif layer_type in ["conv2d", "conv1d"]:
            layer["in_channels"] = random.randint(1, 64)
            layer["out_channels"] = random.randint(1, 128)
            layer["kernel_size"] = random.choice([3, 5, 7])
            layer["stride"] = random.choice([1, 2])
            layer["padding"] = random.choice([0, 1, 2])
        elif layer_type in ["lstm", "gru"]:
            layer["input_size"] = random.randint(64, 512)
            layer["hidden_size"] = random.randint(64, 512)
            layer["num_layers"] = random.randint(1, 3)
            layer["bidirectional"] = random.choice([True, False])
        elif layer_type == "attention":
            layer["embed_dim"] = random.randint(64, 512)
            layer["num_heads"] = random.choice([1, 2, 4, 8])
            layer["dropout"] = random.uniform(0.0, 0.3)
        elif layer_type == "transformer":
            layer["d_model"] = random.randint(64, 512)
            layer["nhead"] = random.choice([1, 2, 4, 8])
            layer["num_layers"] = random.randint(1, 6)
            layer["dim_feedforward"] = random.randint(128, 2048)
        
        return layer
    
    def mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate neural architecture."""
        mutated = copy.deepcopy(architecture)
        
        # Mutate learning rate
        if random.random() < 0.3:
            mutated["learning_rate"] *= random.uniform(0.5, 2.0)
            mutated["learning_rate"] = max(1e-6, min(1e-1, mutated["learning_rate"]))
        
        # Mutate batch size
        if random.random() < 0.2:
            mutated["batch_size"] = random.choice([16, 32, 64, 128])
        
        # Mutate dropout
        if random.random() < 0.3:
            mutated["dropout_rate"] = random.uniform(0.0, 0.5)
        
        # Mutate layers
        if random.random() < 0.4:
            self._mutate_layers(mutated)
        
        return mutated
    
    def _mutate_layers(self, architecture: Dict[str, Any]):
        """Mutate layers in architecture."""
        layers = architecture["layers"]
        
        if random.random() < 0.3 and len(layers) > 1:
            # Remove layer
            layers.pop(random.randint(0, len(layers) - 1))
        elif random.random() < 0.3:
            # Add layer
            new_layer = self._generate_random_layer(len(layers))
            layers.append(new_layer)
        else:
            # Mutate existing layer
            if layers:
                layer_index = random.randint(0, len(layers) - 1)
                layers[layer_index] = self._mutate_layer(layers[layer_index])
    
    def _mutate_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate individual layer."""
        mutated = copy.deepcopy(layer)
        
        # Mutate activation
        if random.random() < 0.3:
            mutated["activation"] = random.choice(self.activation_functions)
        
        # Mutate layer-specific parameters
        if mutated["type"] == "linear":
            if random.random() < 0.3:
                mutated["output_size"] = random.randint(64, 1024)
        elif mutated["type"] in ["conv2d", "conv1d"]:
            if random.random() < 0.3:
                mutated["out_channels"] = random.randint(1, 128)
        elif mutated["type"] in ["lstm", "gru"]:
            if random.random() < 0.3:
                mutated["hidden_size"] = random.randint(64, 512)
        elif mutated["type"] == "attention":
            if random.random() < 0.3:
                mutated["num_heads"] = random.choice([1, 2, 4, 8])
        
        return mutated

class FitnessEvaluator:
    """Fitness evaluator for neural individuals."""
    
    def __init__(self, config: NeuralEvolutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Evaluation metrics
        self.evaluation_history: List[Tuple[str, float]] = []
        
    def evaluate_fitness(self, individual: NeuralIndividual, task_data: Optional[Any] = None) -> float:
        """Evaluate fitness of neural individual."""
        try:
            # Build phenotype if not exists
            if individual.phenotype is None:
                individual.phenotype = self._build_phenotype(individual.genotype)
            
            # Evaluate performance
            fitness = self._evaluate_performance(individual.phenotype, task_data)
            
            # Add complexity penalty
            complexity_penalty = self._calculate_complexity_penalty(individual.genotype)
            fitness -= complexity_penalty
            
            # Store evaluation
            self.evaluation_history.append((individual.id, fitness))
            
            return max(0.0, fitness)
            
        except Exception as e:
            self.logger.error(f"Error evaluating fitness for {individual.id}: {str(e)}")
            return 0.0
    
    def _build_phenotype(self, genotype: Dict[str, Any]) -> nn.Module:
        """Build neural network phenotype from genotype."""
        layers = []
        
        for layer_config in genotype["layers"]:
            layer = self._build_layer(layer_config)
            if layer:
                layers.append(layer)
        
        # Create sequential model
        model = nn.Sequential(*layers)
        return model
    
    def _build_layer(self, layer_config: Dict[str, Any]) -> Optional[nn.Module]:
        """Build individual layer."""
        layer_type = layer_config["type"]
        
        try:
            if layer_type == "linear":
                return nn.Linear(
                    layer_config["input_size"],
                    layer_config["output_size"]
                )
            elif layer_type == "conv2d":
                return nn.Conv2d(
                    layer_config["in_channels"],
                    layer_config["out_channels"],
                    layer_config["kernel_size"],
                    stride=layer_config["stride"],
                    padding=layer_config["padding"]
                )
            elif layer_type == "conv1d":
                return nn.Conv1d(
                    layer_config["in_channels"],
                    layer_config["out_channels"],
                    layer_config["kernel_size"],
                    stride=layer_config["stride"],
                    padding=layer_config["padding"]
                )
            elif layer_type == "lstm":
                return nn.LSTM(
                    layer_config["input_size"],
                    layer_config["hidden_size"],
                    layer_config["num_layers"],
                    bidirectional=layer_config["bidirectional"]
                )
            elif layer_type == "gru":
                return nn.GRU(
                    layer_config["input_size"],
                    layer_config["hidden_size"],
                    layer_config["num_layers"],
                    bidirectional=layer_config["bidirectional"]
                )
            elif layer_type == "attention":
                return nn.MultiheadAttention(
                    layer_config["embed_dim"],
                    layer_config["num_heads"],
                    dropout=layer_config["dropout"]
                )
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error building layer {layer_type}: {str(e)}")
            return None
    
    def _evaluate_performance(self, model: nn.Module, task_data: Optional[Any] = None) -> float:
        """Evaluate model performance."""
        # Simulate performance evaluation
        # In a real implementation, this would use actual task data
        
        # Base performance
        base_performance = random.uniform(0.5, 0.9)
        
        # Model complexity factor
        num_parameters = sum(p.numel() for p in model.parameters())
        complexity_factor = min(1.0, 1000 / num_parameters)  # Prefer simpler models
        
        # Architecture quality factor
        architecture_factor = self._evaluate_architecture_quality(model)
        
        # Combined fitness
        fitness = base_performance * complexity_factor * architecture_factor
        
        return fitness
    
    def _evaluate_architecture_quality(self, model: nn.Module) -> float:
        """Evaluate architecture quality."""
        # Simple heuristics for architecture quality
        quality_score = 1.0
        
        # Check for common issues
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Prefer reasonable layer sizes
                if module.out_features > 2048:
                    quality_score *= 0.9
            elif isinstance(module, nn.Conv2d):
                # Prefer reasonable kernel sizes
                if module.kernel_size[0] > 7:
                    quality_score *= 0.95
        
        return quality_score
    
    def _calculate_complexity_penalty(self, genotype: Dict[str, Any]) -> float:
        """Calculate complexity penalty."""
        num_layers = len(genotype["layers"])
        penalty = num_layers * 0.01  # Small penalty per layer
        
        # Additional penalty for very complex architectures
        if num_layers > 20:
            penalty += 0.1
        
        return penalty

class SelectionOperator:
    """Selection operator for evolutionary algorithms."""
    
    def __init__(self, config: NeuralEvolutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def select(self, population: List[NeuralIndividual], num_parents: int) -> List[NeuralIndividual]:
        """Select parents from population."""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, num_parents)
        elif self.config.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection(population, num_parents)
        elif self.config.selection_method == SelectionMethod.RANK_SELECTION:
            return self._rank_selection(population, num_parents)
        elif self.config.selection_method == SelectionMethod.ELITISM:
            return self._elitism_selection(population, num_parents)
        else:
            return self._tournament_selection(population, num_parents)
    
    def _tournament_selection(self, population: List[NeuralIndividual], num_parents: int) -> List[NeuralIndividual]:
        """Tournament selection."""
        parents = []
        
        for _ in range(num_parents):
            tournament = random.sample(population, self.config.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _roulette_wheel_selection(self, population: List[NeuralIndividual], num_parents: int) -> List[NeuralIndividual]:
        """Roulette wheel selection."""
        # Calculate fitness sum
        fitness_sum = sum(individual.fitness for individual in population)
        
        if fitness_sum == 0:
            return random.sample(population, min(num_parents, len(population)))
        
        # Calculate probabilities
        probabilities = [individual.fitness / fitness_sum for individual in population]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            selected_index = np.random.choice(len(population), p=probabilities)
            parents.append(population[selected_index])
        
        return parents
    
    def _rank_selection(self, population: List[NeuralIndividual], num_parents: int) -> List[NeuralIndividual]:
        """Rank selection."""
        # Sort by fitness
        sorted_population = sorted(population, key=lambda x: x.fitness)
        
        # Assign ranks
        ranks = list(range(1, len(sorted_population) + 1))
        
        # Calculate probabilities based on ranks
        rank_sum = sum(ranks)
        probabilities = [rank / rank_sum for rank in ranks]
        
        # Select parents
        parents = []
        for _ in range(num_parents):
            selected_index = np.random.choice(len(sorted_population), p=probabilities)
            parents.append(sorted_population[selected_index])
        
        return parents
    
    def _elitism_selection(self, population: List[NeuralIndividual], num_parents: int) -> List[NeuralIndividual]:
        """Elitism selection."""
        # Sort by fitness and select best
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:num_parents]

class CrossoverOperator:
    """Crossover operator for evolutionary algorithms."""
    
    def __init__(self, config: NeuralEvolutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def crossover(self, parent1: NeuralIndividual, parent2: NeuralIndividual) -> Tuple[NeuralIndividual, NeuralIndividual]:
        """Perform crossover between two parents."""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2
        
        # Create offspring
        offspring1 = self._create_offspring(parent1, parent2)
        offspring2 = self._create_offspring(parent2, parent1)
        
        return offspring1, offspring2
    
    def _create_offspring(self, parent1: NeuralIndividual, parent2: NeuralIndividual) -> NeuralIndividual:
        """Create offspring from two parents."""
        # Deep copy parent1 as base
        offspring_genotype = copy.deepcopy(parent1.genotype)
        
        # Crossover layers
        if "layers" in offspring_genotype and "layers" in parent2.genotype:
            offspring_genotype["layers"] = self._crossover_layers(
                parent1.genotype["layers"],
                parent2.genotype["layers"]
            )
        
        # Crossover hyperparameters
        offspring_genotype["learning_rate"] = random.choice([
            parent1.genotype["learning_rate"],
            parent2.genotype["learning_rate"]
        ])
        
        offspring_genotype["batch_size"] = random.choice([
            parent1.genotype["batch_size"],
            parent2.genotype["batch_size"]
        ])
        
        # Create offspring individual
        offspring = NeuralIndividual(
            id="",
            genotype=offspring_genotype,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return offspring
    
    def _crossover_layers(self, layers1: List[Dict[str, Any]], layers2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crossover layer configurations."""
        # Single point crossover
        crossover_point = random.randint(1, min(len(layers1), len(layers2)) - 1)
        
        # Combine layers
        new_layers = layers1[:crossover_point] + layers2[crossover_point:]
        
        return new_layers

class MutationOperator:
    """Mutation operator for evolutionary algorithms."""
    
    def __init__(self, config: NeuralEvolutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.architecture_generator = NeuralArchitectureGenerator(config)
    
    def mutate(self, individual: NeuralIndividual) -> NeuralIndividual:
        """Mutate individual."""
        if random.random() > self.config.mutation_rate:
            return individual
        
        # Create mutated individual
        mutated = copy.deepcopy(individual)
        mutated.id = ""  # Generate new ID
        mutated.generation += 1
        mutated.mutations += 1
        
        # Mutate genotype
        mutated.genotype = self.architecture_generator.mutate_architecture(individual.genotype)
        
        # Reset phenotype (will be rebuilt during evaluation)
        mutated.phenotype = None
        
        return mutated

class NeuralEvolutionaryOptimizer:
    """Neural evolutionary optimizer."""
    
    def __init__(self, config: NeuralEvolutionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.architecture_generator = NeuralArchitectureGenerator(config)
        self.fitness_evaluator = FitnessEvaluator(config)
        self.selection_operator = SelectionOperator(config)
        self.crossover_operator = CrossoverOperator(config)
        self.mutation_operator = MutationOperator(config)
        
        # State
        self.population: List[NeuralIndividual] = []
        self.generation = 0
        self.best_individual: Optional[NeuralIndividual] = None
        self.evolution_history: List[EvolutionResult] = []
        
        # Convergence tracking
        self.stagnation_count = 0
        self.last_best_fitness = 0.0
        
        # Optimization state
        self.is_optimizing = False
        self.optimization_thread: Optional[threading.Thread] = None
    
    def initialize_population(self):
        """Initialize population."""
        self.population = []
        
        for i in range(self.config.population_size):
            genotype = self.architecture_generator.generate_random_architecture()
            individual = NeuralIndividual(
                id=f"individual_{i}",
                genotype=genotype,
                generation=0
            )
            self.population.append(individual)
        
        self.logger.info(f"Initialized population of {len(self.population)} individuals")
    
    def start_optimization(self):
        """Start neural evolutionary optimization."""
        if self.is_optimizing:
            return
        
        self.is_optimizing = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        self.logger.info("Neural evolutionary optimization started")
    
    def stop_optimization(self):
        """Stop neural evolutionary optimization."""
        self.is_optimizing = False
        if self.optimization_thread:
            self.optimization_thread.join()
        self.logger.info("Neural evolutionary optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        start_time = time.time()
        
        # Initialize population
        self.initialize_population()
        
        while self.is_optimizing and self.generation < self.config.generations:
            try:
                # Evaluate fitness
                self._evaluate_population()
                
                # Check convergence
                if self._check_convergence():
                    self.logger.info("Convergence achieved")
                    break
                
                # Evolve population
                self._evolve_population()
                
                # Update generation
                self.generation += 1
                
                # Log progress
                if self.generation % 10 == 0:
                    self.logger.info(f"Generation {self.generation}: Best fitness = {self.best_individual.fitness:.4f}")
                
                time.sleep(0.1)  # Brief pause
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {str(e)}")
                break
        
        # Create final result
        optimization_time = time.time() - start_time
        
        result = EvolutionResult(
            best_individual=self.best_individual,
            population_fitness=[ind.fitness for ind in self.population],
            generation=self.generation,
            convergence_achieved=self._check_convergence(),
            optimization_time=optimization_time,
            metadata={
                "strategy": self.config.strategy.value,
                "selection_method": self.config.selection_method.value,
                "mutation_type": self.config.mutation_type.value,
                "population_size": self.config.population_size,
                "final_fitness": self.best_individual.fitness if self.best_individual else 0.0
            }
        )
        
        self.evolution_history.append(result)
        self.logger.info(f"Optimization completed in {optimization_time:.2f}s")
    
    def _evaluate_population(self):
        """Evaluate fitness of entire population."""
        for individual in self.population:
            individual.fitness = self.fitness_evaluator.evaluate_fitness(individual)
            individual.fitness_history.append(individual.fitness)
        
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if not self.best_individual or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
    
    def _evolve_population(self):
        """Evolve population for one generation."""
        new_population = []
        
        # Elitism: keep best individuals
        if self.config.elite_size > 0:
            elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.config.elite_size]
            new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parents = self.selection_operator.select(self.population, 2)
            
            # Crossover
            offspring1, offspring2 = self.crossover_operator.crossover(parents[0], parents[1])
            
            # Mutation
            offspring1 = self.mutation_operator.mutate(offspring1)
            offspring2 = self.mutation_operator.mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Update population
        self.population = new_population[:self.config.population_size]
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if not self.best_individual:
            return False
        
        # Check fitness threshold
        if self.best_individual.fitness >= self.config.fitness_threshold:
            return True
        
        # Check stagnation
        if abs(self.best_individual.fitness - self.last_best_fitness) < self.config.convergence_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        self.last_best_fitness = self.best_individual.fitness
        
        return self.stagnation_count >= self.config.max_stagnation
    
    def get_best_individual(self) -> Optional[NeuralIndividual]:
        """Get best individual."""
        return self.best_individual
    
    def get_evolution_history(self) -> List[EvolutionResult]:
        """Get evolution history."""
        return self.evolution_history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.best_individual:
            return {"status": "No optimization data available"}
        
        return {
            "is_optimizing": self.is_optimizing,
            "generation": self.generation,
            "best_fitness": self.best_individual.fitness,
            "population_size": len(self.population),
            "strategy": self.config.strategy.value,
            "selection_method": self.config.selection_method.value,
            "mutation_type": self.config.mutation_type.value,
            "convergence_achieved": self._check_convergence(),
            "stagnation_count": self.stagnation_count,
            "total_evaluations": len(self.fitness_evaluator.evaluation_history)
        }

# Factory function
def create_neural_evolutionary_optimizer(config: Optional[NeuralEvolutionConfig] = None) -> NeuralEvolutionaryOptimizer:
    """Create neural evolutionary optimizer."""
    if config is None:
        config = NeuralEvolutionConfig()
    return NeuralEvolutionaryOptimizer(config)

# Example usage
if __name__ == "__main__":
    import time
    
    # Create neural evolutionary optimizer
    config = NeuralEvolutionConfig(
        strategy=EvolutionStrategy.GENETIC_ALGORITHM,
        selection_method=SelectionMethod.TOURNAMENT,
        mutation_type=MutationType.GAUSSIAN,
        population_size=50,
        generations=100,
        use_neural_architecture_search=True
    )
    
    optimizer = create_neural_evolutionary_optimizer(config)
    
    # Start optimization
    optimizer.start_optimization()
    
    try:
        # Let it run
        time.sleep(5)
        
        # Get stats
        stats = optimizer.get_stats()
        print("Neural Evolutionary Optimization Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Get best individual
        best = optimizer.get_best_individual()
        if best:
            print(f"\nBest Individual:")
            print(f"  ID: {best.id}")
            print(f"  Fitness: {best.fitness:.4f}")
            print(f"  Generation: {best.generation}")
            print(f"  Mutations: {best.mutations}")
            print(f"  Layers: {len(best.genotype['layers'])}")
    
    finally:
        optimizer.stop_optimization()
    
    print("\nNeural evolutionary optimization completed")


