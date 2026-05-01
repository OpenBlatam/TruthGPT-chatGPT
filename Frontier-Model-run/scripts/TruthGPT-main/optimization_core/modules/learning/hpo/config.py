"""
HPO Configuration
=================

Configuration for hyperparameter optimization systems.
"""
from dataclasses import dataclass, field
from .enums import HpoAlgorithm, SamplerType, PrunerType

@dataclass
class HpoConfig:
    """Configuration for hyperparameter optimization system"""
    # Basic settings
    hpo_algorithm: HpoAlgorithm = HpoAlgorithm.BAYESIAN_OPTIMIZATION
    sampler_type: SamplerType = SamplerType.GAUSSIAN_PROCESS
    pruner_type: PrunerType = PrunerType.MEDIAN_PRUNER
    
    # Optimization settings
    n_trials: int = 100
    n_jobs: int = 1
    timeout: float = 3600.0  # seconds
    
    # Bayesian optimization settings
    acquisition_function: str = "expected_improvement"
    kernel_type: str = "rbf"
    alpha: float = 1e-6
    
    # Evolutionary algorithm settings
    population_size: int = 50
    n_generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # TPE settings
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    
    # Pruning settings
    pruning_threshold: float = 0.1
    pruning_percentile: float = 25.0
    
    # Advanced features
    enable_parallel_evaluation: bool = True
    enable_warm_start: bool = True
    enable_multi_objective: bool = False
    
    def __post_init__(self):
        """Validate HPO configuration"""
        if self.n_trials <= 0:
            raise ValueError("Number of trials must be positive")
        if self.n_jobs <= 0:
            raise ValueError("Number of jobs must be positive")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.alpha <= 0:
            raise ValueError("Alpha must be positive")
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.n_generations <= 0:
            raise ValueError("Number of generations must be positive")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1")
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        if self.n_startup_trials <= 0:
            raise ValueError("Number of startup trials must be positive")
        if self.n_ei_candidates <= 0:
            raise ValueError("Number of EI candidates must be positive")
        if not (0 <= self.pruning_threshold <= 1):
            raise ValueError("Pruning threshold must be between 0 and 1")
        if not (0 <= self.pruning_percentile <= 100):
            raise ValueError("Pruning percentile must be between 0 and 100")

