"""
Bayesian Optimization Configuration
==================================

Configuration dataclasses for Bayesian optimization systems.
"""
from dataclasses import dataclass, field
from .enums import AcquisitionFunction, KernelType, OptimizationStrategy

@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization system"""
    # Basic settings
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT
    kernel_type: KernelType = KernelType.RBF
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.SEQUENTIAL
    
    # Gaussian process settings
    gp_alpha: float = 1e-6
    gp_n_restarts: int = 10
    gp_normalize_y: bool = True
    
    # Acquisition function settings
    acquisition_xi: float = 0.01
    acquisition_kappa: float = 2.576
    acquisition_beta: float = 1.0
    
    # Optimization settings
    n_iterations: int = 100
    n_initial_points: int = 5
    n_candidates: int = 1000
    batch_size: int = 1
    
    # Multi-objective settings
    enable_multi_objective: bool = False
    n_objectives: int = 2
    pareto_front_size: int = 10
    
    # Advanced features
    enable_constraints: bool = False
    enable_noise_estimation: bool = True
    enable_warm_start: bool = True
    enable_parallel_evaluation: bool = False
    
    def __post_init__(self):
        """Validate Bayesian optimization configuration"""
        if self.gp_alpha <= 0:
            raise ValueError("GP alpha must be positive")
        if self.gp_n_restarts <= 0:
            raise ValueError("GP n_restarts must be positive")
        if self.acquisition_xi < 0:
            raise ValueError("Acquisition xi must be non-negative")
        if self.acquisition_kappa <= 0:
            raise ValueError("Acquisition kappa must be positive")
        if self.acquisition_beta <= 0:
            raise ValueError("Acquisition beta must be positive")
        if self.n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        if self.n_initial_points <= 0:
            raise ValueError("Number of initial points must be positive")
        if self.n_candidates <= 0:
            raise ValueError("Number of candidates must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.n_objectives <= 0:
            raise ValueError("Number of objectives must be positive")
        if self.pareto_front_size <= 0:
            raise ValueError("Pareto front size must be positive")

