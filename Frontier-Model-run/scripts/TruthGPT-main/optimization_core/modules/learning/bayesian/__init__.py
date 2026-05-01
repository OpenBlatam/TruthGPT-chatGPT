"""
Bayesian Optimization Package
============================

Advanced Bayesian optimization with Gaussian processes and acquisition functions.
"""
from .enums import AcquisitionFunction, KernelType, OptimizationStrategy
from .config import BayesianOptimizationConfig
from .models import GaussianProcessModel
from .acquisition import AcquisitionFunctionOptimizer
from .optimization import MultiObjectiveOptimizer, ConstrainedOptimizer
from .system import BayesianOptimizer

# Compatibility aliases
BayesianConfig = BayesianOptimizationConfig

# Factory functions
def create_bayesian_optimization_config(**kwargs) -> BayesianOptimizationConfig:
    return BayesianOptimizationConfig(**kwargs)

def create_bayesian_optimizer(config: BayesianOptimizationConfig) -> BayesianOptimizer:
    return BayesianOptimizer(config)

__all__ = [
    'AcquisitionFunction',
    'KernelType',
    'OptimizationStrategy',
    'BayesianOptimizationConfig',
    'BayesianConfig',
    'GaussianProcessModel',
    'AcquisitionFunctionOptimizer',
    'MultiObjectiveOptimizer',
    'ConstrainedOptimizer',
    'BayesianOptimizer',
    'create_bayesian_optimization_config',
    'create_bayesian_optimizer'
]

