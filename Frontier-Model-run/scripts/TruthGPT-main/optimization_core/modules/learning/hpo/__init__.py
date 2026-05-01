"""
Hyperparameter Optimization Package
===================================

Advanced hyperparameter optimization systems with Bayesian, Evolutionary, TPE, and more.
"""
from .enums import HpoAlgorithm, SamplerType, PrunerType
from .config import HpoConfig
from .bayesian import BayesianOptimizer
from .evolutionary import EvolutionaryOptimizer
from .tpe import TPEOptimizer
from .cma_es import CMAESOptimizer
from .optuna import OptunaOptimizer
from .multi_objective import MultiObjectiveOptimizer
from .system import HpoManager

# Compatibility aliases
HyperparameterOptimizer = HpoManager
HPOConfig = HpoConfig

# Factory functions
def create_hpo_config(**kwargs) -> HpoConfig:
    return HpoConfig(**kwargs)

def create_bayesian_optimizer(config: HpoConfig) -> BayesianOptimizer:
    return BayesianOptimizer(config)

def create_evolutionary_optimizer(config: HpoConfig) -> EvolutionaryOptimizer:
    return EvolutionaryOptimizer(config)

def create_tpe_optimizer(config: HpoConfig) -> TPEOptimizer:
    return TPEOptimizer(config)

def create_cmaes_optimizer(config: HpoConfig) -> CMAESOptimizer:
    return CMAESOptimizer(config)

def create_optuna_optimizer(config: HpoConfig) -> OptunaOptimizer:
    return OptunaOptimizer(config)

def create_multi_objective_optimizer(config: HpoConfig) -> MultiObjectiveOptimizer:
    return MultiObjectiveOptimizer(config)

def create_hpo_manager(config: HpoConfig) -> HpoManager:
    return HpoManager(config)

__all__ = [
    'HpoAlgorithm',
    'SamplerType',
    'PrunerType',
    'HpoConfig',
    'BayesianOptimizer',
    'EvolutionaryOptimizer',
    'TPEOptimizer',
    'CMAESOptimizer',
    'OptunaOptimizer',
    'MultiObjectiveOptimizer',
    'HpoManager',
    'create_hpo_config',
    'create_bayesian_optimizer',
    'create_evolutionary_optimizer',
    'create_tpe_optimizer',
    'create_cmaes_optimizer',
    'create_optuna_optimizer',
    'create_multi_objective_optimizer',
    'create_hpo_manager'
]

