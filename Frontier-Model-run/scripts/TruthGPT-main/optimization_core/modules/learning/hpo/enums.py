"""
HPO Enums
=========

Enums for HPO algorithms, samplers, and pruners.
"""
from enum import Enum

class HpoAlgorithm(Enum):
    """Hyperparameter optimization algorithms"""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY_ALGORITHM = "evolutionary_algorithm"
    TPE = "tpe"
    CMA_ES = "cma_es"
    OPTUNA = "optuna"
    HYPEROPT = "hyperopt"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"

class SamplerType(Enum):
    """Sampler types"""
    GAUSSIAN_PROCESS = "gaussian_process"
    TREE_PARZEN_ESTIMATOR = "tree_parzen_estimator"
    CMA_ES_SAMPLER = "cma_es_sampler"
    EVOLUTIONARY_SAMPLER = "evolutionary_sampler"
    RANDOM_SAMPLER = "random_sampler"

class PrunerType(Enum):
    """Pruner types"""
    MEDIAN_PRUNER = "median_pruner"
    PERCENTILE_PRUNER = "percentile_pruner"
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"
    NO_PRUNING = "no_pruning"

