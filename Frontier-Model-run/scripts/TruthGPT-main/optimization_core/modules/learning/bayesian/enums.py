"""
Bayesian Optimization Enums
==========================

Enums for acquisition functions, kernels, and optimization strategies.
"""
from enum import Enum

class AcquisitionFunction(Enum):
    """Acquisition functions"""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    ENTROPY_SEARCH = "entropy_search"
    KNOWLEDGE_GRADIENT = "knowledge_gradient"
    MUTUAL_INFORMATION = "mutual_information"
    THOMPSON_SAMPLING = "thompson_sampling"

class KernelType(Enum):
    """Kernel types"""
    RBF = "rbf"
    MATERN = "matern"
    WHITE = "white"
    CONSTANT = "constant"
    RATIONAL_QUADRATIC = "rational_quadratic"
    EXPONENTIAL = "exponential"
    PERIODIC = "periodic"

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    ASYNC = "async"
    PARALLEL = "parallel"
    MULTI_START = "multi_start"
    GRADIENT_BASED = "gradient_based"

