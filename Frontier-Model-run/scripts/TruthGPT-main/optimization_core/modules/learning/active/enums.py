"""
Active Learning Enums
====================

Strategies, uncertainty measures, and query methods for active learning.
"""
from enum import Enum

class ActiveLearningStrategy(Enum):
    """Active learning strategies"""
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    DIVERSITY_SAMPLING = "diversity_sampling"
    QUERY_BY_COMMITTEE = "query_by_committee"
    EXPECTED_MODEL_CHANGE = "expected_model_change"
    BATCH_ACTIVE_LEARNING = "batch_active_learning"
    HYBRID_SAMPLING = "hybrid_sampling"
    ADAPTIVE_SAMPLING = "adaptive_sampling"
    COST_SENSITIVE_SAMPLING = "cost_sensitive_sampling"

class UncertaintyMeasure(Enum):
    """Uncertainty measures"""
    ENTROPY = "entropy"
    MARGIN = "margin"
    LEAST_CONFIDENT = "least_confident"
    VARIANCE = "variance"
    BALD = "bald"
    MAXIMUM_ENTROPY = "maximum_entropy"
    VARIANCE_REDUCTION = "variance_reduction"

class QueryStrategy(Enum):
    """Query strategies"""
    RANDOM_SAMPLING = "random_sampling"
    UNCERTAINTY_BASED = "uncertainty_based"
    DIVERSITY_BASED = "diversity_based"
    HYBRID_STRATEGY = "hybrid_strategy"
    ADAPTIVE_STRATEGY = "adaptive_strategy"
    COST_AWARE_STRATEGY = "cost_aware_strategy"

