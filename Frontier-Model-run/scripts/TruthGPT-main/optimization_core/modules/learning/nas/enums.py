"""
Neural Architecture Search Enums
================================

Enums for NAS strategies and configurations.
"""
from enum import Enum

class SearchStrategy(Enum):
    """Neural Architecture Search strategies"""
    RANDOM = "random"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    GRADIENT_BASED = "gradient_based"
    DIFFERENTIABLE = "differentiable"

