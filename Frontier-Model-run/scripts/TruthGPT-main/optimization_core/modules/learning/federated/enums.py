"""
Federated Learning Enums
========================

Enums for aggregation methods, client selection strategies, and privacy levels.
"""
from enum import Enum

class AggregationMethod(Enum):
    """Federated aggregation methods"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    SCAFFOLD = "scaffold"
    FEDOPT = "fedopt"
    FEDADAGRAD = "fedadagrad"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"

class ClientSelectionStrategy(Enum):
    """Client selection strategies"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    PROBABILITY_BASED = "probability_based"
    PERFORMANCE_BASED = "performance_based"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"

class PrivacyLevel(Enum):
    """Privacy preservation levels"""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    FEDERATED_LEARNING = "federated_learning"

