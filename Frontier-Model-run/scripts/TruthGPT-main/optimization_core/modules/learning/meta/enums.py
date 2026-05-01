"""
Meta-Learning Enums
===================

Algorithms and task distribution types for meta-learning.
"""
from enum import Enum

class MetaLearningAlgorithm(Enum):
    """Meta-learning algorithms"""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"
    PROTOMAML = "protomaml"
    META_SGD = "meta_sgd"
    LEARNED_INITIALIZATION = "learned_init"
    GRADIENT_BASED_META = "gradient_based"

class TaskDistribution(Enum):
    """Task distribution types"""
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    MULTIMODAL = "multimodal"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"

