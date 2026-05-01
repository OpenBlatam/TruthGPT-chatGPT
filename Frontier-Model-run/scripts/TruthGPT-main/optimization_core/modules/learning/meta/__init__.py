"""
Meta-Learning Package
====================

Algorithms for few-shot learning and rapid adaptation across task distributions.
"""
from .enums import MetaLearningAlgorithm, TaskDistribution
from .config import MetaLearningConfig
from .task_gen import TaskGenerator
from .algorithms import MAML, Reptile
from .system import MetaLearner

# Compatibility aliases
MetaConfig = MetaLearningConfig

# Factory functions
def create_meta_learning_config(**kwargs) -> MetaLearningConfig:
    return MetaLearningConfig(**kwargs)

def create_meta_learner(model, config: MetaLearningConfig) -> MetaLearner:
    return MetaLearner(model, config)

__all__ = [
    'MetaLearningAlgorithm',
    'TaskDistribution',
    'MetaLearningConfig',
    'TaskGenerator',
    'MAML',
    'Reptile',
    'MetaLearner',
    'create_meta_learning_config',
    'create_meta_learner'
]

