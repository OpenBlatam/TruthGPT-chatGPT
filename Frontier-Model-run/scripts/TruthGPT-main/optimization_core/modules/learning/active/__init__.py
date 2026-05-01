"""
Active Learning Package
=======================

Tools for acquiring the most informative data points for training model efficiently.
"""
from .enums import ActiveLearningStrategy, UncertaintyMeasure, QueryStrategy
from .config import ActiveLearningConfig
from .samplers import (
    UncertaintySampler,
    DiversitySampler,
    QueryByCommittee,
    ExpectedModelChange,
    BatchActiveLearning
)
from .system import ActiveLearningSystem

# Factory functions
def create_active_learning_config(**kwargs) -> ActiveLearningConfig:
    return ActiveLearningConfig(**kwargs)

def create_active_learning_system(config: ActiveLearningConfig) -> ActiveLearningSystem:
    return ActiveLearningSystem(config)

__all__ = [
    'ActiveLearningStrategy',
    'UncertaintyMeasure',
    'QueryStrategy',
    'ActiveLearningConfig',
    'UncertaintySampler',
    'DiversitySampler',
    'QueryByCommittee',
    'ExpectedModelChange',
    'BatchActiveLearning',
    'ActiveLearningSystem',
    'create_active_learning_config',
    'create_active_learning_system'
]

