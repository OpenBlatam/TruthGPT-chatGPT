"""
Continual Learning Package
==========================

Advanced continual learning systems with EWC, Replay, and Progressive Networks.
"""
from .enums import CLStrategy, ReplayStrategy, MemoryType
from .config import ContinualLearningConfig
from .ewc import EWC
from .replay import ReplayBuffer
from .progressive import ProgressiveNetwork
from .multitask import MultiTaskLearner
from .lifelong import LifelongLearner
from .system import CLTrainer

# Compatibility aliases
ContinualLearner = CLTrainer
ContinualConfig = ContinualLearningConfig

# Factory functions
def create_cl_config(**kwargs) -> ContinualLearningConfig:
    return ContinualLearningConfig(**kwargs)

def create_cl_trainer(config: ContinualLearningConfig) -> CLTrainer:
    return CLTrainer(config)

__all__ = [
    'CLStrategy',
    'ReplayStrategy',
    'MemoryType',
    'ContinualLearningConfig',
    'EWC',
    'ReplayBuffer',
    'ProgressiveNetwork',
    'MultiTaskLearner',
    'LifelongLearner',
    'CLTrainer',
    'create_cl_config',
    'create_cl_trainer'
]

