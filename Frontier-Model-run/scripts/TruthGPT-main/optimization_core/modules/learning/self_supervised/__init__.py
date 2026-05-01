"""
Self-Supervised Learning Package
================================

Advanced self-supervised learning with contrastive, generative, and pretext task methods.
"""
from .enums import SSLMethod, PretextTaskType, ContrastiveLossType
from .config import SSLConfig
from .contrastive import ContrastiveLearner
from .pretext import PretextTaskModel
from .generative import RepresentationLearner
from .momentum import MomentumEncoder, MemoryBank
from .system import SSLTrainer

# Compatibility aliases
SelfSupervisedTrainer = SSLTrainer
SelfSupervisedConfig = SSLConfig

# Factory functions
def create_ssl_config(**kwargs) -> SSLConfig:
    return SSLConfig(**kwargs)

def create_contrastive_learner(config: SSLConfig) -> ContrastiveLearner:
    return ContrastiveLearner(config)

def create_pretext_task_model(config: SSLConfig) -> PretextTaskModel:
    return PretextTaskModel(config)

def create_representation_learner(config: SSLConfig) -> RepresentationLearner:
    return RepresentationLearner(config)

def create_momentum_encoder(config: SSLConfig) -> MomentumEncoder:
    return MomentumEncoder(config)

def create_memory_bank(config: SSLConfig) -> MemoryBank:
    return MemoryBank(config)

def create_ssl_trainer(config: SSLConfig) -> SSLTrainer:
    return SSLTrainer(config)

__all__ = [
    'SSLMethod',
    'PretextTaskType',
    'ContrastiveLossType',
    'SSLConfig',
    'ContrastiveLearner',
    'PretextTaskModel',
    'RepresentationLearner',
    'MomentumEncoder',
    'MemoryBank',
    'SSLTrainer',
    'create_ssl_config',
    'create_contrastive_learner',
    'create_pretext_task_model',
    'create_representation_learner',
    'create_momentum_encoder',
    'create_memory_bank',
    'create_ssl_trainer'
]

