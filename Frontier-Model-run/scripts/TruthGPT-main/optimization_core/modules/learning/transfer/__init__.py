"""
Transfer Learning Package
=========================

Advanced transfer learning with fine-tuning, distillation, and domain adaptation.
"""
from .enums import TransferStrategy, DomainAdaptationMethod, KnowledgeDistillationType
from .config import TransferLearningConfig
from .fine_tuning import FineTuner
from .feature_extraction import FeatureExtractor
from .distillation import KnowledgeDistiller
from .adaptation import DomainAdapter
from .multitask_adapter import MultiTaskAdapter
from .system import TransferTrainer

# Compatibility aliases
TransferLearningManager = TransferTrainer

# Factory functions
def create_transfer_config(**kwargs) -> TransferLearningConfig:
    return TransferLearningConfig(**kwargs)

def create_fine_tuner(config: TransferLearningConfig) -> FineTuner:
    return FineTuner(config)

def create_feature_extractor(config: TransferLearningConfig) -> FeatureExtractor:
    return FeatureExtractor(config)

def create_knowledge_distiller(config: TransferLearningConfig) -> KnowledgeDistiller:
    return KnowledgeDistiller(config)

def create_domain_adapter(config: TransferLearningConfig) -> DomainAdapter:
    return DomainAdapter(config)

def create_transfer_trainer(config: TransferLearningConfig) -> TransferTrainer:
    return TransferTrainer(config)

__all__ = [
    'TransferStrategy',
    'DomainAdaptationMethod',
    'KnowledgeDistillationType',
    'TransferLearningConfig',
    'FineTuner',
    'FeatureExtractor',
    'KnowledgeDistiller',
    'DomainAdapter',
    'MultiTaskAdapter',
    'TransferTrainer',
    'create_transfer_config',
    'create_fine_tuner',
    'create_feature_extractor',
    'create_knowledge_distiller',
    'create_domain_adapter',
    'create_transfer_trainer'
]

