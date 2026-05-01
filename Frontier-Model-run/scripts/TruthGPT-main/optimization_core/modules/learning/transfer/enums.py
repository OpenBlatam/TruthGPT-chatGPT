"""
Transfer Learning Enums
=======================

Enums for transfer strategies and methods.
"""
from enum import Enum

class TransferStrategy(Enum):
    """Transfer learning strategies"""
    FINE_TUNING = "fine_tuning"
    FEATURE_EXTRACTION = "feature_extraction"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    DOMAIN_ADAPTATION = "domain_adaptation"
    MULTI_TASK_ADAPTER = "multi_task_adapter"
    PROGRESSIVE_TRANSFER = "progressive_transfer"
    GRADIENT_REVERSAL = "gradient_reversal"
    ADVERSARIAL_DOMAIN_ADAPTATION = "adversarial_domain_adaptation"

class DomainAdaptationMethod(Enum):
    """Domain adaptation methods"""
    DANN = "dann"
    CORAL = "coral"
    MMD = "mmd"
    ADDA = "adda"
    CYCLE_GAN = "cycle_gan"
    STARGAN = "stargan"
    UNIT = "unit"
    MUNIT = "munit"

class KnowledgeDistillationType(Enum):
    """Knowledge distillation types"""
    SOFT_DISTILLATION = "soft_distillation"
    HARD_DISTILLATION = "hard_distillation"
    FEATURE_DISTILLATION = "feature_distillation"
    ATTENTION_DISTILLATION = "attention_distillation"
    RELATION_DISTILLATION = "relation_distillation"
    SELF_DISTILLATION = "self_distillation"

