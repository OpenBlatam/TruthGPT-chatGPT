"""
Transfer Learning Configuration
===============================

Configuration for transfer learning systems.
"""
from dataclasses import dataclass, field
from typing import List
from .enums import TransferStrategy, DomainAdaptationMethod, KnowledgeDistillationType

@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning system"""
    # Basic settings
    transfer_strategy: TransferStrategy = TransferStrategy.FINE_TUNING
    domain_adaptation_method: DomainAdaptationMethod = DomainAdaptationMethod.DANN
    distillation_type: KnowledgeDistillationType = KnowledgeDistillationType.SOFT_DISTILLATION
    
    # Model settings
    source_model_path: str = ""
    target_model_path: str = ""
    feature_dim: int = 2048
    num_classes: int = 1000
    
    # Fine-tuning settings
    learning_rate: float = 0.001
    fine_tune_layers: int = 3
    freeze_backbone: bool = False
    gradual_unfreezing: bool = True
    
    # Domain adaptation settings
    domain_loss_weight: float = 1.0
    adversarial_weight: float = 0.1
    adaptation_layers: List[str] = field(default_factory=lambda: ["fc"])
    
    # Knowledge distillation settings
    temperature: float = 3.0
    alpha: float = 0.7
    beta: float = 0.3
    
    # Advanced features
    enable_curriculum_learning: bool = True
    enable_meta_learning: bool = False
    enable_few_shot_learning: bool = True
    
    def __post_init__(self):
        """Validate transfer learning configuration"""
        if self.feature_dim <= 0:
            raise ValueError("Feature dimension must be positive")
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.fine_tune_layers <= 0:
            raise ValueError("Fine-tune layers must be positive")
        if self.domain_loss_weight <= 0:
            raise ValueError("Domain loss weight must be positive")
        if self.adversarial_weight <= 0:
            raise ValueError("Adversarial weight must be positive")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not (0 <= self.alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")
        if not (0 <= self.beta <= 1):
            raise ValueError("Beta must be between 0 and 1")

