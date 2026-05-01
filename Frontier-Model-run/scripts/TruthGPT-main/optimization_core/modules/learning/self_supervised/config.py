"""
Self-Supervised Learning Configuration
======================================

Configuration for self-supervised learning systems.
"""
from dataclasses import dataclass, field
from .enums import SSLMethod, PretextTaskType, ContrastiveLossType

@dataclass
class SSLConfig:
    """Configuration for self-supervised learning system"""
    # Basic settings
    ssl_method: SSLMethod = SSLMethod.SIMCLR
    pretext_task: PretextTaskType = PretextTaskType.CONTRASTIVE_LEARNING
    contrastive_loss: ContrastiveLossType = ContrastiveLossType.INFO_NCE
    
    # Model settings
    encoder_dim: int = 2048
    projection_dim: int = 128
    hidden_dim: int = 512
    
    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 256
    num_epochs: int = 200
    temperature: float = 0.07
    
    # Data augmentation
    enable_augmentation: bool = True
    augmentation_strength: float = 0.5
    num_views: int = 2
    
    # Contrastive learning
    enable_momentum: bool = True
    momentum: float = 0.999
    enable_memory_bank: bool = True
    memory_bank_size: int = 65536
    
    # Advanced features
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_distributed_training: bool = False
    
    def __post_init__(self):
        """Validate SSL configuration"""
        if self.encoder_dim <= 0:
            raise ValueError("Encoder dimension must be positive")
        if self.projection_dim <= 0:
            raise ValueError("Projection dimension must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not (0 <= self.augmentation_strength <= 1):
            raise ValueError("Augmentation strength must be between 0 and 1")
        if self.num_views <= 0:
            raise ValueError("Number of views must be positive")
        if not (0 <= self.momentum <= 1):
            raise ValueError("Momentum must be between 0 and 1")
        if self.memory_bank_size <= 0:
            raise ValueError("Memory bank size must be positive")

