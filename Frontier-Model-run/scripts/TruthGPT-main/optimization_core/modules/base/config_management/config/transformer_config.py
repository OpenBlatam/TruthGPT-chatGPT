"""
Transformer Configuration for TruthGPT Optimization Core
Defines configuration classes for transformer models with validation and type safety
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch

logger = logging.getLogger(__name__)

class AttentionType(Enum):
    """Types of attention mechanisms."""
    STANDARD = "standard"
    FLASH = "flash"
    MEMORY_EFFICIENT = "memory_efficient"
    SPARSE = "sparse"
    LOCAL = "local"

class ActivationType(Enum):
    """Types of activation functions."""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    SWIGLU = "swiglu"
    SILU = "silu"

class NormalizationType(Enum):
    """Types of normalization layers."""
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    GROUP_NORM = "group_norm"

class PositionalEncodingType(Enum):
    """Types of positional encodings."""
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROTARY = "rotary"
    ALIBI = "alibi"
    RELATIVE = "relative"

@dataclass
class ModelConfig:
    """Configuration for transformer model architecture."""
    
    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    vocab_size: int = 50000
    max_seq_length: int = 1024
    
    # Attention configuration
    attention_type: AttentionType = AttentionType.STANDARD
    use_flash_attention: bool = False
    attention_dropout: float = 0.1
    use_causal_mask: bool = True
    
    # Feed-forward configuration
    activation_type: ActivationType = ActivationType.GELU
    ff_dropout: float = 0.1
    use_gated_mlp: bool = False
    
    # Normalization configuration
    normalization_type: NormalizationType = NormalizationType.LAYER_NORM
    layer_norm_eps: float = 1e-5
    use_pre_norm: bool = True
    
    # Positional encoding configuration
    positional_encoding_type: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL
    use_rotary_embeddings: bool = False
    rotary_embedding_dim: Optional[int] = None
    
    # Advanced features
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = False
    use_activation_checkpointing: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate model configuration parameters."""
        errors = []
        
        # Validate dimensions
        if self.d_model <= 0:
            errors.append("d_model must be positive")
        
        if self.n_heads <= 0:
            errors.append("n_heads must be positive")
        
        if self.d_model % self.n_heads != 0:
            errors.append("d_model must be divisible by n_heads")
        
        if self.n_layers <= 0:
            errors.append("n_layers must be positive")
        
        if self.d_ff <= 0:
            errors.append("d_ff must be positive")
        
        if self.vocab_size <= 0:
            errors.append("vocab_size must be positive")
        
        if self.max_seq_length <= 0:
            errors.append("max_seq_length must be positive")
        
        # Validate dropout rates
        if not 0 <= self.attention_dropout <= 1:
            errors.append("attention_dropout must be between 0 and 1")
        
        if not 0 <= self.ff_dropout <= 1:
            errors.append("ff_dropout must be between 0 and 1")
        
        # Validate rotary embedding dimension
        if self.use_rotary_embeddings and self.rotary_embedding_dim is None:
            self.rotary_embedding_dim = self.d_model // self.n_heads
        
        if errors:
            raise ValueError(f"Model configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'attention_type': self.attention_type.value,
            'use_flash_attention': self.use_flash_attention,
            'attention_dropout': self.attention_dropout,
            'use_causal_mask': self.use_causal_mask,
            'activation_type': self.activation_type.value,
            'ff_dropout': self.ff_dropout,
            'use_gated_mlp': self.use_gated_mlp,
            'normalization_type': self.normalization_type.value,
            'layer_norm_eps': self.layer_norm_eps,
            'use_pre_norm': self.use_pre_norm,
            'positional_encoding_type': self.positional_encoding_type.value,
            'use_rotary_embeddings': self.use_rotary_embeddings,
            'rotary_embedding_dim': self.rotary_embedding_dim,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
            'use_mixed_precision': self.use_mixed_precision,
            'use_activation_checkpointing': self.use_activation_checkpointing
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        # Convert string enums back to enum types
        if 'attention_type' in config_dict and isinstance(config_dict['attention_type'], str):
            config_dict['attention_type'] = AttentionType(config_dict['attention_type'])
        
        if 'activation_type' in config_dict and isinstance(config_dict['activation_type'], str):
            config_dict['activation_type'] = ActivationType(config_dict['activation_type'])
        
        if 'normalization_type' in config_dict and isinstance(config_dict['normalization_type'], str):
            config_dict['normalization_type'] = NormalizationType(config_dict['normalization_type'])
        
        if 'positional_encoding_type' in config_dict and isinstance(config_dict['positional_encoding_type'], str):
            config_dict['positional_encoding_type'] = PositionalEncodingType(config_dict['positional_encoding_type'])
        
        return cls(**config_dict)

@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    
    # Learning rate configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Scheduler configuration
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Gradient configuration
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_gradient_clipping: bool = True
    
    # Mixed precision configuration
    use_amp: bool = False
    amp_opt_level: str = "O1"
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    use_activation_checkpointing: bool = False
    
    def __post_init__(self):
        """Validate optimization configuration."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate optimization configuration parameters."""
        errors = []
        
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.weight_decay < 0:
            errors.append("weight_decay must be non-negative")
        
        if not 0 <= self.beta1 < 1:
            errors.append("beta1 must be between 0 and 1")
        
        if not 0 <= self.beta2 < 1:
            errors.append("beta2 must be between 0 and 1")
        
        if self.eps <= 0:
            errors.append("eps must be positive")
        
        if self.warmup_steps < 0:
            errors.append("warmup_steps must be non-negative")
        
        if self.max_steps <= 0:
            errors.append("max_steps must be positive")
        
        if self.max_grad_norm <= 0:
            errors.append("max_grad_norm must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")
        
        if errors:
            raise ValueError(f"Optimization configuration validation failed: {'; '.join(errors)}")

@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Data configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    def __post_init__(self):
        """Validate training configuration."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate training configuration parameters."""
        errors = []
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        
        if self.eval_steps <= 0:
            errors.append("eval_steps must be positive")
        
        if self.save_steps <= 0:
            errors.append("save_steps must be positive")
        
        if self.logging_steps <= 0:
            errors.append("logging_steps must be positive")
        
        # Validate data splits
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            errors.append(f"Data splits must sum to 1.0, got {total_split}")
        
        if any(split < 0 or split > 1 for split in [self.train_split, self.val_split, self.test_split]):
            errors.append("Data splits must be between 0 and 1")
        
        if self.save_total_limit <= 0:
            errors.append("save_total_limit must be positive")
        
        if self.early_stopping_patience <= 0:
            errors.append("early_stopping_patience must be positive")
        
        if self.early_stopping_threshold <= 0:
            errors.append("early_stopping_threshold must be positive")
        
        if errors:
            raise ValueError(f"Training configuration validation failed: {'; '.join(errors)}")

@dataclass
class TransformerConfig:
    """Complete transformer configuration combining all components."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Metadata
    name: str = "transformer"
    version: str = "1.0.0"
    description: str = "TruthGPT Transformer Configuration"
    
    def __post_init__(self):
        """Validate complete configuration."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate complete transformer configuration."""
        # Individual configurations are validated in their __post_init__
        # Additional cross-configuration validation can be added here
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete configuration to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'model': self.model.to_dict(),
            'optimization': self.optimization.__dict__,
            'training': self.training.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create complete configuration from dictionary."""
        model_config = ModelConfig.from_dict(config_dict.get('model', {}))
        optimization_config = OptimizationConfig(**config_dict.get('optimization', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            model=model_config,
            optimization=optimization_config,
            training=training_config,
            name=config_dict.get('name', 'transformer'),
            version=config_dict.get('version', '1.0.0'),
            description=config_dict.get('description', 'TruthGPT Transformer Configuration')
        )

# Factory functions
def create_transformer_config(**kwargs) -> TransformerConfig:
    """Create a new transformer configuration."""
    return TransformerConfig(**kwargs)

def create_optimization_config(**kwargs) -> OptimizationConfig:
    """Create a new optimization configuration."""
    return OptimizationConfig(**kwargs)

def create_training_config(**kwargs) -> TrainingConfig:
    """Create a new training configuration."""
    return TrainingConfig(**kwargs)

def create_model_config(**kwargs) -> ModelConfig:
    """Create a new model configuration."""
    return ModelConfig(**kwargs)





