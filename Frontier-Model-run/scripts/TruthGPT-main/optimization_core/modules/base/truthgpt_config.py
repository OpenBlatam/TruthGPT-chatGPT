"""
TruthGPT Advanced Configuration Module
Advanced configuration management for TruthGPT models with validation and optimization
"""

import yaml
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
import warnings
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTBaseConfig:
    """Base configuration class for TruthGPT components."""
    name: str = "truthgpt"
    version: str = "1.0.0"
    description: str = "TruthGPT Advanced Configuration"
    
    # Environment settings
    device: str = "auto"  # auto, cpu, cuda, cuda:0, etc.
    precision: str = "fp16"  # fp32, fp16, bf16
    deterministic: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_wandb: bool = False
    wandb_project: str = "truthgpt"
    
    # Performance settings
    enable_optimization: bool = True
    enable_profiling: bool = False
    enable_monitoring: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration."""
        return True
    
    def merge(self, other: 'TruthGPTBaseConfig') -> 'TruthGPTBaseConfig':
        """Merge with another configuration."""
        for key, value in other.to_dict().items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return self

@dataclass
class TruthGPTModelConfig(TruthGPTBaseConfig):
    """Advanced model configuration for TruthGPT."""
    # Architecture
    model_type: str = "transformer"
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    
    # Attention configuration
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    attention_type: str = "multi_head"  # multi_head, sparse, local, flash
    use_rotary_embeddings: bool = True
    use_relative_position: bool = False
    
    # Activation and normalization
    activation_function: str = "gelu"  # gelu, relu, swish, silu, mish
    layer_norm_eps: float = 1e-5
    use_layer_norm: bool = True
    
    # Advanced features
    enable_gradient_checkpointing: bool = True
    enable_memory_efficient_attention: bool = True
    enable_flash_attention: bool = False
    enable_quantization: bool = False
    quantization_bits: int = 8
    
    # Initialization
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # Model compression
    enable_pruning: bool = False
    pruning_ratio: float = 0.1
    enable_distillation: bool = False
    teacher_model_path: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate model configuration."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        
        if self.intermediate_size < self.hidden_size:
            raise ValueError(f"intermediate_size ({self.intermediate_size}) must be >= hidden_size ({self.hidden_size})")
        
        if self.attention_dropout < 0 or self.attention_dropout > 1:
            raise ValueError(f"attention_dropout ({self.attention_dropout}) must be between 0 and 1")
        
        if self.hidden_dropout < 0 or self.hidden_dropout > 1:
            raise ValueError(f"hidden_dropout ({self.hidden_dropout}) must be between 0 and 1")
        
        return True

@dataclass
class TruthGPTTrainingConfig(TruthGPTBaseConfig):
    """Advanced training configuration for TruthGPT."""
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    max_steps: int = 100000
    
    # Optimization
    optimizer_type: str = "adamw"  # adam, adamw, sgd, rmsprop, adagrad, lamb
    scheduler_type: str = "cosine"  # linear, cosine, exponential, step, plateau, onecycle
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    
    # Advanced optimization
    enable_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    enable_dynamic_loss_scaling: bool = True
    
    # Learning rate scheduling
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 10
    lr_decay_threshold: float = 0.001
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Advanced features
    enable_gradient_checkpointing: bool = True
    enable_attention_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_compilation: bool = False
    
    # Distributed training
    enable_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    def validate(self) -> bool:
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate ({self.learning_rate}) must be > 0")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be > 0")
        
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs ({self.num_epochs}) must be > 0")
        
        if self.gradient_clip_norm < 0:
            raise ValueError(f"gradient_clip_norm ({self.gradient_clip_norm}) must be >= 0")
        
        return True

@dataclass
class TruthGPTDataConfig(TruthGPTBaseConfig):
    """Advanced data configuration for TruthGPT."""
    # Data paths
    data_path: str = ""
    tokenizer_name: str = "gpt2"
    cache_dir: str = "./cache"
    
    # Data processing
    max_sequence_length: int = 2048
    batch_size: int = 32
    enable_tokenization: bool = True
    enable_padding: bool = True
    enable_truncation: bool = True
    padding_side: str = "right"
    
    # Data augmentation
    enable_augmentation: bool = False
    augmentation_probability: float = 0.1
    augmentation_types: List[str] = field(default_factory=lambda: ["shuffle", "mask", "replace", "insert", "delete"])
    
    # Data splitting
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Advanced features
    enable_caching: bool = True
    enable_streaming: bool = False
    streaming_chunk_size: int = 1000
    enable_parallel_processing: bool = True
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    def validate(self) -> bool:
        """Validate data configuration."""
        if not (0 < self.train_split < 1):
            raise ValueError(f"train_split ({self.train_split}) must be between 0 and 1")
        
        if not (0 < self.val_split < 1):
            raise ValueError(f"val_split ({self.val_split}) must be between 0 and 1")
        
        if not (0 < self.test_split < 1):
            raise ValueError(f"test_split ({self.test_split}) must be between 0 and 1")
        
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must equal 1.0")
        
        return True

@dataclass
class TruthGPTInferenceConfig(TruthGPTBaseConfig):
    """Advanced inference configuration for TruthGPT."""
    # Generation settings
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    # Advanced generation
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    no_repeat_ngram_size: int = 3
    
    # Beam search
    enable_beam_search: bool = False
    beam_size: int = 4
    length_penalty: float = 1.0
    diversity_penalty: float = 0.0
    
    # Performance optimization
    enable_mixed_precision: bool = True
    enable_optimization: bool = True
    enable_compilation: bool = False
    batch_size: int = 1
    
    # Memory optimization
    enable_gradient_checkpointing: bool = False
    enable_memory_efficient_attention: bool = True
    
    def validate(self) -> bool:
        """Validate inference configuration."""
        if self.temperature <= 0:
            raise ValueError(f"temperature ({self.temperature}) must be > 0")
        
        if self.top_k <= 0:
            raise ValueError(f"top_k ({self.top_k}) must be > 0")
        
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p ({self.top_p}) must be between 0 and 1")
        
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty ({self.repetition_penalty}) must be > 0")
        
        return True

class TruthGPTConfigManager:
    """Advanced configuration manager for TruthGPT."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.configs = {}
        self.config_validators = {}
        
        # Register default validators
        self._register_default_validators()
    
    def _register_default_validators(self) -> None:
        """Register default configuration validators."""
        self.config_validators = {
            'model': TruthGPTModelConfig,
            'training': TruthGPTTrainingConfig,
            'data': TruthGPTDataConfig,
            'inference': TruthGPTInferenceConfig
        }
    
    def create_config(self, config_type: str, **kwargs) -> TruthGPTBaseConfig:
        """Create a new configuration."""
        if config_type not in self.config_validators:
            raise ValueError(f"Unknown config type: {config_type}")
        
        config_class = self.config_validators[config_type]
        config = config_class(**kwargs)
        
        # Validate configuration
        if not config.validate():
            raise ValueError(f"Invalid configuration for {config_type}")
        
        self.configs[config_type] = config
        self.logger.info(f"Created {config_type} configuration")
        
        return config
    
    def load_config(self, filepath: str, config_type: str) -> TruthGPTBaseConfig:
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        # Load configuration based on file extension
        if filepath.suffix == '.yaml' or filepath.suffix == '.yml':
            with open(filepath, 'r') as f:
                config_data = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")
        
        # Create configuration
        config = self.create_config(config_type, **config_data)
        
        self.logger.info(f"Loaded {config_type} configuration from {filepath}")
        return config
    
    def save_config(self, config: TruthGPTBaseConfig, filepath: str, format: str = "yaml") -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = config.to_dict()
        
        if format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved configuration to {filepath}")
    
    def merge_configs(self, base_config: TruthGPTBaseConfig, override_config: TruthGPTBaseConfig) -> TruthGPTBaseConfig:
        """Merge two configurations."""
        merged_config = copy.deepcopy(base_config)
        merged_config.merge(override_config)
        
        # Validate merged configuration
        if not merged_config.validate():
            raise ValueError("Invalid merged configuration")
        
        self.logger.info("Merged configurations successfully")
        return merged_config
    
    def get_config(self, config_type: str) -> Optional[TruthGPTBaseConfig]:
        """Get configuration by type."""
        return self.configs.get(config_type)
    
    def list_configs(self) -> List[str]:
        """List all available configurations."""
        return list(self.configs.keys())
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all configurations."""
        results = {}
        for config_type, config in self.configs.items():
            try:
                results[config_type] = config.validate()
            except Exception as e:
                self.logger.error(f"Validation failed for {config_type}: {e}")
                results[config_type] = False
        
        return results
    
    def export_configs(self, output_dir: str, format: str = "yaml") -> None:
        """Export all configurations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for config_type, config in self.configs.items():
            filepath = output_dir / f"{config_type}_config.{format}"
            self.save_config(config, filepath, format)
        
        self.logger.info(f"Exported all configurations to {output_dir}")

class TruthGPTConfigValidator:
    """Advanced configuration validator for TruthGPT."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.validation_rules = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        self.validation_rules = {
            'model': {
                'hidden_size': lambda x: x > 0 and x % 64 == 0,
                'num_attention_heads': lambda x: x > 0 and x <= 64,
                'num_layers': lambda x: x > 0 and x <= 100,
                'vocab_size': lambda x: x > 0 and x <= 1000000
            },
            'training': {
                'learning_rate': lambda x: 1e-6 <= x <= 1e-1,
                'batch_size': lambda x: x > 0 and x <= 1024,
                'num_epochs': lambda x: x > 0 and x <= 1000,
                'weight_decay': lambda x: 0 <= x <= 1
            },
            'data': {
                'max_sequence_length': lambda x: x > 0 and x <= 10000,
                'batch_size': lambda x: x > 0 and x <= 1024,
                'num_workers': lambda x: x >= 0 and x <= 32
            },
            'inference': {
                'max_length': lambda x: x > 0 and x <= 1000,
                'temperature': lambda x: 0.1 <= x <= 2.0,
                'top_k': lambda x: x > 0 and x <= 1000,
                'top_p': lambda x: 0.1 <= x <= 1.0
            }
        }
    
    def validate_config(self, config: TruthGPTBaseConfig, config_type: str) -> Tuple[bool, List[str]]:
        """Validate configuration with detailed feedback."""
        errors = []
        
        if config_type not in self.validation_rules:
            errors.append(f"Unknown config type: {config_type}")
            return False, errors
        
        rules = self.validation_rules[config_type]
        config_dict = config.to_dict()
        
        for field, rule in rules.items():
            if field in config_dict:
                value = config_dict[field]
                if not rule(value):
                    errors.append(f"Invalid {field}: {value}")
        
        # Additional validation
        if hasattr(config, 'validate'):
            try:
                if not config.validate():
                    errors.append("Configuration validation failed")
            except Exception as e:
                errors.append(f"Configuration validation error: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def add_validation_rule(self, config_type: str, field: str, rule: Callable[[Any], bool]) -> None:
        """Add custom validation rule."""
        if config_type not in self.validation_rules:
            self.validation_rules[config_type] = {}
        
        self.validation_rules[config_type][field] = rule
        self.logger.info(f"Added validation rule for {config_type}.{field}")

# Factory functions
def create_truthgpt_config_manager() -> TruthGPTConfigManager:
    """Create TruthGPT configuration manager."""
    return TruthGPTConfigManager()

def create_truthgpt_config_validator() -> TruthGPTConfigValidator:
    """Create TruthGPT configuration validator."""
    return TruthGPTConfigValidator()

def load_truthgpt_config(filepath: str, config_type: str) -> TruthGPTBaseConfig:
    """Quick load TruthGPT configuration."""
    manager = create_truthgpt_config_manager()
    return manager.load_config(filepath, config_type)

def save_truthgpt_config(config: TruthGPTBaseConfig, filepath: str, format: str = "yaml") -> None:
    """Quick save TruthGPT configuration."""
    manager = create_truthgpt_config_manager()
    manager.save_config(config, filepath, format)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT configuration management
    print("🚀 TruthGPT Advanced Configuration Demo")
    print("=" * 50)
    
    # Create configuration manager
    manager = create_truthgpt_config_manager()
    
    # Create model configuration
    model_config = manager.create_config('model', 
        hidden_size=1024, 
        num_layers=24, 
        num_attention_heads=16
    )
    
    # Create training configuration
    training_config = manager.create_config('training',
        learning_rate=2e-4,
        batch_size=64,
        num_epochs=50
    )
    
    # Save configurations
    manager.save_config(model_config, "model_config.yaml")
    manager.save_config(training_config, "training_config.yaml")
    
    # Validate configurations
    validator = create_truthgpt_config_validator()
    is_valid, errors = validator.validate_config(model_config, 'model')
    
    print(f"Model config valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    print("✅ TruthGPT configuration management completed!")



