"""
Configuration loader with validation and best practices
Following deep learning configuration management patterns
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import torch
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """Model configuration with validation"""
    name: str = "truthgpt-optimized"
    architecture: str = "transformer"
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12
    max_sequence_length: int = 2048
    vocab_size: int = 50257
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    def __post_init__(self):
        """Validate model configuration"""
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_attention_heads > 0, "num_attention_heads must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert 0 <= self.attention_dropout <= 1, "attention_dropout must be between 0 and 1"


@dataclass
class TrainingConfig:
    """Training configuration with validation"""
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization techniques
    use_mixed_precision: bool = True
    use_gradient_clipping: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Learning rate scheduling
    lr_scheduler: str = "linear_with_warmup"
    lr_decay_factor: float = 0.1
    lr_patience: int = 5
    
    def __post_init__(self):
        """Validate training configuration"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"


@dataclass
class HardwareConfig:
    """Hardware configuration with auto-detection"""
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    use_cuda_graphs: bool = False
    
    # Multi-GPU settings
    distributed: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "backend": "nccl",
        "world_size": 1,
        "rank": 0
    })
    
    def __post_init__(self):
        """Auto-detect device if set to auto"""
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    log_level: str = "INFO"
    
    # Experiment tracking
    experiment_name: str = "truthgpt_optimization"
    project_name: str = "truthgpt"
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10
    
    def __post_init__(self):
        """Create checkpoint directory"""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class OptimizationConfig:
    """Advanced optimization configuration"""
    # Memory optimizations
    memory: Dict[str, bool] = field(default_factory=lambda: {
        "use_gradient_checkpointing": True,
        "use_activation_checkpointing": True,
        "use_memory_efficient_attention": True
    })
    
    # Speed optimizations
    speed: Dict[str, Any] = field(default_factory=lambda: {
        "use_compile": True,
        "use_torch_script": False,
        "use_onnx": False
    })
    
    # Advanced techniques
    advanced: Dict[str, Any] = field(default_factory=lambda: {
        "use_lora": False,
        "lora_rank": 16,
        "lora_alpha": 32,
        "use_qlora": False,
        "quantization_bits": 4,
        "use_prompt_tuning": False,
        "prompt_length": 20
    })


@dataclass
class FullConfig:
    """Complete configuration with all components"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Additional configurations
    data: Dict[str, Any] = field(default_factory=lambda: {
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True
    })
    
    evaluation: Dict[str, Any] = field(default_factory=lambda: {
        "metrics": ["perplexity", "bleu", "rouge", "accuracy"],
        "generation": {
            "max_length": 100,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "num_beams": 1,
            "early_stopping": True
        }
    })
    
    security: Dict[str, Any] = field(default_factory=lambda: {
        "max_sequence_length": 4096,
        "max_batch_size": 64,
        "timeout_seconds": 300,
        "validate_inputs": True,
        "sanitize_outputs": True
    })


class ConfigLoader:
    """Configuration loader with validation and error handling"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = config_path
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for configuration loader"""
        logger = logging.getLogger("ConfigLoader")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_yaml_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file with validation"""
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading YAML config: {e}")
            raise
    
    def load_json_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file with validation"""
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading JSON config: {e}")
            raise
    
    def create_config_from_dict(self, config_dict: Dict[str, Any]) -> FullConfig:
        """Create configuration object from dictionary"""
        try:
            # Create nested configuration objects
            model_config = ModelConfig(**config_dict.get('model', {}))
            training_config = TrainingConfig(**config_dict.get('training', {}))
            hardware_config = HardwareConfig(**config_dict.get('hardware', {}))
            logging_config = LoggingConfig(**config_dict.get('logging', {}))
            optimization_config = OptimizationConfig(**config_dict.get('optimization', {}))
            
            # Create full configuration
            full_config = FullConfig(
                model=model_config,
                training=training_config,
                hardware=hardware_config,
                logging=logging_config,
                optimization=optimization_config
            )
            
            # Add additional configurations
            full_config.data = config_dict.get('data', full_config.data)
            full_config.evaluation = config_dict.get('evaluation', full_config.evaluation)
            full_config.security = config_dict.get('security', full_config.security)
            
            self.logger.info("Configuration created successfully")
            return full_config
            
        except Exception as e:
            self.logger.error(f"Error creating configuration: {e}")
            raise
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> FullConfig:
        """Load configuration with automatic format detection"""
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            self.logger.warning("No config path provided, using default configuration")
            return FullConfig()
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format and load accordingly
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_dict = self.load_yaml_config(config_path)
        elif config_path.suffix.lower() == '.json':
            config_dict = self.load_json_config(config_path)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        return self.create_config_from_dict(config_dict)
    
    def save_config(self, config: FullConfig, save_path: Union[str, Path], 
                   format: str = "yaml") -> None:
        """Save configuration to file"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            config_dict = self._config_to_dict(config)
            
            if format.lower() == "yaml":
                with open(save_path, 'w', encoding='utf-8') as file:
                    yaml.dump(config_dict, file, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(save_path, 'w', encoding='utf-8') as file:
                    json.dump(config_dict, file, indent=2)
            else:
                raise ValueError(f"Unsupported save format: {format}")
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def _config_to_dict(self, config: FullConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        return {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'hardware': config.hardware.__dict__,
            'logging': config.logging.__dict__,
            'optimization': config.optimization.__dict__,
            'data': config.data,
            'evaluation': config.evaluation,
            'security': config.security
        }
    
    def validate_config(self, config: FullConfig) -> bool:
        """Validate configuration for consistency"""
        try:
            # Validate model configuration
            assert config.model.hidden_size % config.model.num_attention_heads == 0, \
                "hidden_size must be divisible by num_attention_heads"
            
            # Validate training configuration
            assert config.training.batch_size > 0, "batch_size must be positive"
            assert 0 < config.training.learning_rate < 1, "learning_rate must be between 0 and 1"
            
            # Validate hardware configuration
            assert config.hardware.num_workers >= 0, "num_workers must be non-negative"
            
            # Validate logging configuration
            assert config.logging.log_interval > 0, "log_interval must be positive"
            assert config.logging.save_interval > 0, "save_interval must be positive"
            
            self.logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False


# Hydra configuration setup
cs = ConfigStore.instance()
cs.store(name="config", node=FullConfig)


@hydra.main(version_base=None, config_path=".", config_name="optimization_config")
def load_hydra_config(cfg: FullConfig) -> FullConfig:
    """Load configuration using Hydra"""
    return cfg


def get_config(config_path: Optional[Union[str, Path]] = None) -> FullConfig:
    """Get configuration with automatic loading"""
    loader = ConfigLoader(config_path)
    return loader.load_config()


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = get_config("config/optimization_config.yaml")
    
    # Validate configuration
    loader = ConfigLoader()
    if loader.validate_config(config):
        print("Configuration is valid")
    else:
        print("Configuration validation failed")



