"""
Ultra-fast modular configuration system
Following deep learning best practices
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
from dataclasses import dataclass, field
import torch
from omegaconf import OmegaConf
import logging


@dataclass
class BaseConfig:
    """Base configuration with validation"""
    device: str = "auto"
    seed: int = 42
    debug: bool = False
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelConfig(BaseConfig):
    """Model-specific configuration"""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 2048
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12
    vocab_size: int = 50257
    dropout: float = 0.1


@dataclass
class TrainingConfig(BaseConfig):
    """Training-specific configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True


@dataclass
class OptimizationConfig(BaseConfig):
    """Optimization-specific configuration"""
    use_flash_attention: bool = True
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    use_quantization: bool = False
    quantization_bits: int = 8


class ConfigValidator:
    """Fast configuration validator"""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> bool:
        """Validate model configuration"""
        try:
            assert config.hidden_size > 0
            assert config.num_attention_heads > 0
            assert config.num_layers > 0
            assert 0 <= config.dropout <= 1
            return True
        except AssertionError:
            return False
    
    @staticmethod
    def validate_training_config(config: TrainingConfig) -> bool:
        """Validate training configuration"""
        try:
            assert config.batch_size > 0
            assert config.learning_rate > 0
            assert config.num_epochs > 0
            assert config.max_grad_norm > 0
            return True
        except AssertionError:
            return False


class ConfigLoader:
    """Ultra-fast configuration loader"""
    
    def __init__(self):
        self.logger = logging.getLogger("ConfigLoader")
    
    def load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading YAML: {e}")
            raise
    
    def load_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON configuration"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading JSON: {e}")
            raise
    
    def create_config(self, config_dict: Dict[str, Any], 
                     config_type: Type[BaseConfig]) -> BaseConfig:
        """Create configuration object from dictionary"""
        try:
            return config_type(**config_dict)
        except Exception as e:
            self.logger.error(f"Error creating config: {e}")
            raise
