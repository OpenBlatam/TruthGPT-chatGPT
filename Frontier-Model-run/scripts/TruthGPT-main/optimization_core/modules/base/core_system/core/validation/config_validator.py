"""
Configuration validation utilities.
"""
import logging
from typing import Any, Dict

from .validator import Validator, ValidationResult

logger = logging.getLogger(__name__)


class ConfigValidator(Validator):
    """Validator for configuration dictionaries."""
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate configuration.
        
        Args:
            data: Configuration dictionary
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            errors.append("Configuration must be a dictionary")
            return self._create_result(errors)
        
        required_keys = kwargs.get("required_keys", [])
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")
        
        # Validate nested configs
        if "model" in data:
            model_result = self._validate_model_config(data["model"])
            errors.extend(model_result.errors)
            warnings.extend(model_result.warnings)
        
        if "training" in data:
            train_result = self._validate_training_config(data["training"])
            errors.extend(train_result.errors)
            warnings.extend(train_result.warnings)
        
        if "data" in data:
            data_result = self._validate_data_config(data["data"])
            errors.extend(data_result.errors)
            warnings.extend(data_result.warnings)
        
        return self._create_result(errors, warnings)
    
    def _validate_model_config(self, config: Dict) -> ValidationResult:
        """Validate model configuration."""
        errors = []
        
        if "name_or_path" not in config:
            errors.append("model.name_or_path is required")
        
        if "lora" in config and config["lora"].get("enabled", False):
            lora_config = config["lora"]
            if "r" not in lora_config:
                errors.append("model.lora.r is required when LoRA is enabled")
            if "alpha" not in lora_config:
                errors.append("model.lora.alpha is required when LoRA is enabled")
        
        return self._create_result(errors)
    
    def _validate_training_config(self, config: Dict) -> ValidationResult:
        """Validate training configuration."""
        errors = []
        
        if "epochs" in config and config["epochs"] < 1:
            errors.append("training.epochs must be >= 1")
        
        if "learning_rate" in config and config["learning_rate"] <= 0:
            errors.append("training.learning_rate must be > 0")
        
        if "train_batch_size" in config and config["train_batch_size"] < 1:
            errors.append("training.train_batch_size must be >= 1")
        
        return self._create_result(errors)
    
    def _validate_data_config(self, config: Dict) -> ValidationResult:
        """Validate data configuration."""
        errors = []
        
        if "dataset" not in config:
            errors.append("data.dataset is required")
        
        return self._create_result(errors)



