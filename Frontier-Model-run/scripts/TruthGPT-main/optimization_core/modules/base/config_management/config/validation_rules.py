"""
Validation Rules for TruthGPT Optimization Core
Defines validation rules and schemas for configuration validation
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None

@dataclass
class ConfigValidationRule:
    """Base class for configuration validation rules."""
    
    name: str
    description: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    validator: Optional[Callable] = None
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against this rule."""
        if self.validator is None:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Rule '{self.name}' has no validator"
            )
        
        try:
            result = self.validator(config)
            if isinstance(result, bool):
                return ValidationResult(
                    is_valid=result,
                    severity=self.severity,
                    message=f"Rule '{self.name}' {'passed' if result else 'failed'}"
                )
            elif isinstance(result, ValidationResult):
                return result
            else:
                return ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Rule '{self.name}' completed"
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Rule '{self.name}' failed with error: {str(e)}"
            )

class OptimizationValidationRule(ConfigValidationRule):
    """Validation rule for optimization configuration."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description, **kwargs)
        self.validator = self._validate_optimization
    
    def _validate_optimization(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate optimization configuration."""
        optimization = config.get('optimization', {})
        
        # Check learning rate
        learning_rate = optimization.get('learning_rate', 1e-4)
        if learning_rate <= 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Learning rate must be positive",
                field="optimization.learning_rate",
                suggested_fix="Set learning_rate to a positive value (e.g., 1e-4)"
            )
        
        # Check weight decay
        weight_decay = optimization.get('weight_decay', 0.01)
        if weight_decay < 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Weight decay must be non-negative",
                field="optimization.weight_decay",
                suggested_fix="Set weight_decay to a non-negative value (e.g., 0.01)"
            )
        
        # Check beta values
        beta1 = optimization.get('beta1', 0.9)
        beta2 = optimization.get('beta2', 0.999)
        
        if not 0 <= beta1 < 1:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Beta1 must be between 0 and 1",
                field="optimization.beta1",
                suggested_fix="Set beta1 to a value between 0 and 1 (e.g., 0.9)"
            )
        
        if not 0 <= beta2 < 1:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Beta2 must be between 0 and 1",
                field="optimization.beta2",
                suggested_fix="Set beta2 to a value between 0 and 1 (e.g., 0.999)"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Optimization configuration is valid"
        )

class ModelValidationRule(ConfigValidationRule):
    """Validation rule for model configuration."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description, **kwargs)
        self.validator = self._validate_model
    
    def _validate_model(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate model configuration."""
        model = config.get('model', {})
        
        # Check required fields
        required_fields = ['d_model', 'n_heads', 'n_layers', 'd_ff', 'vocab_size']
        missing_fields = [field for field in required_fields if field not in model]
        
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                field="model",
                suggested_fix=f"Add missing fields: {', '.join(missing_fields)}"
            )
        
        # Check d_model divisibility
        d_model = model.get('d_model', 512)
        n_heads = model.get('n_heads', 8)
        
        if d_model % n_heads != 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="d_model must be divisible by n_heads",
                field="model.d_model",
                suggested_fix=f"Set d_model to a multiple of {n_heads} (e.g., {n_heads * 64})"
            )
        
        # Check positive values
        positive_fields = ['d_model', 'n_heads', 'n_layers', 'd_ff', 'vocab_size', 'max_seq_length']
        for field in positive_fields:
            if field in model and model[field] <= 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field} must be positive",
                    field=f"model.{field}",
                    suggested_fix=f"Set {field} to a positive value"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Model configuration is valid"
        )

class TrainingValidationRule(ConfigValidationRule):
    """Validation rule for training configuration."""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description, **kwargs)
        self.validator = self._validate_training
    
    def _validate_training(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration."""
        training = config.get('training', {})
        
        # Check batch size
        batch_size = training.get('batch_size', 32)
        if batch_size <= 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Batch size must be positive",
                field="training.batch_size",
                suggested_fix="Set batch_size to a positive value (e.g., 32)"
            )
        
        # Check epochs
        num_epochs = training.get('num_epochs', 10)
        if num_epochs <= 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Number of epochs must be positive",
                field="training.num_epochs",
                suggested_fix="Set num_epochs to a positive value (e.g., 10)"
            )
        
        # Check data splits
        train_split = training.get('train_split', 0.8)
        val_split = training.get('val_split', 0.1)
        test_split = training.get('test_split', 0.1)
        
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Data splits must sum to 1.0, got {total_split}",
                field="training",
                suggested_fix="Adjust train_split, val_split, and test_split to sum to 1.0"
            )
        
        # Check split ranges
        for split_name, split_value in [('train_split', train_split), ('val_split', val_split), ('test_split', test_split)]:
            if not 0 <= split_value <= 1:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{split_name} must be between 0 and 1",
                    field=f"training.{split_name}",
                    suggested_fix=f"Set {split_name} to a value between 0 and 1"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Training configuration is valid"
        )

def create_validation_rules() -> List[ConfigValidationRule]:
    """Create default validation rules."""
    rules = [
        OptimizationValidationRule(
            name="optimization_validation",
            description="Validate optimization configuration parameters"
        ),
        ModelValidationRule(
            name="model_validation",
            description="Validate model configuration parameters"
        ),
        TrainingValidationRule(
            name="training_validation",
            description="Validate training configuration parameters"
        )
    ]
    
    return rules

def validate_config_with_rules(
    config: Dict[str, Any], 
    rules: List[ConfigValidationRule]
) -> List[ValidationResult]:
    """Validate configuration against a list of rules."""
    results = []
    
    for rule in rules:
        result = rule.validate(config)
        results.append(result)
        
        if result.severity == ValidationSeverity.ERROR and not result.is_valid:
            logger.error(f"Validation failed: {result.message}")
        elif result.severity == ValidationSeverity.WARNING and not result.is_valid:
            logger.warning(f"Validation warning: {result.message}")
        else:
            logger.info(f"Validation passed: {result.message}")
    
    return results

def get_validation_summary(results: List[ValidationResult]) -> Dict[str, Any]:
    """Get summary of validation results."""
    total_rules = len(results)
    passed_rules = sum(1 for r in results if r.is_valid)
    failed_rules = total_rules - passed_rules
    
    error_count = sum(1 for r in results if r.severity == ValidationSeverity.ERROR and not r.is_valid)
    warning_count = sum(1 for r in results if r.severity == ValidationSeverity.WARNING and not r.is_valid)
    
    return {
        'total_rules': total_rules,
        'passed_rules': passed_rules,
        'failed_rules': failed_rules,
        'error_count': error_count,
        'warning_count': warning_count,
        'is_valid': error_count == 0,
        'results': results
    }





