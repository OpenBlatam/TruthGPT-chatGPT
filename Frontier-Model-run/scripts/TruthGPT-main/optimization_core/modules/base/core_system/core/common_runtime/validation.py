"""
Validation system - Refactored validation components
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class ValidationReport:
    """Validation report."""
    test_name: str
    result: ValidationResult
    message: str
    details: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp is None:
            import time
            self.timestamp = time.time()

class ModelValidator:
    """Validates PyTorch models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules: List[Callable[[nn.Module], ValidationReport]] = []
    
    def add_validation_rule(self, rule: Callable[[nn.Module], ValidationReport]):
        """Add a custom validation rule."""
        self.validation_rules.append(rule)
    
    def validate_model(self, model: nn.Module) -> List[ValidationReport]:
        """Validate a model using all registered rules."""
        reports = []
        
        # Basic validation
        basic_report = self._validate_basic_structure(model)
        reports.append(basic_report)
        
        # Custom validation rules
        for rule in self.validation_rules:
            try:
                report = rule(model)
                reports.append(report)
            except Exception as e:
                self.logger.error(f"Validation rule failed: {e}")
                reports.append(ValidationReport(
                    test_name=rule.__name__,
                    result=ValidationResult.FAILED,
                    message=f"Validation rule error: {e}"
                ))
        
        return reports
    
    def _validate_basic_structure(self, model: nn.Module) -> ValidationReport:
        """Validate basic model structure."""
        try:
            # Check if model is a PyTorch module
            if not isinstance(model, nn.Module):
                return ValidationReport(
                    test_name="basic_structure",
                    result=ValidationResult.FAILED,
                    message="Model must be a PyTorch nn.Module"
                )
            
            # Check if model has parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                return ValidationReport(
                    test_name="basic_structure",
                    result=ValidationResult.WARNING,
                    message="Model has no trainable parameters"
                )
            
            # Check if model can be called
            try:
                test_input = torch.randn(1, 10)  # Default input size
                with torch.no_grad():
                    _ = model(test_input)
            except Exception as e:
                return ValidationReport(
                    test_name="basic_structure",
                    result=ValidationResult.FAILED,
                    message=f"Model forward pass failed: {e}"
                )
            
            return ValidationReport(
                test_name="basic_structure",
                result=ValidationResult.PASSED,
                message=f"Model structure is valid ({param_count:,} parameters)"
            )
            
        except Exception as e:
            return ValidationReport(
                test_name="basic_structure",
                result=ValidationResult.FAILED,
                message=f"Basic structure validation failed: {e}"
            )
    
    def validate_model_compatibility(self, original_model: nn.Module, 
                                   optimized_model: nn.Module) -> ValidationReport:
        """Validate that optimized model is compatible with original."""
        try:
            # Check parameter count
            original_params = sum(p.numel() for p in original_model.parameters())
            optimized_params = sum(p.numel() for p in optimized_model.parameters())
            
            if optimized_params > original_params:
                return ValidationReport(
                    test_name="compatibility",
                    result=ValidationResult.WARNING,
                    message=f"Optimized model has more parameters ({optimized_params:,}) than original ({original_params:,})"
                )
            
            # Test with same input
            test_input = torch.randn(1, 10)
            
            with torch.no_grad():
                original_output = original_model(test_input)
                optimized_output = optimized_model(test_input)
            
            # Check output shapes match
            if original_output.shape != optimized_output.shape:
                return ValidationReport(
                    test_name="compatibility",
                    result=ValidationResult.FAILED,
                    message=f"Output shapes don't match: original {original_output.shape} vs optimized {optimized_output.shape}"
                )
            
            # Check output values are reasonable (not NaN or Inf)
            if torch.isnan(optimized_output).any():
                return ValidationReport(
                    test_name="compatibility",
                    result=ValidationResult.FAILED,
                    message="Optimized model output contains NaN values"
                )
            
            if torch.isinf(optimized_output).any():
                return ValidationReport(
                    test_name="compatibility",
                    result=ValidationResult.FAILED,
                    message="Optimized model output contains Inf values"
                )
            
            return ValidationReport(
                test_name="compatibility",
                result=ValidationResult.PASSED,
                message=f"Models are compatible (parameter reduction: {((original_params - optimized_params) / original_params * 100):.1f}%)"
            )
            
        except Exception as e:
            return ValidationReport(
                test_name="compatibility",
                result=ValidationResult.FAILED,
                message=f"Compatibility validation failed: {e}"
            )
    
    def validate_model_performance(self, model: nn.Module, 
                                 test_inputs: List[torch.Tensor],
                                 expected_outputs: Optional[List[torch.Tensor]] = None) -> ValidationReport:
        """Validate model performance with test inputs."""
        try:
            if not test_inputs:
                return ValidationReport(
                    test_name="performance",
                    result=ValidationResult.SKIPPED,
                    message="No test inputs provided"
                )
            
            # Test forward pass
            outputs = []
            for test_input in test_inputs:
                with torch.no_grad():
                    output = model(test_input)
                    outputs.append(output)
            
            # Check outputs are valid
            for i, output in enumerate(outputs):
                if torch.isnan(output).any():
                    return ValidationReport(
                        test_name="performance",
                        result=ValidationResult.FAILED,
                        message=f"Output {i} contains NaN values"
                    )
                
                if torch.isinf(output).any():
                    return ValidationReport(
                        test_name="performance",
                        result=ValidationResult.FAILED,
                        message=f"Output {i} contains Inf values"
                    )
            
            # Compare with expected outputs if provided
            if expected_outputs:
                if len(outputs) != len(expected_outputs):
                    return ValidationReport(
                        test_name="performance",
                        result=ValidationResult.FAILED,
                        message=f"Output count mismatch: got {len(outputs)}, expected {len(expected_outputs)}"
                    )
                
                for i, (output, expected) in enumerate(zip(outputs, expected_outputs)):
                    if output.shape != expected.shape:
                        return ValidationReport(
                            test_name="performance",
                            result=ValidationResult.FAILED,
                            message=f"Output {i} shape mismatch: got {output.shape}, expected {expected.shape}"
                        )
            
            return ValidationReport(
                test_name="performance",
                result=ValidationResult.PASSED,
                message=f"Model performance validation passed ({len(outputs)} outputs tested)"
            )
            
        except Exception as e:
            return ValidationReport(
                test_name="performance",
                result=ValidationResult.FAILED,
                message=f"Performance validation failed: {e}"
            )

class ConfigValidator:
    """Validates configuration objects."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_optimization_config(self, config: Dict[str, Any]) -> List[ValidationReport]:
        """Validate optimization configuration."""
        reports = []
        
        # Check required fields
        required_fields = ['level', 'max_memory_gb', 'max_cpu_cores']
        for field in required_fields:
            if field not in config:
                reports.append(ValidationReport(
                    test_name="config_validation",
                    result=ValidationResult.FAILED,
                    message=f"Required field '{field}' is missing"
                ))
        
        # Validate numeric fields
        if 'max_memory_gb' in config:
            if not isinstance(config['max_memory_gb'], (int, float)) or config['max_memory_gb'] <= 0:
                reports.append(ValidationReport(
                    test_name="config_validation",
                    result=ValidationResult.FAILED,
                    message="max_memory_gb must be a positive number"
                ))
        
        if 'max_cpu_cores' in config:
            if not isinstance(config['max_cpu_cores'], int) or config['max_cpu_cores'] <= 0:
                reports.append(ValidationReport(
                    test_name="config_validation",
                    result=ValidationResult.FAILED,
                    message="max_cpu_cores must be a positive integer"
                ))
        
        # Validate optimization level
        if 'level' in config:
            valid_levels = ['minimal', 'standard', 'aggressive', 'maximum']
            if config['level'] not in valid_levels:
                reports.append(ValidationReport(
                    test_name="config_validation",
                    result=ValidationResult.FAILED,
                    message=f"Invalid optimization level: {config['level']}. Must be one of {valid_levels}"
                ))
        
        # Validate GPU memory fraction
        if 'gpu_memory_fraction' in config:
            fraction = config['gpu_memory_fraction']
            if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
                reports.append(ValidationReport(
                    test_name="config_validation",
                    result=ValidationResult.FAILED,
                    message="gpu_memory_fraction must be between 0 and 1"
                ))
        
        if not reports:
            reports.append(ValidationReport(
                test_name="config_validation",
                result=ValidationResult.PASSED,
                message="Configuration validation passed"
            ))
        
        return reports
    
    def validate_monitoring_config(self, config: Dict[str, Any]) -> List[ValidationReport]:
        """Validate monitoring configuration."""
        reports = []
        
        # Validate profiling interval
        if 'profiling_interval' in config:
            interval = config['profiling_interval']
            if not isinstance(interval, int) or interval <= 0:
                reports.append(ValidationReport(
                    test_name="monitoring_validation",
                    result=ValidationResult.FAILED,
                    message="profiling_interval must be a positive integer"
                ))
        
        # Validate log level
        if 'log_level' in config:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if config['log_level'] not in valid_levels:
                reports.append(ValidationReport(
                    test_name="monitoring_validation",
                    result=ValidationResult.FAILED,
                    message=f"Invalid log level: {config['log_level']}. Must be one of {valid_levels}"
                ))
        
        # Validate thresholds
        threshold_fields = ['cpu_threshold', 'memory_threshold', 'gpu_memory_threshold']
        for field in threshold_fields:
            if field in config:
                threshold = config[field]
                if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 100:
                    reports.append(ValidationReport(
                        test_name="monitoring_validation",
                        result=ValidationResult.FAILED,
                        message=f"{field} must be between 0 and 100"
                    ))
        
        if not reports:
            reports.append(ValidationReport(
                test_name="monitoring_validation",
                result=ValidationResult.PASSED,
                message="Monitoring configuration validation passed"
            ))
        
        return reports

class ResultValidator:
    """Validates optimization results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_optimization_result(self, result: Dict[str, Any]) -> List[ValidationReport]:
        """Validate optimization result."""
        reports = []
        
        # Check required fields
        required_fields = ['success', 'optimization_time', 'memory_usage']
        for field in required_fields:
            if field not in result:
                reports.append(ValidationReport(
                    test_name="result_validation",
                    result=ValidationResult.FAILED,
                    message=f"Required field '{field}' is missing from result"
                ))
        
        # Validate success field
        if 'success' in result:
            if not isinstance(result['success'], bool):
                reports.append(ValidationReport(
                    test_name="result_validation",
                    result=ValidationResult.FAILED,
                    message="success field must be a boolean"
                ))
        
        # Validate numeric fields
        numeric_fields = ['optimization_time', 'memory_usage', 'parameter_reduction']
        for field in numeric_fields:
            if field in result:
                value = result[field]
                if not isinstance(value, (int, float)) or value < 0:
                    reports.append(ValidationReport(
                        test_name="result_validation",
                        result=ValidationResult.FAILED,
                        message=f"{field} must be a non-negative number"
                    ))
        
        # Validate accuracy score if present
        if 'accuracy_score' in result and result['accuracy_score'] is not None:
            score = result['accuracy_score']
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                reports.append(ValidationReport(
                    test_name="result_validation",
                    result=ValidationResult.WARNING,
                    message="accuracy_score should be between 0 and 1"
                ))
        
        if not reports:
            reports.append(ValidationReport(
                test_name="result_validation",
                result=ValidationResult.PASSED,
                message="Optimization result validation passed"
            ))
        
        return reports
    
    def validate_performance_improvement(self, original_metrics: Dict[str, Any],
                                       optimized_metrics: Dict[str, Any]) -> List[ValidationReport]:
        """Validate that optimization actually improved performance."""
        reports = []
        
        # Check memory usage improvement
        if 'memory_usage' in original_metrics and 'memory_usage' in optimized_metrics:
            original_memory = original_metrics['memory_usage']
            optimized_memory = optimized_metrics['memory_usage']
            
            if optimized_memory >= original_memory:
                reports.append(ValidationReport(
                    test_name="performance_improvement",
                    result=ValidationResult.WARNING,
                    message=f"Memory usage did not improve: {original_memory:.2f} -> {optimized_memory:.2f}"
                ))
            else:
                improvement = (original_memory - optimized_memory) / original_memory * 100
                reports.append(ValidationReport(
                    test_name="performance_improvement",
                    result=ValidationResult.PASSED,
                    message=f"Memory usage improved by {improvement:.1f}%"
                ))
        
        # Check parameter reduction
        if 'parameter_count' in original_metrics and 'parameter_count' in optimized_metrics:
            original_params = original_metrics['parameter_count']
            optimized_params = optimized_metrics['parameter_count']
            
            if optimized_params >= original_params:
                reports.append(ValidationReport(
                    test_name="performance_improvement",
                    result=ValidationResult.WARNING,
                    message=f"Parameter count did not decrease: {original_params:,} -> {optimized_params:,}"
                ))
            else:
                reduction = (original_params - optimized_params) / original_params * 100
                reports.append(ValidationReport(
                    test_name="performance_improvement",
                    result=ValidationResult.PASSED,
                    message=f"Parameter count reduced by {reduction:.1f}%"
                ))
        
        return reports

# Factory functions
def create_model_validator() -> ModelValidator:
    """Create a model validator."""
    return ModelValidator()

def create_config_validator() -> ConfigValidator:
    """Create a config validator."""
    return ConfigValidator()

def create_result_validator() -> ResultValidator:
    """Create a result validator."""
    return ResultValidator()

