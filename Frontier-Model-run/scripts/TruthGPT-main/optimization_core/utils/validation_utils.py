"""
Validation Utilities for TruthGPT Optimization Core
Advanced validation and testing utilities
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type
import time
from pydantic import Field
import json
from pathlib import Path
import inspect
import warnings

from .base import BaseOptimizationModel


class ValidationResult(BaseOptimizationModel):
    """Validation result container."""
    # Basic info
    test_name: str
    passed: bool
    message: str
    
    # Metrics
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    
    # Details
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

class TensorValidator:
    """Advanced tensor validation utilities."""
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, 
                       name: str = "tensor",
                       check_nan: bool = True,
                       check_inf: bool = True,
                       check_finite: bool = True,
                       check_shape: Optional[Tuple[int, ...]] = None,
                       check_dtype: Optional[torch.dtype] = None,
                       check_device: Optional[torch.device] = None) -> ValidationResult:
        """Comprehensive tensor validation."""
        start_time = time.time()
        
        try:
            # Basic type check
            if not isinstance(tensor, torch.Tensor):
                return ValidationResult(
                    test_name=f"{name}_type_check",
                    passed=False,
                    message=f"{name} is not a torch.Tensor",
                    execution_time=time.time() - start_time
                )
            
            # NaN check
            if check_nan and torch.isnan(tensor).any():
                return ValidationResult(
                    test_name=f"{name}_nan_check",
                    passed=False,
                    message=f"{name} contains NaN values",
                    execution_time=time.time() - start_time
                )
            
            # Inf check
            if check_inf and torch.isinf(tensor).any():
                return ValidationResult(
                    test_name=f"{name}_inf_check",
                    passed=False,
                    message=f"{name} contains Inf values",
                    execution_time=time.time() - start_time
                )
            
            # Finite check
            if check_finite and not torch.isfinite(tensor).all():
                return ValidationResult(
                    test_name=f"{name}_finite_check",
                    passed=False,
                    message=f"{name} contains non-finite values",
                    execution_time=time.time() - start_time
                )
            
            # Shape check
            if check_shape is not None and tensor.shape != check_shape:
                return ValidationResult(
                    test_name=f"{name}_shape_check",
                    passed=False,
                    message=f"{name} shape {tensor.shape} != expected {check_shape}",
                    execution_time=time.time() - start_time
                )
            
            # Dtype check
            if check_dtype is not None and tensor.dtype != check_dtype:
                return ValidationResult(
                    test_name=f"{name}_dtype_check",
                    passed=False,
                    message=f"{name} dtype {tensor.dtype} != expected {check_dtype}",
                    execution_time=time.time() - start_time
                )
            
            # Device check
            if check_device is not None and tensor.device != check_device:
                return ValidationResult(
                    test_name=f"{name}_device_check",
                    passed=False,
                    message=f"{name} device {tensor.device} != expected {check_device}",
                    execution_time=time.time() - start_time
                )
            
            # All checks passed
            return ValidationResult(
                test_name=f"{name}_validation",
                passed=True,
                message=f"{name} validation passed",
                execution_time=time.time() - start_time,
                details={
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'device': str(tensor.device),
                    'numel': tensor.numel()
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=f"{name}_validation",
                passed=False,
                message=f"Error validating {name}: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    @staticmethod
    def validate_tensor_properties(tensor: torch.Tensor, 
                                 name: str = "tensor",
                                 min_value: Optional[float] = None,
                                 max_value: Optional[float] = None,
                                 mean_range: Optional[Tuple[float, float]] = None,
                                 std_range: Optional[Tuple[float, float]] = None) -> ValidationResult:
        """Validate tensor statistical properties."""
        start_time = time.time()
        
        try:
            details = {}
            
            # Min/Max value checks
            if min_value is not None:
                if tensor.min().item() < min_value:
                    return ValidationResult(
                        test_name=f"{name}_min_value_check",
                        passed=False,
                        message=f"{name} min value {tensor.min().item()} < {min_value}",
                        execution_time=time.time() - start_time
                    )
                details['min_value'] = tensor.min().item()
            
            if max_value is not None:
                if tensor.max().item() > max_value:
                    return ValidationResult(
                        test_name=f"{name}_max_value_check",
                        passed=False,
                        message=f"{name} max value {tensor.max().item()} > {max_value}",
                        execution_time=time.time() - start_time
                    )
                details['max_value'] = tensor.max().item()
            
            # Mean range check
            if mean_range is not None:
                mean_val = tensor.mean().item()
                if not (mean_range[0] <= mean_val <= mean_range[1]):
                    return ValidationResult(
                        test_name=f"{name}_mean_check",
                        passed=False,
                        message=f"{name} mean {mean_val} not in range {mean_range}",
                        execution_time=time.time() - start_time
                    )
                details['mean'] = mean_val
            
            # Std range check
            if std_range is not None:
                std_val = tensor.std().item()
                if not (std_range[0] <= std_val <= std_range[1]):
                    return ValidationResult(
                        test_name=f"{name}_std_check",
                        passed=False,
                        message=f"{name} std {std_val} not in range {std_range}",
                        execution_time=time.time() - start_time
                    )
                details['std'] = std_val
            
            return ValidationResult(
                test_name=f"{name}_properties_validation",
                passed=True,
                message=f"{name} properties validation passed",
                execution_time=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=f"{name}_properties_validation",
                passed=False,
                message=f"Error validating {name} properties: {str(e)}",
                execution_time=time.time() - start_time
            )

class ModelValidator:
    """Advanced model validation utilities."""
    
    def __init__(self):
        self.validation_history: List[ValidationResult] = []
    
    def validate_model_forward(self, model: nn.Module, 
                              input_tensor: torch.Tensor,
                              expected_output_shape: Optional[Tuple[int, ...]] = None,
                              expected_output_dtype: Optional[torch.dtype] = None) -> ValidationResult:
        """Validate model forward pass."""
        start_time = time.time()
        
        try:
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
            
            # Validate output tensor
            output_validation = TensorValidator.validate_tensor(
                output, 
                name="model_output",
                check_shape=expected_output_shape,
                check_dtype=expected_output_dtype
            )
            
            if not output_validation.passed:
                return output_validation
            
            # All checks passed
            return ValidationResult(
                test_name="model_forward_validation",
                passed=True,
                message="Model forward pass validation passed",
                execution_time=time.time() - start_time,
                details={
                    'input_shape': list(input_tensor.shape),
                    'output_shape': list(output.shape),
                    'input_dtype': str(input_tensor.dtype),
                    'output_dtype': str(output.dtype)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="model_forward_validation",
                passed=False,
                message=f"Model forward pass failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_model_gradients(self, model: nn.Module, 
                                 input_tensor: torch.Tensor,
                                 target_tensor: torch.Tensor,
                                 criterion: nn.Module) -> ValidationResult:
        """Validate model gradients."""
        start_time = time.time()
        
        try:
            model.train()
            
            # Forward pass
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            gradient_issues = []
            for name, param in model.named_parameters():
                if param.grad is None:
                    gradient_issues.append(f"{name}: No gradient")
                elif torch.isnan(param.grad).any():
                    gradient_issues.append(f"{name}: NaN gradient")
                elif torch.isinf(param.grad).any():
                    gradient_issues.append(f"{name}: Inf gradient")
            
            if gradient_issues:
                return ValidationResult(
                    test_name="model_gradients_validation",
                    passed=False,
                    message=f"Gradient issues: {', '.join(gradient_issues)}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                test_name="model_gradients_validation",
                passed=True,
                message="Model gradients validation passed",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="model_gradients_validation",
                passed=False,
                message=f"Model gradients validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_model_parameters(self, model: nn.Module) -> ValidationResult:
        """Validate model parameters."""
        start_time = time.time()
        
        try:
            parameter_issues = []
            total_params = 0
            trainable_params = 0
            
            for name, param in model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                
                # Check for NaN/Inf
                if torch.isnan(param).any():
                    parameter_issues.append(f"{name}: NaN parameter")
                elif torch.isinf(param).any():
                    parameter_issues.append(f"{name}: Inf parameter")
            
            if parameter_issues:
                return ValidationResult(
                    test_name="model_parameters_validation",
                    passed=False,
                    message=f"Parameter issues: {', '.join(parameter_issues)}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                test_name="model_parameters_validation",
                passed=True,
                message="Model parameters validation passed",
                execution_time=time.time() - start_time,
                details={
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'non_trainable_parameters': total_params - trainable_params
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="model_parameters_validation",
                passed=False,
                message=f"Model parameters validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_model_compatibility(self, model: nn.Module, 
                                    input_shapes: List[Tuple[int, ...]],
                                    device: str = "cpu") -> ValidationResult:
        """Validate model compatibility with different input shapes."""
        start_time = time.time()
        
        try:
            model = model.to(device)
            model.eval()
            
            compatibility_issues = []
            
            for i, input_shape in enumerate(input_shapes):
                try:
                    test_input = torch.randn(1, *input_shape).to(device)
                    with torch.no_grad():
                        _ = model(test_input)
                except Exception as e:
                    compatibility_issues.append(f"Input shape {input_shape}: {str(e)}")
            
            if compatibility_issues:
                return ValidationResult(
                    test_name="model_compatibility_validation",
                    passed=False,
                    message=f"Compatibility issues: {', '.join(compatibility_issues)}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                test_name="model_compatibility_validation",
                passed=True,
                message="Model compatibility validation passed",
                execution_time=time.time() - start_time,
                details={
                    'tested_shapes': [list(shape) for shape in input_shapes],
                    'device': device
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="model_compatibility_validation",
                passed=False,
                message=f"Model compatibility validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )

class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_dataset(dataset: torch.utils.data.Dataset, 
                        expected_size: Optional[int] = None,
                        expected_classes: Optional[int] = None) -> ValidationResult:
        """Validate dataset properties."""
        start_time = time.time()
        
        try:
            # Check dataset size
            if expected_size is not None and len(dataset) != expected_size:
                return ValidationResult(
                    test_name="dataset_size_validation",
                    passed=False,
                    message=f"Dataset size {len(dataset)} != expected {expected_size}",
                    execution_time=time.time() - start_time
                )
            
            # Check if dataset is not empty
            if len(dataset) == 0:
                return ValidationResult(
                    test_name="dataset_empty_validation",
                    passed=False,
                    message="Dataset is empty",
                    execution_time=time.time() - start_time
                )
            
            # Sample a few items to check structure
            sample_issues = []
            for i in range(min(5, len(dataset))):
                try:
                    sample = dataset[i]
                    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                        sample_issues.append(f"Sample {i}: Invalid structure")
                        continue
                    
                    data, target = sample
                    if not isinstance(data, torch.Tensor) or not isinstance(target, torch.Tensor):
                        sample_issues.append(f"Sample {i}: Invalid tensor types")
                    
                except Exception as e:
                    sample_issues.append(f"Sample {i}: {str(e)}")
            
            if sample_issues:
                return ValidationResult(
                    test_name="dataset_structure_validation",
                    passed=False,
                    message=f"Dataset structure issues: {', '.join(sample_issues)}",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                test_name="dataset_validation",
                passed=True,
                message="Dataset validation passed",
                execution_time=time.time() - start_time,
                details={
                    'dataset_size': len(dataset),
                    'samples_checked': min(5, len(dataset))
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="dataset_validation",
                passed=False,
                message=f"Dataset validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    @staticmethod
    def validate_dataloader(dataloader: torch.utils.data.DataLoader,
                           expected_batch_size: Optional[int] = None) -> ValidationResult:
        """Validate dataloader properties."""
        start_time = time.time()
        
        try:
            # Check batch size
            if expected_batch_size is not None:
                first_batch = next(iter(dataloader))
                if len(first_batch[0]) != expected_batch_size:
                    return ValidationResult(
                        test_name="dataloader_batch_size_validation",
                        passed=False,
                        message=f"Batch size {len(first_batch[0])} != expected {expected_batch_size}",
                        execution_time=time.time() - start_time
                    )
            
            # Check if dataloader is not empty
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count > 10:  # Limit check to first 10 batches
                    break
            
            if batch_count == 0:
                return ValidationResult(
                    test_name="dataloader_empty_validation",
                    passed=False,
                    message="DataLoader is empty",
                    execution_time=time.time() - start_time
                )
            
            return ValidationResult(
                test_name="dataloader_validation",
                passed=True,
                message="DataLoader validation passed",
                execution_time=time.time() - start_time,
                details={
                    'batches_checked': batch_count,
                    'batch_size': len(next(iter(dataloader))[0])
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="dataloader_validation",
                passed=False,
                message=f"DataLoader validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )

class ValidationSuite:
    """Comprehensive validation suite."""
    
    def __init__(self):
        self.tensor_validator = TensorValidator()
        self.model_validator = ModelValidator()
        self.data_validator = DataValidator()
        self.results: List[ValidationResult] = []
    
    def run_tensor_validation(self, tensor: torch.Tensor, 
                            name: str = "tensor",
                            **kwargs) -> ValidationResult:
        """Run comprehensive tensor validation."""
        result = self.tensor_validator.validate_tensor(tensor, name, **kwargs)
        self.results.append(result)
        return result
    
    def run_model_validation(self, model: nn.Module, 
                           input_tensor: torch.Tensor,
                           **kwargs) -> List[ValidationResult]:
        """Run comprehensive model validation."""
        results = []
        
        # Forward pass validation
        forward_result = self.model_validator.validate_model_forward(
            model, input_tensor, **kwargs
        )
        results.append(forward_result)
        self.results.append(forward_result)
        
        # Parameters validation
        params_result = self.model_validator.validate_model_parameters(model)
        results.append(params_result)
        self.results.append(params_result)
        
        return results
    
    def run_data_validation(self, dataset: torch.utils.data.Dataset,
                           dataloader: torch.utils.data.DataLoader,
                           **kwargs) -> List[ValidationResult]:
        """Run comprehensive data validation."""
        results = []
        
        # Dataset validation
        dataset_result = self.data_validator.validate_dataset(dataset, **kwargs)
        results.append(dataset_result)
        self.results.append(dataset_result)
        
        # DataLoader validation
        dataloader_result = self.data_validator.validate_dataloader(dataloader, **kwargs)
        results.append(dataloader_result)
        self.results.append(dataloader_result)
        
        return results
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        if not self.results:
            return {}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Group by test type
        test_types = {}
        for result in self.results:
            test_type = result.test_name.split('_')[0]
            if test_type not in test_types:
                test_types[test_type] = {'passed': 0, 'failed': 0}
            if result.passed:
                test_types[test_type]['passed'] += 1
            else:
                test_types[test_type]['failed'] += 1
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'test_types': test_types,
            'failed_tests': [r for r in self.results if not r.passed]
        }
    
    def save_results(self, filepath: str) -> None:
        """Save validation results to file."""
        results_data = [result.to_dict() for result in self.results]
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Validation results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load validation results from file."""
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        self.results = []
        for data in results_data:
            result = ValidationResult(**data)
            self.results.append(result)
        logger.info(f"Validation results loaded from {filepath}")

# Utility functions
def validate_tensor(tensor: torch.Tensor, name: str = "tensor", **kwargs) -> bool:
    """Quick tensor validation."""
    validator = TensorValidator()
    result = validator.validate_tensor(tensor, name, **kwargs)
    return result.passed

def validate_model(model: nn.Module, input_tensor: torch.Tensor, **kwargs) -> bool:
    """Quick model validation."""
    validator = ModelValidator()
    result = validator.validate_model_forward(model, input_tensor, **kwargs)
    return result.passed

def validate_dataset(dataset: torch.utils.data.Dataset, **kwargs) -> bool:
    """Quick dataset validation."""
    validator = DataValidator()
    result = validator.validate_dataset(dataset, **kwargs)
    return result.passed

# Factory functions
def create_validation_suite() -> ValidationSuite:
    """Create a new validation suite instance."""
    return ValidationSuite()

def create_tensor_validator() -> TensorValidator:
    """Create a new tensor validator instance."""
    return TensorValidator()

def create_model_validator() -> ModelValidator:
    """Create a new model validator instance."""
    return ModelValidator()

def create_data_validator() -> DataValidator:
    """Create a new data validator instance."""
    return DataValidator()

# Example usage
if __name__ == "__main__":
    # Example usage of validation utilities
    print("✅ TruthGPT Validation Utilities Demo")
    print("=" * 50)
    
    # Create validation suite
    validation_suite = create_validation_suite()
    
    # Test tensor validation
    test_tensor = torch.randn(32, 100)
    tensor_result = validation_suite.run_tensor_validation(
        test_tensor, 
        name="test_tensor",
        check_nan=True,
        check_inf=True
    )
    print(f"Tensor validation: {'PASSED' if tensor_result.passed else 'FAILED'}")
    
    # Test model validation
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    input_tensor = torch.randn(32, 100)
    model_results = validation_suite.run_model_validation(model, input_tensor)
    
    for result in model_results:
        print(f"Model validation ({result.test_name}): {'PASSED' if result.passed else 'FAILED'}")
    
    # Get validation report
    report = validation_suite.get_validation_report()
    print(f"Validation report: {report['success_rate']:.2%} success rate")




