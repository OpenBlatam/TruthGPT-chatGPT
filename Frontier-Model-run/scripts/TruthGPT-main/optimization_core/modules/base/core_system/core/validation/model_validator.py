"""
Model validation utilities.
"""
import logging
from typing import Any, Dict
import torch
import torch.nn as nn

from .validator import Validator, ValidationResult

logger = logging.getLogger(__name__)


class ModelValidator(Validator):
    """Validator for PyTorch models."""
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate a model.
        
        Args:
            data: Model to validate
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        if not isinstance(data, nn.Module):
            errors.append("Data is not a PyTorch nn.Module")
            return self._create_result(errors)
        
        # Check for parameters
        if not list(data.parameters()):
            warnings.append("Model has no parameters")
        
        # Check for trainable parameters
        trainable = sum(p.numel() for p in data.parameters() if p.requires_grad)
        if trainable == 0:
            warnings.append("Model has no trainable parameters")
        
        # Check for NaN/Inf in parameters
        for name, param in data.named_parameters():
            if torch.isnan(param).any():
                errors.append(f"Parameter '{name}' contains NaN values")
            if torch.isinf(param).any():
                errors.append(f"Parameter '{name}' contains Inf values")
        
        # Check device consistency
        devices = set(p.device for p in data.parameters())
        if len(devices) > 1:
            warnings.append(f"Model parameters on multiple devices: {devices}")
        
        return self._create_result(errors, warnings)
    
    def validate_inference(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: torch.device
    ) -> ValidationResult:
        """
        Validate model inference.
        
        Args:
            model: Model to validate
            input_shape: Input tensor shape
            device: Device for inference
        
        Returns:
            ValidationResult
        """
        errors = []
        
        try:
            model.eval()
            dummy_input = torch.zeros(input_shape).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output
            if output is None:
                errors.append("Model returned None output")
            elif isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    errors.append("Model output contains NaN")
                if torch.isinf(output).any():
                    errors.append("Model output contains Inf")
            
            model.train()
            
        except Exception as e:
            errors.append(f"Inference validation failed: {str(e)}")
        
        return self._create_result(errors)



