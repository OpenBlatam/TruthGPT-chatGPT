"""
Model Utility Functions
Helper functions for model analysis and comparison
"""

import torch.nn as nn
from typing import Tuple


def calculate_parameter_count(model: nn.Module) -> int:
    """Calculate total parameter count in model."""
    return sum(p.numel() for p in model.parameters())


def calculate_memory_reduction(
    original_model: nn.Module,
    optimized_model: nn.Module
) -> float:
    """Calculate memory reduction percentage between models."""
    original_params = calculate_parameter_count(original_model)
    optimized_params = calculate_parameter_count(optimized_model)
    
    if original_params == 0:
        return 0.0
    
    return (original_params - optimized_params) / original_params


def compare_models(
    original_model: nn.Module,
    optimized_model: nn.Module
) -> Tuple[int, int, float]:
    """Compare two models and return parameter counts and reduction."""
    original_params = calculate_parameter_count(original_model)
    optimized_params = calculate_parameter_count(optimized_model)
    memory_reduction = calculate_memory_reduction(original_model, optimized_model)
    
    return original_params, optimized_params, memory_reduction







