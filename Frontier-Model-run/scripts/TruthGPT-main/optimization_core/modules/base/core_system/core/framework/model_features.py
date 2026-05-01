"""
Model Feature Extraction Module
Extracts features from models for AI analysis
"""

import torch
import torch.nn as nn
from typing import Dict
from collections import defaultdict
import numpy as np
import logging

from .config import DEFAULT_DEPTH_NORMALIZATION
from .model_utils import calculate_parameter_count

logger = logging.getLogger(__name__)


class ModelFeatureExtractor:
    """Extracts features from models for AI analysis."""
    
    def __init__(self, target_feature_size: int = 1024):
        self.target_feature_size = target_feature_size
    
    def extract(self, model: nn.Module) -> torch.Tensor:
        """Extract features from model for AI analysis."""
        features = []
        
        param_count = calculate_parameter_count(model)
        features.append(np.log10(param_count + 1))
        
        layer_types = self._count_layer_types(model)
        total_layers = sum(layer_types.values())
        
        if total_layers > 0:
            features.extend([
                layer_types['linear'] / total_layers,
                layer_types['conv2d'] / total_layers,
                layer_types['lstm'] / total_layers,
                layer_types['transformer'] / total_layers,
                layer_types['attention'] / total_layers
            ])
        else:
            features.extend([0.0] * 5)
        
        depth = len(list(model.modules()))
        features.append(depth / DEFAULT_DEPTH_NORMALIZATION)
        
        memory_usage = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )
        features.append(np.log10(memory_usage + 1))
        
        flops = self._estimate_flops(model)
        features.append(np.log10(flops + 1))
        
        features.extend([0.0] * max(0, self.target_feature_size - len(features)))
        features = features[:self.target_feature_size]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _count_layer_types(self, model: nn.Module) -> Dict[str, int]:
        """Count different layer types in the model."""
        layer_types = defaultdict(int)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_types['linear'] += 1
            elif isinstance(module, nn.Conv2d):
                layer_types['conv2d'] += 1
            elif isinstance(module, nn.LSTM):
                layer_types['lstm'] += 1
            elif isinstance(module, nn.Transformer):
                layer_types['transformer'] += 1
            elif isinstance(module, nn.MultiheadAttention):
                layer_types['attention'] += 1
        return layer_types
    
    def _estimate_flops(self, model: nn.Module) -> int:
        """Estimate FLOPs for model."""
        flops = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                flops += (
                    module.kernel_size[0] * module.kernel_size[1] *
                    module.in_channels * module.out_channels
                )
        return flops


