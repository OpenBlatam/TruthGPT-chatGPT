"""
PyTorch Optimizer Base Class
Migrated from optimization_core.core.pytorch_optimizer_base
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


from abc import ABC, abstractmethod


class OptimizationConfig(BaseModel):
    """Base configuration for python optimizers"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    enabled: bool = True
    verbose: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "float32"

class PyTorchOptimizerBase(ABC):
    """
    Base class for PyTorch-based optimizers.
    Provides common functionality for configuration and device management.
    """
    
    def __init__(self, config: Union[Dict[str, Any], OptimizationConfig] = None):
        if config is None:
            self.config = OptimizationConfig()
        elif isinstance(config, dict):
            self.config = OptimizationConfig(**config)
        else:
            self.config = config
            
        self.device = torch.device(self.config.device if hasattr(self.config, 'device') else 'cpu')
        
    @abstractmethod
    def optimize(self, model: nn.Module) -> nn.Module:
        """
        Apply optimization to the model.
        Must be implemented by subclasses.
        """
        pass
        
    def to(self, device: Union[str, torch.device]):
        """Move optimizer resources to device"""
        self.device = torch.device(device)
        return self


