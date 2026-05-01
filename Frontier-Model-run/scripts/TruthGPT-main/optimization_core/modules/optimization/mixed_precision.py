"""
Ultra-fast mixed precision optimizations
Following deep learning best practices
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from typing import Optional, Union, Any
from dataclasses import dataclass


@dataclass
class MixedPrecisionConfig:
    """Mixed precision configuration"""
    enabled: bool = True
    loss_scale: float = 1.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    init_scale: float = 65536.0


class MixedPrecisionOptimizer:
    """Ultra-fast mixed precision optimizer"""
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scaler = GradScaler(
            init_scale=config.init_scale,
            growth_factor=config.growth_factor,
            backoff_factor=config.backoff_factor,
            growth_interval=config.growth_interval
        ) if config.enabled else None
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients"""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with mixed precision"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def update_scale(self):
        """Update loss scale"""
        if self.scaler is not None:
            self.scaler.update()


class AMPOptimizer:
    """Automatic Mixed Precision optimizer"""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 config: MixedPrecisionConfig):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scaler = GradScaler() if config.enabled else None
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with mixed precision"""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self):
        """Optimizer step with mixed precision"""
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()


def autocast_forward(func):
    """Decorator for automatic mixed precision forward pass"""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'config') and getattr(self.config, 'use_mixed_precision', False):
            with autocast():
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return wrapper


class GradScaler:
    """Gradient scaler for mixed precision training"""
    
    def __init__(self, init_scale: float = 65536.0, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
    
    def scale(self, outputs: Union[torch.Tensor, list]) -> Union[torch.Tensor, list]:
        """Scale outputs"""
        if isinstance(outputs, torch.Tensor):
            return outputs * self.scale
        else:
            return [output * self.scale for output in outputs]
    
    def unscale_(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients"""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Step optimizer"""
        optimizer.step()
    
    def update(self):
        """Update scale"""
        self._growth_tracker += 1
        if self._growth_tracker >= self.growth_interval:
            self.scale *= self.growth_factor
            self._growth_tracker = 0



