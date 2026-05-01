"""
TruthGPT Optimizers Module
Advanced optimization utilities for TruthGPT models following deep learning best practices
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, ExponentialLR, StepLR,
    ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
)
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTOptimizerConfig:
    """Configuration for TruthGPT optimizers."""
    # Optimizer configuration
    optimizer_type: str = "adamw"  # adam, adamw, sgd, rmsprop, adagrad
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # Scheduler configuration
    scheduler_type: str = "cosine"  # linear, cosine, exponential, step, plateau, onecycle
    warmup_steps: int = 1000
    max_steps: int = 100000
    min_lr: float = 1e-6
    
    # Advanced features
    enable_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    enable_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 1
    
    # Learning rate scheduling
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 10
    lr_decay_threshold: float = 0.001
    
    # Optimizer-specific settings
    momentum: float = 0.9  # For SGD
    nesterov: bool = True  # For SGD
    alpha: float = 0.99  # For RMSprop
    centered: bool = False  # For RMSprop
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'eps': self.eps,
            'scheduler_type': self.scheduler_type,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'min_lr': self.min_lr,
            'enable_gradient_clipping': self.enable_gradient_clipping,
            'gradient_clip_norm': self.gradient_clip_norm,
            'enable_gradient_accumulation': self.enable_gradient_accumulation,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'lr_decay_factor': self.lr_decay_factor,
            'lr_decay_patience': self.lr_decay_patience,
            'lr_decay_threshold': self.lr_decay_threshold,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'alpha': self.alpha,
            'centered': self.centered
        }

class TruthGPTOptimizer:
    """Advanced optimizer for TruthGPT models."""
    
    def __init__(self, config: TruthGPTOptimizerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Optimizer state
        self.optimizer = None
        self.scheduler = None
        self.step_count = 0
        self.optimization_stats = {}
        
        self.logger.info(f"TruthGPT Optimizer initialized with {config.optimizer_type}")
    
    def setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Setup optimizer for model."""
        self.logger.info(f"Setting up {self.config.optimizer_type} optimizer")
        
        # Get model parameters
        params = model.parameters()
        
        # Create optimizer based on type
        if self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(
                params,
                lr=self.config.learning_rate,
                alpha=self.config.alpha,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum,
                centered=self.config.centered
            )
        elif self.config.optimizer_type.lower() == "adagrad":
            self.optimizer = optim.Adagrad(
                params,
                lr=self.config.learning_rate,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
        
        self.logger.info(f"✅ {self.config.optimizer_type} optimizer created")
        return self.optimizer
    
    def setup_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        self.logger.info(f"Setting up {self.config.scheduler_type} scheduler")
        
        # Calculate total steps
        total_steps = num_training_steps
        
        # Create scheduler based on type
        if self.config.scheduler_type.lower() == "linear":
            self.scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        elif self.config.scheduler_type.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type.lower() == "exponential":
            self.scheduler = ExponentialLR(
                optimizer,
                gamma=self.config.lr_decay_factor
            )
        elif self.config.scheduler_type.lower() == "step":
            self.scheduler = StepLR(
                optimizer,
                step_size=total_steps // 4,
                gamma=self.config.lr_decay_factor
            )
        elif self.config.scheduler_type.lower() == "plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.lr_decay_factor,
                patience=self.config.lr_decay_patience,
                threshold=self.config.lr_decay_threshold,
                min_lr=self.config.min_lr
            )
        elif self.config.scheduler_type.lower() == "onecycle":
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        elif self.config.scheduler_type.lower() == "cosine_warm_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_steps // 4,
                T_mult=2,
                eta_min=self.config.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        
        self.logger.info(f"✅ {self.config.scheduler_type} scheduler created")
        return self.scheduler
    
    def step(self, loss: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """Perform optimization step."""
        # Gradient clipping
        if self.config.enable_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Scheduler step
        if self.scheduler:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                # For plateau scheduler, we need to pass the loss
                self.scheduler.step(loss.item())
            else:
                self.scheduler.step()
        
        # Update step count
        self.step_count += 1
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Update statistics
        self.optimization_stats = {
            'step': self.step_count,
            'learning_rate': current_lr,
            'loss': loss.item(),
            'gradient_norm': self._get_gradient_norm(model)
        }
        
        return self.optimization_stats
    
    def _get_gradient_norm(self, model: nn.Module) -> float:
        """Get gradient norm."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.optimization_stats
    
    def save_optimizer_state(self, filepath: str) -> None:
        """Save optimizer state."""
        if self.optimizer:
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'step_count': self.step_count,
                'config': self.config.to_dict()
            }, filepath)
            self.logger.info(f"Optimizer state saved to {filepath}")
    
    def load_optimizer_state(self, filepath: str) -> None:
        """Load optimizer state."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step_count = checkpoint['step_count']
        self.logger.info(f"Optimizer state loaded from {filepath}")

class TruthGPTScheduler:
    """Advanced scheduler for TruthGPT models."""
    
    def __init__(self, config: TruthGPTOptimizerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Scheduler state
        self.scheduler = None
        self.scheduler_stats = {}
    
    def create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        self.logger.info(f"Creating {self.config.scheduler_type} scheduler")
        
        # Calculate total steps
        total_steps = num_training_steps
        
        # Create scheduler based on type
        if self.config.scheduler_type.lower() == "linear":
            self.scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        elif self.config.scheduler_type.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type.lower() == "exponential":
            self.scheduler = ExponentialLR(
                optimizer,
                gamma=self.config.lr_decay_factor
            )
        elif self.config.scheduler_type.lower() == "step":
            self.scheduler = StepLR(
                optimizer,
                step_size=total_steps // 4,
                gamma=self.config.lr_decay_factor
            )
        elif self.config.scheduler_type.lower() == "plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.lr_decay_factor,
                patience=self.config.lr_decay_patience,
                threshold=self.config.lr_decay_threshold,
                min_lr=self.config.min_lr
            )
        elif self.config.scheduler_type.lower() == "onecycle":
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        elif self.config.scheduler_type.lower() == "cosine_warm_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_steps // 4,
                T_mult=2,
                eta_min=self.config.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        
        self.logger.info(f"✅ {self.config.scheduler_type} scheduler created")
        return self.scheduler
    
    def step(self, loss: Optional[float] = None) -> Dict[str, Any]:
        """Step scheduler."""
        if not self.scheduler:
            return {}
        
        # Step scheduler
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if loss is not None:
                self.scheduler.step(loss)
            else:
                self.logger.warning("Loss required for ReduceLROnPlateau scheduler")
        else:
            self.scheduler.step()
        
        # Get current learning rate
        current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        
        # Update statistics
        self.scheduler_stats = {
            'learning_rate': current_lr,
            'scheduler_type': self.config.scheduler_type
        }
        
        return self.scheduler_stats
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return self.scheduler_stats

class TruthGPTGradientAccumulator:
    """Gradient accumulator for TruthGPT models."""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.accumulation_count = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.accumulation_count += 1
        return self.accumulation_count % self.accumulation_steps == 0
    
    def reset(self) -> None:
        """Reset accumulation counter."""
        self.accumulation_count = 0
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Get accumulation statistics."""
        return {
            'accumulation_steps': self.accumulation_steps,
            'accumulation_count': self.accumulation_count,
            'should_step': self.should_step()
        }

class TruthGPTOptimizationManager:
    """Comprehensive optimization manager for TruthGPT models."""
    
    def __init__(self, config: TruthGPTOptimizerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Components
        self.optimizer = TruthGPTOptimizer(config)
        self.scheduler = TruthGPTScheduler(config)
        self.gradient_accumulator = TruthGPTGradientAccumulator(config.gradient_accumulation_steps)
        
        # State
        self.optimization_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def setup_optimization(self, model: nn.Module, num_training_steps: int) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Setup complete optimization pipeline."""
        self.logger.info("🚀 Setting up TruthGPT optimization pipeline")
        
        # Setup optimizer
        optimizer = self.optimizer.setup_optimizer(model)
        
        # Setup scheduler
        scheduler = self.scheduler.create_scheduler(optimizer, num_training_steps)
        
        self.logger.info("✅ TruthGPT optimization pipeline setup completed")
        return optimizer, scheduler
    
    def optimization_step(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Perform optimization step."""
        # Check if we should step
        if not self.gradient_accumulator.should_step():
            return {}
        
        # Perform optimization step
        stats = self.optimizer.step(loss, model)
        
        # Step scheduler
        scheduler_stats = self.scheduler.step(loss.item())
        
        # Update history
        step_stats = {**stats, **scheduler_stats}
        self.optimization_history.append(step_stats)
        
        # Check for early stopping
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return step_stats
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history
    
    def get_best_loss(self) -> float:
        """Get best loss achieved."""
        return self.best_loss
    
    def should_early_stop(self, patience: int = 10) -> bool:
        """Check if training should stop early."""
        return self.patience_counter >= patience

# Factory functions
def create_truthgpt_optimizer(config: TruthGPTOptimizerConfig) -> TruthGPTOptimizer:
    """Create TruthGPT optimizer."""
    return TruthGPTOptimizer(config)

def create_truthgpt_scheduler(config: TruthGPTOptimizerConfig) -> TruthGPTScheduler:
    """Create TruthGPT scheduler."""
    return TruthGPTScheduler(config)

def create_truthgpt_optimization_manager(config: TruthGPTOptimizerConfig) -> TruthGPTOptimizationManager:
    """Create TruthGPT optimization manager."""
    return TruthGPTOptimizationManager(config)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT optimization
    print("🚀 TruthGPT Optimizers Demo")
    print("=" * 50)
    
    # Create a sample model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 10000)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create model
    model = TruthGPTModel()
    
    # Create configuration
    config = TruthGPTOptimizerConfig(
        optimizer_type="adamw",
        learning_rate=1e-4,
        scheduler_type="cosine",
        warmup_steps=1000
    )
    
    # Create optimization manager
    optimization_manager = create_truthgpt_optimization_manager(config)
    
    # Setup optimization
    optimizer, scheduler = optimization_manager.setup_optimization(model, 10000)
    
    # Test optimization step
    loss = torch.tensor(1.0, requires_grad=True)
    stats = optimization_manager.optimization_step(model, loss)
    print(f"Optimization stats: {stats}")
    
    print("✅ TruthGPT optimization setup completed!")



