"""
Base Model Architecture
=======================

Abstract base class for all optimization models following PyTorch best practices.
Implements proper weight initialization, mixed precision, and GPU utilization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import math


@dataclass
class ModelConfig:
    """Base configuration for all models"""
    model_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all optimization models.
    
    Features:
    - Proper weight initialization
    - Mixed precision training support
    - Gradient checkpointing
    - Model compilation
    - GPU utilization optimization
    - Comprehensive logging
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Device and dtype setup
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
        # Mixed precision setup
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Model compilation
        self._compiled = False
        
        # Initialize model
        self._initialize_model()
        
        # Move to device
        self.to(self.device)
        
        # Compile if requested
        if config.compile_model:
            self.compile()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize model architecture - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def _initialize_weights(self, module: nn.Module):
        """Initialize weights using best practices"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.Conv2d):
            # Kaiming initialization for conv layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.LayerNorm):
            # Layer normalization initialization
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        
        elif isinstance(module, nn.Embedding):
            # Embedding initialization
            nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def compile(self):
        """Compile model for better performance"""
        if self._compiled:
            return
        
        try:
            if hasattr(torch, 'compile'):
                self._compiled_model = torch.compile(self)
                self._compiled = True
                self.logger.info("Model compiled successfully")
            else:
                self.logger.warning("torch.compile not available")
        except Exception as e:
            self.logger.error(f"Model compilation failed: {e}")
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        if hasattr(self, 'gradient_checkpointing_enable'):
            self.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        if hasattr(self, 'gradient_checkpointing_disable'):
            self.gradient_checkpointing_disable()
            self.logger.info("Gradient checkpointing disabled")
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_memory_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'cached_memory_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
        }
    
    def optimize_for_inference(self):
        """Optimize model for inference"""
        self.eval()
        
        # Disable gradient computation
        for param in self.parameters():
            param.requires_grad = False
        
        # Enable inference optimizations
        if hasattr(torch, 'inference_mode'):
            self._inference_mode = True
        
        self.logger.info("Model optimized for inference")
    
    def optimize_for_training(self):
        """Optimize model for training"""
        self.train()
        
        # Enable gradient computation
        for param in self.parameters():
            param.requires_grad = True
        
        self.logger.info("Model optimized for training")
    
    def save_checkpoint(self, path: str, optimizer_state: Optional[Dict] = None, 
                       scheduler_state: Optional[Dict] = None, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'model_class': self.__class__.__name__,
            **kwargs
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> 'BaseModel':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        # Create model instance
        config = checkpoint['model_config']
        model = cls(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        if device:
            model.to(device)
        
        return model
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights if model has attention mechanism"""
        # This should be overridden by models with attention
        return None
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get model embeddings"""
        # This should be overridden by models that have embeddings
        return x
    
    def freeze_parameters(self, freeze: bool = True):
        """Freeze or unfreeze model parameters"""
        for param in self.parameters():
            param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        self.logger.info(f"Model parameters {status}")
    
    def unfreeze_parameters(self):
        """Unfreeze model parameters"""
        self.freeze_parameters(False)
    
    def get_gradient_norm(self) -> float:
        """Get gradient norm for monitoring"""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients to prevent exploding gradients"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    def zero_grad(self):
        """Zero gradients"""
        self.zero_grad()
    
    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
                   loss_fn: callable, **kwargs) -> Dict[str, float]:
        """Single training step with mixed precision support"""
        self.train()
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.scaler:
            with autocast():
                outputs = self.forward(batch['input'], **kwargs)
                loss = loss_fn(outputs, batch['target'])
            
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = self.forward(batch['input'], **kwargs)
            loss = loss_fn(outputs, batch['target'])
            loss.backward()
            optimizer.step()
        
        return {
            'loss': loss.item(),
            'gradient_norm': self.get_gradient_norm()
        }
    
    def inference_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Single inference step"""
        self.eval()
        
        with torch.no_grad():
            if self.scaler:
                with autocast():
                    outputs = self.forward(batch['input'], **kwargs)
            else:
                outputs = self.forward(batch['input'], **kwargs)
        
        return {'outputs': outputs}
    
    def __repr__(self) -> str:
        """String representation of model"""
        model_size = self.get_model_size()
        return (f"{self.__class__.__name__}(\n"
                f"  config={self.config},\n"
                f"  parameters={model_size['total_parameters']:,},\n"
                f"  device={self.device}\n"
                f")")



