"""
TruthGPT Training Utilities
Advanced training utilities for TruthGPT models with optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import time
from pathlib import Path
import json
from enum import Enum
import math

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTTrainingConfig:
    """TruthGPT training configuration."""
    # Model configuration
    model_name: str = "truthgpt"
    model_size: str = "base"  # base, large, xl
    precision: str = "fp16"  # fp32, fp16, bf16
    
    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization configuration
    optimizer_type: str = "adamw"  # adam, adamw, sgd
    scheduler_type: str = "cosine"  # linear, cosine, constant
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_gradient_accumulation: bool = True
    accumulation_steps: int = 4
    
    # Data configuration
    max_sequence_length: int = 2048
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Monitoring configuration
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'precision': self.precision,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'warmup_steps': self.warmup_steps,
            'max_grad_norm': self.max_grad_norm,
            'optimizer_type': self.optimizer_type,
            'scheduler_type': self.scheduler_type,
            'enable_mixed_precision': self.enable_mixed_precision,
            'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
            'enable_gradient_accumulation': self.enable_gradient_accumulation,
            'accumulation_steps': self.accumulation_steps,
            'max_sequence_length': self.max_sequence_length,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'eval_interval': self.eval_interval,
            'enable_tensorboard': self.enable_tensorboard,
            'enable_wandb': self.enable_wandb
        }

class TruthGPTTrainer:
    """TruthGPT trainer with advanced optimization."""
    
    def __init__(self, config: TruthGPTTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.device = None
        
        # Training metrics
        self.training_metrics = {
            'epoch': 0,
            'step': 0,
            'total_steps': 0,
            'best_loss': float('inf'),
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        # Setup device
        self._setup_device()
        
        # Setup mixed precision
        if self.config.enable_mixed_precision:
            self.scaler = GradScaler()
    
    def _setup_device(self):
        """Setup training device."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU device")
    
    def setup_training(self, model: nn.Module, train_loader, val_loader=None):
        """Setup training components."""
        self.logger.info("🔧 Setting up TruthGPT training")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Apply precision
        if self.config.precision == "fp16":
            self.model = self.model.half()
        elif self.config.precision == "bf16":
            self.model = self.model.bfloat16()
        
        # Enable gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
            self.logger.info("✅ Gradient checkpointing enabled")
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Setup data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.logger.info("✅ TruthGPT training setup completed")
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
        
        self.logger.info(f"✅ {self.config.optimizer_type.upper()} optimizer setup")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.max_epochs
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs
            )
        elif self.config.scheduler_type == "constant":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        
        if self.scheduler:
            self.logger.info(f"✅ {self.config.scheduler_type} scheduler setup")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [item.to(self.device) for item in batch]
            else:
                batch = batch.to(self.device)
            
            # Apply precision
            if self.config.precision == "fp16":
                batch = batch.half() if isinstance(batch, torch.Tensor) else batch
            elif self.config.precision == "bf16":
                batch = batch.bfloat16() if isinstance(batch, torch.Tensor) else batch
            
            # Forward pass with mixed precision
            if self.config.enable_mixed_precision:
                with autocast():
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)
            
            # Backward pass
            if self.config.enable_gradient_accumulation:
                loss = loss / self.config.accumulation_steps
            
            if self.config.enable_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.config.enable_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_grad_norm
                        )
                
                # Optimizer step
                if self.config.enable_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update step counter
                self.training_metrics['step'] += 1
                self.training_metrics['total_steps'] += 1
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if self.training_metrics['step'] % self.config.log_interval == 0:
                self._log_training_progress(epoch, batch_idx, loss.item())
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        # Update metrics
        avg_loss = total_loss / num_batches
        self.training_metrics['train_losses'].append(avg_loss)
        self.training_metrics['learning_rates'].append(
            self.optimizer.param_groups[0]['lr']
        )
        
        return {'train_loss': avg_loss}
    
    def _compute_loss(self, batch) -> torch.Tensor:
        """Compute training loss."""
        if isinstance(batch, (list, tuple)):
            input_ids, labels = batch
            outputs = self.model(input_ids)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            # Assume batch is input_ids and we need to compute language modeling loss
            input_ids = batch
            labels = input_ids.clone()
            outputs = self.model(input_ids)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return loss
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(self.device) for item in batch]
                else:
                    batch = batch.to(self.device)
                
                # Apply precision
                if self.config.precision == "fp16":
                    batch = batch.half() if isinstance(batch, torch.Tensor) else batch
                elif self.config.precision == "bf16":
                    batch = batch.bfloat16() if isinstance(batch, torch.Tensor) else batch
                
                # Compute loss
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.training_metrics['val_losses'].append(avg_loss)
        
        return {'val_loss': avg_loss}
    
    def _log_training_progress(self, epoch: int, batch_idx: int, loss: float):
        """Log training progress."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(
            f"Epoch {epoch}, Batch {batch_idx}, "
            f"Loss: {loss:.4f}, LR: {current_lr:.6f}, "
            f"Step: {self.training_metrics['step']}"
        )
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Train TruthGPT model."""
        if num_epochs is None:
            num_epochs = self.config.max_epochs
        
        self.logger.info(f"🚀 Starting TruthGPT training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.training_metrics['epoch'] = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch} completed - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics.get('val_loss', 'N/A')}"
            )
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        self.logger.info("✅ TruthGPT training completed")
        
        return {
            'final_train_loss': self.training_metrics['train_losses'][-1],
            'final_val_loss': self.training_metrics['val_losses'][-1] if self.training_metrics['val_losses'] else None,
            'total_epochs': num_epochs,
            'total_steps': self.training_metrics['total_steps']
        }
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_metrics': self.training_metrics,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"💾 Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_metrics = checkpoint['training_metrics']
        
        self.logger.info(f"📂 Checkpoint loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return self.training_metrics

class TruthGPTFineTuner:
    """TruthGPT fine-tuning utilities."""
    
    def __init__(self, config: TruthGPTTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.trainer = TruthGPTTrainer(config)
    
    def fine_tune(self, model: nn.Module, train_loader, val_loader=None, 
                 task_type: str = "language_modeling") -> nn.Module:
        """Fine-tune TruthGPT model."""
        self.logger.info(f"🎯 Starting TruthGPT fine-tuning for {task_type}")
        
        # Setup training
        self.trainer.setup_training(model, train_loader, val_loader)
        
        # Apply task-specific modifications
        if task_type == "language_modeling":
            self._setup_language_modeling()
        elif task_type == "classification":
            self._setup_classification()
        elif task_type == "generation":
            self._setup_generation()
        else:
            self.logger.warning(f"Unknown task type: {task_type}")
        
        # Train
        training_results = self.trainer.train()
        
        self.logger.info("✅ TruthGPT fine-tuning completed")
        return self.trainer.model
    
    def _setup_language_modeling(self):
        """Setup for language modeling task."""
        self.logger.info("Setting up language modeling task")
        # Language modeling setup would go here
    
    def _setup_classification(self):
        """Setup for classification task."""
        self.logger.info("Setting up classification task")
        # Classification setup would go here
    
    def _setup_generation(self):
        """Setup for generation task."""
        self.logger.info("Setting up generation task")
        # Generation setup would go here

# Factory functions
def create_truthgpt_trainer(config: TruthGPTTrainingConfig) -> TruthGPTTrainer:
    """Create TruthGPT trainer."""
    return TruthGPTTrainer(config)

def create_truthgpt_finetuner(config: TruthGPTTrainingConfig) -> TruthGPTFineTuner:
    """Create TruthGPT fine-tuner."""
    return TruthGPTFineTuner(config)

def quick_truthgpt_training(model: nn.Module, train_loader, 
                          learning_rate: float = 1e-4,
                          num_epochs: int = 10,
                          precision: str = "fp16") -> nn.Module:
    """Quick TruthGPT training setup."""
    config = TruthGPTTrainingConfig(
        learning_rate=learning_rate,
        max_epochs=num_epochs,
        precision=precision
    )
    
    trainer = create_truthgpt_trainer(config)
    trainer.setup_training(model, train_loader)
    trainer.train()
    
    return trainer.model

# For backward compatibility
TruthGPTTrainingUtils = TruthGPTTrainer

# Context managers
@contextmanager
def truthgpt_training_context(model: nn.Module, train_loader, config: TruthGPTTrainingConfig):
    """Context manager for TruthGPT training."""
    trainer = create_truthgpt_trainer(config)
    trainer.setup_training(model, train_loader)
    try:
        yield trainer
    finally:
        # Cleanup if needed
        pass

# Example usage
if __name__ == "__main__":
    # Example TruthGPT training
    print("🚀 TruthGPT Training Demo")
    print("=" * 50)
    
    # Create a sample TruthGPT-style model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 768)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.1),
                num_layers=12
            )
            self.lm_head = nn.Linear(768, 10000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.lm_head(x)
            return x
    
    # Create model and dummy data
    model = TruthGPTModel()
    
    # Create dummy data loader
    dummy_data = torch.randint(0, 10000, (1000, 512))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_data),
        batch_size=32,
        shuffle=True
    )
    
    # Quick training
    trained_model = quick_truthgpt_training(model, train_loader, num_epochs=2)
    
    print("✅ TruthGPT training completed!")



