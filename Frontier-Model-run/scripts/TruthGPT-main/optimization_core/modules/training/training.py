"""
TruthGPT Training Module
Advanced training utilities for TruthGPT models following deep learning best practices
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
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTTrainingConfig:
    """Configuration for TruthGPT training."""
    # Model configuration
    model_name: str = "truthgpt"
    model_size: str = "base"
    precision: str = "fp16"  # fp32, fp16, bf16
    device: str = "auto"
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    max_sequence_length: int = 2048
    
    # Optimization configuration
    optimizer_type: str = "adamw"  # adam, adamw, sgd, rmsprop
    scheduler_type: str = "cosine"  # linear, cosine, exponential, step
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Mixed precision training
    enable_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Logging and monitoring
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    enable_wandb: bool = False
    wandb_project: str = "truthgpt"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Advanced features
    enable_gradient_checkpointing: bool = True
    enable_attention_optimization: bool = True
    enable_memory_optimization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'precision': self.precision,
            'device': self.device,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'max_sequence_length': self.max_sequence_length,
            'optimizer_type': self.optimizer_type,
            'scheduler_type': self.scheduler_type,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'enable_mixed_precision': self.enable_mixed_precision,
            'gradient_clip_norm': self.gradient_clip_norm,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'eval_interval': self.eval_interval,
            'enable_wandb': self.enable_wandb,
            'wandb_project': self.wandb_project,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'prefetch_factor': self.prefetch_factor,
            'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
            'enable_attention_optimization': self.enable_attention_optimization,
            'enable_memory_optimization': self.enable_memory_optimization
        }

@dataclass
class TruthGPTTrainingMetrics:
    """Training metrics for TruthGPT."""
    # Training metrics
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    
    # Performance metrics
    throughput: float = 0.0
    memory_used_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    
    # Training time
    epoch_time: float = 0.0
    step_time: float = 0.0
    total_time: float = 0.0
    
    # Validation metrics
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_perplexity: float = 0.0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'epoch': self.epoch,
            'step': self.step,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'throughput': self.throughput,
            'memory_used_mb': self.memory_used_mb,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'epoch_time': self.epoch_time,
            'step_time': self.step_time,
            'total_time': self.total_time,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy,
            'val_perplexity': self.val_perplexity,
            'timestamp': self.timestamp
        }

class TruthGPTTrainer:
    """Advanced trainer for TruthGPT models."""
    
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
        self.training_metrics: List[TruthGPTTrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Setup training
        self._setup_training()
    
    def _setup_training(self) -> None:
        """Setup training components."""
        self.logger.info("🚀 Setting up TruthGPT training")
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup mixed precision
        if self.config.enable_mixed_precision:
            self.scaler = GradScaler()
            self.logger.info("✅ Mixed precision training enabled")
        
        # Setup wandb if enabled
        if self.config.enable_wandb:
            try:
                import wandb
                wandb.init(project=self.config.wandb_project, config=self.config.to_dict())
                self.logger.info("✅ Wandb logging enabled")
            except ImportError:
                self.logger.warning("Wandb not available, skipping logging")
    
    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
            self.logger.info(f"Using specified device: {device}")
        
        return device
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for training."""
        self.logger.info("🔧 Setting up TruthGPT model")
        
        # Move model to device
        model = model.to(self.device)
        
        # Apply precision
        if self.config.precision == "fp16":
            model = model.half()
        elif self.config.precision == "bf16":
            model = model.bfloat16()
        
        # Enable gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
            self.logger.info("✅ Gradient checkpointing enabled")
        
        # Enable memory optimizations
        if self.config.enable_memory_optimization:
            if hasattr(model, 'enable_memory_efficient_attention'):
                model.enable_memory_efficient_attention()
                self.logger.info("✅ Memory efficient attention enabled")
        
        self.model = model
        return model
    
    def setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Setup optimizer for training."""
        self.logger.info(f"🔧 Setting up {self.config.optimizer_type} optimizer")
        
        # Get parameters
        params = model.parameters()
        
        # Create optimizer
        if self.config.optimizer_type.lower() == "adam":
            optimizer = optim.Adam(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type.lower() == "rmsprop":
            optimizer = optim.RMSprop(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
        
        self.optimizer = optimizer
        return optimizer
    
    def setup_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        self.logger.info(f"🔧 Setting up {self.config.scheduler_type} scheduler")
        
        # Calculate total steps
        total_steps = num_training_steps
        
        # Create scheduler
        if self.config.scheduler_type.lower() == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                total_iters=self.config.warmup_steps
            )
        elif self.config.scheduler_type.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type.lower() == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=0.95
            )
        elif self.config.scheduler_type.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=total_steps // 4, 
                gamma=0.5
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        
        self.scheduler = scheduler
        return scheduler
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, 
                   epoch: int) -> TruthGPTTrainingMetrics:
        """Train one epoch."""
        self.model.train()
        
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            step_start_time = time.time()
            
            # Prepare batch
            if isinstance(batch, (list, tuple)):
                input_ids, labels = batch
            else:
                input_ids = batch
                labels = batch
            
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Apply precision
            if self.config.precision == "fp16":
                input_ids = input_ids.half()
                labels = labels.half()
            elif self.config.precision == "bf16":
                input_ids = input_ids.bfloat16()
                labels = labels.bfloat16()
            
            # Forward pass with mixed precision
            if self.config.enable_mixed_precision:
                with autocast():
                    outputs = self.model(input_ids)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(input_ids)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.enable_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    if self.config.enable_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                # Optimizer step
                if self.config.enable_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Calculate step time
            step_time = time.time() - step_start_time
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'step_time': f"{step_time:.3f}s"
            })
            
            # Log metrics
            if (batch_idx + 1) % self.config.log_interval == 0:
                self._log_training_metrics(epoch, batch_idx, loss.item(), step_time)
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches
        
        # Create metrics
        metrics = TruthGPTTrainingMetrics(
            epoch=epoch,
            step=epoch * len(train_loader),
            loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            throughput=len(train_loader) / epoch_time,
            epoch_time=epoch_time,
            step_time=epoch_time / len(train_loader)
        )
        
        self.training_metrics.append(metrics)
        return metrics
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> TruthGPTTrainingMetrics:
        """Validate model."""
        self.model.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Prepare batch
                if isinstance(batch, (list, tuple)):
                    input_ids, labels = batch
                else:
                    input_ids = batch
                    labels = batch
                
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Apply precision
                if self.config.precision == "fp16":
                    input_ids = input_ids.half()
                    labels = labels.half()
                elif self.config.precision == "bf16":
                    input_ids = input_ids.bfloat16()
                    labels = labels.bfloat16()
                
                # Forward pass
                if self.config.enable_mixed_precision:
                    with autocast():
                        outputs = self.model(input_ids)
                        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                else:
                    outputs = self.model(input_ids)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == labels).float().mean()
                
                val_loss += loss.item()
                val_accuracy += accuracy.item()
                num_batches += 1
        
        # Calculate average metrics
        avg_val_loss = val_loss / num_batches
        avg_val_accuracy = val_accuracy / num_batches
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        # Create metrics
        metrics = TruthGPTTrainingMetrics(
            val_loss=avg_val_loss,
            val_accuracy=avg_val_accuracy,
            val_perplexity=val_perplexity
        )
        
        return metrics
    
    def _log_training_metrics(self, epoch: int, batch_idx: int, loss: float, step_time: float) -> None:
        """Log training metrics."""
        metrics = {
            'epoch': epoch,
            'batch': batch_idx,
            'loss': loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            'step_time': step_time
        }
        
        self.logger.info(f"Training metrics: {metrics}")
        
        # Log to wandb
        if self.config.enable_wandb:
            try:
                import wandb
                wandb.log(metrics)
            except ImportError:
                pass
    
    def save_checkpoint(self, filepath: str, epoch: int, step: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config.to_dict(),
            'metrics': [m.to_dict() for m in self.training_metrics]
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def get_training_metrics(self) -> List[TruthGPTTrainingMetrics]:
        """Get training metrics."""
        return self.training_metrics

# Factory functions
def create_truthgpt_trainer(config: TruthGPTTrainingConfig) -> TruthGPTTrainer:
    """Create TruthGPT trainer."""
    return TruthGPTTrainer(config)

def quick_truthgpt_training(model: nn.Module, 
                           train_loader: torch.utils.data.DataLoader,
                           val_loader: torch.utils.data.DataLoader,
                           config: TruthGPTTrainingConfig) -> TruthGPTTrainer:
    """Quick TruthGPT training setup."""
    trainer = create_truthgpt_trainer(config)
    
    # Setup model
    trainer.setup_model(model)
    
    # Setup optimizer
    trainer.setup_optimizer(model)
    
    # Setup scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    trainer.setup_scheduler(trainer.optimizer, num_training_steps)
    
    return trainer

# Context managers
@contextmanager
def truthgpt_training_context(model: nn.Module, config: TruthGPTTrainingConfig):
    """Context manager for TruthGPT training."""
    trainer = create_truthgpt_trainer(config)
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
    
    # Create model
    model = TruthGPTModel()
    
    # Create configuration
    config = TruthGPTTrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=10,
        enable_mixed_precision=True
    )
    
    # Create trainer
    trainer = create_truthgpt_trainer(config)
    
    # Setup model
    trainer.setup_model(model)
    
    # Setup optimizer
    trainer.setup_optimizer(model)
    
    print("✅ TruthGPT training setup completed!")



