"""
TruthGPT Advanced Training System
Production-ready training utilities with state-of-the-art techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, TypeVar, Iterator
from dataclasses import dataclass, field
from contextlib import contextmanager
import time
import json
import pickle
from pathlib import Path
import math
import warnings
import psutil
import gc
from collections import defaultdict, deque
import hashlib
import uuid
from datetime import datetime, timezone
import functools
import inspect
import traceback
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from abc import ABC, abstractmethod
import wandb
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)
T = TypeVar('T')

@dataclass
class TruthGPTTrainingConfig:
    """Advanced TruthGPT training configuration."""
    # Model configuration
    model_name: str = "truthgpt"
    model_size: str = "base"  # base, large, xl, xxl
    architecture: str = "transformer"  # transformer, gpt, bert, etc.
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    
    # Advanced training techniques
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    data_parallel: bool = True
    distributed_training: bool = False
    deepspeed: bool = False
    
    # Optimization techniques
    optimizer: str = "adamw"  # adam, adamw, sgd, adagrad, rmsprop
    scheduler: str = "cosine"  # linear, cosine, exponential, plateau
    lr_scheduler_warmup: bool = True
    lr_scheduler_decay: float = 0.1
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    label_smoothing: float = 0.0
    
    # Advanced features
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True
    use_relative_position: bool = False
    use_gradient_accumulation: bool = True
    use_ema: bool = False  # Exponential Moving Average
    
    # Monitoring and logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    wandb_project: str = "truthgpt"
    wandb_entity: str = None
    
    # Data configuration
    max_sequence_length: int = 4096
    vocab_size: int = 50257
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda, cuda:0, etc.
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    checkpoint_format: str = "pytorch"  # pytorch, safetensors, onnx
    resume_from_checkpoint: Optional[str] = None
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_metric: str = "loss"
    early_stopping_mode: str = "min"  # min, max
    
    # Validation
    validation_split: float = 0.1
    validation_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy", "perplexity"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'warmup_steps': self.warmup_steps,
            'max_grad_norm': self.max_grad_norm,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'mixed_precision': self.mixed_precision,
            'gradient_checkpointing': self.gradient_checkpointing,
            'data_parallel': self.data_parallel,
            'distributed_training': self.distributed_training,
            'deepspeed': self.deepspeed,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'lr_scheduler_warmup': self.lr_scheduler_warmup,
            'lr_scheduler_decay': self.lr_scheduler_decay,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'label_smoothing': self.label_smoothing,
            'use_flash_attention': self.use_flash_attention,
            'use_rotary_embeddings': self.use_rotary_embeddings,
            'use_relative_position': self.use_relative_position,
            'use_gradient_accumulation': self.use_gradient_accumulation,
            'use_ema': self.use_ema,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval,
            'save_interval': self.save_interval,
            'tensorboard_logging': self.tensorboard_logging,
            'wandb_logging': self.wandb_logging,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'max_sequence_length': self.max_sequence_length,
            'vocab_size': self.vocab_size,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'device': self.device,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'save_dir': self.save_dir,
            'checkpoint_format': self.checkpoint_format,
            'resume_from_checkpoint': self.resume_from_checkpoint,
            'early_stopping': self.early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_metric': self.early_stopping_metric,
            'early_stopping_mode': self.early_stopping_mode,
            'validation_split': self.validation_split,
            'validation_metrics': self.validation_metrics
        }

class TruthGPTAdvancedTrainer:
    """Advanced TruthGPT trainer with state-of-the-art techniques."""
    
    def __init__(self, config: TruthGPTTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ema = None
        
        # Training metrics
        self.training_metrics = defaultdict(list)
        self.validation_metrics = defaultdict(list)
        self.best_metric = float('inf') if config.early_stopping_mode == 'min' else float('-inf')
        self.early_stopping_counter = 0
        
        # Logging and monitoring
        self.writer = None
        self.wandb_run = None
        
        # Device setup
        self.device = self._setup_device()
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("🚀 Advanced TruthGPT trainer initialized")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("💻 Using CPU device")
        else:
            device = torch.device(self.config.device)
            self.logger.info(f"🔧 Using specified device: {device}")
        
        return device
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        # TensorBoard logging
        if self.config.tensorboard_logging:
            log_dir = Path(self.config.save_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.logger.info("📊 TensorBoard logging enabled")
        
        # Weights & Biases logging
        if self.config.wandb_logging:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    config=self.config.to_dict(),
                    name=f"truthgpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.wandb_run = wandb
                self.logger.info("🔬 Weights & Biases logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for training."""
        self.logger.info("🔧 Setting up TruthGPT model for training")
        
        # Move model to device
        model = model.to(self.device)
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.logger.info("✅ Gradient checkpointing enabled")
        
        # Data parallel training
        if self.config.data_parallel and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            self.logger.info(f"🔄 Data parallel training enabled on {torch.cuda.device_count()} GPUs")
        
        # Mixed precision scaler
        if self.config.mixed_precision:
            self.scaler = GradScaler()
            self.logger.info("⚡ Mixed precision training enabled")
        
        # Exponential Moving Average
        if self.config.use_ema:
            self.ema = self._create_ema(model)
            self.logger.info("📈 Exponential Moving Average enabled")
        
        self.model = model
        return model
    
    def _create_ema(self, model: nn.Module, decay: float = 0.999) -> Dict[str, torch.Tensor]:
        """Create Exponential Moving Average for model parameters."""
        ema_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema_params[name] = param.data.clone()
        return ema_params
    
    def _update_ema(self):
        """Update Exponential Moving Average parameters."""
        if self.ema is None:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ema:
                self.ema[name] = 0.999 * self.ema[name] + 0.001 * param.data
    
    def setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Setup optimizer with advanced techniques."""
        self.logger.info(f"🔧 Setting up {self.config.optimizer} optimizer")
        
        # Get model parameters
        if hasattr(model, 'module'):  # DataParallel case
            parameters = model.module.parameters()
        else:
            parameters = model.parameters()
        
        # Create optimizer
        if self.config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        self.optimizer = optimizer
        return optimizer
    
    def setup_scheduler(self, optimizer: optim.Optimizer, 
                       total_steps: int) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        self.logger.info(f"📈 Setting up {self.config.scheduler} scheduler")
        
        if self.config.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        elif self.config.scheduler.lower() == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
        elif self.config.scheduler.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
        
        self.scheduler = scheduler
        return scheduler
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   epoch: int) -> Dict[str, float]:
        """Train one epoch with advanced techniques."""
        model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = len(dataloader)
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [item.to(self.device) for item in batch]
            else:
                batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    loss = self._compute_loss(model, batch)
            else:
                loss = self._compute_loss(model, batch)
            
            # Backward pass
            if self.config.use_gradient_accumulation:
                loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                # Update EMA
                if self.config.use_ema:
                    self._update_ema()
                
                # Clear gradients
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            if isinstance(batch, (list, tuple)):
                total_tokens += batch[0].size(0) * batch[0].size(1)
            else:
                total_tokens += batch.size(0) * batch.size(1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self._log_training_metrics(epoch, batch_idx, loss.item(), 
                                         self.optimizer.param_groups[0]['lr'])
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        epoch_metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens_per_second': total_tokens / (time.time() - getattr(self, '_epoch_start_time', time.time()))
        }
        
        return epoch_metrics
    
    def _compute_loss(self, model: nn.Module, batch: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Compute loss for a batch."""
        if isinstance(batch, (list, tuple)):
            input_ids, labels = batch
        else:
            input_ids = batch
            labels = batch  # For language modeling, labels are the same as input_ids
        
        # Forward pass
        outputs = model(input_ids)
        
        # Compute loss
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Shift labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def _log_training_metrics(self, epoch: int, batch_idx: int, loss: float, lr: float):
        """Log training metrics."""
        metrics = {
            'epoch': epoch,
            'batch': batch_idx,
            'loss': loss,
            'learning_rate': lr,
            'timestamp': time.time()
        }
        
        # Store metrics
        for key, value in metrics.items():
            if key != 'timestamp':
                self.training_metrics[key].append(value)
        
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Training/Loss', loss, epoch * len(self.training_metrics['batch']) + batch_idx)
            self.writer.add_scalar('Training/LearningRate', lr, epoch * len(self.training_metrics['batch']) + batch_idx)
        
        # Weights & Biases logging
        if self.wandb_run:
            self.wandb_run.log({
                'training/loss': loss,
                'training/learning_rate': lr,
                'training/epoch': epoch,
                'training/batch': batch_idx
            })
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(self.device) for item in batch]
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast():
                        loss = self._compute_loss(model, batch)
                else:
                    loss = self._compute_loss(model, batch)
                
                total_loss += loss.item()
                
                # Calculate accuracy (simplified)
                if isinstance(batch, (list, tuple)):
                    input_ids, labels = batch
                    total_tokens += input_ids.size(0) * input_ids.size(1)
                else:
                    total_tokens += batch.size(0) * batch.size(1)
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        eval_metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'tokens_per_second': total_tokens / (time.time() - getattr(self, '_eval_start_time', time.time()))
        }
        
        return eval_metrics
    
    def train(self, model: nn.Module, train_dataloader: DataLoader, 
              val_dataloader: Optional[DataLoader] = None) -> nn.Module:
        """Complete training loop with advanced features."""
        self.logger.info("🚀 Starting TruthGPT training")
        
        # Setup model
        model = self.setup_model(model)
        
        # Setup optimizer
        self.setup_optimizer(model)
        
        # Setup scheduler
        total_steps = len(train_dataloader) * self.config.max_epochs
        self.setup_scheduler(self.optimizer, total_steps)
        
        # Resume from checkpoint
        start_epoch = 0
        if self.config.resume_from_checkpoint:
            start_epoch = self._load_checkpoint(model, self.config.resume_from_checkpoint)
        
        # Training loop
        for epoch in range(start_epoch, self.config.max_epochs):
            self._epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(model, train_dataloader, epoch)
            
            # Validation
            if val_dataloader and epoch % (self.config.eval_interval // len(train_dataloader)) == 0:
                self._eval_start_time = time.time()
                val_metrics = self.evaluate(model, val_dataloader)
                self.validation_metrics.update(val_metrics)
                
                # Log validation metrics
                self._log_validation_metrics(epoch, val_metrics)
                
                # Early stopping
                if self.config.early_stopping:
                    if self._check_early_stopping(val_metrics):
                        self.logger.info(f"🛑 Early stopping at epoch {epoch}")
                        break
            
            # Save checkpoint
            if epoch % (self.config.save_interval // len(train_dataloader)) == 0:
                self._save_checkpoint(model, epoch, train_metrics)
            
            # Log epoch metrics
            self._log_epoch_metrics(epoch, train_metrics)
        
        self.logger.info("✅ TruthGPT training completed")
        return model
    
    def _log_validation_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log validation metrics."""
        # Store metrics
        for key, value in metrics.items():
            self.validation_metrics[key].append(value)
        
        # TensorBoard logging
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Validation/{key.title()}', value, epoch)
        
        # Weights & Biases logging
        if self.wandb_run:
            wandb_metrics = {f'validation/{key}': value for key, value in metrics.items()}
            wandb_metrics['validation/epoch'] = epoch
            self.wandb_run.log(wandb_metrics)
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics."""
        self.logger.info(f"Epoch {epoch} - " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        
        # TensorBoard logging
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Training/{key.title()}', value, epoch)
        
        # Weights & Biases logging
        if self.wandb_run:
            wandb_metrics = {f'training/{key}': value for key, value in metrics.items()}
            wandb_metrics['training/epoch'] = epoch
            self.wandb_run.log(wandb_metrics)
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria are met."""
        metric_value = metrics.get(self.config.early_stopping_metric, float('inf'))
        
        if self.config.early_stopping_mode == 'min':
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
        else:  # max
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema if self.ema else None,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if metrics.get('loss', float('inf')) < self.best_metric:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_metric = metrics['loss']
        
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load EMA state
        if 'ema_state_dict' in checkpoint and self.ema:
            self.ema = checkpoint['ema_state_dict']
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        self.logger.info(f"📂 Checkpoint loaded from epoch {start_epoch - 1}")
        
        return start_epoch
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer:
            self.writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()
        
        self.logger.info("🧹 Training resources cleaned up")

# Advanced factory functions
def create_advanced_trainer(config: TruthGPTTrainingConfig) -> TruthGPTAdvancedTrainer:
    """Create advanced TruthGPT trainer."""
    return TruthGPTAdvancedTrainer(config)

def quick_advanced_training(model: nn.Module, 
                          train_dataloader: DataLoader,
                          val_dataloader: Optional[DataLoader] = None,
                          learning_rate: float = 1e-4,
                          max_epochs: int = 100,
                          mixed_precision: bool = True) -> nn.Module:
    """Quick advanced training setup."""
    config = TruthGPTTrainingConfig(
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        mixed_precision=mixed_precision
    )
    
    trainer = create_advanced_trainer(config)
    return trainer.train(model, train_dataloader, val_dataloader)

# Advanced context managers
@contextmanager
def advanced_training_context(model: nn.Module, config: TruthGPTTrainingConfig):
    """Advanced training context manager."""
    trainer = create_advanced_trainer(config)
    try:
        yield trainer
    finally:
        trainer.cleanup()

# Example usage
if __name__ == "__main__":
    # Advanced TruthGPT training example
    print("🚀 Advanced TruthGPT Training Demo")
    print("=" * 60)
    
    # Create a sample TruthGPT model
    class TruthGPTModel(nn.Module):
        def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, 12, 3072, dropout=0.1),
                num_layers=num_layers
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.lm_head(x)
            return x
    
    # Create model and config
    model = TruthGPTModel()
    config = TruthGPTTrainingConfig(
        learning_rate=1e-4,
        max_epochs=10,
        mixed_precision=True,
        gradient_checkpointing=True
    )
    
    # Advanced training
    with advanced_training_context(model, config) as trainer:
        # Setup model
        model = trainer.setup_model(model)
        
        print("✅ Advanced TruthGPT training setup completed!")
        print("🔧 Model configured with advanced features:")
        print(f"   - Mixed precision: {config.mixed_precision}")
        print(f"   - Gradient checkpointing: {config.gradient_checkpointing}")
        print(f"   - Data parallel: {config.data_parallel}")
        print(f"   - EMA: {config.use_ema}")



