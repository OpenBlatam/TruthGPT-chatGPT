"""
Advanced Training System
========================

Production-ready training system with:
- Mixed precision training
- Gradient accumulation
- Distributed training support
- Comprehensive logging
- Experiment tracking
- Callback system
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from .data_loader import DataLoader as CustomDataLoader
from .scheduler import LearningRateScheduler
from .callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor
from .metrics import MetricsCalculator
from ..models.base import BaseModel


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Mixed precision
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd, rmsprop
    scheduler: str = "cosine"  # cosine, step, exponential, plateau
    warmup_epochs: int = 5
    
    # Regularization
    dropout: float = 0.1
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Model checkpointing
    save_best_model: bool = True
    save_every_n_epochs: int = 10
    checkpoint_dir: str = "checkpoints"
    
    # Logging and monitoring
    log_every_n_steps: int = 10
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "truthgpt-optimization"
    
    # Validation
    validate_every_n_epochs: int = 1
    validation_split: float = 0.2
    
    # Device and distributed training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    seed: Optional[int] = None
    deterministic: bool = True


class Trainer:
    """
    Advanced training system with comprehensive features.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Distributed training
    - Comprehensive logging
    - Experiment tracking
    - Callback system
    - Model checkpointing
    """
    
    def __init__(self, model: BaseModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Mixed precision setup
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.metrics_calculator = MetricsCalculator()
        
        # Callbacks
        self.callbacks: List[Callback] = []
        self._setup_default_callbacks()
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=config.__dict__)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.training_history = []
        
        # Setup reproducibility
        if config.seed is not None:
            self._set_seed(config.seed)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config"""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif self.config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.early_stopping_patience // 2,
                factor=0.5
            )
        else:
            return None
    
    def _setup_default_callbacks(self):
        """Setup default callbacks"""
        # Early stopping
        if self.config.early_stopping_patience > 0:
            self.add_callback(EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta
            ))
        
        # Model checkpointing
        if self.config.save_best_model:
            self.add_callback(ModelCheckpoint(
                save_dir=self.config.checkpoint_dir,
                save_best_only=True
            ))
        
        # Learning rate monitoring
        self.add_callback(LearningRateMonitor())
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def add_callback(self, callback: Callback):
        """Add training callback"""
        callback.trainer = self
        self.callbacks.append(callback)
        self.logger.info(f"Added callback: {callback.__class__.__name__}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              loss_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            
        Returns:
            Training history and metrics
        """
        self.logger.info("Starting training...")
        
        # Setup loss function
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        # Call training start callbacks
        for callback in self.callbacks:
            callback.on_train_begin()
        
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Call epoch start callbacks
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch)
                
                # Training phase
                train_metrics = self._train_epoch(train_loader, loss_fn)
                
                # Validation phase
                val_metrics = {}
                if val_loader is not None and epoch % self.config.validate_every_n_epochs == 0:
                    val_metrics = self._validate_epoch(val_loader, loss_fn)
                
                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                    else:
                        self.scheduler.step()
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                epoch_metrics['epoch'] = epoch
                epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                # Store training history
                self.training_history.append(epoch_metrics)
                
                # Log metrics
                self._log_metrics(epoch_metrics)
                
                # Call epoch end callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, epoch_metrics)
                
                # Check for early stopping
                if any(callback.should_stop for callback in self.callbacks):
                    self.logger.info("Early stopping triggered")
                    break
            
            # Call training end callbacks
            for callback in self.callbacks:
                callback.on_train_end()
            
            self.logger.info("Training completed")
            
            return {
                'history': self.training_history,
                'best_metric': self.best_metric,
                'total_epochs': self.current_epoch + 1
            }
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            return {
                'history': self.training_history,
                'best_metric': self.best_metric,
                'total_epochs': self.current_epoch + 1,
                'interrupted': True
            }
    
    def _train_epoch(self, train_loader: DataLoader, loss_fn: Callable) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(batch['input'])
                    loss = loss_fn(outputs, batch['target'])
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                outputs = self.model(batch['input'])
                loss = loss_fn(outputs, batch['target'])
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_every_n_steps == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            # Call step callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, {'loss': loss.item()})
        
        return {
            'train_loss': total_loss / num_batches,
            'train_batches': num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader, loss_fn: Callable) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(batch['input'])
                        loss = loss_fn(outputs, batch['target'])
                else:
                    outputs = self.model(batch['input'])
                    loss = loss_fn(outputs, batch['target'])
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_batches': num_batches
        }
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()}
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to various backends"""
        # TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, self.current_epoch)
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.log(metrics, step=self.current_epoch)
        
        # Console logging
        self.logger.info(f"Epoch {self.current_epoch}: {metrics}")
    
    def evaluate(self, test_loader: DataLoader, loss_fn: Optional[Callable] = None) -> Dict[str, float]:
        """Evaluate model on test data"""
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = self._move_batch_to_device(batch)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(batch['input'])
                        loss = loss_fn(outputs, batch['target'])
                else:
                    outputs = self.model(batch['input'])
                    loss = loss_fn(outputs, batch['target'])
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Test loss: {avg_loss:.4f}")
        
        return {'test_loss': avg_loss}
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            **kwargs
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def cleanup(self):
        """Cleanup training resources"""
        if hasattr(self, 'writer'):
            self.writer.close()
        
        if self.config.use_wandb:
            wandb.finish()



