"""
Training service for orchestrating training workflows.
"""
import logging
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader

from .base_service import BaseService
from ..event_system import EventType
from ...training.training_loop import TrainingLoop
from ...training.evaluator import Evaluator
from ...training.checkpoint_manager import CheckpointManager
from ...training.ema_manager import EMAManager
from ...training.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class TrainingService(BaseService):
    """
    Service for training orchestration.
    Coordinates training loop, evaluation, checkpointing, and tracking.
    """
    
    def __init__(self, **kwargs):
        """Initialize training service."""
        super().__init__(name="TrainingService", **kwargs)
        self.training_loop: Optional[TrainingLoop] = None
        self.evaluator: Optional[Evaluator] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.ema_manager: Optional[EMAManager] = None
        self.tracker: Optional[ExperimentTracker] = None
    
    def _do_initialize(self) -> None:
        """Initialize training components."""
        # Components will be created on demand or via configure method
        pass
    
    def configure(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        device: torch.device,
        output_dir: str,
    ) -> None:
        """
        Configure training service with all components.
        
        Args:
            config: Training configuration
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: GradScaler
            device: Training device
            output_dir: Output directory
        """
        # Training loop
        self.training_loop = TrainingLoop(
            use_amp=config.get("use_amp", True),
            amp_dtype=torch.bfloat16 if config.get("mixed_precision") == "bf16" else torch.float16,
            max_grad_norm=config.get("max_grad_norm", 1.0),
            grad_accum_steps=config.get("grad_accum_steps", 1),
        )
        
        # Evaluator
        self.evaluator = Evaluator(
            use_amp=config.get("use_amp", True),
            amp_dtype=torch.bfloat16 if config.get("mixed_precision") == "bf16" else torch.float16,
            device=device,
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(output_dir=output_dir)
        
        # EMA manager
        if config.get("ema_enabled", False):
            self.ema_manager = EMAManager(
                decay=config.get("ema_decay", 0.999),
                model=model,
            )
        
        # Experiment tracker
        if config.get("tracking_enabled", False):
            self.tracker = ExperimentTracker(
                trackers=config.get("trackers", []),
                project=config.get("project"),
                run_name=config.get("run_name"),
                log_dir=config.get("log_dir"),
            )
        
        logger.info("Training service configured")
    
    def train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        epoch: int,
    ) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: GradScaler
            epoch: Current epoch number
        
        Returns:
            Dictionary with epoch metrics
        """
        if not self.training_loop:
            raise RuntimeError("Training service not configured")
        
        self.emit(EventType.TRAINING_EPOCH, {"epoch": epoch, "status": "started"})
        
        # Define step callback
        def step_callback(step: int, metrics: Dict, learning_rate: float):
            step_data = {
                "step": step,
                "loss": metrics.get("loss"),
                "learning_rate": learning_rate,
            }
            
            # Emit event
            self.emit(EventType.TRAINING_STEP, step_data)
            
            # Log to tracker
            if self.tracker:
                self.tracker.log(step_data, step=step)
            
            # Update EMA
            if self.ema_manager:
                self.ema_manager.update(model)
        
        # Train epoch
        epoch_metrics = self.training_loop.train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step_callback=step_callback,
        )
        
        self.emit(EventType.TRAINING_EPOCH, {
            "epoch": epoch,
            "status": "finished",
            "metrics": epoch_metrics,
        })
        
        return epoch_metrics
    
    def evaluate(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            device: Device for evaluation
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.evaluator:
            raise RuntimeError("Training service not configured")
        
        self.emit(EventType.EVALUATION_STARTED, {})
        
        # Apply EMA if enabled
        if self.ema_manager:
            self.ema_manager.apply_to_model(model)
        
        # Evaluate
        metrics = self.evaluator.evaluate(model, val_loader, device)
        
        # Restore model weights
        if self.ema_manager:
            self.ema_manager.restore_from_backup(model)
        
        # Emit event
        self.emit(EventType.EVALUATION_FINISHED, metrics)
        
        # Log to tracker
        if self.tracker:
            self.tracker.log(metrics)
        
        return metrics
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        step: int,
        path: str,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer
            scheduler: Scheduler
            scaler: GradScaler
            step: Training step
            path: Checkpoint path
            tokenizer: Optional tokenizer
        """
        if not self.checkpoint_manager:
            raise RuntimeError("Training service not configured")
        
        ema_state = None
        if self.ema_manager:
            ema_state = self.ema_manager.state_dict()
        
        self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            path=path,
            tokenizer=tokenizer,
            ema_state=ema_state,
        )
        
        self.emit(EventType.CHECKPOINT_SAVED, {"path": path, "step": step})
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        path: str,
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer
            scheduler: Scheduler
            scaler: GradScaler
            path: Checkpoint path
        
        Returns:
            Dictionary with loaded state
        """
        if not self.checkpoint_manager:
            raise RuntimeError("Training service not configured")
        
        state = self.checkpoint_manager.load_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        
        # Restore EMA if present
        if self.ema_manager and "ema_state" in state:
            self.ema_manager.load_state_dict(state["ema_state"])
        
        self.emit(EventType.CHECKPOINT_LOADED, {"path": path, "step": state.get("step", 0)})
        
        return state
    
    def finish(self) -> None:
        """Finish training and cleanup."""
        if self.tracker:
            self.tracker.finish()
        
        self.emit(EventType.TRAINING_FINISHED, {})

