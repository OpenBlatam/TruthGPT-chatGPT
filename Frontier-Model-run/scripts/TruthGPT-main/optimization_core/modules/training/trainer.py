"""
Advanced Training Module
========================

Modular training engine with mixed precision, gradient accumulation,
EMA, distributed training, and pluggable experiment tracking.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import TrainingConfig, TrainingStrategy, OptimizerType, SchedulerType, create_training_config
from .components import EMAManager, CurriculumScheduler, ExperimentLogger, TrainingStep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AdvancedTrainer
# ---------------------------------------------------------------------------
class AdvancedTrainer:
    """
    Production-grade training engine.

    Features:
    - Mixed-precision training via ``torch.amp``
    - Gradient accumulation & clipping
    - Optional EMA weight averaging
    - Early stopping with configurable patience
    - Pluggable experiment tracking (wandb / tensorboard / mlflow)
    - Checkpoint save / load
    - Curriculum learning scheduler
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Components
        self.criterion = criterion or nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler() if config.use_mixed_precision else None
        self.ema = EMAManager(model, config.ema_decay) if config.use_ema else None
        self.curriculum = (
            CurriculumScheduler(config.epochs)
            if config.strategy == TrainingStrategy.CURRICULUM
            else None
        )

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf") if config.mode == "min" else float("-inf")
        self.patience_counter = 0
        self.training_history: List[Dict[str, Any]] = []

        # Logging
        self.experiment_logger = ExperimentLogger(config)

        # Checkpoint dir
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ----- builder helpers ------------------------------------------------
    def _build_optimizer(self) -> optim.Optimizer:
        params = self.model.parameters()
        cfg = self.config
        builders = {
            OptimizerType.ADAM: lambda: optim.Adam(
                params, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps, weight_decay=cfg.weight_decay,
            ),
            OptimizerType.ADAMW: lambda: optim.AdamW(
                params, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps, weight_decay=cfg.weight_decay,
            ),
            OptimizerType.SGD: lambda: optim.SGD(
                params, lr=cfg.learning_rate, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            ),
            OptimizerType.RMSPROP: lambda: optim.RMSprop(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            ),
        }
        builder = builders.get(cfg.optimizer)
        if builder is None:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
        return builder()

    def _build_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        cfg = self.config
        opt = self.optimizer
        builders = {
            SchedulerType.STEP: lambda: optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma),
            SchedulerType.EXPONENTIAL: lambda: optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.gamma),
            SchedulerType.COSINE: lambda: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.total_steps, eta_min=cfg.min_lr),
            SchedulerType.LINEAR: lambda: optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=cfg.total_steps),
            SchedulerType.PLATEAU: lambda: optim.lr_scheduler.ReduceLROnPlateau(opt, mode=cfg.mode, factor=cfg.gamma, patience=cfg.patience, min_lr=cfg.min_lr),
        }
        builder = builders.get(cfg.scheduler)
        return builder() if builder else None

    # ----- core training loop ---------------------------------------------
    def train(self) -> List[Dict[str, Any]]:
        """Run the full training loop. Returns training history."""
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            train_metrics = self._train_epoch(epoch)

            val_metrics = self._validate_epoch(epoch) if self.val_dataloader else {}

            combined = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.training_history.append(combined)
            self.experiment_logger.log(combined, step=epoch)

            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics.get('loss', 0):.4f}  "
                f"val_loss={val_metrics.get('val_loss', 0):.4f}"
            )

            # Early stopping
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Checkpoint
            if self.config.save_checkpoints:
                self._save_checkpoint(epoch, val_metrics)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.config.monitor_metric, 0))
                else:
                    self.scheduler.step()

        self.experiment_logger.finish()
        return self.training_history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = defaultdict(float)
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            batch = self._to_device(batch)

            # Forward
            with torch.amp.autocast(device_type=self.device.type, enabled=self.config.use_mixed_precision):
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                loss = self._extract_loss(outputs, batch)

            # Scale for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (at accumulation boundary)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema:
                    self.ema.update(self.model)

            totals["loss"] += loss.item() * self.config.gradient_accumulation_steps
            totals["lr"] = self.optimizer.param_groups[0]["lr"]
            n_batches += 1
            self.global_step += 1

            if batch_idx % self.config.log_frequency == 0:
                logger.debug(f"  batch {batch_idx} loss={loss.item():.4f}")

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        if self.ema:
            self.ema.apply(self.model)

        totals: Dict[str, float] = defaultdict(float)
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._to_device(batch)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.config.use_mixed_precision):
                    outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                    loss = self._extract_loss(outputs, batch)
                totals["val_loss"] += loss.item()
                n_batches += 1

        if self.ema:
            self.ema.restore(self.model)

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # ----- helpers --------------------------------------------------------
    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    @staticmethod
    def _extract_loss(outputs: Any, batch: Any) -> torch.Tensor:
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        # Fallback — caller should provide a criterion externally
        raise ValueError("Model output does not contain a loss. Pass a custom criterion.")

    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        if not self.config.early_stopping or not val_metrics:
            return False
        current = val_metrics.get(self.config.monitor_metric, float("inf"))
        improved = (
            current < self.best_metric - self.config.min_delta
            if self.config.mode == "min"
            else current > self.best_metric + self.config.min_delta
        )
        if improved:
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return self.patience_counter >= self.config.patience

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "best_metric": self.best_metric,
        }
        if self.ema:
            state["ema_shadow"] = self.ema.state_dict()

        ckpt_dir = Path(self.config.checkpoint_dir)
        torch.save(state, ckpt_dir / f"checkpoint_epoch_{epoch}.pt")

        current = metrics.get(self.config.monitor_metric, float("inf"))
        is_best = (
            (self.config.mode == "min" and current <= self.best_metric)
            or (self.config.mode == "max" and current >= self.best_metric)
        )
        if is_best:
            torch.save(state, ckpt_dir / "best_model.pt")
            logger.info(f"Saved best model at epoch {epoch}")

    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, float]]:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt["epoch"], ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# Aliases & backward compat
# ---------------------------------------------------------------------------
FastTrainer = AdvancedTrainer


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def create_trainer(
    config: TrainingConfig,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
) -> AdvancedTrainer:
    """Create an AdvancedTrainer instance."""
    return AdvancedTrainer(config, model, train_dataloader, val_dataloader)
