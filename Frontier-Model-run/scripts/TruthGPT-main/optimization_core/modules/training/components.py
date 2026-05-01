import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EMA Manager
# ---------------------------------------------------------------------------
class EMAManager:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self.shadow)


# ---------------------------------------------------------------------------
# Curriculum Scheduler
# ---------------------------------------------------------------------------
class CurriculumScheduler:
    """Simple difficulty scheduler for curriculum learning."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = max(total_epochs, 1)

    def get_difficulty(self, epoch: int) -> float:
        return min(epoch / self.total_epochs, 1.0)


# ---------------------------------------------------------------------------
# Experiment Logger
# ---------------------------------------------------------------------------
class ExperimentLogger:
    """Unified experiment logger that lazily loads optional backends."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._wandb = None
        self._tb_writer = None
        self._mlflow = None
        self._init_backends()

    def _init_backends(self) -> None:
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(project="truthgpt-training", reinit=True)
                self._wandb = wandb
            except Exception as e:
                logger.warning(f"wandb unavailable: {e}")

        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=self.config.log_dir)
                self._tb_writer = SummaryWriter(log_dir=self.config.log_dir)
            except Exception as e:
                logger.warning(f"tensorboard unavailable: {e}")

        if self.config.use_mlflow:
            try:
                import mlflow
                mlflow.start_run()
                self._mlflow = mlflow
            except Exception as e:
                logger.warning(f"mlflow unavailable: {e}")

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self._wandb:
            self._wandb.log(metrics, step=step)
        if self._tb_writer:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, step)
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)

    def finish(self) -> None:
        if self._wandb:
            self._wandb.finish()
        if self._tb_writer:
            self._tb_writer.close()
        if self._mlflow:
            self._mlflow.end_run()


@dataclass
class TrainingStep:
    """Lightweight container for a single training step result."""
    loss: float
    lr: float
    step: int
    epoch: int

