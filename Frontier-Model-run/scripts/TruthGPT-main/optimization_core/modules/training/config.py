"""
Training Configuration and Enums
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TrainingStrategy(Enum):
    STANDARD = "standard"
    GRADIENT_ACCUMULATION = "gradient_accumulation"
    MIXED_PRECISION = "mixed_precision"
    DISTRIBUTED = "distributed"
    CURRICULUM = "curriculum"


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    NONE = "none"
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    LINEAR = "linear"
    PLATEAU = "plateau"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Single source of truth for all training parameters."""

    # Core
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer: OptimizerType = OptimizerType.ADAMW
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.9

    # Scheduler
    scheduler: SchedulerType = SchedulerType.COSINE
    warmup_steps: int = 100
    total_steps: int = 1000
    gamma: float = 0.1
    step_size: int = 30
    min_lr: float = 1e-6

    # Strategy
    strategy: TrainingStrategy = TrainingStrategy.STANDARD
    use_mixed_precision: bool = False
    use_distributed: bool = False

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.999

    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 1e-4
    monitor_metric: str = "val_loss"
    mode: str = "min"

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Logging — all optional, no crash if lib missing
    use_wandb: bool = False
    use_tensorboard: bool = False
    use_mlflow: bool = False
    log_frequency: int = 10
    log_dir: str = "logs"

    # Performance
    num_workers: int = 4
    pin_memory: bool = True

    # Device (auto-detected)
    device: str = "auto"

    def __post_init__(self) -> None:
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

def create_training_config(**kwargs) -> TrainingConfig:
    """Create a TrainingConfig from keyword arguments."""
    return TrainingConfig(**kwargs)

