"""
Trainers module - Modular training components.

This module provides:
- TrainerConfig: Configuration system with composition
- ModelManager: Model loading and configuration
- OptimizerManager: Optimizer and scheduler management
- DataManager: Data loading and preprocessing
- EMAManager: Exponential Moving Average
- Evaluator: Model evaluation
- CheckpointManager: Checkpoint management
- GenericTrainer: Main training orchestrator
"""

from optimization_core.trainers.config import (
    TrainerConfig,
    ModelConfig,
    TrainingConfig,
    HardwareConfig,
    CheckpointConfig,
    EMAConfig,
)
from optimization_core.trainers.model_manager import ModelManager
from optimization_core.trainers.optimizer_manager import OptimizerManager
from optimization_core.trainers.data_manager import DataManager
from optimization_core.trainers.ema_manager import EMAManager
from optimization_core.trainers.evaluator import Evaluator, EvaluationResult, EvaluationMetrics, MetricStrategy
from optimization_core.trainers.checkpoint_manager import CheckpointManager

# Import trainer last to avoid circular dependencies
try:
    from optimization_core.trainers.trainer import GenericTrainer
except ImportError:
    # GenericTrainer might not be updated yet
    GenericTrainer = None

__all__ = [
    "TrainerConfig",
    "ModelConfig",
    "TrainingConfig",
    "HardwareConfig",
    "CheckpointConfig",
    "EMAConfig",
    "ModelManager",
    "OptimizerManager",
    "DataManager",
    "EMAManager",
    "Evaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "MetricStrategy",
    "CheckpointManager",
    "GenericTrainer",
]

