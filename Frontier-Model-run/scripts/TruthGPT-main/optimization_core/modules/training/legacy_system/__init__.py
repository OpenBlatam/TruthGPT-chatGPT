"""
Unified Training System
=======================
Centralized access to all training components in optimization_core.
"""

# Import trainers components
try:
    from ..trainers.config import (
        TrainerConfig,
        ModelConfig,
        TrainingConfig,
        HardwareConfig,
        CheckpointConfig,
        EMAConfig,
    )
except ImportError:
    TrainerConfig = None
    ModelConfig = None
    TrainingConfig = None
    HardwareConfig = None
    CheckpointConfig = None
    EMAConfig = None

try:
    from ..trainers.model_manager import ModelManager as TrainersModelManager
except ImportError:
    TrainersModelManager = None

try:
    from ..trainers.optimizer_manager import OptimizerManager as TrainersOptimizerManager
except ImportError:
    TrainersOptimizerManager = None

try:
    from ..trainers.data_manager import DataManager as TrainersDataManager
except ImportError:
    TrainersDataManager = None

try:
    from ..trainers.ema_manager import EMAManager as TrainersEMAManager
except ImportError:
    TrainersEMAManager = None

try:
    from ..trainers.evaluator import Evaluator as TrainersEvaluator
except ImportError:
    TrainersEvaluator = None

try:
    from ..trainers.checkpoint_manager import CheckpointManager as TrainersCheckpointManager
except ImportError:
    TrainersCheckpointManager = None

try:
    from ..trainers.trainer import GenericTrainer
except ImportError:
    GenericTrainer = None

try:
    from ..trainers.callbacks import (
        Callback,
        PrintLogger,
        WandbLogger,
        TensorBoardLogger,
    )
except ImportError:
    Callback = None
    PrintLogger = None
    WandbLogger = None
    TensorBoardLogger = None

# Import training components
try:
    from ..training.evaluator import Evaluator as TrainingEvaluator
except ImportError:
    TrainingEvaluator = None

try:
    from ..training.checkpoint_manager import CheckpointManager as TrainingCheckpointManager
except ImportError:
    TrainingCheckpointManager = None

try:
    from ..training.ema_manager import EMAManager as TrainingEMAManager
except ImportError:
    TrainingEMAManager = None

try:
    from ..training.training_loop import TrainingLoop
except ImportError:
    TrainingLoop = None

try:
    from ..training.experiment_tracker import ExperimentTracker
except ImportError:
    ExperimentTracker = None


# Unified training component factory
def create_training_component(
    component_type: str = "trainer",
    config: dict = None
):
    """
    Unified factory function to create training components.
    
    Args:
        component_type: Type of component to create. Options:
            - "trainer" - GenericTrainer
            - "training_loop" - TrainingLoop
            - "model_manager" - ModelManager (trainers)
            - "optimizer_manager" - OptimizerManager (trainers)
            - "data_manager" - DataManager (trainers)
            - "ema_manager" - EMAManager (trainers)
            - "evaluator" - Evaluator (trainers)
            - "checkpoint_manager" - CheckpointManager (trainers)
            - "experiment_tracker" - ExperimentTracker
        config: Optional configuration dictionary
    
    Returns:
        The requested training component instance
    """
    if config is None:
        config = {}
    
    component_type = component_type.lower()
    
    factory_map = {
        "trainer": lambda cfg: GenericTrainer(cfg) if GenericTrainer else None,
        "training_loop": lambda cfg: TrainingLoop(**cfg) if TrainingLoop else None,
        "model_manager": lambda cfg: TrainersModelManager(cfg) if TrainersModelManager else None,
        "optimizer_manager": lambda cfg: TrainersOptimizerManager(cfg) if TrainersOptimizerManager else None,
        "data_manager": lambda cfg: TrainersDataManager(cfg) if TrainersDataManager else None,
        "ema_manager": lambda cfg: TrainersEMAManager(cfg) if TrainersEMAManager else None,
        "evaluator": lambda cfg: TrainersEvaluator(cfg) if TrainersEvaluator else None,
        "checkpoint_manager": lambda cfg: TrainersCheckpointManager(cfg) if TrainersCheckpointManager else None,
        "experiment_tracker": lambda cfg: ExperimentTracker(cfg) if ExperimentTracker else None,
    }
    
    if component_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown training component type: '{component_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[component_type]
    component = factory(config)
    
    if component is None:
        raise ImportError(f"Training component type '{component_type}' is not available (module not found)")
    
    return component


# Registry of all available training components
TRAINING_COMPONENT_REGISTRY = {
    "trainer": {
        "class": GenericTrainer,
        "module": "trainers.trainer",
        "description": "Generic trainer for model training",
    },
    "training_loop": {
        "class": TrainingLoop,
        "module": "training.training_loop",
        "description": "Training loop implementation",
    },
    "model_manager": {
        "class": TrainersModelManager,
        "module": "trainers.model_manager",
        "description": "Model manager for trainers",
    },
    "optimizer_manager": {
        "class": TrainersOptimizerManager,
        "module": "trainers.optimizer_manager",
        "description": "Optimizer manager for trainers",
    },
    "data_manager": {
        "class": TrainersDataManager,
        "module": "trainers.data_manager",
        "description": "Data manager for trainers",
    },
    "ema_manager": {
        "class": TrainersEMAManager,
        "module": "trainers.ema_manager",
        "description": "EMA manager for trainers",
    },
    "evaluator": {
        "class": TrainersEvaluator,
        "module": "trainers.evaluator",
        "description": "Evaluator for trainers",
    },
    "checkpoint_manager": {
        "class": TrainersCheckpointManager,
        "module": "trainers.checkpoint_manager",
        "description": "Checkpoint manager for trainers",
    },
    "experiment_tracker": {
        "class": ExperimentTracker,
        "module": "training.experiment_tracker",
        "description": "Experiment tracker",
    },
}


def list_available_training_components() -> list:
    """List all available training component types."""
    return [k for k, v in TRAINING_COMPONENT_REGISTRY.items() if v["class"] is not None]


def get_training_component_info(component_type: str) -> dict:
    """
    Get information about a specific training component.
    
    Args:
        component_type: Type of training component
    
    Returns:
        Dictionary with training component information
    """
    if component_type not in TRAINING_COMPONENT_REGISTRY:
        raise ValueError(f"Unknown training component type: {component_type}")
    
    registry_entry = TRAINING_COMPONENT_REGISTRY[component_type]
    
    if registry_entry["class"] is None:
        raise ImportError(f"Training component type '{component_type}' is not available (module not found)")
    
    return {
        "type": component_type,
        "class": registry_entry["class"].__name__,
        "module": registry_entry["module"],
        "description": registry_entry["description"],
    }


__all__ = [
    # Trainer configs
    "TrainerConfig",
    "ModelConfig",
    "TrainingConfig",
    "HardwareConfig",
    "CheckpointConfig",
    "EMAConfig",
    # Trainer components
    "GenericTrainer",
    "TrainersModelManager",
    "TrainersOptimizerManager",
    "TrainersDataManager",
    "TrainersEMAManager",
    "TrainersEvaluator",
    "TrainersCheckpointManager",
    # Training components
    "TrainingLoop",
    "TrainingEvaluator",
    "TrainingCheckpointManager",
    "TrainingEMAManager",
    "ExperimentTracker",
    # Callbacks
    "Callback",
    "PrintLogger",
    "WandbLogger",
    "TensorBoardLogger",
    # Unified factory
    "create_training_component",
    # Registry
    "TRAINING_COMPONENT_REGISTRY",
    "list_available_training_components",
    "get_training_component_info",
]


