"""
Unified Manager System
=======================
Centralized access to all manager classes in optimization_core.
"""

# Import all manager modules
try:
    from ..config.config_manager import (
        ConfigurationManager,
        ConfigManager,
    )
except ImportError:
    ConfigurationManager = None
    ConfigManager = None

try:
    from ..data.dataset_manager import (
        DatasetManager,
    )
except ImportError:
    DatasetManager = None

try:
    from ..trainers.checkpoint_manager import (
        CheckpointManager as TrainersCheckpointManager,
    )
except ImportError:
    TrainersCheckpointManager = None

try:
    from ..trainers.model_manager import (
        ModelManager as TrainersModelManager,
    )
except ImportError:
    TrainersModelManager = None

try:
    from ..trainers.ema_manager import (
        EMAManager as TrainersEMAManager,
    )
except ImportError:
    TrainersEMAManager = None

try:
    from ..trainers.data_manager import (
        DataManager as TrainersDataManager,
    )
except ImportError:
    TrainersDataManager = None

try:
    from ..trainers.optimizer_manager import (
        OptimizerManager as TrainersOptimizerManager,
    )
except ImportError:
    TrainersOptimizerManager = None

try:
    from ..inference.cache_manager import (
        CacheManager as InferenceCacheManager,
    )
except ImportError:
    InferenceCacheManager = None

try:
    from ..models.diffusion_manager import (
        DiffusionManager,
    )
except ImportError:
    DiffusionManager = None

try:
    from ..training.ema_manager import (
        EMAManager as TrainingEMAManager,
    )
except ImportError:
    TrainingEMAManager = None

try:
    from ..training.checkpoint_manager import (
        CheckpointManager as TrainingCheckpointManager,
    )
except ImportError:
    TrainingCheckpointManager = None

try:
    from ..models.model_manager import (
        ModelManager as ModelsModelManager,
    )
except ImportError:
    ModelsModelManager = None

try:
    from ..modules.memory.advanced_memory_manager import (
        AdvancedMemoryManager,
        create_advanced_memory_manager,
    )
except ImportError:
    AdvancedMemoryManager = None
    create_advanced_memory_manager = None

try:
    from ..modules.module_manager import (
        ModuleManager,
    )
except ImportError:
    ModuleManager = None

try:
    from ..modules.feed_forward.refactored_config_manager import (
        ConfigurationManager as FeedForwardConfigManager,
    )
except ImportError:
    FeedForwardConfigManager = None

try:
    from ..commit_tracker.version_manager import (
        VersionManager,
    )
except ImportError:
    VersionManager = None


# Unified manager factory
def create_manager(manager_type: str = "config", config: dict = None):
    """
    Unified factory function to create managers.
    
    Args:
        manager_type: Type of manager to create. Options:
            - "config" - ConfigurationManager
            - "dataset" - DatasetManager
            - "checkpoint" - CheckpointManager (trainers)
            - "model" - ModelManager (trainers)
            - "ema" - EMAManager (trainers)
            - "data" - DataManager (trainers)
            - "optimizer" - OptimizerManager (trainers)
            - "cache" - CacheManager (inference)
            - "diffusion" - DiffusionManager
            - "memory" - AdvancedMemoryManager
            - "module" - ModuleManager
            - "version" - VersionManager
        config: Optional configuration dictionary
    
    Returns:
        The requested manager instance
    """
    if config is None:
        config = {}
    
    manager_type = manager_type.lower()
    
    factory_map = {
        "config": lambda cfg: ConfigurationManager(cfg) if ConfigurationManager else None,
        "dataset": lambda cfg: DatasetManager(cfg) if DatasetManager else None,
        "checkpoint": lambda cfg: TrainersCheckpointManager(cfg) if TrainersCheckpointManager else None,
        "model": lambda cfg: TrainersModelManager(cfg) if TrainersModelManager else None,
        "ema": lambda cfg: TrainersEMAManager(cfg) if TrainersEMAManager else None,
        "data": lambda cfg: TrainersDataManager(cfg) if TrainersDataManager else None,
        "optimizer": lambda cfg: TrainersOptimizerManager(cfg) if TrainersOptimizerManager else None,
        "cache": lambda cfg: InferenceCacheManager(cfg) if InferenceCacheManager else None,
        "diffusion": lambda cfg: DiffusionManager(cfg) if DiffusionManager else None,
        "memory": lambda cfg: create_advanced_memory_manager(cfg) if create_advanced_memory_manager else None,
        "module": lambda cfg: ModuleManager(cfg) if ModuleManager else None,
        "version": lambda cfg: VersionManager(cfg) if VersionManager else None,
    }
    
    if manager_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown manager type: '{manager_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[manager_type]
    manager = factory(config)
    
    if manager is None:
        raise ImportError(f"Manager type '{manager_type}' is not available (module not found)")
    
    return manager


# Registry of all available managers
MANAGER_REGISTRY = {
    "config": {
        "class": ConfigurationManager,
        "module": "config.config_manager",
        "description": "Configuration manager",
    },
    "dataset": {
        "class": DatasetManager,
        "module": "data.dataset_manager",
        "description": "Dataset manager",
    },
    "checkpoint": {
        "class": TrainersCheckpointManager,
        "module": "trainers.checkpoint_manager",
        "description": "Checkpoint manager for trainers",
    },
    "model": {
        "class": TrainersModelManager,
        "module": "trainers.model_manager",
        "description": "Model manager for trainers",
    },
    "ema": {
        "class": TrainersEMAManager,
        "module": "trainers.ema_manager",
        "description": "EMA manager for trainers",
    },
    "data": {
        "class": TrainersDataManager,
        "module": "trainers.data_manager",
        "description": "Data manager for trainers",
    },
    "optimizer": {
        "class": TrainersOptimizerManager,
        "module": "trainers.optimizer_manager",
        "description": "Optimizer manager for trainers",
    },
    "cache": {
        "class": InferenceCacheManager,
        "module": "inference.cache_manager",
        "description": "Cache manager for inference",
    },
    "diffusion": {
        "class": DiffusionManager,
        "module": "models.diffusion_manager",
        "description": "Diffusion model manager",
    },
    "memory": {
        "class": AdvancedMemoryManager,
        "module": "modules.memory.advanced_memory_manager",
        "description": "Advanced memory manager",
    },
    "module": {
        "class": ModuleManager,
        "module": "modules.module_manager",
        "description": "Module manager",
    },
    "version": {
        "class": VersionManager,
        "module": "commit_tracker.version_manager",
        "description": "Version manager",
    },
}


def list_available_managers() -> list:
    """List all available manager types."""
    return [k for k, v in MANAGER_REGISTRY.items() if v["class"] is not None]


def get_manager_info(manager_type: str) -> dict:
    """
    Get information about a specific manager.
    
    Args:
        manager_type: Type of manager
    
    Returns:
        Dictionary with manager information
    """
    if manager_type not in MANAGER_REGISTRY:
        raise ValueError(f"Unknown manager type: {manager_type}")
    
    registry_entry = MANAGER_REGISTRY[manager_type]
    
    if registry_entry["class"] is None:
        raise ImportError(f"Manager type '{manager_type}' is not available (module not found)")
    
    return {
        "type": manager_type,
        "class": registry_entry["class"].__name__,
        "module": registry_entry["module"],
        "description": registry_entry["description"],
    }


__all__ = [
    # Configuration managers
    "ConfigurationManager",
    "ConfigManager",
    "FeedForwardConfigManager",
    # Dataset managers
    "DatasetManager",
    # Trainer managers
    "TrainersCheckpointManager",
    "TrainersModelManager",
    "TrainersEMAManager",
    "TrainersDataManager",
    "TrainersOptimizerManager",
    # Inference managers
    "InferenceCacheManager",
    # Model managers
    "DiffusionManager",
    "ModelsModelManager",
    # Training managers
    "TrainingEMAManager",
    "TrainingCheckpointManager",
    # Memory managers
    "AdvancedMemoryManager",
    "create_advanced_memory_manager",
    # Module managers
    "ModuleManager",
    # Version managers
    "VersionManager",
    # Unified factory
    "create_manager",
    # Registry
    "MANAGER_REGISTRY",
    "list_available_managers",
    "get_manager_info",
]


