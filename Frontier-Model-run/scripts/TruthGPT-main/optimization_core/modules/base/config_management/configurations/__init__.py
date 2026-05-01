"""
Unified Configuration System
============================
Centralized access to all configuration classes in optimization_core.
"""

# Import main config module
from ..config import (
    ConfigManager,
    ConfigValidationError,
    ConfigLoadError,
    create_config_manager,
    load_config_from_file,
    load_config_from_env,
    validate_config,
    TransformerConfig,
    OptimizationConfig as ConfigOptimizationConfig,
    TrainingConfig as ConfigTrainingConfig,
    ModelConfig as ConfigModelConfig,
    create_transformer_config,
    create_optimization_config,
    create_training_config,
    create_model_config,
    EnvironmentConfig,
    DevelopmentConfig,
    ProductionConfig as ConfigProductionConfig,
    TestingConfig,
    create_environment_config,
    ConfigValidationRule,
    OptimizationValidationRule,
    ModelValidationRule,
    TrainingValidationRule,
    create_validation_rules,
)

# Import architecture config
try:
    from ..config.architecture import (
        ArchitectureConfig,
        ConfigurationManager,
        ArchitectureValidator,
        ArchitectureBuilder,
        create_configuration_manager,
        create_architecture_validator,
        create_architecture_builder,
    )
except ImportError:
    ArchitectureConfig = None
    ConfigurationManager = None
    ArchitectureValidator = None
    ArchitectureBuilder = None
    create_configuration_manager = None
    create_architecture_validator = None
    create_architecture_builder = None

# Import trainer configs
try:
    from ..trainers.config import (
        TrainerConfig,
        ModelConfig as TrainerModelConfig,
        TrainingConfig as TrainerTrainingConfig,
        HardwareConfig,
        CheckpointConfig,
        EMAConfig,
    )
except ImportError:
    TrainerConfig = None
    TrainerModelConfig = None
    TrainerTrainingConfig = None
    HardwareConfig = None
    CheckpointConfig = None
    EMAConfig = None

# Import production config
try:
    from ..production.production_config import (
        ProductionConfig as ProductionProductionConfig,
    )
except ImportError:
    ProductionProductionConfig = None

# Import feed forward config manager
try:
    from ..modules.feed_forward.refactored_config_manager import (
        ConfigurationManager as FeedForwardConfigManager,
        ConfigurationFactory,
        ConfigTemplates,
        ConfigValidators,
        EnvironmentConfigBuilder,
        ConfigSource,
        ConfigFormat,
        ConfigValidationRule as FeedForwardConfigValidationRule,
        ConfigSourceInfo,
        create_configuration_demo,
    )
except ImportError:
    FeedForwardConfigManager = None
    ConfigurationFactory = None
    ConfigTemplates = None
    ConfigValidators = None
    EnvironmentConfigBuilder = None
    ConfigSource = None
    ConfigFormat = None
    FeedForwardConfigValidationRule = None
    ConfigSourceInfo = None
    create_configuration_demo = None


# Unified configuration factory
def create_configuration(
    config_type: str = "transformer",
    config: dict = None
):
    """
    Unified factory function to create configurations.
    
    Args:
        config_type: Type of configuration to create. Options:
            - "transformer" - TransformerConfig
            - "optimization" - OptimizationConfig
            - "training" - TrainingConfig
            - "model" - ModelConfig
            - "environment" - EnvironmentConfig
            - "trainer" - TrainerConfig
            - "architecture" - ArchitectureConfig
            - "production" - ProductionConfig
        config: Optional configuration dictionary
    
    Returns:
        The requested configuration instance
    """
    if config is None:
        config = {}
    
    config_type = config_type.lower()
    
    factory_map = {
        "transformer": lambda cfg: create_transformer_config(**cfg),
        "optimization": lambda cfg: create_optimization_config(**cfg),
        "training": lambda cfg: create_training_config(**cfg),
        "model": lambda cfg: create_model_config(**cfg),
        "environment": lambda cfg: create_environment_config(**cfg),
        "trainer": lambda cfg: TrainerConfig(**cfg) if TrainerConfig else None,
        "architecture": lambda cfg: ArchitectureConfig(**cfg) if ArchitectureConfig else None,
        "production": lambda cfg: ProductionProductionConfig(**cfg) if ProductionProductionConfig else None,
    }
    
    if config_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown configuration type: '{config_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[config_type]
    configuration = factory(config)
    
    if configuration is None:
        raise ImportError(f"Configuration type '{config_type}' is not available (module not found)")
    
    return configuration


# Registry of all available configurations
CONFIGURATION_REGISTRY = {
    "transformer": {
        "class": TransformerConfig,
        "module": "config.transformer_config",
        "description": "Transformer model configuration",
    },
    "optimization": {
        "class": ConfigOptimizationConfig,
        "module": "config.transformer_config",
        "description": "Optimization configuration",
    },
    "training": {
        "class": ConfigTrainingConfig,
        "module": "config.transformer_config",
        "description": "Training configuration",
    },
    "model": {
        "class": ConfigModelConfig,
        "module": "config.transformer_config",
        "description": "Model configuration",
    },
    "environment": {
        "class": EnvironmentConfig,
        "module": "config.environment_config",
        "description": "Environment configuration",
    },
    "trainer": {
        "class": TrainerConfig,
        "module": "trainers.config",
        "description": "Trainer configuration",
    },
    "architecture": {
        "class": ArchitectureConfig,
        "module": "config.architecture",
        "description": "Architecture configuration",
    },
    "production": {
        "class": ProductionProductionConfig,
        "module": "production.production_config",
        "description": "Production configuration",
    },
}


def list_available_configurations() -> list:
    """List all available configuration types."""
    return [k for k, v in CONFIGURATION_REGISTRY.items() if v["class"] is not None]


def get_configuration_info(config_type: str) -> dict:
    """
    Get information about a specific configuration.
    
    Args:
        config_type: Type of configuration
    
    Returns:
        Dictionary with configuration information
    """
    if config_type not in CONFIGURATION_REGISTRY:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    registry_entry = CONFIGURATION_REGISTRY[config_type]
    
    if registry_entry["class"] is None:
        raise ImportError(f"Configuration type '{config_type}' is not available (module not found)")
    
    return {
        "type": config_type,
        "class": registry_entry["class"].__name__,
        "module": registry_entry["module"],
        "description": registry_entry["description"],
    }


__all__ = [
    # Main config
    "ConfigManager",
    "ConfigValidationError",
    "ConfigLoadError",
    "create_config_manager",
    "load_config_from_file",
    "load_config_from_env",
    "validate_config",
    # Transformer configs
    "TransformerConfig",
    "ConfigOptimizationConfig",
    "ConfigTrainingConfig",
    "ConfigModelConfig",
    "create_transformer_config",
    "create_optimization_config",
    "create_training_config",
    "create_model_config",
    # Environment configs
    "EnvironmentConfig",
    "DevelopmentConfig",
    "ConfigProductionConfig",
    "TestingConfig",
    "create_environment_config",
    # Validation rules
    "ConfigValidationRule",
    "OptimizationValidationRule",
    "ModelValidationRule",
    "TrainingValidationRule",
    "create_validation_rules",
    # Architecture configs
    "ArchitectureConfig",
    "ConfigurationManager",
    "ArchitectureValidator",
    "ArchitectureBuilder",
    "create_configuration_manager",
    "create_architecture_validator",
    "create_architecture_builder",
    # Trainer configs
    "TrainerConfig",
    "TrainerModelConfig",
    "TrainerTrainingConfig",
    "HardwareConfig",
    "CheckpointConfig",
    "EMAConfig",
    # Production configs
    "ProductionProductionConfig",
    # Feed forward configs
    "FeedForwardConfigManager",
    "ConfigurationFactory",
    "ConfigTemplates",
    "ConfigValidators",
    "EnvironmentConfigBuilder",
    "ConfigSource",
    "ConfigFormat",
    "FeedForwardConfigValidationRule",
    "ConfigSourceInfo",
    "create_configuration_demo",
    # Unified factory
    "create_configuration",
    # Registry
    "CONFIGURATION_REGISTRY",
    "list_available_configurations",
    "get_configuration_info",
]


