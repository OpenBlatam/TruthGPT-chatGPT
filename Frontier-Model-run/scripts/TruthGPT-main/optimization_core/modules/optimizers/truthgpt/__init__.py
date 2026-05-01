"""
TruthGPT-Specific Optimizers
=============================
Optimizers specifically designed for TruthGPT models.
"""

# Import TruthGPT-specific optimizers
from .truthgpt_dynamo_optimizer import (
    TruthGPTDynamoOptimizer,
    TruthGPTDynamoLevel,
    TruthGPTDynamoResult,
    create_truthgpt_dynamo_optimizer,
)

from .truthgpt_inductor_optimizer import (
    TruthGPTInductorOptimizer,
    TruthGPTInductorLevel,
    TruthGPTInductorResult,
    create_truthgpt_inductor_optimizer,
)

from .truthgpt_quantization_optimizer import (
    TruthGPTQuantizationOptimizer,
    TruthGPTQuantizationLevel,
    TruthGPTQuantizationResult,
    create_truthgpt_quantization_optimizer,
)

from .supreme_truthgpt_optimizer import (
    SupremeTruthGPTOptimizer,
    SupremeOptimizationLevel,
    SupremeOptimizationResult,
)

from ..transformer.transformer_optimizer import (
    TransformerOptimizer,
)


# Unified factory function for TruthGPT optimizers
def create_truthgpt_optimizer_by_type(
    optimizer_type: str = "dynamo",
    config: dict = None
):
    """
    Unified factory function to create TruthGPT-specific optimizers.
    
    Args:
        optimizer_type: Type of optimizer to create. Options:
            - "dynamo" - TruthGPTDynamoOptimizer
            - "inductor" - TruthGPTInductorOptimizer
            - "quantization" - TruthGPTQuantizationOptimizer
            - "supreme" - SupremeTruthGPTOptimizer
            - "transformer" - TransformerOptimizer
        config: Optional configuration dictionary
    
    Returns:
        The requested optimizer instance
    
    Example:
        >>> optimizer = create_truthgpt_optimizer_by_type("dynamo", {"level": "advanced"})
        >>> result = optimizer.optimize(model)
    """
    if config is None:
        config = {}
    
    optimizer_type = optimizer_type.lower()
    
    factory_map = {
        "dynamo": create_truthgpt_dynamo_optimizer,
        "inductor": create_truthgpt_inductor_optimizer,
        "quantization": create_truthgpt_quantization_optimizer,
        "supreme": lambda cfg: SupremeTruthGPTOptimizer(cfg),
        "transformer": lambda cfg: TransformerOptimizer(cfg),
    }
    
    if optimizer_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown TruthGPT optimizer type: '{optimizer_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[optimizer_type]
    return factory(config)


# Registry of all available TruthGPT optimizers
TRUTHGPT_OPTIMIZER_REGISTRY = {
    "dynamo": {
        "class": TruthGPTDynamoOptimizer,
        "level_enum": TruthGPTDynamoLevel,
        "result_class": TruthGPTDynamoResult,
        "factory": create_truthgpt_dynamo_optimizer,
    },
    "inductor": {
        "class": TruthGPTInductorOptimizer,
        "level_enum": TruthGPTInductorLevel,
        "result_class": TruthGPTInductorResult,
        "factory": create_truthgpt_inductor_optimizer,
    },
    "quantization": {
        "class": TruthGPTQuantizationOptimizer,
        "level_enum": TruthGPTQuantizationLevel,
        "result_class": TruthGPTQuantizationResult,
        "factory": create_truthgpt_quantization_optimizer,
    },
    "supreme": {
        "class": SupremeTruthGPTOptimizer,
        "level_enum": SupremeOptimizationLevel,
        "result_class": SupremeOptimizationResult,
        "factory": lambda cfg: SupremeTruthGPTOptimizer(cfg),
    },
    "transformer": {
        "class": TransformerOptimizer,
        "level_enum": None,
        "result_class": None,
        "factory": lambda cfg: TransformerOptimizer(cfg),
    },
}


def list_available_truthgpt_optimizers() -> list:
    """List all available TruthGPT optimizer types."""
    return list(TRUTHGPT_OPTIMIZER_REGISTRY.keys())


def get_truthgpt_optimizer_info(optimizer_type: str) -> dict:
    """
    Get information about a specific TruthGPT optimizer.
    
    Args:
        optimizer_type: Type of optimizer
    
    Returns:
        Dictionary with optimizer information
    """
    if optimizer_type not in TRUTHGPT_OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    registry_entry = TRUTHGPT_OPTIMIZER_REGISTRY[optimizer_type]
    return {
        "type": optimizer_type,
        "class": registry_entry["class"].__name__,
        "level_enum": registry_entry["level_enum"].__name__ if registry_entry["level_enum"] else None,
        "result_class": registry_entry["result_class"].__name__ if registry_entry["result_class"] else None,
        "factory": registry_entry["factory"].__name__ if hasattr(registry_entry["factory"], "__name__") else "lambda",
    }


__all__ = [
    # Dynamo optimizer
    "TruthGPTDynamoOptimizer",
    "TruthGPTDynamoLevel",
    "TruthGPTDynamoResult",
    "create_truthgpt_dynamo_optimizer",
    # Inductor optimizer
    "TruthGPTInductorOptimizer",
    "TruthGPTInductorLevel",
    "TruthGPTInductorResult",
    "create_truthgpt_inductor_optimizer",
    # Quantization optimizer
    "TruthGPTQuantizationOptimizer",
    "TruthGPTQuantizationLevel",
    "TruthGPTQuantizationResult",
    "create_truthgpt_quantization_optimizer",
    # Supreme optimizer
    "SupremeTruthGPTOptimizer",
    "SupremeOptimizationLevel",
    "SupremeOptimizationResult",
    # Transformer optimizer
    "TransformerOptimizer",
    # Unified factory
    "create_truthgpt_optimizer_by_type",
    # Registry
    "TRUTHGPT_OPTIMIZER_REGISTRY",
    "list_available_truthgpt_optimizers",
    "get_truthgpt_optimizer_info",
]


