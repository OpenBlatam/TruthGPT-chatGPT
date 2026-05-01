"""
Core Optimizers
===============
Unified exports for all optimizers in the core directory.
"""

# Import ops optimizers
from ..ops.extreme_optimizer import (
    ExtremeOptimizer,
    ExtremeOptimizationLevel,
    ExtremeOptimizationResult,
    QuantumNeuralOptimizer,
    CosmicOptimizer,
    TranscendentOptimizer,
)

from ..ops.quantum_extreme_optimizer import (
    QuantumOptimizer,
)

from ..ops.ultra_fast_optimizer import (
    UltraFastOptimizer,
    ParallelOptimizer,
    CacheOptimizer,
)

# Import util optimizers
from ..util.enhanced_optimizer import (
    EnhancedOptimizer,
    EnhancedOptimizationLevel,
    EnhancedOptimizationResult,
)

from ..util.complementary_optimizer import (
    ComplementaryOptimizer,
    ComplementaryOptimizationLevel,
    ComplementaryOptimizationResult,
)

from ..util.advanced_complementary_optimizer import (
    AdvancedComplementaryOptimizer,
)

from ..util.microservices_optimizer import (
    MicroservicesOptimizer,
    OptimizerService,
)

# Import framework optimizers
from ..framework.ai_extreme_optimizer import (
    AIExtremeOptimizer,
)

# Import advanced optimizations
from ..advanced_optimizations import (
    QuantumInspiredOptimizer,
    EvolutionaryOptimizer,
    MetaLearningOptimizer,
)

# Import other core optimizers
from optimization_core.modules.optimizers.core.pytorch_optimizer_base import (
    PyTorchOptimizerBase,
)


# Unified factory function for core optimizers
def create_core_optimizer(
    optimizer_type: str = "enhanced",
    config: dict = None
):
    """
    Unified factory function to create core optimizers.
    
    Args:
        optimizer_type: Type of optimizer to create. Options:
            - "extreme" - ExtremeOptimizer
            - "quantum" - QuantumOptimizer
            - "ultra_fast" - UltraFastOptimizer
            - "enhanced" - EnhancedOptimizer
            - "complementary" - ComplementaryOptimizer
            - "advanced_complementary" - AdvancedComplementaryOptimizer
            - "microservices" - MicroservicesOptimizer
            - "ai_extreme" - AIExtremeOptimizer
            - "quantum_inspired" - QuantumInspiredOptimizer
            - "evolutionary" - EvolutionaryOptimizer
            - "meta_learning" - MetaLearningOptimizer
        config: Optional configuration dictionary
    
    Returns:
        The requested optimizer instance
    """
    if config is None:
        config = {}
    
    optimizer_type = optimizer_type.lower()
    
    factory_map = {
        "extreme": lambda cfg: ExtremeOptimizer(cfg),
        "quantum": lambda cfg: QuantumOptimizer(cfg),
        "ultra_fast": lambda cfg: UltraFastOptimizer(cfg),
        "enhanced": lambda cfg: EnhancedOptimizer(cfg),
        "complementary": lambda cfg: ComplementaryOptimizer(cfg),
        "advanced_complementary": lambda cfg: AdvancedComplementaryOptimizer(cfg),
        "microservices": lambda cfg: MicroservicesOptimizer(cfg),
        "ai_extreme": lambda cfg: AIExtremeOptimizer(cfg),
        "quantum_inspired": lambda cfg: QuantumInspiredOptimizer(cfg),
        "evolutionary": lambda cfg: EvolutionaryOptimizer(cfg),
        "meta_learning": lambda cfg: MetaLearningOptimizer(cfg),
    }
    
    if optimizer_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown core optimizer type: '{optimizer_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[optimizer_type]
    return factory(config)


# Registry of all available core optimizers
CORE_OPTIMIZER_REGISTRY = {
    "extreme": {
        "class": ExtremeOptimizer,
        "module": "core.ops.extreme_optimizer",
    },
    "quantum": {
        "class": QuantumOptimizer,
        "module": "core.ops.quantum_extreme_optimizer",
    },
    "ultra_fast": {
        "class": UltraFastOptimizer,
        "module": "core.ops.ultra_fast_optimizer",
    },
    "enhanced": {
        "class": EnhancedOptimizer,
        "module": "core.util.enhanced_optimizer",
    },
    "complementary": {
        "class": ComplementaryOptimizer,
        "module": "core.util.complementary_optimizer",
    },
    "advanced_complementary": {
        "class": AdvancedComplementaryOptimizer,
        "module": "core.util.advanced_complementary_optimizer",
    },
    "microservices": {
        "class": MicroservicesOptimizer,
        "module": "core.util.microservices_optimizer",
    },
    "ai_extreme": {
        "class": AIExtremeOptimizer,
        "module": "core.framework.ai_extreme_optimizer",
    },
    "quantum_inspired": {
        "class": QuantumInspiredOptimizer,
        "module": "core.advanced_optimizations",
    },
    "evolutionary": {
        "class": EvolutionaryOptimizer,
        "module": "core.advanced_optimizations",
    },
    "meta_learning": {
        "class": MetaLearningOptimizer,
        "module": "core.advanced_optimizations",
    },
}


def list_available_core_optimizers() -> list:
    """List all available core optimizer types."""
    return list(CORE_OPTIMIZER_REGISTRY.keys())


def get_core_optimizer_info(optimizer_type: str) -> dict:
    """
    Get information about a specific core optimizer.
    
    Args:
        optimizer_type: Type of optimizer
    
    Returns:
        Dictionary with optimizer information
    """
    if optimizer_type not in CORE_OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    registry_entry = CORE_OPTIMIZER_REGISTRY[optimizer_type]
    return {
        "type": optimizer_type,
        "class": registry_entry["class"].__name__,
        "module": registry_entry["module"],
    }


__all__ = [
    # Ops optimizers
    "ExtremeOptimizer",
    "ExtremeOptimizationLevel",
    "ExtremeOptimizationResult",
    "QuantumNeuralOptimizer",
    "CosmicOptimizer",
    "TranscendentOptimizer",
    "QuantumOptimizer",
    "UltraFastOptimizer",
    "ParallelOptimizer",
    "CacheOptimizer",
    # Util optimizers
    "EnhancedOptimizer",
    "EnhancedOptimizationLevel",
    "EnhancedOptimizationResult",
    "ComplementaryOptimizer",
    "ComplementaryOptimizationLevel",
    "ComplementaryOptimizationResult",
    "AdvancedComplementaryOptimizer",
    "MicroservicesOptimizer",
    "OptimizerService",
    # Framework optimizers
    "AIExtremeOptimizer",
    # Advanced optimizations
    "QuantumInspiredOptimizer",
    "EvolutionaryOptimizer",
    "MetaLearningOptimizer",
    # Other core optimizers
    "PyTorchOptimizerBase",
    # Unified factory
    "create_core_optimizer",
    # Registry
    "CORE_OPTIMIZER_REGISTRY",
    "list_available_core_optimizers",
    "get_core_optimizer_info",
]

