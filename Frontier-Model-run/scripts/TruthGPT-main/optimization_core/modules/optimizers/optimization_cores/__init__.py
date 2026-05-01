"""
Unified Optimization Cores
==========================
Centralized exports for all optimization core implementations.
"""

from typing import Dict, Any, Optional
import warnings

# Import all optimization cores
from .enhanced_optimization_core import (
    EnhancedOptimizationCore,
    EnhancedOptimizationConfig,
    create_enhanced_optimization_core,
)

from .ultra_enhanced_optimization_core import (
    UltraEnhancedOptimizationCore,
    UltraEnhancedOptimizationConfig,
    create_ultra_enhanced_optimization_core,
)

from .mega_enhanced_optimization_core import (
    MegaEnhancedOptimizationCore,
    MegaEnhancedOptimizationConfig,
    create_mega_enhanced_optimization_core,
)

from .supreme_optimization_core import (
    SupremeOptimizationCore,
    SupremeOptimizationConfig,
    create_supreme_optimization_core,
)

from .transcendent_optimization_core import (
    TranscendentOptimizationCore,
    TranscendentOptimizationConfig,
    create_transcendent_optimization_core,
)

from .hybrid_optimization_core import (
    HybridOptimizationCore,
    HybridOptimizationConfig,
    create_hybrid_optimization_core,
)

from .ultra_fast_optimization_core import (
    UltraFastOptimizationCore,
    UltraFastOptimizationResult,
    create_ultra_fast_optimization_core,
)

# Use result as config if needed, or just placeholder
UltraFastOptimizationConfig = UltraFastOptimizationResult


# Unified factory function
def create_optimization_core(
    core_type: str = "enhanced",
    config: Optional[Dict[str, Any]] = None
):
    """
    Unified factory function to create any optimization core.
    
    Args:
        core_type: Type of optimization core to create. Options:
            - "enhanced"
            - "ultra_enhanced"
            - "mega_enhanced"
            - "supreme"
            - "transcendent"
            - "hybrid"
            - "ultra_fast"
        config: Optional configuration dictionary
    
    Returns:
        The requested optimization core instance
    
    Example:
        >>> core = create_optimization_core("enhanced", {"optimization_aggressiveness": 0.9})
        >>> optimized_module, stats = core.enhance_optimization_module(module)
    """
    if config is None:
        config = {}
    
    core_type = core_type.lower()
    
    factory_map = {
        "enhanced": create_enhanced_optimization_core,
        "ultra_enhanced": create_ultra_enhanced_optimization_core,
        "mega_enhanced": create_mega_enhanced_optimization_core,
        "supreme": create_supreme_optimization_core,
        "transcendent": create_transcendent_optimization_core,
        "hybrid": create_hybrid_optimization_core,
        "ultra_fast": create_ultra_fast_optimization_core,
    }
    
    if core_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown optimization core type: '{core_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[core_type]
    return factory(config)


# Registry of all available optimization cores
OPTIMIZATION_CORE_REGISTRY = {
    "enhanced": {
        "class": EnhancedOptimizationCore,
        "config": EnhancedOptimizationConfig,
        "factory": create_enhanced_optimization_core,
    },
    "ultra_enhanced": {
        "class": UltraEnhancedOptimizationCore,
        "config": UltraEnhancedOptimizationConfig,
        "factory": create_ultra_enhanced_optimization_core,
    },
    "mega_enhanced": {
        "class": MegaEnhancedOptimizationCore,
        "config": MegaEnhancedOptimizationConfig,
        "factory": create_mega_enhanced_optimization_core,
    },
    "supreme": {
        "class": SupremeOptimizationCore,
        "config": SupremeOptimizationConfig,
        "factory": create_supreme_optimization_core,
    },
    "transcendent": {
        "class": TranscendentOptimizationCore,
        "config": TranscendentOptimizationConfig,
        "factory": create_transcendent_optimization_core,
    },
    "hybrid": {
        "class": HybridOptimizationCore,
        "config": HybridOptimizationConfig,
        "factory": create_hybrid_optimization_core,
    },
    "ultra_fast": {
        "class": UltraFastOptimizationCore,
        "config": UltraFastOptimizationConfig,
        "factory": create_ultra_fast_optimization_core,
    },
}


def list_available_cores() -> list:
    """List all available optimization core types."""
    return list(OPTIMIZATION_CORE_REGISTRY.keys())


def get_core_info(core_type: str) -> Dict[str, Any]:
    """
    Get information about a specific optimization core.
    
    Args:
        core_type: Type of optimization core
    
    Returns:
        Dictionary with core information
    """
    if core_type not in OPTIMIZATION_CORE_REGISTRY:
        raise ValueError(f"Unknown core type: {core_type}")
    
    registry_entry = OPTIMIZATION_CORE_REGISTRY[core_type]
    return {
        "type": core_type,
        "class": registry_entry["class"].__name__,
        "config_class": registry_entry["config"].__name__,
        "factory": registry_entry["factory"].__name__,
    }


__all__ = [
    # Core classes
    "EnhancedOptimizationCore",
    "UltraEnhancedOptimizationCore",
    "MegaEnhancedOptimizationCore",
    "SupremeOptimizationCore",
    "TranscendentOptimizationCore",
    "HybridOptimizationCore",
    "UltraFastOptimizationCore",
    # Config classes
    "EnhancedOptimizationConfig",
    "UltraEnhancedOptimizationConfig",
    "MegaEnhancedOptimizationConfig",
    "SupremeOptimizationConfig",
    "TranscendentOptimizationConfig",
    "HybridOptimizationConfig",
    "UltraFastOptimizationConfig",
    # Factory functions
    "create_enhanced_optimization_core",
    "create_ultra_enhanced_optimization_core",
    "create_mega_enhanced_optimization_core",
    "create_supreme_optimization_core",
    "create_transcendent_optimization_core",
    "create_hybrid_optimization_core",
    "create_ultra_fast_optimization_core",
    # Unified factory
    "create_optimization_core",
    # Registry
    "OPTIMIZATION_CORE_REGISTRY",
    "list_available_cores",
    "get_core_info",
]


