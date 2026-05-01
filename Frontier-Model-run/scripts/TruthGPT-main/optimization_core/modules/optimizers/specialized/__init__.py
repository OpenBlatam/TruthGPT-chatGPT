"""
Specialized Optimizers
======================
Optimizers for specific use cases and techniques.
"""

# Import specialized optimizers
from ..mcts_optimization import (
    MCTSOptimizer,
    MCTSOptimizationArgs,
    create_mcts_optimizer,
)

from ..enhanced_mcts_optimizer import (
    EnhancedMCTSWithBenchmarks,
    EnhancedMCTSBenchmarkArgs,
    create_enhanced_mcts_with_benchmarks,
    benchmark_mcts_comparison,
)

from .. import (
    LibraryOptimizer,
    create_library_optimizer,
)

from ...core.framework.ai_extreme_optimizer import (
    AIExtremeOptimizer,
    create_ai_extreme_optimizer,
)




# Unified factory function for specialized optimizers
def create_specialized_optimizer(
    optimizer_type: str = "mcts",
    config: dict = None
):
    """
    Unified factory function to create specialized optimizers.
    
    Args:
        optimizer_type: Type of optimizer to create. Options:
            - "mcts" - MCTSOptimizer
            - "enhanced_mcts" - EnhancedMCTSWithBenchmarks
            - "library" - LibraryOptimizer
            - "ai_extreme" - AIExtremeOptimizer
        config: Optional configuration dictionary
    
    Returns:
        The requested optimizer instance
    """
    if config is None:
        config = {}
    
    optimizer_type = optimizer_type.lower()
    
    factory_map = {
        "mcts": create_mcts_optimizer,
        "enhanced_mcts": create_enhanced_mcts_with_benchmarks,
        "library": create_library_optimizer,
        "ai_extreme": create_ai_extreme_optimizer,
    }
    
    if optimizer_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown specialized optimizer type: '{optimizer_type}'. "
            f"Available types: {available}"
        )
    
    factory = factory_map[optimizer_type]
    return factory(config)


# Registry of all available specialized optimizers
SPECIALIZED_OPTIMIZER_REGISTRY = {
    "mcts": {
        "class": MCTSOptimizer,
        "factory": create_mcts_optimizer,
    },
    "enhanced_mcts": {
        "class": EnhancedMCTSWithBenchmarks,
        "factory": create_enhanced_mcts_with_benchmarks,
    },
    "library": {
        "class": LibraryOptimizer,
        "factory": create_library_optimizer,
    },
    "ai_extreme": {
        "class": AIExtremeOptimizer,
        "factory": create_ai_extreme_optimizer,
    },
}


def list_available_specialized_optimizers() -> list:
    """List all available specialized optimizer types."""
    return list(SPECIALIZED_OPTIMIZER_REGISTRY.keys())


def get_specialized_optimizer_info(optimizer_type: str) -> dict:
    """
    Get information about a specific specialized optimizer.
    
    Args:
        optimizer_type: Type of optimizer
    
    Returns:
        Dictionary with optimizer information
    """
    if optimizer_type not in SPECIALIZED_OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    registry_entry = SPECIALIZED_OPTIMIZER_REGISTRY[optimizer_type]
    return {
        "type": optimizer_type,
        "class": registry_entry["class"].__name__,
        "factory": registry_entry["factory"].__name__ if hasattr(registry_entry["factory"], "__name__") else "lambda",
    }


__all__ = [
    # MCTS optimizers
    "MCTSOptimizer",
    "MCTSOptimizationArgs",
    "create_mcts_optimizer",
    "EnhancedMCTSWithBenchmarks",
    "EnhancedMCTSBenchmarkArgs",
    "create_enhanced_mcts_with_benchmarks",
    "benchmark_mcts_comparison",
    # Unified factory
    "create_specialized_optimizer",
    # Registry
    "SPECIALIZED_OPTIMIZER_REGISTRY",
    "list_available_specialized_optimizers",
    "get_specialized_optimizer_info",
]


