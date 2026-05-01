from .truthgpt import create_truthgpt_optimizer_by_type as create_truthgpt_optimizer
from .core.generic_optimizer import create_generic_optimizer as create_module_optimizer
from .core.generic_optimizer import create_generic_optimizer
from .production.production_optimizer import create_production_optimizer
from .optimization_cores import EnhancedOptimizationCore
from .techniques import computational_optimizations, triton_optimizations

def list_available_module_optimizers():
    """Returns a list of available module optimizers for backward compatibility."""
    return ["cuda", "gpu", "memory"]

__all__ = [
    'create_truthgpt_optimizer', 
    'create_generic_optimizer', 
    'create_production_optimizer',
    'create_module_optimizer',
    'EnhancedOptimizationCore',
    'list_available_module_optimizers',
    'computational_optimizations',
    'triton_optimizations'
]

