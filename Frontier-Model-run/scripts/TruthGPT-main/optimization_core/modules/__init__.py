"""
Modules Package for TruthGPT Optimization Core
Modular system following deep learning best practices
"""

from __future__ import annotations
import logging
from optimization_core.utils.dependency_manager import resolve_lazy_import

_logger = logging.getLogger(__name__)

# Lazy imports for all submodules and backward compatible exports
_LAZY_IMPORTS = {
    # Submodules
    'optimizers': '.optimizers',
    'advanced': '.advanced',
    'attention': '.attention',
    'embeddings': '.embeddings',
    'feed_forward': '.feed_forward',
    'model': '.model',
    'optimization': '.optimization',
    'training': '.training',
    'transformer': '.transformer',
    'base': '.base',
    'interface': '.interface',
    'memory': '.memory',
    'monitoring': '.monitoring',
    'libraries': '.libraries',
    'advanced_libraries': '.libraries', # Lazy alias
    
    # Backward compatible exports from optimizers
    'CudaKernelConfig': '.optimizers.cuda_optimizer',
    'CudaKernelType': '.optimizers.cuda_optimizer',
    'CudaKernelOptimizer': '.optimizers.cuda_optimizer',
    'CudaKernelManager': '.optimizers.cuda_optimizer',
    'create_cuda_optimizer': '.optimizers.cuda_optimizer',
    'create_cuda_kernel_manager': '.optimizers.cuda_optimizer',
    'create_cuda_kernel_config': '.optimizers.cuda_optimizer',
    
    'GPUOptimizationConfig': '.optimizers.gpu_optimizer',
    'GPUOptimizationLevel': '.optimizers.gpu_optimizer',
    'GPUOptimizer': '.optimizers.gpu_optimizer',
    'GPUMemoryManager': '.optimizers.gpu_optimizer',
    'create_gpu_optimizer': '.optimizers.gpu_optimizer',
    'create_gpu_optimization_config': '.optimizers.gpu_optimizer',
    'create_gpu_memory_manager': '.optimizers.gpu_optimizer',
    
    'MemoryOptimizationConfig': '.optimizers.memory_optimizer',
    'MemoryOptimizationLevel': '.optimizers.memory_optimizer',
    'MemoryOptimizer': '.optimizers.memory_optimizer',
    'MemoryProfiler': '.optimizers.memory_optimizer',
    'create_memory_optimizer': '.optimizers.memory_optimizer',
    'create_memory_optimization_config': '.optimizers.memory_optimizer',
    'create_memory_profiler': '.optimizers.memory_optimizer',
    
    # Advanced Libraries Base Classes (Forwarded from libraries.core)
    'BaseModule': '.libraries.core',
    'ModelModule': '.libraries.models',
    'DataModule': '.libraries.data',
    'TrainingModule': '.libraries.training',
    'TransformerModule': '.libraries.models', # Map appropriately if needed
    'DiffusionModule': '.libraries.models',
    'LoRAModule': '.libraries.models',
    'QuantizedModule': '.libraries.models',
}

def __getattr__(name: str):
    """Lazy import system for module submodules and components."""
    return resolve_lazy_import(name, __package__ or 'modules', _LAZY_IMPORTS)

def list_available_module_submodules() -> list[str]:
    """List all available module submodules."""
    return list(_LAZY_IMPORTS.keys())

__all__ = [
    'optimizers', 'advanced', 'attention', 'embeddings', 'feed_forward',
    'model', 'optimization', 'training', 'transformer', 'base', 'interface',
    'memory', 'monitoring', 'libraries', 'advanced_libraries', 'learning',
    'list_available_module_submodules',
    
    # Exported classes (Lazy)
    'CudaKernelOptimizer', 'GPUOptimizer', 'MemoryOptimizer',
    'BaseModule', 'ModelModule', 'DataModule', 'TrainingModule',
]

# Version information
__version__ = "1.1.0"
__author__ = "TruthGPT Optimization Core Team"
__description__ = "Modular optimization system for TruthGPT with standardized lazy loading"
