"""
Memory management shim for modules.base.
"""
from ..acceleration.gpu.memory import (
    GPUMemoryManager as MemoryManager,
)
# Assuming a generic MemoryConfig might be needed
from ..acceleration.gpu.config import GPUAcceleratorConfig as MemoryConfig

def optimize_memory_usage():
    """Legacy optimization call."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

__all__ = ['MemoryManager', 'MemoryConfig', 'optimize_memory_usage']
