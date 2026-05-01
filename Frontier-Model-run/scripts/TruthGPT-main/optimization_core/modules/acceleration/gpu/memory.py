"""
GPU Memory Management
"""
import torch
import logging
from typing import Dict, Any, Tuple

from .config import GPUAcceleratorConfig

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """Advanced GPU memory management with pooling and optimization."""
    
    def __init__(self, config: GPUAcceleratorConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.memory_pool = {}
        self.memory_stats = {
            'allocations': 0,
            'deallocations': 0,
            'peak_memory': 0,
            'current_memory': 0,
            'pooled_memory': 0
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup memory management
        if self.config.enable_memory_pool and self.device.type == "cuda":
            self._setup_memory_pool()
        
        self.logger.info("✅ GPU Memory Manager initialized")
    
    def _setup_memory_pool(self):
        """Setup memory pool for efficient allocation."""
        if not torch.cuda.is_available():
            return
        # Clear cache
        torch.cuda.empty_cache()
        self.logger.info("Memory pool initialized")
    
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                 pin_memory: bool = True) -> torch.Tensor:
        """Allocate GPU memory efficiently."""
        # Check memory pool
        if self.config.enable_memory_pool and shape in self.memory_pool:
            if self.memory_pool[shape]:
                tensor = self.memory_pool[shape].pop()
                self.memory_stats['allocations'] += 1
                return tensor
        
        # Allocate new memory
        tensor = torch.empty(shape, dtype=dtype, device=self.device, 
                           pin_memory=pin_memory and self.device.type == "cuda")
        
        self.memory_stats['allocations'] += 1
        tensor_bytes = tensor.numel() * tensor.element_size()
        self.memory_stats['current_memory'] += tensor_bytes
        self.memory_stats['peak_memory'] = max(self.memory_stats['peak_memory'], 
                                              self.memory_stats['current_memory'])
        
        return tensor
    
    def deallocate(self, tensor: torch.Tensor):
        """Deallocate GPU memory efficiently."""
        shape = tensor.shape
        tensor_bytes = tensor.numel() * tensor.element_size()
        
        # Cache memory for reuse
        if self.config.enable_memory_pool:
            if shape not in self.memory_pool:
                self.memory_pool[shape] = []
            
            # Limit cache size
            if len(self.memory_pool[shape]) < 10:
                self.memory_pool[shape].append(tensor.detach())
                self.memory_stats['pooled_memory'] += tensor_bytes
        
        self.memory_stats['deallocations'] += 1
        self.memory_stats['current_memory'] -= tensor_bytes
    
    def clear_pool(self):
        """Clear memory pool."""
        self.memory_pool.clear()
        self.memory_stats['pooled_memory'] = 0
        self.memory_stats['current_memory'] = 0
        
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self.logger.info("Memory pool cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {**self.memory_stats}
        if torch.cuda.is_available() and self.device.type == "cuda":
            stats['gpu_allocated'] = torch.cuda.memory_allocated(self.config.device_id)
            stats['gpu_cached'] = torch.cuda.memory_reserved(self.config.device_id)
        else:
            stats['gpu_allocated'] = 0
            stats['gpu_cached'] = 0
        return stats

