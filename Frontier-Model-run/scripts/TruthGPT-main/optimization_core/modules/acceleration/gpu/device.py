"""
GPU Device Management
"""
import torch
import logging
from typing import Dict, Any

from .config import GPUAcceleratorConfig

logger = logging.getLogger(__name__)

class GPUDeviceManager:
    """Advanced GPU device management with automatic configuration."""
    
    def __init__(self, config: GPUAcceleratorConfig):
        self.config = config
        self.device = self._setup_device()
        self.device_properties = self._get_device_properties()
        self.memory_info = self._get_memory_info()
        
        # Setup CUDA optimizations
        self._setup_cuda_optimizations()
        
        logger.info(f"✅ GPU Device Manager initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup device for GPU operations."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.device_id}")
            
            # Set CUDA device
            torch.cuda.set_device(self.config.device_id)
            
            # Enable memory management
            if self.config.enable_memory_pool:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.max_memory_fraction,
                    self.config.device_id
                )
            
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(self.config.device_id)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    def _setup_cuda_optimizations(self):
        """Setup CUDA optimizations following best practices."""
        if not torch.cuda.is_available() or self.config.device != "cuda":
            return
        
        # Enable cuDNN optimizations
        if self.config.enable_cudnn:
            torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
            torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
        
        # Enable Tensor Core optimizations
        if self.config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache
        torch.cuda.empty_cache()
        
        logger.info("CUDA optimizations enabled")
    
    def _get_device_properties(self) -> Dict[str, Any]:
        """Get GPU device properties."""
        if not torch.cuda.is_available() or self.config.device != "cuda":
            return {}
        
        props = torch.cuda.get_device_properties(self.config.device_id)
        return {
            'name': props.name,
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count,
            'total_memory': props.total_memory,
            'max_threads_per_block': props.max_threads_per_block,
            'warp_size': props.warp_size
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not torch.cuda.is_available() or self.config.device != "cuda":
            return {}
        
        total_memory = torch.cuda.get_device_properties(self.config.device_id).total_memory
        allocated = torch.cuda.memory_allocated(self.config.device_id)
        cached = torch.cuda.memory_reserved(self.config.device_id)
        
        return {
            'total_memory': total_memory,
            'allocated_memory': allocated,
            'cached_memory': cached,
            'free_memory': total_memory - allocated,
            'utilization': allocated / total_memory if total_memory > 0 else 0.0
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        return {
            'device': str(self.device),
            'properties': self.device_properties,
            'memory': self._get_memory_info()
        }

