"""
CUDA Optimizer Support
"""
import torch
import torch.nn as nn
import logging

from .config import GPUAcceleratorConfig

logger = logging.getLogger(__name__)

class CUDAOptimizer:
    """Advanced CUDA optimization with kernel fusion and optimization."""
    
    def __init__(self, config: GPUAcceleratorConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize optimizations
        self._initialize_optimizations()
        
        logger.info("✅ CUDA Optimizer initialized")
    
    def _initialize_optimizations(self):
        """Initialize CUDA optimizations."""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return
        
        # Setup cuDNN
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
        
        # Setup Tensor Cores
        if self.config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        logger.info("CUDA optimizations initialized")
    
    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor operations for GPU."""
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        
        # Ensure tensor is contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        return tensor
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model for GPU execution."""
        # Move to device
        model = model.to(self.device)
        
        # Enable gradient checkpointing if available
        if self.config.enable_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
        
        # Apply kernel fusion if enabled
        if self.config.enable_kernel_fusion:
            model = self._apply_kernel_fusion(model)
        
        return model
    
    def _apply_kernel_fusion(self, model: nn.Module) -> nn.Module:
        """Apply kernel fusion to model."""
        try:
            # Requires PyTorch >= 2.0
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
                self.logger.info("Kernel fusion applied via torch.compile")
        except Exception as e:
            self.logger.warning(f"Kernel fusion failed: {e}")
        
        return model

