"""
TruthGPT Core Utilities
Core components for TruthGPT optimization following deep learning best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager
import time
from pathlib import Path
import json
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Optimization levels for TruthGPT."""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"
    LEGENDARY = "legendary"
    TRANSCENDENT = "transcendent"
    DIVINE = "divine"
    OMNIPOTENT = "omnipotent"
    INFINITE = "infinite"
    ULTIMATE = "ultimate"

class DeviceType(Enum):
    """Device types for TruthGPT."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

class PrecisionType(Enum):
    """Precision types for TruthGPT."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"

@dataclass
class TruthGPTConfig:
    """Configuration for TruthGPT optimization."""
    # Model configuration
    model_name: str = "truthgpt"
    model_size: str = "base"
    precision: PrecisionType = PrecisionType.FP16
    device: DeviceType = DeviceType.AUTO
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    
    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_sequence_length: int = 2048
    num_epochs: int = 100
    
    # Optimization configuration
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_attention_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_distillation: bool = False
    
    # Performance configuration
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Monitoring configuration
    enable_monitoring: bool = True
    enable_profiling: bool = True
    log_interval: int = 100
    save_interval: int = 1000
    
    # Advanced configuration
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"  # linear, cosine, exponential
    early_stopping_patience: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'precision': self.precision.value,
            'device': self.device.value,
            'optimization_level': self.optimization_level.value,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_sequence_length': self.max_sequence_length,
            'num_epochs': self.num_epochs,
            'enable_mixed_precision': self.enable_mixed_precision,
            'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
            'enable_attention_optimization': self.enable_attention_optimization,
            'enable_memory_optimization': self.enable_memory_optimization,
            'enable_quantization': self.enable_quantization,
            'enable_pruning': self.enable_pruning,
            'enable_distillation': self.enable_distillation,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'prefetch_factor': self.prefetch_factor,
            'enable_monitoring': self.enable_monitoring,
            'enable_profiling': self.enable_profiling,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'gradient_clip_norm': self.gradient_clip_norm,
            'warmup_steps': self.warmup_steps,
            'scheduler_type': self.scheduler_type,
            'early_stopping_patience': self.early_stopping_patience
        }

class BaseTruthGPTOptimizer(ABC):
    """Base class for TruthGPT optimizers."""
    
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.optimization_stats = {}
    
    @abstractmethod
    def optimize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
        """Optimize model."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.optimization_stats
    
    def _log_optimization(self, message: str, level: str = "info") -> None:
        """Log optimization message."""
        getattr(self.logger, level)(f"🔧 {message}")

class TruthGPTDeviceManager:
    """Device management for TruthGPT."""
    
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.device = self._setup_device()
        self.scaler = GradScaler() if config.enable_mixed_precision else None
    
    def _setup_device(self) -> torch.device:
        """Setup device for TruthGPT."""
        if self.config.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU device")
        elif self.config.device == DeviceType.CUDA:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                raise RuntimeError("CUDA not available but requested")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU device")
        
        return device
    
    def move_to_device(self, model: nn.Module) -> nn.Module:
        """Move model to device."""
        return model.to(self.device)
    
    def get_device(self) -> torch.device:
        """Get current device."""
        return self.device
    
    def get_scaler(self) -> Optional[GradScaler]:
        """Get gradient scaler."""
        return self.scaler

class TruthGPTPrecisionManager:
    """Precision management for TruthGPT."""
    
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def apply_precision(self, model: nn.Module) -> nn.Module:
        """Apply precision to model."""
        if self.config.precision == PrecisionType.FP16:
            model = model.half()
            self.logger.info("✅ Applied FP16 precision")
        elif self.config.precision == PrecisionType.BF16:
            model = model.bfloat16()
            self.logger.info("✅ Applied BF16 precision")
        elif self.config.precision == PrecisionType.INT8:
            # This would require more sophisticated quantization
            self.logger.warning("INT8 precision requires quantization setup")
        elif self.config.precision == PrecisionType.INT4:
            # This would require more sophisticated quantization
            self.logger.warning("INT4 precision requires quantization setup")
        
        return model
    
    def get_precision_dtype(self) -> torch.dtype:
        """Get precision dtype."""
        if self.config.precision == PrecisionType.FP16:
            return torch.float16
        elif self.config.precision == PrecisionType.BF16:
            return torch.bfloat16
        elif self.config.precision == PrecisionType.INT8:
            return torch.qint8
        elif self.config.precision == PrecisionType.INT4:
            return torch.qint4
        else:
            return torch.float32

class TruthGPTMemoryManager:
    """Memory management for TruthGPT."""
    
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.memory_stats = {}
    
    def optimize_memory(self, model: nn.Module) -> nn.Module:
        """Optimize memory usage."""
        if not self.config.enable_memory_optimization:
            return model
        
        self.logger.info("🧠 Optimizing memory usage")
        
        # Enable gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
            self.logger.info("✅ Gradient checkpointing enabled")
        
        # Enable memory efficient attention
        if hasattr(model, 'enable_memory_efficient_attention'):
            model.enable_memory_efficient_attention()
            self.logger.info("✅ Memory efficient attention enabled")
        
        # Calculate memory usage
        self._calculate_memory_stats(model)
        
        return model
    
    def _calculate_memory_stats(self, model: nn.Module) -> None:
        """Calculate memory statistics."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        self.memory_stats = {
            'parameters_mb': param_size / (1024 * 1024),
            'buffers_mb': buffer_size / (1024 * 1024),
            'total_mb': (param_size + buffer_size) / (1024 * 1024),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        if torch.cuda.is_available():
            self.memory_stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_cached_mb': torch.cuda.memory_reserved() / (1024 * 1024)
            })
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory_stats

class TruthGPTPerformanceManager:
    """Performance management for TruthGPT."""
    
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.performance_stats = {}
    
    def optimize_performance(self, model: nn.Module) -> nn.Module:
        """Optimize performance."""
        self.logger.info("⚡ Optimizing performance")
        
        # Enable cuDNN optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.logger.info("✅ cuDNN optimizations enabled")
        
        # Enable JIT compilation
        if hasattr(model, 'eval'):
            model.eval()
            try:
                # Create sample input for JIT compilation
                sample_input = torch.randn(1, self.config.max_sequence_length, 768)
                if self.config.precision == PrecisionType.FP16:
                    sample_input = sample_input.half()
                elif self.config.precision == PrecisionType.BF16:
                    sample_input = sample_input.bfloat16()
                
                # JIT compile
                model = torch.jit.optimize_for_inference(model)
                self.logger.info("✅ JIT compilation applied")
            except Exception as e:
                self.logger.warning(f"JIT compilation failed: {e}")
        
        # Enable tensor core optimizations
        if torch.cuda.is_available() and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("✅ Tensor Core optimizations enabled")
        
        self.performance_stats = {
            'mixed_precision': self.config.enable_mixed_precision,
            'jit_compilation': True,
            'cudnn_optimizations': torch.cuda.is_available()
        }
        
        return model
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats

class TruthGPTAttentionOptimizer(BaseTruthGPTOptimizer):
    """Attention optimization for TruthGPT."""
    
    def __init__(self, config: TruthGPTConfig):
        super().__init__(config)
        self.attention_stats = {}
    
    def optimize(self, model: nn.Module) -> nn.Module:
        """Optimize attention mechanisms."""
        if not self.config.enable_attention_optimization:
            return model
        
        self._log_optimization("Optimizing attention mechanisms")
        
        # Optimize attention layers
        for name, module in model.named_modules():
            if hasattr(module, 'attention_dropout'):
                module.attention_dropout = 0.0  # Disable for inference
            if hasattr(module, 'dropout'):
                module.dropout = 0.0  # Disable for inference
        
        # Enable memory efficient attention
        if hasattr(model, 'enable_memory_efficient_attention'):
            model.enable_memory_efficient_attention()
            self._log_optimization("Memory efficient attention enabled")
        
        # Enable flash attention if available
        if hasattr(model, 'enable_flash_attention'):
            model.enable_flash_attention()
            self._log_optimization("Flash attention enabled")
        
        self.attention_stats = {
            'attention_optimization_enabled': True,
            'memory_efficient_attention': hasattr(model, 'enable_memory_efficient_attention'),
            'flash_attention': hasattr(model, 'enable_flash_attention')
        }
        
        return model

class TruthGPTQuantizationOptimizer(BaseTruthGPTOptimizer):
    """Quantization optimization for TruthGPT."""
    
    def __init__(self, config: TruthGPTConfig):
        super().__init__(config)
        self.quantization_stats = {}
    
    def optimize(self, model: nn.Module) -> nn.Module:
        """Optimize quantization."""
        if not self.config.enable_quantization:
            return model
        
        self._log_optimization("Optimizing quantization")
        
        try:
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.LSTM, nn.GRU}, 
                dtype=torch.qint8
            )
            
            self.quantization_stats = {
                'quantization_enabled': True,
                'method': 'dynamic',
                'dtype': 'qint8'
            }
            
            self._log_optimization("Dynamic quantization applied")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            return model

class TruthGPTPruningOptimizer(BaseTruthGPTOptimizer):
    """Pruning optimization for TruthGPT."""
    
    def __init__(self, config: TruthGPTConfig):
        super().__init__(config)
        self.pruning_stats = {}
    
    def optimize(self, model: nn.Module) -> nn.Module:
        """Optimize pruning."""
        if not self.config.enable_pruning:
            return model
        
        self._log_optimization("Optimizing pruning")
        
        # Apply magnitude-based pruning
        pruned_params = 0
        total_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weights = module.weight.data
                total_params += weights.numel()
                
                # Calculate threshold
                threshold = torch.quantile(torch.abs(weights), 0.1)  # 10% pruning
                
                # Create mask
                mask = torch.abs(weights) > threshold
                pruned_params += (mask == 0).sum().item()
                
                # Apply pruning
                module.weight.data *= mask.float()
        
        self.pruning_stats = {
            'pruning_enabled': True,
            'method': 'magnitude',
            'pruned_params': pruned_params,
            'total_params': total_params,
            'sparsity': pruned_params / total_params if total_params > 0 else 0.0
        }
        
        self._log_optimization(f"Pruning applied: {pruned_params}/{total_params} parameters pruned")
        return model

class TruthGPTIntegratedOptimizer:
    """Integrated TruthGPT optimizer."""
    
    def __init__(self, config: TruthGPTConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize managers
        self.device_manager = TruthGPTDeviceManager(config)
        self.precision_manager = TruthGPTPrecisionManager(config)
        self.memory_manager = TruthGPTMemoryManager(config)
        self.performance_manager = TruthGPTPerformanceManager(config)
        
        # Initialize optimizers
        self.attention_optimizer = TruthGPTAttentionOptimizer(config)
        self.quantization_optimizer = TruthGPTQuantizationOptimizer(config)
        self.pruning_optimizer = TruthGPTPruningOptimizer(config)
        
        self.integration_stats = {}
    
    def optimize_comprehensive(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive optimization."""
        self.logger.info("🚀 Starting comprehensive TruthGPT optimization")
        
        # Move to device
        model = self.device_manager.move_to_device(model)
        
        # Apply precision
        model = self.precision_manager.apply_precision(model)
        
        # Apply memory optimization
        model = self.memory_manager.optimize_memory(model)
        
        # Apply performance optimization
        model = self.performance_manager.optimize_performance(model)
        
        # Apply attention optimization
        model = self.attention_optimizer.optimize(model)
        
        # Apply quantization
        model = self.quantization_optimizer.optimize(model)
        
        # Apply pruning
        model = self.pruning_optimizer.optimize(model)
        
        # Compile statistics
        self._compile_stats()
        
        self.logger.info("✅ Comprehensive TruthGPT optimization completed")
        return model
    
    def _compile_stats(self) -> None:
        """Compile optimization statistics."""
        self.integration_stats = {
            'device': str(self.device_manager.get_device()),
            'precision': self.config.precision.value,
            'memory': self.memory_manager.get_memory_stats(),
            'performance': self.performance_manager.get_performance_stats(),
            'attention': self.attention_optimizer.get_stats(),
            'quantization': self.quantization_optimizer.get_stats(),
            'pruning': self.pruning_optimizer.get_stats()
        }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return self.integration_stats

# Factory functions
def create_truthgpt_config(**kwargs) -> TruthGPTConfig:
    """Create TruthGPT configuration."""
    return TruthGPTConfig(**kwargs)

def create_truthgpt_optimizer(config: TruthGPTConfig) -> TruthGPTIntegratedOptimizer:
    """Create TruthGPT optimizer."""
    return TruthGPTIntegratedOptimizer(config)

def quick_truthgpt_optimization(model: nn.Module, 
                              optimization_level: str = "advanced",
                              precision: str = "fp16",
                              device: str = "auto") -> nn.Module:
    """Quick TruthGPT optimization."""
    config = TruthGPTConfig(
        optimization_level=OptimizationLevel(optimization_level),
        precision=PrecisionType(precision),
        device=DeviceType(device)
    )
    
    optimizer = create_truthgpt_optimizer(config)
    return optimizer.optimize_comprehensive(model)

# Context managers
@contextmanager
def truthgpt_optimization_context(model: nn.Module, config: TruthGPTConfig):
    """Context manager for TruthGPT optimization."""
    optimizer = create_truthgpt_optimizer(config)
    try:
        yield optimizer
    finally:
        # Cleanup if needed
        pass

# Example usage
if __name__ == "__main__":
    # Example TruthGPT optimization
    print("🚀 TruthGPT Core Utilities Demo")
    print("=" * 50)
    
    # Create a sample TruthGPT-style model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 768)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.1),
                num_layers=12
            )
            self.lm_head = nn.Linear(768, 10000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.lm_head(x)
            return x
    
    # Create model
    model = TruthGPTModel()
    
    # Create configuration
    config = create_truthgpt_config(
        optimization_level=OptimizationLevel.ADVANCED,
        precision=PrecisionType.FP16,
        device=DeviceType.AUTO
    )
    
    # Optimize model
    with truthgpt_optimization_context(model, config) as optimizer:
        optimized_model = optimizer.optimize_comprehensive(model)
        
        # Get statistics
        stats = optimizer.get_integration_stats()
        print(f"Optimization statistics: {stats}")
    
    print("✅ TruthGPT optimization completed!")



