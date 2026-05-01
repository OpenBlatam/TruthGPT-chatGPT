"""
Ultra-fast optimization modules
Following deep learning best practices
"""

from .attention import FlashAttention, MultiHeadAttention, AttentionOptimizer
from .memory import MemoryOptimizer, GradientCheckpointing, MemoryEfficientAttention
from .quantization import QuantizationOptimizer, DynamicQuantization, StaticQuantization
from .mixed_precision import MixedPrecisionOptimizer, AMPOptimizer, GradScaler
from .lora import LoRAOptimizer, LoRALayer, LoRAConfig
from .compilation import ModelCompiler, TorchCompile, JITOptimizer
from .distributed import DistributedOptimizer, DataParallelOptimizer, DDPOptimizer
from .profiling import PerformanceProfiler, MemoryProfiler, SpeedProfiler

__all__ = [
    # Attention optimizations
    'FlashAttention', 'MultiHeadAttention', 'AttentionOptimizer',
    
    # Memory optimizations
    'MemoryOptimizer', 'GradientCheckpointing', 'MemoryEfficientAttention',
    
    # Quantization
    'QuantizationOptimizer', 'DynamicQuantization', 'StaticQuantization',
    
    # Mixed precision
    'MixedPrecisionOptimizer', 'AMPOptimizer', 'GradScaler',
    
    # LoRA
    'LoRAOptimizer', 'LoRALayer', 'LoRAConfig',
    
    # Compilation
    'ModelCompiler', 'TorchCompile', 'JITOptimizer',
    
    # Distributed
    'DistributedOptimizer', 'DataParallelOptimizer', 'DDPOptimizer',
    
    # Profiling
    'PerformanceProfiler', 'MemoryProfiler', 'SpeedProfiler'
]



