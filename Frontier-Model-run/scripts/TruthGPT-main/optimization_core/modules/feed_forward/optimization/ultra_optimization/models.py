"""
Unified models for Ultra-Optimization System.
"""
from typing import Optional, Dict, List, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict
import torch

from optimization_core.modules.acceleration.gpu import GPUAcceleratorConfig as GPUConfig

class ZeroCopyConfig(BaseModel):
    """Configuration for zero-copy optimization."""
    enable_zero_copy: bool = True
    max_buffer_size: int = 1024 * 1024 * 1024  # 1GB
    use_memory_mapping: bool = True
    use_pinned_memory: bool = True
    enable_in_place_operations: bool = True
    enable_tensor_views: bool = True
    memory_alignment: int = 64  # bytes
    cache_size: int = 1000
    enable_memory_pool: bool = True
    memory_pool_size: int = 100 * 1024 * 1024  # 100MB
    enable_compression: bool = False
    compression_algorithm: str = 'lz4'
    enable_encryption: bool = False
    encryption_algorithm: str = 'aes256'

class BatchingConfig(BaseModel):
    """Configuration for dynamic batching."""
    max_batch_size: int = 32
    min_batch_size: int = 1
    max_wait_time: float = 0.1  # seconds
    enable_priority_batching: bool = True
    enable_adaptive_batching: bool = True
    enable_load_balancing: bool = True
    num_workers: int = 4
    enable_pipeline_optimization: bool = True
    enable_batch_compression: bool = False
    compression_ratio: float = 0.5
    enable_memory_optimization: bool = True
    memory_threshold: float = 0.8
    enable_statistics: bool = True
    statistics_interval: float = 1.0
    enable_auto_scaling: bool = True
    scaling_threshold: float = 0.8
    max_workers: int = 16

class CompilationConfig(BaseModel):
    """Configuration for model compilation."""
    target: str = "torchscript" # Default
    optimization_level: str = 'default'  # default, trace, script, optimize
    backend: str = 'inductor'  # inductor, aot_eager, nvfuser
    max_batch_size: int = 32
    enable_fusion: bool = True
    enable_memory_optimization: bool = True
    enable_quantization: bool = False
    quantization_type: str = 'dynamic'  # dynamic, static
    enable_caching: bool = True
    cache_size: int = 100
    enable_benchmarking: bool = True
    benchmark_runs: int = 100
    enable_profiling: bool = False
    profile_output: str = 'profile.json'
    enable_optimization_reports: bool = True
    optimization_reports_dir: str = 'optimization_reports'

class UltraOptimizationConfig(BaseModel):
    """Unified configuration for the entire Ultra-Optimization system."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    zero_copy: ZeroCopyConfig = Field(default_factory=ZeroCopyConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    compilation: CompilationConfig = Field(default_factory=CompilationConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    
    enable_all: bool = True
    logging_level: str = "INFO"
