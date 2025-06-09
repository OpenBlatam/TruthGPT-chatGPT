"""
Optimized variant models with enhanced performance benchmarking.
"""

from .benchmark_suite import BenchmarkSuite, ModelBenchmark
from .optimized_deepseek import OptimizedDeepSeekV3, create_optimized_deepseek_model
from .optimized_viral_clipper import OptimizedViralClipper, create_optimized_viral_clipper_model
from .optimized_brandkit import OptimizedBrandAnalyzer, OptimizedContentGenerator
from .performance_utils import PerformanceProfiler
from .benchmark_suite import MemoryTracker, SpeedBenchmark
from .advanced_optimizations import (
    AdvancedOptimizationSuite, DynamicQuantization, KernelFusion,
    MemoryOptimizer, OptimizedAttentionKernels, ComputeOptimizer,
    BatchOptimizer, CacheOptimizer, ModelCompiler, apply_advanced_optimizations
)
from .ultra_optimized_models import (
    UltraOptimizedModel, create_ultra_optimized_deepseek,
    create_ultra_optimized_viral_clipper, create_ultra_optimized_brandkit
)

__all__ = [
    'BenchmarkSuite',
    'ModelBenchmark', 
    'OptimizedDeepSeekV3',
    'OptimizedViralClipper',
    'OptimizedBrandAnalyzer',
    'OptimizedContentGenerator',
    'PerformanceProfiler',
    'MemoryTracker',
    'SpeedBenchmark',
    'create_optimized_deepseek_model',
    'create_optimized_viral_clipper_model',
    'AdvancedOptimizationSuite',
    'DynamicQuantization',
    'KernelFusion',
    'MemoryOptimizer',
    'OptimizedAttentionKernels',
    'ComputeOptimizer',
    'BatchOptimizer',
    'CacheOptimizer',
    'ModelCompiler',
    'apply_advanced_optimizations',
    'UltraOptimizedModel',
    'create_ultra_optimized_deepseek',
    'create_ultra_optimized_viral_clipper',
    'create_ultra_optimized_brandkit'
]
