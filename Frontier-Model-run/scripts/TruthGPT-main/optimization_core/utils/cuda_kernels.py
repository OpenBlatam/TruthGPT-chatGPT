"""
Backward compatibility shim for CUDA kernels.
Redirects to optimization_core.modules.acceleration.gpu.cuda_kernels
"""

from optimization_core.modules.acceleration.gpu.cuda_kernels import (
    CudaKernelConfig,
    CudaKernelType,
    PerformanceMonitor,
    CudaKernelManager,
    CudaKernelOptimizer,
    AdvancedCudaKernelOptimizer,
    create_cuda_kernel_config,
    create_cuda_kernel_optimizer,
    create_advanced_cuda_kernel_optimizer,
    optimize_model_with_cuda_kernels,
    cuda_kernel_context,
    CUDAOptimizations,
    OptimizedLayerNorm,
    OptimizedRMSNorm
)

__all__ = [
    'CudaKernelConfig',
    'CudaKernelType',
    'PerformanceMonitor',
    'CudaKernelManager',
    'CudaKernelOptimizer',
    'AdvancedCudaKernelOptimizer',
    'create_cuda_kernel_config',
    'create_cuda_kernel_optimizer',
    'create_advanced_cuda_kernel_optimizer',
    'optimize_model_with_cuda_kernels',
    'cuda_kernel_context',
    'CUDAOptimizations',
    'OptimizedLayerNorm',
    'OptimizedRMSNorm'
]
