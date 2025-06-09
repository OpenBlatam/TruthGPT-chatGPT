"""
CUDA-optimized kernels for TruthGPT variants.
Integrated from QKV-CUDA.PY and triton.py optimization files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.cpp_extension import load_inline
from typing import Optional, Union, Tuple
import warnings
import math
import os
import platform

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    warnings.warn("Triton not available. Some optimizations will be disabled.")
    TRITON_AVAILABLE = False
    triton = None
    tl = None

class CUDAConfig:
    """CUDA configuration and utilities."""
    WARP_SIZE = 32
    MAX_THREADS_PER_BLOCK = 1024
    MIN_THREADS_PER_BLOCK = 128
    VECTOR_SIZE = 4
    MAX_WARPS_PER_BLOCK = MAX_THREADS_PER_BLOCK // WARP_SIZE
    TILE_SIZE = 16

    @staticmethod
    def get_optimal_block_size(dim: int) -> int:
        """Calculate optimal block size for given dimension."""
        return min(CUDAConfig.MAX_THREADS_PER_BLOCK,
                  max(CUDAConfig.MIN_THREADS_PER_BLOCK,
                      (dim + CUDAConfig.WARP_SIZE - 1) & ~(CUDAConfig.WARP_SIZE - 1)))

    @staticmethod
    def get_compilation_flags() -> list:
        """Get optimal compilation flags based on system."""
        flags = ["-O3", "-DNDEBUG", "-Xfatbin", "-compress-all"]
        
        flags.append("-arch=compute_75,code=sm_75")
        
        flags.extend(["-DUSE_TENSOR_CORES", "-D__CUDA_ARCH__=750"])
        
        if platform.system() == "Linux":
            flags.extend(["-D__linux__", "-D__GNUC__"])
        
        return flags

layer_norm_source = """

namespace cg = cooperative_groups;

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int VECTOR_SIZE = 4;

template<typename T>
__device__ __forceinline__ T warp_reduce(cg::thread_block_tile<WARP_SIZE>& tile, T val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float fast_rsqrt(float x) {
    float xhalf = 0.5f * x;
    int i = __float_as_int(x);
    i = 0x5f3759df - (i >> 1);
    x = __int_as_float(i);
    x = x * (1.5f - xhalf * x * x);
    return x;
}

__global__ void layer_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int D, float eps
) {
    extern __shared__ float sdata[];
    float* mean_shared = sdata;
    float* var_shared = sdata + blockDim.x;

    int n = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> tile = cg::tiled_partition<WARP_SIZE>(block);

    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int d = tid * VECTOR_SIZE; d < D; d += blockDim.x * VECTOR_SIZE) {
        if (d + VECTOR_SIZE <= D) {
            float4 vals = reinterpret_cast<const float4*>(&x[n * D + d])[0];
            sum += vals.x + vals.y + vals.z + vals.w;
            sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
        } else {
            for (int i = 0; i < VECTOR_SIZE && d + i < D; i++) {
                float val = x[n * D + d + i];
                sum += val;
                sum_sq += val * val;
            }
        }
    }

    sum = warp_reduce(tile, sum);
    sum_sq = warp_reduce(tile, sum_sq);

    if (lane_id == 0) {
        mean_shared[warp_id] = sum;
        var_shared[warp_id] = sum_sq;
    }
    block.sync();

    if (tid < WARP_SIZE) {
        sum = mean_shared[tid];
        sum_sq = var_shared[tid];
        sum = warp_reduce(tile, sum);
        sum_sq = warp_reduce(tile, sum_sq);
        
        if (tid == 0) {
            float mean = sum / D;
            float variance = (sum_sq / D) - mean * mean;
            float inv_std = fast_rsqrt(variance + eps);
            mean_shared[0] = mean;
            var_shared[0] = inv_std;
        }
    }
    block.sync();

    float mean = mean_shared[0];
    float inv_std = var_shared[0];

    for (int d = tid * VECTOR_SIZE; d < D; d += blockDim.x * VECTOR_SIZE) {
        if (d + VECTOR_SIZE <= D) {
            float4 vals = reinterpret_cast<const float4*>(&x[n * D + d])[0];
            float4 gammas = reinterpret_cast<const float4*>(&gamma[d])[0];
            float4 betas = reinterpret_cast<const float4*>(&beta[d])[0];
            
            float4 norm_vals;
            norm_vals.x = (vals.x - mean) * inv_std * gammas.x + betas.x;
            norm_vals.y = (vals.y - mean) * inv_std * gammas.y + betas.y;
            norm_vals.z = (vals.z - mean) * inv_std * gammas.z + betas.z;
            norm_vals.w = (vals.w - mean) * inv_std * gammas.w + betas.w;
            
            reinterpret_cast<float4*>(&y[n * D + d])[0] = norm_vals;
        } else {
            for (int i = 0; i < VECTOR_SIZE && d + i < D; i++) {
                int idx = n * D + d + i;
                float norm_val = (x[idx] - mean) * inv_std;
                y[idx] = norm_val * gamma[d + i] + beta[d + i];
            }
        }
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int N, int D, float eps) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(gamma.is_cuda(), "Gamma tensor must be on CUDA device");
    TORCH_CHECK(beta.is_cuda(), "Beta tensor must be on CUDA device");

    auto y = torch.empty_like(x);

    int block_size = min(1024, max(128, (D + 31) & ~31));
    dim3 grid(N);
    dim3 block(block_size);
    const int shared_memory_size = 2 * block_size * sizeof(float);
    
    layer_norm_kernel<<<grid, block, shared_memory_size>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), 
        y.data_ptr<float>(), N, D, eps
    );

    return y;
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int N, int D, float eps);"
)

try:
    layer_norm_cuda = load_inline(
        name="layer_norm_cuda",
        cpp_sources=layer_norm_cpp_source,
        cuda_sources=layer_norm_source,
        functions=["layer_norm_cuda"],
        verbose=False,
        extra_cflags=CUDAConfig.get_compilation_flags(),
        extra_ldflags=[""],
    )
    CUDA_AVAILABLE = True
except Exception as e:
    warnings.warn(f"CUDA kernel compilation failed: {e}. Falling back to PyTorch implementation.")
    CUDA_AVAILABLE = False
    layer_norm_cuda = None

class OptimizedLayerNorm(nn.Module):
    """
    Optimized Layer Normalization with CUDA kernels and fallback to PyTorch.
    """
    
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5, 
                 elementwise_affine: bool = True, device: Optional[str] = None):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = Parameter(torch.ones(normalized_shape, device=device))
            self.bias = Parameter(torch.zeros(normalized_shape, device=device))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        original_shape = x.shape
        if x.dim() == 2:
            N, D = x.size()
        else:
            N = x.size(0)
            D = self.normalized_shape[0]
            x = x.view(N, -1)
        
        if (CUDA_AVAILABLE and layer_norm_cuda is not None and 
            x.dtype == torch.float32 and self.elementwise_affine and
            D == self.normalized_shape[0]):
            
            try:
                result = layer_norm_cuda(x, self.weight, self.bias, N, D, self.eps)
                return result.view(original_shape)
            except Exception as e:
                warnings.warn(f"CUDA kernel failed: {e}. Falling back to PyTorch.")
        
        result = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return result.view(original_shape)
    
    def extra_repr(self) -> str:
        return (f'normalized_shape={self.normalized_shape}, '
                f'eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}, '
                f'cuda_available={CUDA_AVAILABLE}')

class OptimizedRMSNorm(nn.Module):
    """Optimized RMS Normalization with CUDA acceleration."""
    
    def __init__(self, dim: int, eps: float = 1e-6, device: Optional[str] = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CUDAOptimizations:
    """Utility class for applying CUDA optimizations to models."""
    
    @staticmethod
    def replace_layer_norm(model: nn.Module, eps: float = 1e-5) -> nn.Module:
        """Replace all LayerNorm modules with OptimizedLayerNorm."""
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                else:
                    parent = model
                    child_name = name
                
                optimized_norm = OptimizedLayerNorm(
                    normalized_shape=module.normalized_shape,
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine,
                    device=next(module.parameters()).device if module.elementwise_affine else None
                )
                
                if module.elementwise_affine and optimized_norm.elementwise_affine:
                    optimized_norm.weight.data.copy_(module.weight.data)
                    optimized_norm.bias.data.copy_(module.bias.data)
                
                setattr(parent, child_name, optimized_norm)
        
        return model
    
    @staticmethod
    def replace_rms_norm(model: nn.Module) -> nn.Module:
        """Replace RMSNorm modules with OptimizedRMSNorm."""
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and 'RMSNorm' in module.__class__.__name__:
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                else:
                    parent = model
                    child_name = name
                
                if hasattr(module, 'weight') and hasattr(module, 'eps'):
                    optimized_norm = OptimizedRMSNorm(
                        dim=module.weight.shape[0],
                        eps=module.eps,
                        device=module.weight.device
                    )
                    optimized_norm.weight.data.copy_(module.weight.data)
                    setattr(parent, child_name, optimized_norm)
        
        return model
    
    @staticmethod
    def get_optimization_report(model: nn.Module) -> dict:
        """Get a report of optimization status."""
        total_modules = 0
        optimized_modules = 0
        layer_norm_modules = 0
        
        for module in model.modules():
            total_modules += 1
            if isinstance(module, nn.LayerNorm):
                layer_norm_modules += 1
            elif isinstance(module, (OptimizedLayerNorm, OptimizedRMSNorm)):
                optimized_modules += 1
                layer_norm_modules += 1
        
        return {
            'total_modules': total_modules,
            'layer_norm_modules': layer_norm_modules,
            'optimized_modules': optimized_modules,
            'optimization_ratio': optimized_modules / layer_norm_modules if layer_norm_modules > 0 else 0,
            'cuda_available': CUDA_AVAILABLE
        }
