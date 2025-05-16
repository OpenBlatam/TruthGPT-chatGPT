from torch.utils.cpp_extension import load_inline
import torch
from torch.nn import Parameter

cuda_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ scale,
                                  const float* __restrict__ bias,
                                  float* __restrict__ y,
                                  int N,
                                  int D,
                                  float eps) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float shared[];
    float* mean_var = shared;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int d = tid; d < D; d += block_size) {
        float val = x[n * D + d];
        sum += val;
        sum_sq += val * val;
    }

    __shared__ float total_sum, total_sum_sq;
    atomicAdd(&total_sum, sum);
    atomicAdd(&total_sum_sq, sum_sq);
    __syncthreads();

    if (tid == 0) {
        float mean = total_sum / D;
        float var = total_sum_sq / D - mean * mean;
        mean_var[0] = mean;
        mean_var[1] = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = mean_var[0];
    float inv_std = mean_var[1];

    for (int d = tid; d < D; d += block_size) {
        float val = x[n * D + d];
        val = (val - mean) * inv_std;
        val = val * scale[d] + bias[d];
        y[n * D + d] = val;
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, int N, int D, float eps) {
    auto y = torch::empty_like(x);
    const int threads = 256;
    const int shared_mem = 2 * sizeof(float);

    layer_norm_kernel<<<N, threads, shared_mem>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        N, D, eps
    );

    return y;
}
"""

cpp_src = "torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, int N, int D, float eps);"

layer_norm = load_inline(
    name="layer_norm_cuda_mod",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-lineinfo"],
)


## Use the CUDA instead of triton 
class CUDALayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, device: str = "cuda"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.scale = Parameter(torch.ones(normalized_shape, device=device))
        self.bias = Parameter(torch.zeros(normalized_shape, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        return layer_norm.layer_norm_cuda(x, self.scale, self.bias, N, D, self.eps)
