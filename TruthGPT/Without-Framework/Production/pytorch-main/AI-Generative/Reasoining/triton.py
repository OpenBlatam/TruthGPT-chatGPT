import torch
from torch.nn import Parameter
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl

# Define the custom CUDA kernel for Layer Normalization
layer_norm_source = """
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
    const int tid = threadIdx.x;
    const int block_size = 1024;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int d = tid; d < D; d += block_size) {
        float x_val = x[n * D + d];
        sum += x_val;
        sum_sq += x_val * x_val;
    }

    // Warp-level reduction using shuffle instructions
    int warp_id = tid % 32;
    int warp_lane = tid / 32;

    float warp_sum = sum;
    float warp_sum_sq = sum_sq;

    for (int delta = 1; delta <= 16; delta <<= 1) {
        float other_sum = __shfl_xor_sync(0xFFFFFFFF, warp_sum, delta);
        float other_sum_sq = __shfl_xor_sync(0xFFFFFFFF, warp_sum_sq, delta);
        warp_sum += other_sum;
        warp_sum_sq += other_sum_sq;
    }

    __shared__ float sum_warp[32];
    __shared__ float sum_sq_warp[32];
    __shared__ float results[2]; // [mean, inv_std]

    if (warp_id == 0) {
        sum_warp[warp_lane] = warp_sum;
        sum_sq_warp[warp_lane] = warp_sum_sq;
    }
    __syncthreads();

    // Final reduction within the first warp (tid 0-31)
    if (tid < 32) {
        float my_sum = sum_warp[tid];
        float my_sum_sq = sum_sq_warp[tid];

        // Reduce within the first warp (32 threads)
        for (int s = 16; s >= 1; s >>= 1) {
            my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, s);
            my_sum_sq += __shfl_xor_sync(0xFFFFFFFF, my_sum_sq, s);
        }

        if (tid == 0) {
            float total_sum = my_sum;
            float total_sum_sq = my_sum_sq;
            float mean = total_sum / D;
            float variance = (total_sum_sq / D) - mean * mean;
            float inv_std = rsqrtf(variance + eps);
            results[0] = mean;
            results[1] = inv_std;
        }
    }
    __syncthreads();

    float mean = results[0];
    float inv_std = results[1];

    for (int d = tid; d < D; d += block_size) {
        float x_val = x[n * D + d];
        float y_val = (x_val - mean) * inv_std;
        y_val = y_val * scale[d] + bias[d];
        y[n * D + d] = y_val;
    }
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, int N, int D, float eps);"
)

# Compile the inline CUDA code for Layer Normalization
layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class TritonLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.scale = Parameter(torch.ones(normalized_shape, device="cuda"))
        self.bias = Parameter(torch.zeros(normalized_shape, device="cuda"))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.size(0), self.normalized_shape
        return layer_norm.layer_norm_cuda(x, self.scale, self.bias, N, D, self.eps)


# Example usage of TritonLayerNorm
if __name__ == "__main__":
    input_data = torch.randn((32, 1024), device="cuda")
    model = TritonLayerNorm(1024)
    output = model(input_data)
    print(output)