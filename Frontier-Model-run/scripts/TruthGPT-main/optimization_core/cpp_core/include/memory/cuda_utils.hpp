#pragma once

/**
 * @file cuda_utils.hpp
 * @brief CUDA utility functions for error handling and operations
 */

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#endif

namespace optimization_core {
namespace memory {
namespace cuda {

#ifdef HAVE_CUDA

/**
 * @brief Check CUDA error and throw exception if failed
 * 
 * @param err CUDA error code
 * @param operation Description of operation that failed
 * @throws std::runtime_error if error occurred
 */
inline void check_cuda_error(cudaError_t err, const std::string& operation) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            "CUDA " + operation + " failed: " + 
            std::string(cudaGetErrorString(err))
        );
    }
}

/**
 * @brief Allocate GPU memory with error handling
 * 
 * @param size Size in bytes
 * @return Allocated pointer
 * @throws std::runtime_error if allocation failed
 */
inline void* malloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    check_cuda_error(err, "allocation");
    return ptr;
}

/**
 * @brief Allocate pinned (page-locked) host memory
 * 
 * @param size Size in bytes
 * @return Allocated pointer
 * @throws std::runtime_error if allocation failed
 */
inline void* malloc_host(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    check_cuda_error(err, "pinned allocation");
    return ptr;
}

/**
 * @brief Free GPU memory
 * 
 * @param ptr Pointer to free
 */
inline void free(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

/**
 * @brief Free pinned host memory
 * 
 * @param ptr Pointer to free
 */
inline void free_host(void* ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

#else

/**
 * @brief Throw error if CUDA is not available
 * 
 * @param operation Description of operation
 * @throws std::runtime_error always
 */
inline void require_cuda(const std::string& operation) {
    throw std::runtime_error("CUDA not available for: " + operation);
}

#endif // HAVE_CUDA

} // namespace cuda
} // namespace memory
} // namespace optimization_core








