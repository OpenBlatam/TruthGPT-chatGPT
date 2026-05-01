/**
 * @file allocator.cpp
 * @brief High-performance memory allocator implementation
 * 
 * Provides memory pooling and allocation strategies optimized for ML workloads.
 * Integrates with mimalloc/jemalloc when available for superior performance.
 */

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <algorithm>

#ifdef HAVE_MIMALLOC
#include <mimalloc.h>
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "memory/cuda_utils.hpp"
#include "memory/aligned_alloc.hpp"

namespace optimization_core {
namespace memory {

/**
 * @brief Memory block header for tracking allocations
 */
struct BlockHeader {
    size_t size;
    size_t alignment;
    uint32_t pool_id;
    bool is_gpu;
};

/**
 * @brief Memory pool for efficient small allocations
 */
class MemoryPool {
public:
    explicit MemoryPool(size_t block_size, size_t initial_blocks = 1024)
        : block_size_(block_size), 
          total_blocks_(initial_blocks),
          free_blocks_(initial_blocks) {
        
        // Allocate pool memory
        pool_memory_ = aligned::allocate(block_size * initial_blocks, 64);
        
        // Initialize free list
        free_list_.reserve(initial_blocks);
        for (size_t i = 0; i < initial_blocks; ++i) {
            free_list_.push_back(static_cast<char*>(pool_memory_) + i * block_size);
        }
    }
    
    ~MemoryPool() {
        aligned::deallocate(pool_memory_);
    }
    
    void* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (free_list_.empty()) {
            expand();
        }
        
        void* ptr = free_list_.back();
        free_list_.pop_back();
        free_blocks_--;
        
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_.push_back(ptr);
        free_blocks_++;
    }
    
    size_t block_size() const { return block_size_; }
    size_t total_blocks() const { return total_blocks_; }
    size_t free_blocks() const { return free_blocks_; }
    
private:
    size_t block_size_;
    size_t total_blocks_;
    std::atomic<size_t> free_blocks_;
    void* pool_memory_;
    std::vector<void*> free_list_;
    std::mutex mutex_;
    
    void expand() {
        size_t new_blocks = total_blocks_;  // Double the pool
        void* new_memory = aligned::allocate(block_size_ * new_blocks, 64);
        
        for (size_t i = 0; i < new_blocks; ++i) {
            free_list_.push_back(static_cast<char*>(new_memory) + i * block_size_);
        }
        
        total_blocks_ += new_blocks;
        free_blocks_ += new_blocks;
    }
};

/**
 * @brief High-performance allocator with pooling
 */
class HighPerformanceAllocator {
public:
    static HighPerformanceAllocator& instance() {
        static HighPerformanceAllocator allocator;
        return allocator;
    }
    
    /**
     * @brief Allocate memory with specified alignment
     */
    void* allocate(size_t size, size_t alignment = 64) {
        // Try pool allocation for small sizes
        if (size <= max_pool_size_) {
            auto it = pools_.find(round_up_to_pool_size(size));
            if (it != pools_.end()) {
                return it->second->allocate();
            }
        }
        
        // Fall back to aligned allocation
        void* ptr = aligned::allocate(size + sizeof(BlockHeader), alignment);
        
        // Store header
        auto* header = static_cast<BlockHeader*>(ptr);
        header->size = size;
        header->alignment = alignment;
        header->pool_id = 0;
        header->is_gpu = false;
        
        total_allocated_ += size;
        
        return static_cast<char*>(ptr) + sizeof(BlockHeader);
    }
    
    /**
     * @brief Deallocate memory
     */
    void deallocate(void* ptr) {
        if (ptr == nullptr) return;
        
        auto* header = reinterpret_cast<BlockHeader*>(
            static_cast<char*>(ptr) - sizeof(BlockHeader));
        
        // Check if from pool
        if (header->pool_id > 0) {
            auto it = pools_.find(header->size);
            if (it != pools_.end()) {
                it->second->deallocate(ptr);
                return;
            }
        }
        
        total_allocated_ -= header->size;
        aligned::deallocate(header);
    }
    
    /**
     * @brief Allocate GPU memory
     */
    void* allocate_gpu(size_t size) {
#ifdef HAVE_CUDA
        void* ptr = cuda::malloc(size);
        gpu_allocated_ += size;
        return ptr;
#else
        cuda::require_cuda("GPU allocation");
        return nullptr;  // Unreachable
#endif
    }
    
    /**
     * @brief Deallocate GPU memory
     */
    void deallocate_gpu(void* ptr) {
#ifdef HAVE_CUDA
        cuda::free(ptr);
#endif
    }
    
    /**
     * @brief Allocate pinned (page-locked) memory for fast GPU transfers
     */
    void* allocate_pinned(size_t size) {
#ifdef HAVE_CUDA
        return cuda::malloc_host(size);
#else
        return allocate(size);
#endif
    }
    
    /**
     * @brief Deallocate pinned memory
     */
    void deallocate_pinned(void* ptr) {
#ifdef HAVE_CUDA
        cuda::free_host(ptr);
#else
        deallocate(ptr);
#endif
    }
    
    /**
     * @brief Get allocation statistics
     */
    struct Stats {
        size_t total_allocated;
        size_t gpu_allocated;
        size_t pool_hits;
        size_t pool_misses;
    };
    
    Stats get_stats() const {
        return {
            total_allocated_.load(),
            gpu_allocated_.load(),
            pool_hits_.load(),
            pool_misses_.load()
        };
    }
    
    /**
     * @brief Clear all pools and free memory
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pools_.clear();
        total_allocated_ = 0;
    }

private:
    HighPerformanceAllocator() {
        // Initialize pools for common sizes
        std::vector<size_t> pool_sizes = {
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
        };
        
        for (size_t size : pool_sizes) {
            pools_[size] = std::make_unique<MemoryPool>(size);
        }
    }
    
    size_t round_up_to_pool_size(size_t size) {
        // Round up to next power of 2
        size--;
        size |= size >> 1;
        size |= size >> 2;
        size |= size >> 4;
        size |= size >> 8;
        size |= size >> 16;
        size++;
        return std::max(size, static_cast<size_t>(64));
    }
    
    
    static constexpr size_t max_pool_size_ = 65536;  // 64KB
    
    std::unordered_map<size_t, std::unique_ptr<MemoryPool>> pools_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> gpu_allocated_{0};
    std::atomic<size_t> pool_hits_{0};
    std::atomic<size_t> pool_misses_{0};
    std::mutex mutex_;
};

// Global allocation functions
void* fast_alloc(size_t size, size_t alignment) {
    return HighPerformanceAllocator::instance().allocate(size, alignment);
}

void fast_free(void* ptr) {
    HighPerformanceAllocator::instance().deallocate(ptr);
}

void* gpu_alloc(size_t size) {
    return HighPerformanceAllocator::instance().allocate_gpu(size);
}

void gpu_free(void* ptr) {
    HighPerformanceAllocator::instance().deallocate_gpu(ptr);
}

} // namespace memory
} // namespace optimization_core





