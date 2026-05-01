#pragma once

/**
 * @file aligned_alloc.hpp
 * @brief Platform-agnostic aligned memory allocation
 */

#include <cstdlib>
#include <cstddef>

#ifdef HAVE_MIMALLOC
#include <mimalloc.h>
#endif

namespace optimization_core {
namespace memory {
namespace aligned {

/**
 * @brief Allocate aligned memory
 * 
 * @param size Size in bytes
 * @param alignment Alignment requirement (must be power of 2)
 * @return Allocated pointer (nullptr on failure)
 */
inline void* allocate(size_t size, size_t alignment) {
#ifdef HAVE_MIMALLOC
    return mi_malloc_aligned(size, alignment);
#elif defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/**
 * @brief Deallocate aligned memory
 * 
 * @param ptr Pointer to deallocate
 */
inline void deallocate(void* ptr) {
    if (ptr == nullptr) return;
    
#ifdef HAVE_MIMALLOC
    mi_free(ptr);
#elif defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace aligned
} // namespace memory
} // namespace optimization_core








