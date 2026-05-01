#pragma once

/**
 * @file parallel_utils.hpp
 * @brief Parallel execution utilities for TBB and OpenMP
 * 
 * Provides unified interface for parallel loops that works with
 * both TBB and OpenMP, with fallback to sequential execution.
 */

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

namespace optimization_core {
namespace attention {
namespace parallel {

/**
 * @brief Execute function in parallel over range
 * 
 * @tparam Func Function type (callable)
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to execute for each index
 * 
 * @example
 *   parallel_for(0, n_heads, [&](int h) {
 *       // Process head h
 *   });
 */
template<typename Func>
void parallel_for(int start, int end, Func&& func) {
#ifdef HAVE_TBB
    tbb::parallel_for(start, end, std::forward<Func>(func));
#elif defined(HAVE_OPENMP)
    #pragma omp parallel for
    for (int i = start; i < end; ++i) {
        func(i);
    }
#else
    // Sequential fallback
    for (int i = start; i < end; ++i) {
        func(i);
    }
#endif
}

/**
 * @brief Execute function in parallel over blocked range (TBB-style)
 * 
 * @tparam Func Function type (callable)
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to execute for each range
 * 
 * @example
 *   parallel_for_blocked(0, scores.rows(), [&](int begin, int end) {
 *       for (int i = begin; i < end; ++i) {
 *           // Process row i
 *       }
 *   });
 */
template<typename Func>
void parallel_for_blocked(int start, int end, Func&& func) {
#ifdef HAVE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(start, end),
        [&](const tbb::blocked_range<int>& r) {
            func(r.begin(), r.end());
        });
#elif defined(HAVE_OPENMP)
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = (end - start + num_threads - 1) / num_threads;
        int thread_start = start + thread_id * chunk_size;
        int thread_end = std::min(thread_start + chunk_size, end);
        
        if (thread_start < end) {
            func(thread_start, thread_end);
        }
    }
#else
    // Sequential fallback
    func(start, end);
#endif
}

} // namespace parallel
} // namespace attention
} // namespace optimization_core








