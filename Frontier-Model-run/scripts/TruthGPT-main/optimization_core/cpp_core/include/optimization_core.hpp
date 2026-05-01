#pragma once

/**
 * @file optimization_core.hpp
 * @brief Unified header for optimization_core C++ library
 * 
 * This is the main include file for using the optimization_core C++ extensions.
 * 
 * ## Quick Start
 * 
 * ```cpp
 * #include <optimization_core.hpp>
 * 
 * using namespace optimization_core;
 * 
 * // Configure attention
 * attention::AttentionConfig config;
 * config.with_heads(12).with_head_dim(64).with_flash(128);
 * 
 * // Create attention module
 * auto attn = attention::create_attention(config);
 * 
 * // Create KV cache
 * memory::CacheConfig cache_config;
 * cache_config.with_size(8192).with_strategy(memory::EvictionStrategy::LRU);
 * auto cache = memory::create_cache(cache_config);
 * 
 * // Create inference engine
 * auto engine = inference::create_engine(42);
 * ```
 */

// Version info
#define OPTIMIZATION_CORE_VERSION_MAJOR 1
#define OPTIMIZATION_CORE_VERSION_MINOR 0
#define OPTIMIZATION_CORE_VERSION_PATCH 0
#define OPTIMIZATION_CORE_VERSION "1.0.0"

// Common types
#include "common/types.hpp"

// Attention module
#include "attention/attention.hpp"

// Memory management
#include "memory/cache.hpp"

// Inference engine
#include "inference/engine.hpp"

namespace optimization_core {

/**
 * @brief Library version information
 */
struct Version {
    static constexpr int major = OPTIMIZATION_CORE_VERSION_MAJOR;
    static constexpr int minor = OPTIMIZATION_CORE_VERSION_MINOR;
    static constexpr int patch = OPTIMIZATION_CORE_VERSION_PATCH;
    static constexpr const char* string = OPTIMIZATION_CORE_VERSION;
    
    static std::string full() {
        return std::string(string) + " (C++ extensions)";
    }
};

/**
 * @brief Get available backends
 */
inline std::vector<std::string> available_backends() {
    std::vector<std::string> backends;
    
    #ifdef HAVE_EIGEN
    backends.push_back("eigen");
    #endif
    
    #ifdef HAVE_CUDA
    backends.push_back("cuda");
    #endif
    
    #ifdef HAVE_CUTLASS
    backends.push_back("cutlass");
    #endif
    
    #ifdef HAVE_ONEDNN
    backends.push_back("onednn");
    #endif
    
    #ifdef HAVE_TBB
    backends.push_back("tbb");
    #endif
    
    #ifdef HAVE_MIMALLOC
    backends.push_back("mimalloc");
    #endif
    
    return backends;
}

/**
 * @brief Check if a backend is available
 */
inline bool has_backend(const std::string& name) {
    auto backends = available_backends();
    return std::find(backends.begin(), backends.end(), name) != backends.end();
}

} // namespace optimization_core












