/**
 * @file example_cache.cpp
 * @brief Practical example of using the KV cache module
 * 
 * This example demonstrates:
 * - Basic cache operations (put, get, remove)
 * - Different eviction strategies (LRU, LFU, FIFO)
 * - Concurrent cache for multi-threaded scenarios
 * - Performance monitoring and statistics
 */

#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <chrono>

#include "../include/optimization_core.hpp"

using namespace optimization_core;
using namespace optimization_core::memory;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

// ============================================================================
// Example 1: Basic Cache Operations
// ============================================================================

void example_basic_operations() {
    print_separator("Example 1: Basic Cache Operations");
    
    // Create cache with default config
    auto config = CacheConfig()
        .with_size(100)
        .with_strategy(EvictionStrategy::LRU);
    
    KVCache cache(config);
    
    std::cout << "Cache created with max size: " << cache.max_size() << "\n\n";
    
    // Store KV pairs for different layers
    std::cout << "Storing key-value pairs:\n";
    
    for (usize layer = 0; layer < 3; ++layer) {
        for (usize head = 0; head < 4; ++head) {
            std::vector<u8> key_data(64, static_cast<u8>(layer * 10 + head));
            std::vector<u8> value_data(128, static_cast<u8>(layer * 10 + head + 50));
            
            cache.put(layer, head, key_data, "key");
            cache.put(layer, head, value_data, "value");
        }
    }
    
    std::cout << "  - Stored " << cache.size() << " entries\n";
    std::cout << "  - Cache is " << (cache.empty() ? "empty" : "not empty") << "\n\n";
    
    // Retrieve data
    std::cout << "Retrieving data:\n";
    
    auto key = cache.get(0, 0, "key");
    auto value = cache.get(0, 0, "value");
    
    if (key && value) {
        std::cout << "  - Layer 0, Head 0:\n";
        std::cout << "    Key size: " << key->size() << " bytes\n";
        std::cout << "    Value size: " << value->size() << " bytes\n";
    }
    
    // Try to get non-existent entry
    auto missing = cache.get(99, 99, "key");
    std::cout << "  - Non-existent entry: " << (missing ? "found" : "not found") << "\n\n";
    
    // Statistics
    std::cout << "Hit rate: " << std::fixed << std::setprecision(2) 
              << (cache.hit_rate() * 100) << "%\n";
}

// ============================================================================
// Example 2: Eviction Strategies
// ============================================================================

void example_eviction_strategies() {
    print_separator("Example 2: Eviction Strategies");
    
    auto test_eviction = [](EvictionStrategy strategy, const std::string& name) {
        std::cout << name << " Eviction:\n";
        
        CacheConfig config;
        config.max_size = 3;
        config.eviction_strategy = strategy;
        config.eviction_threshold = 1.0f;  // Evict when full
        
        KVCache cache(config);
        
        // Fill cache
        cache.put(0, 0, {1});
        cache.put(0, 1, {2});
        cache.put(0, 2, {3});
        
        std::cout << "  Initial entries: ";
        for (usize i = 0; i < 3; ++i) {
            std::cout << (cache.get(0, i) ? "✓" : "✗");
        }
        std::cout << "\n";
        
        // Access pattern (for LRU/LFU)
        cache.get(0, 0);  // Access entry 0
        cache.get(0, 0);  // Access entry 0 again
        cache.get(0, 2);  // Access entry 2
        
        // Add new entry (triggers eviction)
        cache.put(0, 3, {4});
        
        std::cout << "  After adding entry 3: ";
        for (usize i = 0; i < 4; ++i) {
            auto result = cache.get(0, i);
            std::cout << (result ? "✓" : "✗");
        }
        std::cout << "\n";
        std::cout << "  Evictions: " << cache.stats().eviction_count.load() << "\n\n";
    };
    
    test_eviction(EvictionStrategy::LRU, "LRU (Least Recently Used)");
    test_eviction(EvictionStrategy::LFU, "LFU (Least Frequently Used)");
    test_eviction(EvictionStrategy::FIFO, "FIFO (First In First Out)");
}

// ============================================================================
// Example 3: LLM Inference Simulation
// ============================================================================

void example_llm_inference() {
    print_separator("Example 3: LLM Inference Simulation");
    
    // Simulate KV cache for a transformer with 12 layers
    const usize num_layers = 12;
    const usize num_heads = 8;
    const usize head_dim = 64;
    const usize max_seq_len = 2048;
    
    auto config = CacheConfig()
        .with_size(num_layers * num_heads * max_seq_len)
        .with_strategy(EvictionStrategy::Adaptive);
    
    KVCache cache(config);
    
    std::cout << "Simulating autoregressive generation:\n";
    std::cout << "  - Model: " << num_layers << " layers, " << num_heads << " heads\n";
    std::cout << "  - Head dim: " << head_dim << "\n";
    std::cout << "  - Max sequence: " << max_seq_len << " tokens\n\n";
    
    // Simulate generating 100 tokens
    auto start = std::chrono::high_resolution_clock::now();
    
    const usize gen_tokens = 100;
    for (usize token_idx = 0; token_idx < gen_tokens; ++token_idx) {
        for (usize layer = 0; layer < num_layers; ++layer) {
            for (usize head = 0; head < num_heads; ++head) {
                // Store KV for this token
                std::vector<u8> kv_data(head_dim * sizeof(f32), 0);
                cache.put(layer, token_idx * num_heads + head, kv_data, "kv");
                
                // Read previous KVs (simulate attention computation)
                for (usize prev = 0; prev < token_idx; ++prev) {
                    cache.get(layer, prev * num_heads + head, "kv");
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Results:\n";
    std::cout << "  - Tokens generated: " << gen_tokens << "\n";
    std::cout << "  - Time: " << duration.count() << " ms\n";
    std::cout << "  - Cache entries: " << cache.size() << "\n";
    std::cout << "  - Hit rate: " << std::fixed << std::setprecision(2) 
              << (cache.hit_rate() * 100) << "%\n";
    std::cout << "  - Hits: " << cache.stats().hit_count.load() << "\n";
    std::cout << "  - Misses: " << cache.stats().miss_count.load() << "\n";
}

// ============================================================================
// Example 4: Concurrent Cache
// ============================================================================

void example_concurrent_cache() {
    print_separator("Example 4: Concurrent Cache (Multi-threaded)");
    
    auto config = CacheConfig()
        .with_size(10000)
        .with_strategy(EvictionStrategy::LRU);
    
    ConcurrentKVCache cache(config);
    
    const int num_threads = 4;
    const int ops_per_thread = 1000;
    
    std::cout << "Running " << num_threads << " threads with " 
              << ops_per_thread << " operations each:\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&cache, t, ops_per_thread]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                usize key = t * ops_per_thread + i;
                
                // 70% writes, 30% reads
                if (i % 10 < 7) {
                    std::vector<u8> data = {
                        static_cast<u8>(t),
                        static_cast<u8>(i % 256)
                    };
                    cache.put(0, key, data);
                } else {
                    cache.get(0, key % (ops_per_thread * num_threads / 2));
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    i64 total_ops = static_cast<i64>(num_threads) * ops_per_thread;
    f64 ops_per_sec = total_ops * 1e6 / duration.count();
    
    std::cout << "Results:\n";
    std::cout << "  - Total operations: " << total_ops << "\n";
    std::cout << "  - Time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "  - Throughput: " << std::fixed << std::setprecision(0) 
              << ops_per_sec << " ops/sec\n";
    std::cout << "  - Final cache size: " << cache.size() << "\n";
    std::cout << "  - Hit rate: " << std::setprecision(2) 
              << (cache.hit_rate() * 100) << "%\n";
}

// ============================================================================
// Example 5: Cache Warming and Prefetching
// ============================================================================

void example_cache_warming() {
    print_separator("Example 5: Cache Warming");
    
    KVCache cache(CacheConfig().with_size(1000));
    
    std::cout << "Warming cache with frequently used data:\n\n";
    
    // Warm cache with common prompts/prefixes
    std::vector<std::string> common_prefixes = {
        "The quick brown fox",
        "Hello, how are you",
        "Please summarize",
        "What is the meaning"
    };
    
    std::cout << "Pre-loading " << common_prefixes.size() << " common prefixes:\n";
    for (usize i = 0; i < common_prefixes.size(); ++i) {
        const auto& prefix = common_prefixes[i];
        std::vector<u8> data(prefix.begin(), prefix.end());
        cache.put(0, i, data, "prompt");
        std::cout << "  " << (i + 1) << ". \"" << prefix << "...\"\n";
    }
    
    std::cout << "\nCache state after warming:\n";
    std::cout << "  - Entries: " << cache.size() << "\n";
    std::cout << "  - Ready for inference!\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║                      KV Cache Examples                         ║
║                     Optimization Core C++                      ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;

    example_basic_operations();
    example_eviction_strategies();
    example_llm_inference();
    example_concurrent_cache();
    example_cache_warming();
    
    print_separator("Summary");
    std::cout << "KV Cache features demonstrated:\n";
    std::cout << "  1. Basic CRUD operations (put, get, remove, clear)\n";
    std::cout << "  2. Multiple eviction strategies (LRU, LFU, FIFO, Adaptive)\n";
    std::cout << "  3. Thread-safe concurrent cache (ConcurrentKVCache)\n";
    std::cout << "  4. Performance statistics and monitoring\n";
    std::cout << "  5. Cache warming for common patterns\n";
    std::cout << "\nUse cases:\n";
    std::cout << "  • Autoregressive LLM generation\n";
    std::cout << "  • Batched inference with KV reuse\n";
    std::cout << "  • Multi-request serving with shared cache\n";
    std::cout << "\n";
    
    return 0;
}












