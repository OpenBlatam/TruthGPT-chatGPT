/**
 * @file test_cache.cpp
 * @brief Unit tests for KV cache module
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "../include/optimization_core.hpp"

using namespace optimization_core;
using namespace optimization_core::memory;

class CacheTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(CacheTest, ConfigBuilder) {
    auto config = CacheConfig()
        .with_size(4096)
        .with_strategy(EvictionStrategy::Adaptive)
        .with_compression(true, 512);
    
    EXPECT_EQ(config.max_size, 4096);
    EXPECT_EQ(config.eviction_strategy, EvictionStrategy::Adaptive);
    EXPECT_TRUE(config.enable_compression);
    EXPECT_EQ(config.compression_threshold, 512);
}

TEST_F(CacheTest, ConfigValidation) {
    CacheConfig config;
    config.max_size = 0;
    
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST_F(CacheTest, PutAndGet) {
    KVCache cache(CacheConfig().with_size(100));
    
    std::vector<u8> data = {1, 2, 3, 4};
    cache.put(0, 0, data);
    
    auto result = cache.get(0, 0);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, data);
}

TEST_F(CacheTest, GetMissing) {
    KVCache cache(CacheConfig().with_size(100));
    
    auto result = cache.get(0, 0);
    
    EXPECT_FALSE(result.has_value());
}

TEST_F(CacheTest, Remove) {
    KVCache cache(CacheConfig().with_size(100));
    
    cache.put(0, 0, {1, 2, 3, 4});
    EXPECT_EQ(cache.size(), 1);
    
    bool removed = cache.remove(0, 0);
    
    EXPECT_TRUE(removed);
    EXPECT_EQ(cache.size(), 0);
    EXPECT_FALSE(cache.get(0, 0).has_value());
}

TEST_F(CacheTest, Clear) {
    KVCache cache(CacheConfig().with_size(100));
    
    cache.put(0, 0, {1, 2, 3, 4});
    cache.put(0, 1, {5, 6, 7, 8});
    EXPECT_EQ(cache.size(), 2);
    
    cache.clear();
    
    EXPECT_EQ(cache.size(), 0);
    EXPECT_TRUE(cache.empty());
}

TEST_F(CacheTest, WithTag) {
    KVCache cache(CacheConfig().with_size(100));
    
    cache.put(0, 0, {1, 2, 3}, "tag1");
    cache.put(0, 0, {4, 5, 6}, "tag2");
    
    auto result1 = cache.get(0, 0, "tag1");
    auto result2 = cache.get(0, 0, "tag2");
    
    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());
    EXPECT_NE(*result1, *result2);
}

// ============================================================================
// Eviction Tests
// ============================================================================

TEST_F(CacheTest, LRUEviction) {
    CacheConfig config;
    config.max_size = 3;
    config.eviction_strategy = EvictionStrategy::LRU;
    config.eviction_threshold = 1.0f;  // Evict when full
    
    KVCache cache(config);
    
    cache.put(0, 0, {1});
    cache.put(0, 1, {2});
    cache.put(0, 2, {3});
    
    // Access first entry to make it recent
    cache.get(0, 0);
    
    // Add new entry, should evict entry 1 (least recent)
    cache.put(0, 3, {4});
    
    EXPECT_TRUE(cache.get(0, 0).has_value());   // Was accessed
    EXPECT_FALSE(cache.get(0, 1).has_value());  // Evicted
    EXPECT_TRUE(cache.get(0, 2).has_value());
    EXPECT_TRUE(cache.get(0, 3).has_value());
}

TEST_F(CacheTest, EvictionStats) {
    CacheConfig config;
    config.max_size = 2;
    config.eviction_threshold = 1.0f;
    
    KVCache cache(config);
    
    cache.put(0, 0, {1});
    cache.put(0, 1, {2});
    cache.put(0, 2, {3});  // Triggers eviction
    
    EXPECT_EQ(cache.stats().eviction_count.load(), 1);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(CacheTest, HitRate) {
    KVCache cache(CacheConfig().with_size(100));
    
    cache.put(0, 0, {1, 2, 3, 4});
    
    // 2 hits
    cache.get(0, 0);
    cache.get(0, 0);
    
    // 1 miss
    cache.get(0, 1);
    
    f64 hit_rate = cache.hit_rate();
    EXPECT_NEAR(hit_rate, 2.0 / 3.0, 0.01);
}

TEST_F(CacheTest, SizeTracking) {
    KVCache cache(CacheConfig().with_size(100));
    
    EXPECT_EQ(cache.size(), 0);
    EXPECT_TRUE(cache.empty());
    
    cache.put(0, 0, {1, 2, 3, 4});
    EXPECT_EQ(cache.size(), 1);
    EXPECT_FALSE(cache.empty());
    
    cache.put(0, 1, {5, 6, 7, 8});
    EXPECT_EQ(cache.size(), 2);
}

// ============================================================================
// Concurrent Cache Tests
// ============================================================================

TEST_F(CacheTest, ConcurrentPutGet) {
    ConcurrentKVCache cache(CacheConfig().with_size(1000));
    
    const int num_threads = 4;
    const int ops_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&cache, t, ops_per_thread]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                usize key = t * ops_per_thread + i;
                std::vector<u8> data = {static_cast<u8>(key % 256)};
                cache.put(0, key, data);
                cache.get(0, key);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_GT(cache.size(), 0);
}

TEST_F(CacheTest, ConcurrentHitRate) {
    ConcurrentKVCache cache(CacheConfig().with_size(100));
    
    // Pre-populate
    for (int i = 0; i < 10; ++i) {
        cache.put(0, i, {static_cast<u8>(i)});
    }
    
    // Concurrent reads
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&cache]() {
            for (int i = 0; i < 100; ++i) {
                cache.get(0, i % 10);  // Mostly hits
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_GT(cache.hit_rate(), 0.5);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(CacheTest, EmptyData) {
    KVCache cache(CacheConfig().with_size(100));
    
    cache.put(0, 0, {});
    
    auto result = cache.get(0, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST_F(CacheTest, LargeData) {
    KVCache cache(CacheConfig().with_size(100));
    
    std::vector<u8> large_data(1024 * 1024, 42);  // 1MB
    cache.put(0, 0, large_data);
    
    auto result = cache.get(0, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), large_data.size());
}

TEST_F(CacheTest, UpdateExisting) {
    KVCache cache(CacheConfig().with_size(100));
    
    cache.put(0, 0, {1, 2, 3});
    cache.put(0, 0, {4, 5, 6});  // Update
    
    auto result = cache.get(0, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, std::vector<u8>({4, 5, 6}));
    EXPECT_EQ(cache.size(), 1);  // Still 1 entry
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST_F(CacheTest, CreateCache) {
    auto cache = create_cache(CacheConfig().with_size(100));
    EXPECT_NE(cache, nullptr);
    EXPECT_EQ(cache->max_size(), 100);
}

TEST_F(CacheTest, CreateConcurrentCache) {
    auto cache = create_concurrent_cache(CacheConfig().with_size(100));
    EXPECT_NE(cache, nullptr);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}












