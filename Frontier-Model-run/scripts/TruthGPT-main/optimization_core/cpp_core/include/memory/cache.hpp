#pragma once

/**
 * @file cache.hpp
 * @brief Refactored KV Cache with clean interfaces
 * 
 * Aligned with Rust implementation patterns:
 * - Builder pattern for configuration
 * - Strategy pattern for eviction
 * - RAII resource management
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/types.hpp"

namespace optimization_core {
namespace memory {

// ============================================================================
// Eviction Strategy
// ============================================================================

enum class EvictionStrategy {
    LRU,        // Least Recently Used
    LFU,        // Least Frequently Used
    FIFO,       // First In First Out
    Adaptive,   // Combines LRU + LFU
};

inline std::string to_string(EvictionStrategy s) {
    switch (s) {
        case EvictionStrategy::LRU: return "LRU";
        case EvictionStrategy::LFU: return "LFU";
        case EvictionStrategy::FIFO: return "FIFO";
        case EvictionStrategy::Adaptive: return "Adaptive";
        default: return "Unknown";
    }
}

// ============================================================================
// Cache Configuration
// ============================================================================

struct CacheConfig {
    usize max_size = 8192;
    EvictionStrategy eviction_strategy = EvictionStrategy::LRU;
    bool enable_compression = true;
    usize compression_threshold = 1024;
    usize num_shards = 16;
    f32 eviction_threshold = 0.8f;
    
    // Builder pattern
    CacheConfig& with_size(usize size) { max_size = size; return *this; }
    CacheConfig& with_strategy(EvictionStrategy s) { eviction_strategy = s; return *this; }
    CacheConfig& with_compression(bool enable, usize threshold = 1024) {
        enable_compression = enable;
        compression_threshold = threshold;
        return *this;
    }
    CacheConfig& with_shards(usize n) { num_shards = n; return *this; }
    
    void validate() const {
        if (max_size == 0) throw std::invalid_argument("max_size must be > 0");
        if (num_shards == 0) throw std::invalid_argument("num_shards must be > 0");
    }
};

// ============================================================================
// Cache Key
// ============================================================================

struct CacheKey {
    usize layer_idx;
    usize position;
    std::string tag;
    
    CacheKey(usize layer, usize pos, std::string t = "")
        : layer_idx(layer), position(pos), tag(std::move(t)) {}
    
    bool operator==(const CacheKey& other) const {
        return layer_idx == other.layer_idx 
            && position == other.position 
            && tag == other.tag;
    }
    
    std::string to_string() const {
        return std::to_string(layer_idx) + "_" + std::to_string(position) 
             + (tag.empty() ? "" : "_" + tag);
    }
};

struct CacheKeyHash {
    usize operator()(const CacheKey& k) const {
        usize h1 = std::hash<usize>{}(k.layer_idx);
        usize h2 = std::hash<usize>{}(k.position);
        usize h3 = std::hash<std::string>{}(k.tag);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// ============================================================================
// Cache Entry
// ============================================================================

struct CacheEntry {
    std::vector<u8> data;
    u64 access_count = 1;
    TimePoint last_access;
    TimePoint created_at;
    bool compressed = false;
    usize original_size = 0;
    
    CacheEntry() : last_access(Clock::now()), created_at(Clock::now()) {}
    
    explicit CacheEntry(std::vector<u8> d, bool comp = false, usize orig = 0)
        : data(std::move(d))
        , last_access(Clock::now())
        , created_at(Clock::now())
        , compressed(comp)
        , original_size(orig > 0 ? orig : data.size()) {}
    
    void touch() {
        ++access_count;
        last_access = Clock::now();
    }
    
    f64 age_seconds() const {
        return Duration(Clock::now() - last_access).count();
    }
    
    usize stored_size() const { return data.size(); }
};

// ============================================================================
// Cache Statistics
// ============================================================================

struct CacheStats {
    std::atomic<u64> hit_count{0};
    std::atomic<u64> miss_count{0};
    std::atomic<u64> eviction_count{0};
    std::atomic<usize> compression_savings{0};
    std::atomic<usize> total_stored{0};
    std::atomic<usize> total_original{0};
    std::atomic<usize> current_size{0};
    
    f64 hit_rate() const {
        u64 total = hit_count + miss_count;
        return total > 0 ? static_cast<f64>(hit_count) / total : 0.0;
    }
    
    f64 compression_ratio() const {
        return total_original > 0 
            ? static_cast<f64>(total_stored) / total_original : 1.0;
    }
    
    void reset() {
        hit_count = 0;
        miss_count = 0;
        eviction_count = 0;
        compression_savings = 0;
        total_stored = 0;
        total_original = 0;
        current_size = 0;
    }
};

// ============================================================================
// KV Cache Implementation
// ============================================================================

class KVCache {
public:
    explicit KVCache(CacheConfig config = {})
        : config_(std::move(config)) {
        config_.validate();
        entries_.reserve(config_.max_size);
    }
    
    /**
     * @brief Get entry from cache
     */
    std::optional<std::vector<u8>> get(usize layer_idx, usize position, 
                                        const std::string& tag = "") {
        CacheKey key(layer_idx, position, tag);
        
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.touch();
            frequency_[key]++;
            stats_.hit_count++;
            return it->second.data;
        }
        
        stats_.miss_count++;
        return std::nullopt;
    }
    
    /**
     * @brief Put entry in cache
     */
    void put(usize layer_idx, usize position, std::vector<u8> data,
             const std::string& tag = "") {
        CacheKey key(layer_idx, position, tag);
        usize original_size = data.size();
        bool compressed = false;
        
        // Compression (placeholder - would use LZ4/zstd)
        if (config_.enable_compression && data.size() > config_.compression_threshold) {
            // TODO: Implement actual compression
            compressed = false;
        }
        
        // Eviction check
        if (entries_.size() >= config_.max_size * config_.eviction_threshold) {
            evict();
        }
        
        // Remove old entry stats if exists
        auto existing = entries_.find(key);
        if (existing != entries_.end()) {
            stats_.total_stored -= existing->second.stored_size();
            stats_.total_original -= existing->second.original_size;
        }
        
        // Insert new entry
        usize stored_size = data.size();
        entries_[key] = CacheEntry(std::move(data), compressed, original_size);
        frequency_[key]++;
        
        // Update stats
        stats_.total_stored += stored_size;
        stats_.total_original += original_size;
        stats_.current_size = entries_.size();
        
        if (compressed) {
            stats_.compression_savings += (original_size - stored_size);
        }
    }
    
    /**
     * @brief Remove entry from cache
     */
    bool remove(usize layer_idx, usize position, const std::string& tag = "") {
        CacheKey key(layer_idx, position, tag);
        
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            stats_.total_stored -= it->second.stored_size();
            stats_.total_original -= it->second.original_size;
            entries_.erase(it);
            frequency_.erase(key);
            stats_.current_size = entries_.size();
            return true;
        }
        return false;
    }
    
    /**
     * @brief Clear all entries
     */
    void clear() {
        entries_.clear();
        frequency_.clear();
        stats_.reset();
    }
    
    // Accessors
    usize size() const { return entries_.size(); }
    usize max_size() const { return config_.max_size; }
    bool empty() const { return entries_.empty(); }
    bool full() const { return entries_.size() >= config_.max_size; }
    f64 hit_rate() const { return stats_.hit_rate(); }
    const CacheStats& stats() const { return stats_; }

private:
    CacheConfig config_;
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> entries_;
    std::unordered_map<CacheKey, u64, CacheKeyHash> frequency_;
    CacheStats stats_;
    
    void evict() {
        switch (config_.eviction_strategy) {
            case EvictionStrategy::LRU:
                evict_lru();
                break;
            case EvictionStrategy::LFU:
                evict_lfu();
                break;
            case EvictionStrategy::FIFO:
                evict_fifo();
                break;
            case EvictionStrategy::Adaptive:
                evict_adaptive();
                break;
        }
    }
    
    void evict_lru() {
        if (entries_.empty()) return;
        
        auto oldest = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.last_access < oldest->second.last_access) {
                oldest = it;
            }
        }
        
        remove_entry(oldest);
    }
    
    void evict_lfu() {
        if (frequency_.empty()) return;
        
        auto least_freq = frequency_.begin();
        for (auto it = frequency_.begin(); it != frequency_.end(); ++it) {
            if (it->second < least_freq->second) {
                least_freq = it;
            }
        }
        
        auto entry_it = entries_.find(least_freq->first);
        if (entry_it != entries_.end()) {
            remove_entry(entry_it);
        }
    }
    
    void evict_fifo() {
        if (entries_.empty()) return;
        
        auto oldest = entries_.begin();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.created_at < oldest->second.created_at) {
                oldest = it;
            }
        }
        
        remove_entry(oldest);
    }
    
    void evict_adaptive() {
        if (entries_.empty()) return;
        
        // Score = frequency / (age + 1)
        // Lower score = better eviction candidate
        
        auto best = entries_.begin();
        f64 best_score = std::numeric_limits<f64>::max();
        
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            f64 age = it->second.age_seconds();
            f64 freq = static_cast<f64>(it->second.access_count);
            f64 score = freq / (age + 1.0);
            
            if (score < best_score) {
                best_score = score;
                best = it;
            }
        }
        
        remove_entry(best);
    }
    
    template<typename Iter>
    void remove_entry(Iter it) {
        stats_.total_stored -= it->second.stored_size();
        stats_.total_original -= it->second.original_size;
        stats_.eviction_count++;
        frequency_.erase(it->first);
        entries_.erase(it);
        stats_.current_size = entries_.size();
    }
};

// ============================================================================
// Thread-Safe Cache
// ============================================================================

class ConcurrentKVCache {
public:
    explicit ConcurrentKVCache(CacheConfig config = {})
        : cache_(std::move(config)) {}
    
    std::optional<std::vector<u8>> get(usize layer_idx, usize position,
                                        const std::string& tag = "") {
        std::shared_lock lock(mutex_);
        return cache_.get(layer_idx, position, tag);
    }
    
    void put(usize layer_idx, usize position, std::vector<u8> data,
             const std::string& tag = "") {
        std::unique_lock lock(mutex_);
        cache_.put(layer_idx, position, std::move(data), tag);
    }
    
    bool remove(usize layer_idx, usize position, const std::string& tag = "") {
        std::unique_lock lock(mutex_);
        return cache_.remove(layer_idx, position, tag);
    }
    
    void clear() {
        std::unique_lock lock(mutex_);
        cache_.clear();
    }
    
    usize size() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }
    
    f64 hit_rate() const {
        std::shared_lock lock(mutex_);
        return cache_.hit_rate();
    }

private:
    mutable std::shared_mutex mutex_;
    KVCache cache_;
};

// ============================================================================
// Factory
// ============================================================================

inline std::unique_ptr<KVCache> create_cache(const CacheConfig& config = {}) {
    return std::make_unique<KVCache>(config);
}

inline std::unique_ptr<ConcurrentKVCache> create_concurrent_cache(
    const CacheConfig& config = {}
) {
    return std::make_unique<ConcurrentKVCache>(config);
}

} // namespace memory
} // namespace optimization_core












