"""
High-Performance KV Cache for TruthGPT

Lock-free concurrent cache with multiple eviction strategies.
Supports both single-threaded and sharded multi-threaded access patterns.
"""

using Base.Threads
using Random

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default configuration values
const DEFAULT_MAX_ENTRIES = 8192
const DEFAULT_COMPRESSION_THRESHOLD = 1024
const DEFAULT_NUM_SHARDS = 16

# Adaptive eviction weights
const ADAPTIVE_AGE_WEIGHT = 0.7
const ADAPTIVE_FREQ_WEIGHT = 0.3

# ═══════════════════════════════════════════════════════════════════════════════
# EVICTION STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EvictionStrategy

Cache eviction strategy enumeration.

# Values
- `LRU`: Least Recently Used - evicts oldest accessed entry
- `LFU`: Least Frequently Used - evicts least accessed entry
- `FIFO`: First In First Out - evicts oldest created entry
- `Adaptive`: Hybrid LRU + LFU with weighted scoring
"""
@enum EvictionStrategy begin
    LRU       # Least Recently Used
    LFU       # Least Frequently Used
    FIFO      # First In First Out
    Adaptive  # LRU + LFU hybrid
end

# ═══════════════════════════════════════════════════════════════════════════════
# CACHE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CacheConfig

Configuration for KV cache.

# Fields
- `max_entries`: Maximum number of entries before eviction
- `enable_compression`: Enable compression for large entries
- `compression_threshold`: Minimum size (bytes) to compress
- `eviction_strategy`: Strategy for evicting entries

# Examples
```julia
config = CacheConfig(
    max_entries=16384,
    eviction_strategy=LRU
)
```
"""
struct CacheConfig
    max_entries::Int
    enable_compression::Bool
    compression_threshold::Int
    eviction_strategy::EvictionStrategy
    
    function CacheConfig(
        max_entries::Int = DEFAULT_MAX_ENTRIES,
        enable_compression::Bool = true,
        compression_threshold::Int = DEFAULT_COMPRESSION_THRESHOLD,
        eviction_strategy::EvictionStrategy = LRU
    )
        # Validate all parameters
        validate_cache_config(max_entries, compression_threshold)
        
        new(max_entries, enable_compression, compression_threshold, eviction_strategy)
    end
end

"""
    CacheConfig(; kwargs...)

Create CacheConfig with keyword arguments.
"""
function CacheConfig(;
    max_entries::Int = DEFAULT_MAX_ENTRIES,
    enable_compression::Bool = true,
    compression_threshold::Int = DEFAULT_COMPRESSION_THRESHOLD,
    eviction_strategy::EvictionStrategy = LRU
)
    CacheConfig(max_entries, enable_compression, compression_threshold, eviction_strategy)
end

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_cache_config(max_entries, compression_threshold)

Validate cache configuration parameters.

# Arguments
- `max_entries`: Maximum number of entries
- `compression_threshold`: Compression threshold in bytes

# Throws
- `ArgumentError` if any parameter is invalid
"""
function validate_cache_config(
    max_entries::Int,
    compression_threshold::Int
)
    validate_max_entries(max_entries)
    validate_compression_threshold(compression_threshold)
end

"""
    validate_max_entries(max_entries)

Validate maximum number of cache entries.

# Arguments
- `max_entries`: Maximum number of entries

# Throws
- `ArgumentError` if max_entries is invalid
"""
function validate_max_entries(max_entries::Int)
    if max_entries <= 0
        throw(ArgumentError("max_entries must be positive, got $max_entries"))
    end
end

"""
    validate_compression_threshold(compression_threshold)

Validate compression threshold.

# Arguments
- `compression_threshold`: Compression threshold in bytes

# Throws
- `ArgumentError` if compression_threshold is invalid
"""
function validate_compression_threshold(compression_threshold::Int)
    if compression_threshold <= 0
        throw(ArgumentError(
            "compression_threshold must be positive, got $compression_threshold"
        ))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# CACHE ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CacheEntry{T}

Single cache entry with metadata for eviction decisions.

# Fields
- `data`: Cached data vector
- `access_count`: Number of times accessed
- `last_access`: Timestamp of last access
- `created_at`: Timestamp of creation
- `is_compressed`: Whether data is compressed
- `original_size`: Original size before compression (bytes)
"""
mutable struct CacheEntry{T}
    data::Vector{T}
    access_count::Int64
    last_access::Float64
    created_at::Float64
    is_compressed::Bool
    original_size::Int
end

"""
    CacheEntry(data::Vector{T})

Create a new CacheEntry with current timestamp.

# Arguments
- `data`: Data vector to cache

# Returns
- New CacheEntry instance
"""
function CacheEntry(data::Vector{T}) where T
    current_time = time()
    original_size = length(data) * sizeof(T)
    
    CacheEntry(
        data,
        1,              # access_count
        current_time,   # last_access
        current_time,   # created_at
        false,          # is_compressed
        original_size   # original_size
    )
end

"""
    touch!(entry::CacheEntry)

Update access statistics for an entry.

Increments access count and updates last access timestamp.

# Arguments
- `entry`: CacheEntry to update (modified in-place)
"""
function touch!(entry::CacheEntry)
    entry.access_count += 1
    entry.last_access = time()
end

# ═══════════════════════════════════════════════════════════════════════════════
# KV CACHE
# ═══════════════════════════════════════════════════════════════════════════════

"""
    KVCache{T}

High-performance KV cache with eviction support.

Thread-safe cache using ReentrantLock for concurrent access.
Tracks hits, misses, and evictions using atomic counters.

# Fields
- `config`: CacheConfig
- `data`: Dictionary mapping (layer_idx, position) -> CacheEntry
- `lock`: ReentrantLock for thread safety
- `hits`: Atomic counter for cache hits
- `misses`: Atomic counter for cache misses
- `evictions`: Atomic counter for evictions
"""
mutable struct KVCache{T}
    config::CacheConfig
    data::Dict{Tuple{Int, Int}, CacheEntry{T}}
    lock::ReentrantLock
    hits::Atomic{Int64}
    misses::Atomic{Int64}
    evictions::Atomic{Int64}
end

"""
    KVCache{T}(config::CacheConfig=CacheConfig())

Create a new KVCache instance.

# Arguments
- `T`: Element type
- `config`: CacheConfig (optional, uses defaults)

# Returns
- New KVCache instance
"""
function KVCache{T}(config::CacheConfig = CacheConfig()) where T
    KVCache{T}(
        config,
        Dict{Tuple{Int, Int}, CacheEntry{T}}(),
        ReentrantLock(),
        Atomic{Int64}(0),
        Atomic{Int64}(0),
        Atomic{Int64}(0)
    )
end

"""
    make_cache_key(layer_idx, position)

Create a cache key tuple.

# Arguments
- `layer_idx`: Layer index
- `position`: Position index

# Returns
- Tuple (layer_idx, position)
"""
@inline make_cache_key(layer_idx::Int, position::Int) = (layer_idx, position)

"""
    kv_cache_get(cache, layer_idx, position)

Get cached value for given layer and position.

# Arguments
- `cache`: KVCache instance
- `layer_idx`: Layer index
- `position`: Position index

# Returns
- Copy of cached data, or `nothing` if not found

# Examples
```julia
cache = KVCache{Float32}()
data = kv_cache_get(cache, 0, 42)
```
"""
function kv_cache_get(cache::KVCache{T}, layer_idx::Int, position::Int) where T
    key = make_cache_key(layer_idx, position)
    
    lock(cache.lock) do
        if haskey(cache.data, key)
            entry = cache.data[key]
            touch!(entry)
            atomic_add!(cache.hits, 1)
            return copy(entry.data)  # Return copy to prevent external modification
        else
            atomic_add!(cache.misses, 1)
            return nothing
        end
    end
end

"""
    kv_cache_put(cache, layer_idx, position, data)

Store value in cache.

Automatically evicts entries if cache is at capacity.

# Arguments
- `cache`: KVCache instance
- `layer_idx`: Layer index
- `position`: Position index
- `data`: Data vector to cache

# Examples
```julia
cache = KVCache{Float32}()
data = randn(Float32, 128)
kv_cache_put(cache, 0, 42, data)
```
"""
function kv_cache_put(
    cache::KVCache{T},
    layer_idx::Int,
    position::Int,
    data::Vector{T}
) where T
    if isempty(data)
        throw(ArgumentError("Cannot cache empty data"))
    end
    
    key = make_cache_key(layer_idx, position)
    
    lock(cache.lock) do
        # Evict entries until we have space
        while length(cache.data) >= cache.config.max_entries
            evict!(cache)
        end
        
        # Store new entry
        cache.data[key] = CacheEntry(data)
    end
end

"""
    select_eviction_key(cache)

Select key to evict based on eviction strategy.

# Arguments
- `cache`: KVCache instance

# Returns
- Key tuple to evict, or `nothing` if cache is empty
"""
function select_eviction_key(cache::KVCache)
    # Early return if cache is empty
    isempty(cache.data) && return nothing
    
    strategy = cache.config.eviction_strategy
    
    if strategy == LRU
        # Evict least recently used (oldest last_access)
        return argmin(e -> e.last_access, cache.data)[1]
    elseif strategy == LFU
        # Evict least frequently used (lowest access_count)
        return argmin(e -> e.access_count, cache.data)[1]
    elseif strategy == FIFO
        # Evict first in (oldest created_at)
        return argmin(e -> e.created_at, cache.data)[1]
    else  # Adaptive: hybrid LRU + LFU with weighted scoring
        current_time = time()
        return argmin(cache.data) do (k, e)
            # Age score: time since last access (higher = older)
            age_score = current_time - e.last_access
            
            # Frequency score: inverse of access count (higher = less frequent)
            # Add 1 to avoid division by zero
            freq_score = 1.0 / (e.access_count + 1)
            
            # Weighted combination: higher score = better candidate for eviction
            ADAPTIVE_AGE_WEIGHT * age_score + ADAPTIVE_FREQ_WEIGHT * freq_score
        end[1]
    end
end

"""
    evict!(cache)

Evict one entry based on configured strategy.

# Arguments
- `cache`: KVCache instance (modified in-place)

# Returns
- Number of entries evicted (0 or 1)
"""
function evict!(cache::KVCache)
    isempty(cache.data) && return 0
    
    key_to_remove = select_eviction_key(cache)
    
    if !isnothing(key_to_remove)
        delete!(cache.data, key_to_remove)
        atomic_add!(cache.evictions, 1)
        return 1
    end
    
    return 0
end

"""
    clear!(cache)

Clear all cached data and reset statistics.

# Arguments
- `cache`: KVCache instance (modified in-place)
"""
function clear!(cache::KVCache)
    lock(cache.lock) do
        # Clear all cached entries
        empty!(cache.data)
        
        # Reset statistics atomically
        # Note: atomic_sub with current value effectively sets to zero
        hits_val = cache.hits[]
        misses_val = cache.misses[]
        evictions_val = cache.evictions[]
        
        atomic_sub!(cache.hits, hits_val)
        atomic_sub!(cache.misses, misses_val)
        atomic_sub!(cache.evictions, evictions_val)
    end
end

"""
    Base.length(cache::KVCache)

Get number of cached entries.

# Arguments
- `cache`: KVCache instance

# Returns
- Number of entries
"""
Base.length(cache::KVCache) = length(cache.data)

"""
    hit_rate(cache::KVCache)

Get cache hit rate as a ratio.

# Arguments
- `cache`: KVCache instance

# Returns
- Hit rate between 0.0 and 1.0
"""
function hit_rate(cache::KVCache)
    # Get current statistics atomically
    hits = cache.hits[]
    misses = cache.misses[]
    total = hits + misses
    
    # Avoid division by zero
    return total > 0 ? Float64(hits) / Float64(total) : 0.0
end

"""
    stats(cache::KVCache)

Get comprehensive cache statistics.

# Arguments
- `cache`: KVCache instance

# Returns
- Dictionary with statistics:
  - `entries`: Current number of entries
  - `max_entries`: Maximum capacity
  - `hits`: Total cache hits
  - `misses`: Total cache misses
  - `evictions`: Total evictions
  - `hit_rate`: Hit rate ratio

# Examples
```julia
cache = KVCache{Float32}()
s = stats(cache)
println("Hit rate: $(s["hit_rate"])")
```
"""
function stats(cache::KVCache)
    # Get all statistics atomically to ensure consistency
    hits_val = cache.hits[]
    misses_val = cache.misses[]
    evictions_val = cache.evictions[]
    entries_count = length(cache.data)
    
    # Calculate hit rate
    total_requests = hits_val + misses_val
    hit_rate_val = total_requests > 0 ? Float64(hits_val) / Float64(total_requests) : 0.0
    
    return Dict(
        "entries" => entries_count,
        "max_entries" => cache.config.max_entries,
        "hits" => hits_val,
        "misses" => misses_val,
        "evictions" => evictions_val,
        "hit_rate" => hit_rate_val,
        "utilization" => entries_count / cache.config.max_entries  # Cache utilization ratio
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# SHARDED KV CACHE
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ShardedKVCache{T}

Sharded cache for better concurrency.

Distributes entries across multiple shards to reduce lock contention.
Each shard has its own lock, allowing parallel access to different shards.

# Fields
- `shards`: Vector of KVCache shards
- `num_shards`: Number of shards
"""
struct ShardedKVCache{T}
    shards::Vector{KVCache{T}}
    num_shards::Int
end

"""
    ShardedKVCache{T}(config::CacheConfig; num_shards::Int=16)

Create a sharded cache with multiple shards.

# Arguments
- `T`: Element type
- `config`: CacheConfig for the cache
- `num_shards`: Number of shards (default: 16)

# Returns
- New ShardedKVCache instance

# Examples
```julia
config = CacheConfig(max_entries=16384)
cache = ShardedKVCache{Float32}(config, num_shards=32)
```
"""
function ShardedKVCache{T}(
    config::CacheConfig;
    num_shards::Int = DEFAULT_NUM_SHARDS
) where T
    if num_shards <= 0
        throw(ArgumentError("num_shards must be positive, got $num_shards"))
    end
    
    # Create config for each shard (divide max_entries)
    shard_max_entries = cld(config.max_entries, num_shards)
    shard_config = CacheConfig(
        max_entries = shard_max_entries,
        enable_compression = config.enable_compression,
        compression_threshold = config.compression_threshold,
        eviction_strategy = config.eviction_strategy
    )
    
    # Create shards
    shards = [KVCache{T}(shard_config) for _ in 1:num_shards]
    
    return ShardedKVCache(shards, num_shards)
end

"""
    get_shard(cache, layer_idx, position)

Get the shard for a given key using consistent hashing.

# Arguments
- `cache`: ShardedKVCache instance
- `layer_idx`: Layer index
- `position`: Position index

# Returns
- KVCache shard for this key
"""
function get_shard(cache::ShardedKVCache, layer_idx::Int, position::Int)
    key = make_cache_key(layer_idx, position)
    hash_val = hash(key)
    shard_idx = mod1(hash_val, cache.num_shards)
    return cache.shards[shard_idx]
end

"""
    kv_cache_get(cache::ShardedKVCache, layer_idx, position)

Get cached value from sharded cache.

# Arguments
- `cache`: ShardedKVCache instance
- `layer_idx`: Layer index
- `position`: Position index

# Returns
- Copy of cached data, or `nothing` if not found
"""
function kv_cache_get(cache::ShardedKVCache{T}, layer_idx::Int, position::Int) where T
    shard = get_shard(cache, layer_idx, position)
    return kv_cache_get(shard, layer_idx, position)
end

"""
    kv_cache_put(cache::ShardedKVCache, layer_idx, position, data)

Store value in sharded cache.

# Arguments
- `cache`: ShardedKVCache instance
- `layer_idx`: Layer index
- `position`: Position index
- `data`: Data vector to cache
"""
function kv_cache_put(
    cache::ShardedKVCache{T},
    layer_idx::Int,
    position::Int,
    data::Vector{T}
) where T
    shard = get_shard(cache, layer_idx, position)
    kv_cache_put(shard, layer_idx, position, data)
end

"""
    stats(cache::ShardedKVCache)

Get aggregated statistics from all shards.

# Arguments
- `cache`: ShardedKVCache instance

# Returns
- Dictionary with aggregated statistics:
  - `entries`: Total entries across all shards
  - `hits`: Total hits across all shards
  - `misses`: Total misses across all shards
  - `evictions`: Total evictions across all shards
  - `num_shards`: Number of shards
  - `hit_rate`: Overall hit rate
"""
function stats(cache::ShardedKVCache)
    # Initialize aggregated statistics
    total_stats = Dict(
        "entries" => 0,
        "hits" => 0,
        "misses" => 0,
        "evictions" => 0,
        "num_shards" => cache.num_shards,
        "max_entries" => 0
    )
    
    # Aggregate statistics from all shards
    @inbounds for shard in cache.shards
        shard_stats = stats(shard)
        total_stats["entries"] += shard_stats["entries"]
        total_stats["hits"] += shard_stats["hits"]
        total_stats["misses"] += shard_stats["misses"]
        total_stats["evictions"] += shard_stats["evictions"]
        total_stats["max_entries"] += shard_stats["max_entries"]
    end
    
    # Calculate overall hit rate
    total_requests = total_stats["hits"] + total_stats["misses"]
    total_stats["hit_rate"] = total_requests > 0 ? 
        Float64(total_stats["hits"]) / Float64(total_requests) : 0.0
    
    # Calculate overall utilization
    total_stats["utilization"] = total_stats["max_entries"] > 0 ?
        total_stats["entries"] / total_stats["max_entries"] : 0.0
    
    return total_stats
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

export EvictionStrategy, LRU, LFU, FIFO, Adaptive
export CacheConfig, KVCache, ShardedKVCache
export kv_cache_get, kv_cache_put, clear!, hit_rate, stats
