#pragma once

/**
 * @file compression.hpp
 * @brief High-performance compression module
 * 
 * Provides ultra-fast compression/decompression using LZ4 and Zstd.
 * Optimized for tensor data and KV cache storage.
 * 
 * ## Performance
 * 
 * | Algorithm | Compress | Decompress | Ratio |
 * |-----------|----------|------------|-------|
 * | LZ4 | 5 GB/s | 8 GB/s | 2-3x |
 * | Zstd (level 3) | 400 MB/s | 1 GB/s | 3-5x |
 * | Zstd (level 9) | 100 MB/s | 1 GB/s | 4-6x |
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../common/types.hpp"

// Optional: LZ4 and Zstd integration
#ifdef HAVE_LZ4
#include <lz4.h>
#include <lz4hc.h>
#endif

#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

namespace optimization_core {
namespace compression {

// ============================================================================
// Algorithm Enum
// ============================================================================

enum class Algorithm {
    None,   // No compression
    LZ4,    // Fastest (~5GB/s)
    Zstd,   // Balanced speed/ratio
};

inline std::string to_string(Algorithm algo) {
    switch (algo) {
        case Algorithm::None: return "none";
        case Algorithm::LZ4: return "lz4";
        case Algorithm::Zstd: return "zstd";
        default: return "unknown";
    }
}

inline Algorithm from_string(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "lz4") return Algorithm::LZ4;
    if (lower == "zstd" || lower == "zstandard") return Algorithm::Zstd;
    return Algorithm::None;
}

// ============================================================================
// Compression Statistics
// ============================================================================

struct CompressionStats {
    usize original_size = 0;
    usize compressed_size = 0;
    f64 compression_time_us = 0.0;
    f64 decompression_time_us = 0.0;
    std::string algorithm;
    
    f64 compression_ratio() const {
        return original_size > 0 
            ? static_cast<f64>(compressed_size) / original_size : 0.0;
    }
    
    f64 space_savings() const {
        return 1.0 - compression_ratio();
    }
    
    usize bytes_saved() const {
        return original_size > compressed_size 
            ? original_size - compressed_size : 0;
    }
    
    f64 compression_throughput_mbps() const {
        return compression_time_us > 0 
            ? (original_size / 1e6) / (compression_time_us / 1e6) : 0.0;
    }
    
    f64 decompression_throughput_mbps() const {
        return decompression_time_us > 0 
            ? (original_size / 1e6) / (decompression_time_us / 1e6) : 0.0;
    }
};

// ============================================================================
// Compression Result
// ============================================================================

struct CompressionResult {
    std::vector<u8> data;
    CompressionStats stats;
    bool success = false;
    std::string error;
    
    static CompressionResult ok(std::vector<u8> d, CompressionStats s) {
        return {std::move(d), std::move(s), true, ""};
    }
    
    static CompressionResult err(std::string msg) {
        return {{}, {}, false, std::move(msg)};
    }
};

// ============================================================================
// LZ4 Implementation (Fallback if library not available)
// ============================================================================

namespace lz4 {

#ifdef HAVE_LZ4

inline CompressionResult compress(const std::vector<u8>& data, i32 level = 1) {
    Timer timer;
    
    // Calculate max compressed size
    i32 max_size = LZ4_compressBound(data.size());
    std::vector<u8> compressed(max_size + sizeof(u32));
    
    // Store original size in first 4 bytes
    u32 original_size = static_cast<u32>(data.size());
    std::memcpy(compressed.data(), &original_size, sizeof(u32));
    
    // Compress
    i32 compressed_size;
    if (level > 1) {
        compressed_size = LZ4_compress_HC(
            reinterpret_cast<const char*>(data.data()),
            reinterpret_cast<char*>(compressed.data() + sizeof(u32)),
            data.size(),
            max_size,
            level
        );
    } else {
        compressed_size = LZ4_compress_default(
            reinterpret_cast<const char*>(data.data()),
            reinterpret_cast<char*>(compressed.data() + sizeof(u32)),
            data.size(),
            max_size
        );
    }
    
    if (compressed_size <= 0) {
        return CompressionResult::err("LZ4 compression failed");
    }
    
    compressed.resize(compressed_size + sizeof(u32));
    
    CompressionStats stats;
    stats.original_size = data.size();
    stats.compressed_size = compressed.size();
    stats.compression_time_us = timer.elapsed_us();
    stats.algorithm = "lz4";
    
    return CompressionResult::ok(std::move(compressed), stats);
}

inline CompressionResult decompress(const std::vector<u8>& data) {
    if (data.size() < sizeof(u32)) {
        return CompressionResult::err("Invalid LZ4 data: too short");
    }
    
    Timer timer;
    
    // Read original size
    u32 original_size;
    std::memcpy(&original_size, data.data(), sizeof(u32));
    
    std::vector<u8> decompressed(original_size);
    
    i32 result = LZ4_decompress_safe(
        reinterpret_cast<const char*>(data.data() + sizeof(u32)),
        reinterpret_cast<char*>(decompressed.data()),
        data.size() - sizeof(u32),
        original_size
    );
    
    if (result < 0) {
        return CompressionResult::err("LZ4 decompression failed");
    }
    
    CompressionStats stats;
    stats.original_size = original_size;
    stats.compressed_size = data.size();
    stats.decompression_time_us = timer.elapsed_us();
    stats.algorithm = "lz4";
    
    return CompressionResult::ok(std::move(decompressed), stats);
}

#else

// Fallback: Simple RLE-like compression
inline CompressionResult compress(const std::vector<u8>& data, i32 level = 1) {
    (void)level;
    Timer timer;
    
    // Simple copy with size prefix (no actual compression without library)
    std::vector<u8> result(data.size() + sizeof(u32));
    u32 size = static_cast<u32>(data.size());
    std::memcpy(result.data(), &size, sizeof(u32));
    std::memcpy(result.data() + sizeof(u32), data.data(), data.size());
    
    CompressionStats stats;
    stats.original_size = data.size();
    stats.compressed_size = result.size();
    stats.compression_time_us = timer.elapsed_us();
    stats.algorithm = "none (lz4 unavailable)";
    
    return CompressionResult::ok(std::move(result), stats);
}

inline CompressionResult decompress(const std::vector<u8>& data) {
    if (data.size() < sizeof(u32)) {
        return CompressionResult::err("Invalid data");
    }
    
    Timer timer;
    
    u32 size;
    std::memcpy(&size, data.data(), sizeof(u32));
    
    std::vector<u8> result(data.begin() + sizeof(u32), data.end());
    
    CompressionStats stats;
    stats.original_size = size;
    stats.compressed_size = data.size();
    stats.decompression_time_us = timer.elapsed_us();
    stats.algorithm = "none";
    
    return CompressionResult::ok(std::move(result), stats);
}

#endif // HAVE_LZ4

} // namespace lz4

// ============================================================================
// Zstd Implementation
// ============================================================================

namespace zstd {

#ifdef HAVE_ZSTD

inline CompressionResult compress(const std::vector<u8>& data, i32 level = 3) {
    Timer timer;
    
    usize max_size = ZSTD_compressBound(data.size());
    std::vector<u8> compressed(max_size);
    
    usize compressed_size = ZSTD_compress(
        compressed.data(),
        max_size,
        data.data(),
        data.size(),
        level
    );
    
    if (ZSTD_isError(compressed_size)) {
        return CompressionResult::err(
            std::string("Zstd compression failed: ") + ZSTD_getErrorName(compressed_size)
        );
    }
    
    compressed.resize(compressed_size);
    
    CompressionStats stats;
    stats.original_size = data.size();
    stats.compressed_size = compressed_size;
    stats.compression_time_us = timer.elapsed_us();
    stats.algorithm = "zstd";
    
    return CompressionResult::ok(std::move(compressed), stats);
}

inline CompressionResult decompress(const std::vector<u8>& data) {
    Timer timer;
    
    usize original_size = ZSTD_getFrameContentSize(data.data(), data.size());
    
    if (original_size == ZSTD_CONTENTSIZE_ERROR) {
        return CompressionResult::err("Invalid Zstd frame");
    }
    if (original_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        return CompressionResult::err("Unknown Zstd content size");
    }
    
    std::vector<u8> decompressed(original_size);
    
    usize result = ZSTD_decompress(
        decompressed.data(),
        original_size,
        data.data(),
        data.size()
    );
    
    if (ZSTD_isError(result)) {
        return CompressionResult::err(
            std::string("Zstd decompression failed: ") + ZSTD_getErrorName(result)
        );
    }
    
    CompressionStats stats;
    stats.original_size = original_size;
    stats.compressed_size = data.size();
    stats.decompression_time_us = timer.elapsed_us();
    stats.algorithm = "zstd";
    
    return CompressionResult::ok(std::move(decompressed), stats);
}

#else

// Fallback
inline CompressionResult compress(const std::vector<u8>& data, i32 level = 3) {
    return lz4::compress(data, level);
}

inline CompressionResult decompress(const std::vector<u8>& data) {
    return lz4::decompress(data);
}

#endif // HAVE_ZSTD

} // namespace zstd

// ============================================================================
// Unified Interface
// ============================================================================

inline CompressionResult compress(const std::vector<u8>& data, 
                                   Algorithm algo, 
                                   i32 level = 3) {
    switch (algo) {
        case Algorithm::None: {
            CompressionStats stats;
            stats.original_size = data.size();
            stats.compressed_size = data.size();
            stats.algorithm = "none";
            return CompressionResult::ok(data, stats);
        }
        case Algorithm::LZ4:
            return lz4::compress(data, level);
        case Algorithm::Zstd:
            return zstd::compress(data, level);
        default:
            return CompressionResult::err("Unknown algorithm");
    }
}

inline CompressionResult decompress(const std::vector<u8>& data, Algorithm algo) {
    switch (algo) {
        case Algorithm::None: {
            CompressionStats stats;
            stats.original_size = data.size();
            stats.compressed_size = data.size();
            stats.algorithm = "none";
            return CompressionResult::ok(data, stats);
        }
        case Algorithm::LZ4:
            return lz4::decompress(data);
        case Algorithm::Zstd:
            return zstd::decompress(data);
        default:
            return CompressionResult::err("Unknown algorithm");
    }
}

// ============================================================================
// Compressor Class
// ============================================================================

class Compressor {
public:
    explicit Compressor(Algorithm algo = Algorithm::LZ4, i32 level = 3)
        : algorithm_(algo), level_(level) {}
    
    // Builder pattern
    static Compressor lz4() { return Compressor(Algorithm::LZ4, 1); }
    static Compressor zstd(i32 level = 3) { return Compressor(Algorithm::Zstd, level); }
    
    Compressor& with_level(i32 level) {
        level_ = std::clamp(level, 1, 22);
        return *this;
    }
    
    CompressionResult compress(const std::vector<u8>& data) const {
        return compression::compress(data, algorithm_, level_);
    }
    
    CompressionResult decompress(const std::vector<u8>& data) const {
        return compression::decompress(data, algorithm_);
    }
    
    Algorithm algorithm() const { return algorithm_; }
    i32 level() const { return level_; }

private:
    Algorithm algorithm_;
    i32 level_;
};

// ============================================================================
// Streaming Compressor
// ============================================================================

class StreamingCompressor {
public:
    StreamingCompressor(Algorithm algo, usize chunk_size = 65536)
        : compressor_(algo), chunk_size_(chunk_size) {
        buffer_.reserve(chunk_size);
    }
    
    std::vector<std::vector<u8>> write(const std::vector<u8>& data) {
        buffer_.insert(buffer_.end(), data.begin(), data.end());
        total_input_ += data.size();
        
        std::vector<std::vector<u8>> chunks;
        
        while (buffer_.size() >= chunk_size_) {
            std::vector<u8> chunk(buffer_.begin(), buffer_.begin() + chunk_size_);
            buffer_.erase(buffer_.begin(), buffer_.begin() + chunk_size_);
            
            auto result = compressor_.compress(chunk);
            if (result.success) {
                total_output_ += result.data.size();
                chunks.push_back(std::move(result.data));
            }
        }
        
        return chunks;
    }
    
    std::optional<std::vector<u8>> flush() {
        if (buffer_.empty()) {
            return std::nullopt;
        }
        
        auto result = compressor_.compress(buffer_);
        buffer_.clear();
        
        if (result.success) {
            total_output_ += result.data.size();
            return result.data;
        }
        
        return std::nullopt;
    }
    
    usize total_input() const { return total_input_; }
    usize total_output() const { return total_output_; }
    
    f64 overall_ratio() const {
        return total_input_ > 0 
            ? static_cast<f64>(total_output_) / total_input_ : 0.0;
    }

private:
    Compressor compressor_;
    usize chunk_size_;
    std::vector<u8> buffer_;
    usize total_input_ = 0;
    usize total_output_ = 0;
};

// ============================================================================
// Batch Compressor (Parallel)
// ============================================================================

class BatchCompressor {
public:
    BatchCompressor(Algorithm algo, i32 level = 3)
        : compressor_(algo, level) {}
    
    std::vector<CompressionResult> compress_batch(
        const std::vector<std::vector<u8>>& items
    ) {
        std::vector<CompressionResult> results(items.size());
        
        #ifdef HAVE_TBB
        #include <tbb/parallel_for.h>
        tbb::parallel_for(usize(0), items.size(), [&](usize i) {
            results[i] = compressor_.compress(items[i]);
        });
        #else
        #pragma omp parallel for
        for (usize i = 0; i < items.size(); ++i) {
            results[i] = compressor_.compress(items[i]);
        }
        #endif
        
        return results;
    }
    
    std::vector<CompressionResult> decompress_batch(
        const std::vector<std::vector<u8>>& items
    ) {
        std::vector<CompressionResult> results(items.size());
        
        #pragma omp parallel for
        for (usize i = 0; i < items.size(); ++i) {
            results[i] = compressor_.decompress(items[i]);
        }
        
        return results;
    }

private:
    Compressor compressor_;
};

} // namespace compression
} // namespace optimization_core












