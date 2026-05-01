/**
 * @file test_compression.cpp
 * @brief Unit tests for compression module
 */

#include <gtest/gtest.h>
#include <numeric>
#include <random>

#include "../include/compression/compression.hpp"

using namespace optimization_core;
using namespace optimization_core::compression;

class CompressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_ = std::mt19937(42);
    }
    
    std::vector<u8> random_data(usize size) {
        std::vector<u8> data(size);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& b : data) b = static_cast<u8>(dist(gen_));
        return data;
    }
    
    std::vector<u8> compressible_data(usize size) {
        std::vector<u8> data(size);
        for (usize i = 0; i < size; ++i) {
            data[i] = static_cast<u8>(i % 10);  // Highly repetitive
        }
        return data;
    }
    
    std::mt19937 gen_;
};

// ============================================================================
// Algorithm Enum Tests
// ============================================================================

TEST_F(CompressionTest, AlgorithmToString) {
    EXPECT_EQ(to_string(Algorithm::None), "none");
    EXPECT_EQ(to_string(Algorithm::LZ4), "lz4");
    EXPECT_EQ(to_string(Algorithm::Zstd), "zstd");
}

TEST_F(CompressionTest, AlgorithmFromString) {
    EXPECT_EQ(from_string("lz4"), Algorithm::LZ4);
    EXPECT_EQ(from_string("LZ4"), Algorithm::LZ4);
    EXPECT_EQ(from_string("zstd"), Algorithm::Zstd);
    EXPECT_EQ(from_string("zstandard"), Algorithm::Zstd);
    EXPECT_EQ(from_string("none"), Algorithm::None);
    EXPECT_EQ(from_string("unknown"), Algorithm::None);
}

// ============================================================================
// Compression Stats Tests
// ============================================================================

TEST_F(CompressionTest, StatsCompressionRatio) {
    CompressionStats stats;
    stats.original_size = 1000;
    stats.compressed_size = 500;
    
    EXPECT_DOUBLE_EQ(stats.compression_ratio(), 0.5);
    EXPECT_DOUBLE_EQ(stats.space_savings(), 0.5);
    EXPECT_EQ(stats.bytes_saved(), 500);
}

TEST_F(CompressionTest, StatsThroughput) {
    CompressionStats stats;
    stats.original_size = 1'000'000;  // 1 MB
    stats.compression_time_us = 1'000;  // 1 ms
    
    f64 throughput = stats.compression_throughput_mbps();
    EXPECT_NEAR(throughput, 1000.0, 1.0);  // 1000 MB/s
}

TEST_F(CompressionTest, StatsZeroHandling) {
    CompressionStats stats;
    stats.original_size = 0;
    stats.compressed_size = 0;
    
    EXPECT_DOUBLE_EQ(stats.compression_ratio(), 0.0);
    EXPECT_DOUBLE_EQ(stats.compression_throughput_mbps(), 0.0);
}

// ============================================================================
// No Compression Tests
// ============================================================================

TEST_F(CompressionTest, NoCompressionRoundtrip) {
    std::vector<u8> data = {1, 2, 3, 4, 5};
    
    auto compressed = compress(data, Algorithm::None);
    ASSERT_TRUE(compressed.success);
    EXPECT_EQ(compressed.data, data);
    
    auto decompressed = decompress(compressed.data, Algorithm::None);
    ASSERT_TRUE(decompressed.success);
    EXPECT_EQ(decompressed.data, data);
}

// ============================================================================
// LZ4 Tests
// ============================================================================

TEST_F(CompressionTest, LZ4Roundtrip) {
    std::vector<u8> data = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!'};
    
    auto compressed = compress(data, Algorithm::LZ4);
    ASSERT_TRUE(compressed.success);
    
    auto decompressed = decompress(compressed.data, Algorithm::LZ4);
    ASSERT_TRUE(decompressed.success);
    EXPECT_EQ(decompressed.data, data);
}

TEST_F(CompressionTest, LZ4CompressibleData) {
    auto data = compressible_data(10000);
    
    auto compressed = compress(data, Algorithm::LZ4);
    ASSERT_TRUE(compressed.success);
    
    // Highly repetitive data should compress well
    EXPECT_LT(compressed.data.size(), data.size());
    EXPECT_LT(compressed.stats.compression_ratio(), 1.0);
    
    auto decompressed = decompress(compressed.data, Algorithm::LZ4);
    ASSERT_TRUE(decompressed.success);
    EXPECT_EQ(decompressed.data, data);
}

TEST_F(CompressionTest, LZ4LargeData) {
    auto data = random_data(1024 * 1024);  // 1 MB
    
    auto compressed = compress(data, Algorithm::LZ4);
    ASSERT_TRUE(compressed.success);
    EXPECT_GT(compressed.stats.compression_throughput_mbps(), 0.0);
    
    auto decompressed = decompress(compressed.data, Algorithm::LZ4);
    ASSERT_TRUE(decompressed.success);
    EXPECT_EQ(decompressed.data, data);
}

TEST_F(CompressionTest, LZ4EmptyData) {
    std::vector<u8> data;
    
    auto compressed = compress(data, Algorithm::LZ4);
    ASSERT_TRUE(compressed.success);
    
    auto decompressed = decompress(compressed.data, Algorithm::LZ4);
    ASSERT_TRUE(decompressed.success);
    EXPECT_TRUE(decompressed.data.empty());
}

// ============================================================================
// Zstd Tests
// ============================================================================

TEST_F(CompressionTest, ZstdRoundtrip) {
    std::vector<u8> data = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '!'};
    
    auto compressed = compress(data, Algorithm::Zstd);
    ASSERT_TRUE(compressed.success);
    
    auto decompressed = decompress(compressed.data, Algorithm::Zstd);
    ASSERT_TRUE(decompressed.success);
    EXPECT_EQ(decompressed.data, data);
}

TEST_F(CompressionTest, ZstdBetterRatio) {
    auto data = compressible_data(10000);
    
    auto lz4_result = compress(data, Algorithm::LZ4);
    auto zstd_result = compress(data, Algorithm::Zstd, 9);  // High level
    
    // Zstd at high level should have better ratio
    EXPECT_LE(zstd_result.data.size(), lz4_result.data.size());
}

TEST_F(CompressionTest, ZstdLevels) {
    auto data = compressible_data(10000);
    
    auto low_level = compress(data, Algorithm::Zstd, 1);
    auto high_level = compress(data, Algorithm::Zstd, 19);
    
    // Both should decompress correctly
    auto decompressed1 = decompress(low_level.data, Algorithm::Zstd);
    auto decompressed2 = decompress(high_level.data, Algorithm::Zstd);
    
    ASSERT_TRUE(decompressed1.success);
    ASSERT_TRUE(decompressed2.success);
    EXPECT_EQ(decompressed1.data, data);
    EXPECT_EQ(decompressed2.data, data);
    
    // Higher level should give better compression
    EXPECT_LE(high_level.data.size(), low_level.data.size());
}

// ============================================================================
// Compressor Class Tests
// ============================================================================

TEST_F(CompressionTest, CompressorBuilder) {
    auto compressor = Compressor::zstd(9);
    
    EXPECT_EQ(compressor.algorithm(), Algorithm::Zstd);
    EXPECT_EQ(compressor.level(), 9);
}

TEST_F(CompressionTest, CompressorWithLevel) {
    auto compressor = Compressor::lz4().with_level(5);
    
    EXPECT_EQ(compressor.level(), 5);
}

TEST_F(CompressionTest, CompressorLevelClamping) {
    auto compressor = Compressor::zstd(100);  // Over max
    compressor.with_level(100);
    
    EXPECT_LE(compressor.level(), 22);
}

TEST_F(CompressionTest, CompressorRoundtrip) {
    Compressor compressor(Algorithm::LZ4);
    
    std::vector<u8> data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    
    auto compressed = compressor.compress(data);
    ASSERT_TRUE(compressed.success);
    
    auto decompressed = compressor.decompress(compressed.data);
    ASSERT_TRUE(decompressed.success);
    EXPECT_EQ(decompressed.data, data);
}

// ============================================================================
// Streaming Compressor Tests
// ============================================================================

TEST_F(CompressionTest, StreamingCompressor) {
    StreamingCompressor streamer(Algorithm::LZ4, 100);
    
    // Write data in chunks
    auto chunks1 = streamer.write(std::vector<u8>(50, 1));
    EXPECT_EQ(chunks1.size(), 0);  // Not enough for a chunk
    
    auto chunks2 = streamer.write(std::vector<u8>(60, 2));
    EXPECT_EQ(chunks2.size(), 1);  // Should produce one chunk
    
    auto final_chunk = streamer.flush();
    EXPECT_TRUE(final_chunk.has_value());
    
    EXPECT_EQ(streamer.total_input(), 110);
    EXPECT_GT(streamer.total_output(), 0);
}

TEST_F(CompressionTest, StreamingCompressionRatio) {
    StreamingCompressor streamer(Algorithm::LZ4, 1024);
    
    // Write highly compressible data
    for (int i = 0; i < 10; ++i) {
        streamer.write(compressible_data(1024));
    }
    streamer.flush();
    
    f64 ratio = streamer.overall_ratio();
    EXPECT_LT(ratio, 1.0);  // Should compress
}

TEST_F(CompressionTest, StreamingEmpty) {
    StreamingCompressor streamer(Algorithm::LZ4, 100);
    
    auto final_chunk = streamer.flush();
    EXPECT_FALSE(final_chunk.has_value());
}

// ============================================================================
// Batch Compressor Tests
// ============================================================================

TEST_F(CompressionTest, BatchCompression) {
    BatchCompressor batch(Algorithm::LZ4);
    
    std::vector<std::vector<u8>> items;
    for (int i = 0; i < 10; ++i) {
        items.push_back(compressible_data(1000));
    }
    
    auto compressed = batch.compress_batch(items);
    
    EXPECT_EQ(compressed.size(), items.size());
    for (const auto& result : compressed) {
        EXPECT_TRUE(result.success);
        EXPECT_LT(result.data.size(), 1000);  // Should compress
    }
}

TEST_F(CompressionTest, BatchDecompression) {
    BatchCompressor batch(Algorithm::LZ4);
    
    // First compress
    std::vector<std::vector<u8>> original;
    for (int i = 0; i < 5; ++i) {
        original.push_back({static_cast<u8>(i), static_cast<u8>(i + 1)});
    }
    
    auto compressed = batch.compress_batch(original);
    
    // Extract just the data
    std::vector<std::vector<u8>> compressed_data;
    for (const auto& r : compressed) {
        compressed_data.push_back(r.data);
    }
    
    // Then decompress
    auto decompressed = batch.decompress_batch(compressed_data);
    
    EXPECT_EQ(decompressed.size(), original.size());
    for (usize i = 0; i < original.size(); ++i) {
        ASSERT_TRUE(decompressed[i].success);
        EXPECT_EQ(decompressed[i].data, original[i]);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(CompressionTest, SingleByte) {
    std::vector<u8> data = {42};
    
    for (Algorithm algo : {Algorithm::LZ4, Algorithm::Zstd}) {
        auto compressed = compress(data, algo);
        ASSERT_TRUE(compressed.success);
        
        auto decompressed = decompress(compressed.data, algo);
        ASSERT_TRUE(decompressed.success);
        EXPECT_EQ(decompressed.data, data);
    }
}

TEST_F(CompressionTest, AllSameBytes) {
    std::vector<u8> data(10000, 0x42);  // All same byte
    
    auto compressed = compress(data, Algorithm::LZ4);
    ASSERT_TRUE(compressed.success);
    
    // Should compress extremely well
    EXPECT_LT(compressed.stats.compression_ratio(), 0.1);
    
    auto decompressed = decompress(compressed.data, Algorithm::LZ4);
    ASSERT_TRUE(decompressed.success);
    EXPECT_EQ(decompressed.data, data);
}

TEST_F(CompressionTest, HighEntropy) {
    auto data = random_data(1000);  // Random = high entropy
    
    auto compressed = compress(data, Algorithm::LZ4);
    ASSERT_TRUE(compressed.success);
    
    // High entropy data doesn't compress well
    // Compressed size might even be larger due to header
    
    auto decompressed = decompress(compressed.data, Algorithm::LZ4);
    ASSERT_TRUE(decompressed.success);
    EXPECT_EQ(decompressed.data, data);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}












