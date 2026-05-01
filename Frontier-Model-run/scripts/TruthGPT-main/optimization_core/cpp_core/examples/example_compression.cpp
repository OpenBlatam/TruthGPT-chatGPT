/**
 * @file example_compression.cpp
 * @brief Practical example of using the compression module
 * 
 * This example demonstrates:
 * - LZ4 and Zstd compression
 * - Streaming compression
 * - Batch compression
 * - Compression for KV cache storage
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

#include "../include/compression/compression.hpp"

using namespace optimization_core;
using namespace optimization_core::compression;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

// Generate test data
std::vector<u8> generate_tensor_data(usize size, f32 sparsity = 0.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<f32> sparse_dist(0.0f, 1.0f);
    
    std::vector<u8> data(size);
    
    for (usize i = 0; i < size; i += sizeof(f32)) {
        f32 value = (sparse_dist(gen) < sparsity) ? 0.0f : dist(gen);
        std::memcpy(data.data() + i, &value, sizeof(f32));
    }
    
    return data;
}

std::vector<u8> generate_repetitive_data(usize size) {
    std::vector<u8> data(size);
    for (usize i = 0; i < size; ++i) {
        data[i] = static_cast<u8>(i % 16);  // Very repetitive
    }
    return data;
}

// ============================================================================
// Example 1: Basic Compression
// ============================================================================

void example_basic_compression() {
    print_separator("Example 1: Basic LZ4 and Zstd Compression");
    
    std::vector<u8> data = {
        'H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!',
        'H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!',
        'H', 'e', 'l', 'l', 'o', ',', ' ', 'W', 'o', 'r', 'l', 'd', '!',
    };
    
    std::cout << "Original data: \"Hello, World!\" (repeated 3x)\n";
    std::cout << "Original size: " << data.size() << " bytes\n\n";
    
    // LZ4 compression
    auto lz4_result = compress(data, Algorithm::LZ4);
    if (lz4_result.success) {
        std::cout << "LZ4 compression:\n";
        std::cout << "  Compressed size: " << lz4_result.stats.compressed_size << " bytes\n";
        std::cout << "  Ratio: " << std::fixed << std::setprecision(2) 
                  << (lz4_result.stats.compression_ratio() * 100) << "%\n";
        std::cout << "  Time: " << lz4_result.stats.compression_time_us << " μs\n";
        
        // Verify roundtrip
        auto decompressed = decompress(lz4_result.data, Algorithm::LZ4);
        std::cout << "  Roundtrip: " << (decompressed.data == data ? "✓" : "✗") << "\n\n";
    }
    
    // Zstd compression
    auto zstd_result = compress(data, Algorithm::Zstd);
    if (zstd_result.success) {
        std::cout << "Zstd compression:\n";
        std::cout << "  Compressed size: " << zstd_result.stats.compressed_size << " bytes\n";
        std::cout << "  Ratio: " << std::fixed << std::setprecision(2) 
                  << (zstd_result.stats.compression_ratio() * 100) << "%\n";
        std::cout << "  Time: " << zstd_result.stats.compression_time_us << " μs\n";
        
        auto decompressed = decompress(zstd_result.data, Algorithm::Zstd);
        std::cout << "  Roundtrip: " << (decompressed.data == data ? "✓" : "✗") << "\n";
    }
}

// ============================================================================
// Example 2: Compression Levels
// ============================================================================

void example_compression_levels() {
    print_separator("Example 2: Zstd Compression Levels");
    
    auto data = generate_repetitive_data(100000);  // 100KB
    
    std::cout << "Data: 100KB repetitive pattern\n\n";
    std::cout << std::setw(8) << "Level" << std::setw(12) << "Size" 
              << std::setw(12) << "Ratio" << std::setw(12) << "Time (μs)\n";
    std::cout << std::string(44, '-') << "\n";
    
    for (i32 level : {1, 3, 5, 9, 15, 19}) {
        auto result = compress(data, Algorithm::Zstd, level);
        if (result.success) {
            std::cout << std::setw(8) << level
                      << std::setw(12) << result.stats.compressed_size
                      << std::setw(11) << std::fixed << std::setprecision(1)
                      << (result.stats.compression_ratio() * 100) << "%"
                      << std::setw(12) << static_cast<i64>(result.stats.compression_time_us)
                      << "\n";
        }
    }
    
    std::cout << "\n→ Higher levels = better ratio but slower\n";
    std::cout << "→ Level 3 is usually a good balance\n";
}

// ============================================================================
// Example 3: Tensor Data Compression
// ============================================================================

void example_tensor_compression() {
    print_separator("Example 3: Tensor Data Compression");
    
    std::cout << "Compressing different tensor types:\n\n";
    
    struct TestCase {
        std::string name;
        std::vector<u8> data;
    };
    
    std::vector<TestCase> cases = {
        {"Dense tensor (random)", generate_tensor_data(10000 * sizeof(f32), 0.0f)},
        {"50% sparse tensor", generate_tensor_data(10000 * sizeof(f32), 0.5f)},
        {"90% sparse tensor", generate_tensor_data(10000 * sizeof(f32), 0.9f)},
        {"Repetitive pattern", generate_repetitive_data(10000 * sizeof(f32))},
    };
    
    std::cout << std::setw(22) << "Type" << std::setw(12) << "Original"
              << std::setw(10) << "LZ4" << std::setw(10) << "Zstd"
              << std::setw(12) << "Best Ratio\n";
    std::cout << std::string(66, '-') << "\n";
    
    for (const auto& tc : cases) {
        auto lz4 = compress(tc.data, Algorithm::LZ4);
        auto zstd = compress(tc.data, Algorithm::Zstd, 3);
        
        f64 best_ratio = std::min(lz4.stats.compression_ratio(), 
                                   zstd.stats.compression_ratio());
        
        std::cout << std::setw(22) << tc.name
                  << std::setw(12) << tc.data.size()
                  << std::setw(10) << lz4.stats.compressed_size
                  << std::setw(10) << zstd.stats.compressed_size
                  << std::setw(11) << std::fixed << std::setprecision(1)
                  << (best_ratio * 100) << "%\n";
    }
    
    std::cout << "\n→ Sparse data compresses much better!\n";
    std::cout << "→ Quantized models often have sparse weights\n";
}

// ============================================================================
// Example 4: Streaming Compression
// ============================================================================

void example_streaming() {
    print_separator("Example 4: Streaming Compression");
    
    std::cout << "Streaming compression for large data:\n\n";
    
    StreamingCompressor streamer(Algorithm::LZ4, 4096);  // 4KB chunks
    
    std::cout << "Processing 1MB in 100 chunks of 10KB each:\n";
    
    usize total_chunks = 0;
    std::vector<std::vector<u8>> compressed_chunks;
    
    for (int i = 0; i < 100; ++i) {
        auto chunk = generate_repetitive_data(10240);
        auto outputs = streamer.write(chunk);
        
        for (auto& output : outputs) {
            compressed_chunks.push_back(std::move(output));
            total_chunks++;
        }
        
        if ((i + 1) % 25 == 0) {
            std::cout << "  Processed " << (i + 1) * 10 << " KB, "
                      << "output chunks: " << total_chunks << "\n";
        }
    }
    
    // Flush remaining
    auto final = streamer.flush();
    if (final) {
        compressed_chunks.push_back(std::move(*final));
        total_chunks++;
    }
    
    std::cout << "\nResults:\n";
    std::cout << "  Total input: " << streamer.total_input() / 1024 << " KB\n";
    std::cout << "  Total output: " << streamer.total_output() / 1024 << " KB\n";
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(2)
              << (streamer.overall_ratio() * 100) << "%\n";
    std::cout << "  Output chunks: " << total_chunks << "\n";
}

// ============================================================================
// Example 5: Batch Compression
// ============================================================================

void example_batch() {
    print_separator("Example 5: Batch Compression (Parallel)");
    
    std::cout << "Compressing multiple KV cache entries in parallel:\n\n";
    
    const int num_entries = 100;
    const usize entry_size = 4096;  // 4KB per entry (typical KV size)
    
    // Generate batch
    std::vector<std::vector<u8>> entries;
    for (int i = 0; i < num_entries; ++i) {
        entries.push_back(generate_tensor_data(entry_size, 0.3f));
    }
    
    BatchCompressor batch(Algorithm::LZ4);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto compressed = batch.compress_batch(entries);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    usize total_original = num_entries * entry_size;
    usize total_compressed = 0;
    for (const auto& c : compressed) {
        total_compressed += c.data.size();
    }
    
    std::cout << "Batch size: " << num_entries << " entries × " 
              << entry_size / 1024 << " KB = " << total_original / 1024 << " KB\n\n";
    std::cout << "Results:\n";
    std::cout << "  Original: " << total_original / 1024 << " KB\n";
    std::cout << "  Compressed: " << total_compressed / 1024 << " KB\n";
    std::cout << "  Ratio: " << std::fixed << std::setprecision(1)
              << (100.0 * total_compressed / total_original) << "%\n";
    std::cout << "  Time: " << duration.count() / 1000.0 << " ms\n";
    std::cout << "  Throughput: " << std::setprecision(0)
              << (total_original / 1e6) / (duration.count() / 1e6) << " MB/s\n";
}

// ============================================================================
// Example 6: KV Cache Compression
// ============================================================================

void example_kv_cache_compression() {
    print_separator("Example 6: KV Cache Compression for LLM");
    
    std::cout << "Simulating KV cache compression for a transformer:\n\n";
    
    // Model parameters
    const usize num_layers = 32;
    const usize num_heads = 32;
    const usize head_dim = 128;
    const usize batch_size = 4;
    const usize seq_len = 2048;
    
    // KV size per token per layer
    usize kv_size = 2 * num_heads * head_dim * sizeof(f32);  // K + V
    usize total_kv = batch_size * seq_len * num_layers * kv_size;
    
    std::cout << "Model config:\n";
    std::cout << "  Layers: " << num_layers << "\n";
    std::cout << "  Heads: " << num_heads << "\n";
    std::cout << "  Head dim: " << head_dim << "\n";
    std::cout << "  Batch × Seq: " << batch_size << " × " << seq_len << "\n\n";
    
    std::cout << "Memory usage:\n";
    std::cout << "  KV per token per layer: " << kv_size / 1024.0 << " KB\n";
    std::cout << "  Total KV cache: " << total_kv / (1024.0 * 1024.0) << " MB\n\n";
    
    // Simulate compression for one layer
    std::vector<u8> layer_kv = generate_tensor_data(
        batch_size * seq_len * kv_size / num_layers, 0.2f
    );
    
    Compressor compressor = Compressor::lz4();
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = compressor.compress(layer_kv);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto compress_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    auto decompressed = compressor.decompress(result.data);
    end = std::chrono::high_resolution_clock::now();
    
    auto decompress_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    f64 ratio = result.stats.compression_ratio();
    f64 estimated_savings = (1.0 - ratio) * total_kv / (1024.0 * 1024.0);
    
    std::cout << "Compression results (per layer):\n";
    std::cout << "  Original: " << layer_kv.size() / 1024.0 << " KB\n";
    std::cout << "  Compressed: " << result.data.size() / 1024.0 << " KB\n";
    std::cout << "  Ratio: " << std::fixed << std::setprecision(1) 
              << (ratio * 100) << "%\n";
    std::cout << "  Compress time: " << compress_time.count() << " μs\n";
    std::cout << "  Decompress time: " << decompress_time.count() << " μs\n\n";
    
    std::cout << "Estimated total savings:\n";
    std::cout << "  Memory saved: " << std::setprecision(1) << estimated_savings << " MB\n";
    std::cout << "  New total: " << (total_kv * ratio) / (1024.0 * 1024.0) << " MB\n";
    std::cout << "\n→ Compression enables longer contexts with same memory!\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║                    Compression Module Examples                 ║
║                     Optimization Core C++                      ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;

    example_basic_compression();
    example_compression_levels();
    example_tensor_compression();
    example_streaming();
    example_batch();
    example_kv_cache_compression();
    
    print_separator("Summary");
    std::cout << "Compression algorithms:\n";
    std::cout << "  • LZ4: ~5 GB/s, 2-3x ratio (fastest)\n";
    std::cout << "  • Zstd: ~400 MB/s, 3-5x ratio (balanced)\n";
    std::cout << "\nUse cases:\n";
    std::cout << "  1. KV cache compression for longer contexts\n";
    std::cout << "  2. Model weight storage (quantized models)\n";
    std::cout << "  3. Checkpoint serialization\n";
    std::cout << "  4. Network transfer optimization\n";
    std::cout << "\nRecommendations:\n";
    std::cout << "  • Use LZ4 for real-time (inference)\n";
    std::cout << "  • Use Zstd level 3-5 for storage\n";
    std::cout << "  • Use Zstd level 15+ for archival\n";
    std::cout << "\n";
    
    return 0;
}












