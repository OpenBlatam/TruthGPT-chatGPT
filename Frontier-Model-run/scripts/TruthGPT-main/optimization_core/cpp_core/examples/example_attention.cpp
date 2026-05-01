/**
 * @file example_attention.cpp
 * @brief Practical example of using the attention module
 * 
 * This example demonstrates:
 * - Creating attention configurations
 * - Running scaled dot-product attention
 * - Using Flash Attention for memory efficiency
 * - Sparse attention patterns
 */

#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

#include "../include/optimization_core.hpp"

using namespace optimization_core;
using namespace optimization_core::attention;

// Helper to generate random tensor data
std::vector<f32> random_tensor(usize size, u32 seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    
    std::vector<f32> tensor(size);
    for (auto& v : tensor) v = dist(gen);
    return tensor;
}

// Benchmark helper
template<typename Func>
f64 benchmark_ms(Func&& fn, int iterations = 10) {
    // Warmup
    fn();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0 / iterations;
}

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";
}

// ============================================================================
// Example 1: Basic Attention
// ============================================================================

void example_basic_attention() {
    print_separator("Example 1: Basic Scaled Dot-Product Attention");
    
    // Configure attention
    auto config = AttentionConfig()
        .with_heads(8)
        .with_head_dim(64)
        .with_scale(0.0f);  // Auto-calculate scale
    
    std::cout << "Configuration:\n";
    std::cout << "  - Heads: " << config.num_heads << "\n";
    std::cout << "  - Head dim: " << config.head_dim << "\n";
    std::cout << "  - Model dim: " << config.d_model() << "\n";
    std::cout << "  - Scale: " << config.get_scale() << "\n";
    
    #ifdef HAVE_EIGEN
    ScaledDotProductAttention attn(config);
    
    usize batch = 4;
    usize seq_len = 128;
    usize dim = config.d_model();
    
    auto q = random_tensor(batch * seq_len * dim);
    auto k = random_tensor(batch * seq_len * dim);
    auto v = random_tensor(batch * seq_len * dim);
    
    std::cout << "\nInput shapes:\n";
    std::cout << "  - Q, K, V: [" << batch << ", " << seq_len << ", " << dim << "]\n";
    
    f64 time_ms = benchmark_ms([&]() {
        auto output = attn.forward(q, k, v, batch, seq_len, std::nullopt);
    });
    
    std::cout << "\nPerformance:\n";
    std::cout << "  - Time: " << std::fixed << std::setprecision(2) << time_ms << " ms\n";
    std::cout << "  - Throughput: " << (batch * seq_len / time_ms * 1000) << " tokens/sec\n";
    #else
    std::cout << "\n[Note: Eigen not available, skipping actual computation]\n";
    #endif
}

// ============================================================================
// Example 2: Flash Attention
// ============================================================================

void example_flash_attention() {
    print_separator("Example 2: Flash Attention (Memory Efficient)");
    
    auto config = AttentionConfig()
        .with_heads(12)
        .with_head_dim(64)
        .with_flash(64)     // Block size for tiling
        .with_causal();     // Causal masking for autoregressive models
    
    std::cout << "Configuration:\n";
    std::cout << "  - Flash Attention: " << (config.use_flash ? "enabled" : "disabled") << "\n";
    std::cout << "  - Block size: " << config.block_size << "\n";
    std::cout << "  - Causal mask: " << (config.use_causal_mask ? "yes" : "no") << "\n";
    
    #ifdef HAVE_EIGEN
    FlashAttention attn(config);
    
    usize batch = 2;
    usize seq_len = 512;
    usize dim = config.d_model();
    
    auto q = random_tensor(batch * seq_len * dim);
    auto k = random_tensor(batch * seq_len * dim);
    auto v = random_tensor(batch * seq_len * dim);
    
    std::cout << "\nInput shapes:\n";
    std::cout << "  - Q, K, V: [" << batch << ", " << seq_len << ", " << dim << "]\n";
    
    // Calculate memory statistics
    auto stats = AttentionStats::compute(batch, seq_len, config.num_heads, config.head_dim);
    std::cout << "\nMemory statistics:\n";
    std::cout << "  - Standard attention would use: " 
              << std::fixed << std::setprecision(2) << stats.memory_peak_mb << " MB\n";
    std::cout << "  - Flash attention uses: ~"
              << std::setprecision(2) 
              << (stats.memory_peak_mb * config.block_size * config.block_size) / (seq_len * seq_len) 
              << " MB\n";
    
    f64 time_ms = benchmark_ms([&]() {
        auto output = attn.forward(q, k, v, batch, seq_len, std::nullopt);
    });
    
    std::cout << "\nPerformance:\n";
    std::cout << "  - Time: " << time_ms << " ms\n";
    #else
    std::cout << "\n[Note: Eigen not available, skipping actual computation]\n";
    #endif
}

// ============================================================================
// Example 3: Sparse Attention
// ============================================================================

void example_sparse_attention() {
    print_separator("Example 3: Sparse Attention (Local + Global)");
    
    auto config = AttentionConfig()
        .with_heads(8)
        .with_head_dim(64);
    
    usize local_window = 64;    // Attend to 64 nearby tokens
    usize global_tokens = 8;    // Plus 8 global tokens (e.g., [CLS])
    
    std::cout << "Sparse pattern:\n";
    std::cout << "  - Local window: " << local_window << " tokens\n";
    std::cout << "  - Global tokens: " << global_tokens << "\n";
    
    #ifdef HAVE_EIGEN
    SparseAttention attn(config, local_window, global_tokens);
    
    usize batch = 2;
    usize seq_len = 1024;  // Long sequence
    usize dim = config.d_model();
    
    auto q = random_tensor(batch * seq_len * dim);
    auto k = random_tensor(batch * seq_len * dim);
    auto v = random_tensor(batch * seq_len * dim);
    
    std::cout << "\nComplexity comparison:\n";
    std::cout << "  - Full attention: O(" << seq_len << " × " << seq_len << ") = O(" 
              << (seq_len * seq_len) << ")\n";
    std::cout << "  - Sparse attention: O(" << seq_len << " × " << (local_window + global_tokens) 
              << ") = O(" << (seq_len * (local_window + global_tokens)) << ")\n";
    std::cout << "  - Speedup: ~" << std::fixed << std::setprecision(1) 
              << (f64)(seq_len) / (local_window + global_tokens) << "x\n";
    
    f64 time_ms = benchmark_ms([&]() {
        auto output = attn.forward(q, k, v, batch, seq_len, std::nullopt);
    });
    
    std::cout << "\nPerformance:\n";
    std::cout << "  - Time: " << time_ms << " ms\n";
    #else
    std::cout << "\n[Note: Eigen not available, skipping actual computation]\n";
    #endif
}

// ============================================================================
// Example 4: Attention Masks
// ============================================================================

void example_attention_masks() {
    print_separator("Example 4: Attention Masks");
    
    // Causal mask (for autoregressive generation)
    std::cout << "Causal mask (4x4):\n";
    auto causal = mask::create_causal(4);
    for (usize i = 0; i < 4; ++i) {
        std::cout << "  [";
        for (usize j = 0; j < 4; ++j) {
            f32 v = causal[i * 4 + j];
            if (v < -1e8f) {
                std::cout << " -inf";
            } else {
                std::cout << std::setw(5) << std::fixed << std::setprecision(1) << v;
            }
        }
        std::cout << " ]\n";
    }
    
    // Padding mask
    std::cout << "\nPadding mask (sequences of length 3, 2, 4 with max_len=5):\n";
    std::vector<usize> lengths = {3, 2, 4};
    auto padding = mask::create_padding(lengths, 5);
    
    for (usize batch = 0; batch < 3; ++batch) {
        std::cout << "  Batch " << batch << " (len=" << lengths[batch] << "): [";
        for (usize j = 0; j < 5; ++j) {
            f32 v = padding[batch * 5 + j];
            if (v < -1e8f) {
                std::cout << " MASK";
            } else {
                std::cout << "  ok ";
            }
        }
        std::cout << " ]\n";
    }
}

// ============================================================================
// Example 5: Factory Pattern
// ============================================================================

void example_factory_pattern() {
    print_separator("Example 5: Factory Pattern");
    
    std::cout << "Creating attention implementations via factory:\n\n";
    
    // Standard attention
    auto config_std = AttentionConfig()
        .with_heads(8)
        .with_head_dim(64);
    config_std.use_flash = false;
    
    auto std_attn = create_attention(config_std);
    std::cout << "1. Standard attention created: " << (std_attn != nullptr ? "✓" : "✗") << "\n";
    
    // Flash attention
    auto config_flash = AttentionConfig()
        .with_heads(8)
        .with_head_dim(64)
        .with_flash(64);
    
    auto flash_attn = create_attention(config_flash);
    std::cout << "2. Flash attention created: " << (flash_attn != nullptr ? "✓" : "✗") << "\n";
    
    std::cout << "\nBoth implement the same interface (IAttention).\n";
    std::cout << "Switch implementations without changing calling code!\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║                    Attention Module Examples                   ║
║                     Optimization Core C++                      ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;

    example_basic_attention();
    example_flash_attention();
    example_sparse_attention();
    example_attention_masks();
    example_factory_pattern();
    
    print_separator("Summary");
    std::cout << "Available attention implementations:\n";
    std::cout << "  1. ScaledDotProductAttention - Standard O(n²) attention\n";
    std::cout << "  2. FlashAttention - Memory-efficient tiled attention\n";
    std::cout << "  3. SparseAttention - Local + global attention patterns\n";
    std::cout << "\nKey features:\n";
    std::cout << "  • Builder pattern for configuration\n";
    std::cout << "  • Causal and padding mask support\n";
    std::cout << "  • Factory pattern for implementation selection\n";
    std::cout << "  • Memory and performance statistics\n";
    std::cout << "\n";
    
    return 0;
}












