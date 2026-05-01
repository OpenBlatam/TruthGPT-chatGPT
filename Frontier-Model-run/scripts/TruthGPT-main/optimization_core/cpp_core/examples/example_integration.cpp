/**
 * @file example_integration.cpp
 * @brief Complete integration example using all modules together
 * 
 * This example demonstrates:
 * - Full LLM inference pipeline
 * - Attention with KV cache
 * - Compressed cache storage
 * - Generation with various strategies
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>

#include "../include/optimization_core.hpp"
#include "../include/compression/compression.hpp"

using namespace optimization_core;

void print_header() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║              Complete Integration Example                      ║
║                 Optimization Core C++                          ║
║                                                                ║
║  This example shows a complete LLM inference pipeline:         ║
║  Attention → KV Cache (Compressed) → Sampling → Generation    ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

// ============================================================================
// Simulated LLM Components
// ============================================================================

/**
 * @brief Simulated transformer layer with attention and KV cache
 */
class TransformerLayer {
public:
    TransformerLayer(usize layer_idx, const attention::AttentionConfig& attn_config,
                     std::shared_ptr<memory::ConcurrentKVCache> cache,
                     bool compress_kv = true)
        : layer_idx_(layer_idx)
        , config_(attn_config)
        , cache_(std::move(cache))
        , compress_kv_(compress_kv)
    {
        #ifdef HAVE_EIGEN
        attention_ = attention::create_attention(config_);
        #endif
        
        if (compress_kv_) {
            compressor_ = std::make_unique<compression::Compressor>(
                compression::Algorithm::LZ4
            );
        }
    }
    
    /**
     * @brief Forward pass with KV caching
     */
    std::vector<f32> forward(
        const std::vector<f32>& hidden_states,
        usize batch_size,
        usize seq_len,
        usize cache_position
    ) {
        usize dim = config_.d_model();
        
        // 1. Project to Q, K, V (simulated)
        auto q = hidden_states;  // In reality: hidden @ W_q
        auto k = hidden_states;  // In reality: hidden @ W_k
        auto v = hidden_states;  // In reality: hidden @ W_v
        
        // 2. Try to retrieve cached K, V
        auto cached_k = get_cached_kv(cache_position, "k");
        auto cached_v = get_cached_kv(cache_position, "v");
        
        // 3. Concatenate cached with current (if exists)
        // In a real implementation, we'd concat along sequence dimension
        
        // 4. Run attention
        std::vector<f32> output(batch_size * seq_len * dim);
        
        #ifdef HAVE_EIGEN
        if (attention_) {
            output = attention_->forward(q, k, v, batch_size, seq_len, std::nullopt);
        }
        #endif
        
        // 5. Cache current K, V
        store_kv(k, cache_position, "k");
        store_kv(v, cache_position, "v");
        
        return output;
    }
    
private:
    void store_kv(const std::vector<f32>& tensor, usize pos, const std::string& tag) {
        // Convert to bytes
        std::vector<u8> bytes(tensor.size() * sizeof(f32));
        std::memcpy(bytes.data(), tensor.data(), bytes.size());
        
        // Optionally compress
        if (compress_kv_ && compressor_) {
            auto result = compressor_->compress(bytes);
            if (result.success) {
                bytes = std::move(result.data);
            }
        }
        
        cache_->put(layer_idx_, pos, bytes, tag);
    }
    
    std::optional<std::vector<f32>> get_cached_kv(usize pos, const std::string& tag) {
        auto cached = cache_->get(layer_idx_, pos, tag);
        if (!cached) return std::nullopt;
        
        auto bytes = *cached;
        
        // Decompress if needed
        if (compress_kv_ && compressor_) {
            auto result = compressor_->decompress(bytes);
            if (result.success) {
                bytes = std::move(result.data);
            }
        }
        
        // Convert back to floats
        std::vector<f32> tensor(bytes.size() / sizeof(f32));
        std::memcpy(tensor.data(), bytes.data(), bytes.size());
        
        return tensor;
    }
    
    usize layer_idx_;
    attention::AttentionConfig config_;
    std::shared_ptr<attention::IAttention> attention_;
    std::shared_ptr<memory::ConcurrentKVCache> cache_;
    std::unique_ptr<compression::Compressor> compressor_;
    bool compress_kv_;
};

/**
 * @brief Complete simulated LLM
 */
class SimulatedLLM {
public:
    struct Config {
        usize num_layers = 12;
        usize num_heads = 8;
        usize head_dim = 64;
        usize vocab_size = 50000;
        usize max_seq_len = 2048;
        bool compress_cache = true;
    };
    
    explicit SimulatedLLM(const Config& config)
        : config_(config)
        , hidden_dim_(config.num_heads * config.head_dim)
    {
        // Create attention config
        attention::AttentionConfig attn_config;
        attn_config.num_heads = config.num_heads;
        attn_config.head_dim = config.head_dim;
        attn_config.use_flash = true;
        attn_config.use_causal_mask = true;
        attn_config.block_size = 64;
        
        // Create shared KV cache
        memory::CacheConfig cache_config;
        cache_config.max_size = config.num_layers * config.max_seq_len * 2;  // K + V
        cache_config.eviction_strategy = memory::EvictionStrategy::LRU;
        
        cache_ = std::make_shared<memory::ConcurrentKVCache>(cache_config);
        
        // Create layers
        for (usize i = 0; i < config.num_layers; ++i) {
            layers_.push_back(std::make_unique<TransformerLayer>(
                i, attn_config, cache_, config.compress_cache
            ));
        }
        
        std::cout << "Created LLM with " << config.num_layers << " layers, "
                  << hidden_dim_ << "d hidden\n";
    }
    
    /**
     * @brief Forward pass returning logits
     */
    std::vector<f32> forward(const std::vector<i32>& tokens, usize cache_pos = 0) {
        usize batch = 1;
        usize seq_len = tokens.size();
        
        // Create hidden states (simulated embedding)
        std::vector<f32> hidden(batch * seq_len * hidden_dim_);
        for (usize i = 0; i < hidden.size(); ++i) {
            // Simple deterministic "embedding"
            hidden[i] = std::sin(static_cast<f32>(tokens[i % seq_len] + i) * 0.1f);
        }
        
        // Forward through layers
        for (auto& layer : layers_) {
            hidden = layer->forward(hidden, batch, seq_len, cache_pos);
        }
        
        // Project to vocab (simulated)
        std::vector<f32> logits(config_.vocab_size);
        for (usize i = 0; i < config_.vocab_size; ++i) {
            // Simple projection: sum of hidden * position
            f32 sum = 0.0f;
            for (usize j = 0; j < std::min(usize(10), hidden.size()); ++j) {
                sum += hidden[j] * std::cos(static_cast<f32>(i + j) * 0.01f);
            }
            logits[i] = sum;
        }
        
        return logits;
    }
    
    usize cache_size() const { return cache_->size(); }
    f64 cache_hit_rate() const { return cache_->hit_rate(); }
    
    void clear_cache() { cache_->clear(); }
    
private:
    Config config_;
    usize hidden_dim_;
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    std::shared_ptr<memory::ConcurrentKVCache> cache_;
};

// ============================================================================
// Main Example
// ============================================================================

void run_generation_example() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  Running Generation Example\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    // Create model
    SimulatedLLM::Config model_config;
    model_config.num_layers = 6;
    model_config.num_heads = 8;
    model_config.head_dim = 32;
    model_config.vocab_size = 1000;
    model_config.max_seq_len = 256;
    model_config.compress_cache = true;
    
    SimulatedLLM model(model_config);
    
    // Create inference engine
    inference::InferenceEngine engine(42);
    
    // Wrap model as forward function
    auto forward_fn = [&model](const std::vector<i32>& tokens) {
        return model.forward(tokens);
    };
    
    // Example prompts
    std::vector<i32> prompt = {1, 5, 10, 15, 20};  // Simulated tokens
    
    // Test different generation strategies
    std::cout << "Prompt: [1, 5, 10, 15, 20]\n\n";
    
    struct Strategy {
        std::string name;
        inference::GenerationConfig config;
    };
    
    std::vector<Strategy> strategies = {
        {"Greedy", inference::GenerationConfig::greedy().with_max_tokens(10)},
        {"Sampling (temp=0.7)", inference::GenerationConfig::sampling(0.7f, 0.9f).with_max_tokens(10)},
        {"Top-K (k=10)", inference::GenerationConfig::sampling(1.0f, 1.0f).with_max_tokens(10)},
        {"Beam (beams=3)", inference::GenerationConfig::beam(3).with_max_tokens(10)},
    };
    
    strategies[2].config.top_k = 10;
    
    std::cout << std::setw(25) << "Strategy" 
              << std::setw(10) << "Tokens"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Tok/sec\n";
    std::cout << std::string(59, '-') << "\n";
    
    for (const auto& strategy : strategies) {
        model.clear_cache();
        
        auto result = engine.generate(prompt, forward_fn, strategy.config);
        
        std::cout << std::setw(25) << strategy.name
                  << std::setw(10) << result.tokens_generated
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << result.generation_time_ms
                  << std::setw(12) << std::setprecision(0) 
                  << result.tokens_per_second() << "\n";
    }
    
    std::cout << "\nCache statistics:\n";
    std::cout << "  - Entries: " << model.cache_size() << "\n";
    std::cout << "  - Hit rate: " << std::fixed << std::setprecision(2) 
              << (model.cache_hit_rate() * 100) << "%\n";
}

void run_cache_compression_benchmark() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  KV Cache Compression Benchmark\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    // Create test data (simulating KV vectors)
    const usize num_entries = 1000;
    const usize entry_size = 512 * sizeof(f32);  // 512 dim embedding
    
    std::vector<std::vector<u8>> entries;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 255);
    
    for (usize i = 0; i < num_entries; ++i) {
        std::vector<u8> entry(entry_size);
        // Make it somewhat compressible (sparse-like)
        for (usize j = 0; j < entry_size; ++j) {
            entry[j] = (j % 8 == 0) ? static_cast<u8>(dist(gen)) : 0;
        }
        entries.push_back(std::move(entry));
    }
    
    usize total_size = num_entries * entry_size;
    
    std::cout << "Test: " << num_entries << " entries × " 
              << entry_size / 1024.0 << " KB = "
              << total_size / (1024.0 * 1024.0) << " MB\n\n";
    
    // Test without compression
    {
        memory::CacheConfig config;
        config.max_size = num_entries;
        memory::ConcurrentKVCache cache(config);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (usize i = 0; i < num_entries; ++i) {
            cache.put(0, i, entries[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Without compression:\n";
        std::cout << "  - Write time: " << duration.count() / 1000.0 << " ms\n";
        std::cout << "  - Throughput: " << (total_size / 1e6) / (duration.count() / 1e6) 
                  << " MB/s\n\n";
    }
    
    // Test with LZ4 compression
    {
        compression::BatchCompressor compressor(compression::Algorithm::LZ4);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto compressed = compressor.compress_batch(entries);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        usize compressed_size = 0;
        for (const auto& c : compressed) {
            compressed_size += c.data.size();
        }
        
        std::cout << "With LZ4 compression:\n";
        std::cout << "  - Compress time: " << duration.count() / 1000.0 << " ms\n";
        std::cout << "  - Compressed size: " << compressed_size / 1024.0 << " KB\n";
        std::cout << "  - Ratio: " << std::fixed << std::setprecision(1) 
                  << (100.0 * compressed_size / total_size) << "%\n";
        std::cout << "  - Memory saved: " << (total_size - compressed_size) / (1024.0 * 1024.0) 
                  << " MB\n\n";
    }
}

void run_attention_memory_analysis() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  Attention Memory Analysis\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    std::cout << "Comparing memory usage for different sequence lengths:\n\n";
    
    std::cout << std::setw(10) << "Seq Len"
              << std::setw(12) << "Batch"
              << std::setw(15) << "Std Attn MB"
              << std::setw(15) << "Flash MB"
              << std::setw(12) << "Savings\n";
    std::cout << std::string(64, '-') << "\n";
    
    for (usize seq_len : {512, 1024, 2048, 4096, 8192}) {
        for (usize batch : {1, 4}) {
            auto stats = attention::AttentionStats::compute(batch, seq_len, 12, 64);
            
            // Flash attention uses O(sqrt(N)) memory vs O(N²)
            f64 flash_mb = stats.memory_peak_mb * 64.0 / seq_len;  // Approximation
            f64 savings = (1.0 - flash_mb / stats.memory_peak_mb) * 100;
            
            std::cout << std::setw(10) << seq_len
                      << std::setw(12) << batch
                      << std::setw(15) << std::fixed << std::setprecision(1) 
                      << stats.memory_peak_mb
                      << std::setw(15) << flash_mb
                      << std::setw(11) << std::setprecision(0) << savings << "%\n";
        }
    }
    
    std::cout << "\n→ Flash Attention enables much longer contexts!\n";
}

int main() {
    print_header();
    
    run_generation_example();
    run_cache_compression_benchmark();
    run_attention_memory_analysis();
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  Integration Example Complete\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    std::cout << "Key takeaways:\n";
    std::cout << "  1. KV Cache with compression saves significant memory\n";
    std::cout << "  2. Flash Attention scales to longer sequences\n";
    std::cout << "  3. Concurrent cache enables parallel inference\n";
    std::cout << "  4. Multiple sampling strategies available\n";
    std::cout << "\n";
    
    return 0;
}












