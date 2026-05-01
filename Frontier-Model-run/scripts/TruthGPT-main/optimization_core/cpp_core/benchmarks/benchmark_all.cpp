/**
 * @file benchmark_all.cpp
 * @brief Comprehensive benchmarks for optimization_core
 * 
 * Benchmarks for:
 * - Attention mechanisms
 * - KV cache operations
 * - Compression algorithms
 * - Inference sampling
 */

#include <benchmark/benchmark.h>
#include <random>
#include <vector>

#include "../include/optimization_core.hpp"
#include "../include/compression/compression.hpp"

using namespace optimization_core;

// ============================================================================
// Utilities
// ============================================================================

static std::vector<f32> random_tensor(usize size, u32 seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    std::vector<f32> tensor(size);
    for (auto& v : tensor) v = dist(gen);
    return tensor;
}

static std::vector<u8> random_bytes(usize size, u32 seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<u8> data(size);
    for (auto& b : data) b = static_cast<u8>(dist(gen));
    return data;
}

static std::vector<u8> repetitive_bytes(usize size) {
    std::vector<u8> data(size);
    for (usize i = 0; i < size; ++i) data[i] = static_cast<u8>(i % 16);
    return data;
}

// ============================================================================
// Attention Benchmarks
// ============================================================================

#ifdef HAVE_EIGEN

static void BM_Softmax(benchmark::State& state) {
    usize size = state.range(0);
    auto logits = random_tensor(size);
    
    for (auto _ : state) {
        auto result = attention::math::softmax(logits);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_Softmax)->RangeMultiplier(4)->Range(64, 4096);

static void BM_CausalMask(benchmark::State& state) {
    usize seq_len = state.range(0);
    
    for (auto _ : state) {
        auto mask = attention::mask::create_causal(seq_len);
        benchmark::DoNotOptimize(mask);
    }
    
    state.SetItemsProcessed(state.iterations() * seq_len * seq_len);
}
BENCHMARK(BM_CausalMask)->RangeMultiplier(2)->Range(64, 2048);

static void BM_ScaledDotProductAttention(benchmark::State& state) {
    usize batch = 4;
    usize seq_len = state.range(0);
    usize num_heads = 8;
    usize head_dim = 64;
    usize dim = num_heads * head_dim;
    
    attention::AttentionConfig config;
    config.num_heads = num_heads;
    config.head_dim = head_dim;
    
    attention::ScaledDotProductAttention attn(config);
    
    auto q = random_tensor(batch * seq_len * dim);
    auto k = random_tensor(batch * seq_len * dim);
    auto v = random_tensor(batch * seq_len * dim);
    
    for (auto _ : state) {
        auto output = attn.forward(q, k, v, batch, seq_len, std::nullopt);
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * batch * seq_len);
    state.SetBytesProcessed(state.iterations() * batch * seq_len * dim * 3 * sizeof(f32));
}
BENCHMARK(BM_ScaledDotProductAttention)->RangeMultiplier(2)->Range(32, 512);

static void BM_FlashAttention(benchmark::State& state) {
    usize batch = 4;
    usize seq_len = state.range(0);
    usize num_heads = 8;
    usize head_dim = 64;
    usize dim = num_heads * head_dim;
    
    attention::AttentionConfig config;
    config.num_heads = num_heads;
    config.head_dim = head_dim;
    config.block_size = 64;
    
    attention::FlashAttention attn(config);
    
    auto q = random_tensor(batch * seq_len * dim);
    auto k = random_tensor(batch * seq_len * dim);
    auto v = random_tensor(batch * seq_len * dim);
    
    for (auto _ : state) {
        auto output = attn.forward(q, k, v, batch, seq_len, std::nullopt);
        benchmark::DoNotOptimize(output);
    }
    
    state.SetItemsProcessed(state.iterations() * batch * seq_len);
}
BENCHMARK(BM_FlashAttention)->RangeMultiplier(2)->Range(64, 2048);

#endif // HAVE_EIGEN

// ============================================================================
// Cache Benchmarks
// ============================================================================

static void BM_CachePut(benchmark::State& state) {
    memory::CacheConfig config;
    config.max_size = 100000;
    
    memory::KVCache cache(config);
    
    std::vector<u8> data(256, 42);
    usize key = 0;
    
    for (auto _ : state) {
        cache.put(0, key++, data);
        if (key >= config.max_size) {
            cache.clear();
            key = 0;
        }
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CachePut);

static void BM_CacheGet(benchmark::State& state) {
    memory::CacheConfig config;
    config.max_size = state.range(0);
    
    memory::KVCache cache(config);
    
    // Pre-populate
    std::vector<u8> data(256, 42);
    for (usize i = 0; i < config.max_size; ++i) {
        cache.put(0, i, data);
    }
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<usize> dist(0, config.max_size - 1);
    
    for (auto _ : state) {
        auto result = cache.get(0, dist(gen));
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CacheGet)->RangeMultiplier(10)->Range(100, 100000);

static void BM_ConcurrentCacheGet(benchmark::State& state) {
    memory::CacheConfig config;
    config.max_size = 10000;
    
    memory::ConcurrentKVCache cache(config);
    
    // Pre-populate
    std::vector<u8> data(256, 42);
    for (usize i = 0; i < config.max_size; ++i) {
        cache.put(0, i, data);
    }
    
    for (auto _ : state) {
        auto result = cache.get(0, state.iterations() % config.max_size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ConcurrentCacheGet)->Threads(1)->Threads(2)->Threads(4)->Threads(8);

// ============================================================================
// Compression Benchmarks
// ============================================================================

static void BM_LZ4Compress(benchmark::State& state) {
    usize size = state.range(0);
    auto data = repetitive_bytes(size);
    
    for (auto _ : state) {
        auto result = compression::compress(data, compression::Algorithm::LZ4);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_LZ4Compress)->RangeMultiplier(4)->Range(1024, 1024 * 1024);

static void BM_LZ4Decompress(benchmark::State& state) {
    usize size = state.range(0);
    auto data = repetitive_bytes(size);
    auto compressed = compression::compress(data, compression::Algorithm::LZ4);
    
    for (auto _ : state) {
        auto result = compression::decompress(compressed.data, compression::Algorithm::LZ4);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_LZ4Decompress)->RangeMultiplier(4)->Range(1024, 1024 * 1024);

static void BM_ZstdCompress(benchmark::State& state) {
    usize size = state.range(0);
    auto data = repetitive_bytes(size);
    
    for (auto _ : state) {
        auto result = compression::compress(data, compression::Algorithm::Zstd, 3);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * size);
}
BENCHMARK(BM_ZstdCompress)->RangeMultiplier(4)->Range(1024, 1024 * 1024);

static void BM_ZstdCompressLevel(benchmark::State& state) {
    i32 level = state.range(0);
    usize size = 65536;
    auto data = repetitive_bytes(size);
    
    for (auto _ : state) {
        auto result = compression::compress(data, compression::Algorithm::Zstd, level);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * size);
    state.SetLabel("level=" + std::to_string(level));
}
BENCHMARK(BM_ZstdCompressLevel)->DenseRange(1, 19, 3);

static void BM_CompressionRatio(benchmark::State& state) {
    // Compare compression ratio for different data types
    int data_type = state.range(0);
    usize size = 65536;
    
    std::vector<u8> data;
    switch (data_type) {
        case 0: data = random_bytes(size); break;         // Random (incompressible)
        case 1: data = repetitive_bytes(size); break;     // Repetitive
        case 2: data = std::vector<u8>(size, 0); break;   // All zeros
    }
    
    for (auto _ : state) {
        auto result = compression::compress(data, compression::Algorithm::LZ4);
        benchmark::DoNotOptimize(result);
        state.counters["ratio"] = static_cast<double>(result.stats.compressed_size) / size;
    }
}
BENCHMARK(BM_CompressionRatio)->DenseRange(0, 2);

// ============================================================================
// Inference Benchmarks
// ============================================================================

static void BM_SamplerGreedy(benchmark::State& state) {
    usize vocab_size = state.range(0);
    auto logits = random_tensor(vocab_size);
    
    inference::TokenSampler sampler(42);
    inference::GenerationConfig config;
    config.do_sample = false;
    
    for (auto _ : state) {
        auto token = sampler.sample(logits, config);
        benchmark::DoNotOptimize(token);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SamplerGreedy)->RangeMultiplier(4)->Range(1000, 128000);

static void BM_SamplerTopK(benchmark::State& state) {
    usize vocab_size = 50000;
    i32 k = state.range(0);
    auto logits = random_tensor(vocab_size);
    
    inference::TokenSampler sampler(42);
    auto config = inference::GenerationConfig::sampling(1.0f, 1.0f);
    config.top_k = k;
    
    for (auto _ : state) {
        auto token = sampler.sample(logits, config);
        benchmark::DoNotOptimize(token);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SamplerTopK)->Arg(10)->Arg(50)->Arg(100)->Arg(500);

static void BM_SamplerTopP(benchmark::State& state) {
    usize vocab_size = 50000;
    auto logits = random_tensor(vocab_size);
    
    inference::TokenSampler sampler(42);
    auto config = inference::GenerationConfig::sampling(1.0f, 0.9f);
    
    for (auto _ : state) {
        auto token = sampler.sample(logits, config);
        benchmark::DoNotOptimize(token);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SamplerTopP);

static void BM_RepetitionPenalty(benchmark::State& state) {
    usize vocab_size = 50000;
    usize context_len = state.range(0);
    
    auto logits = random_tensor(vocab_size);
    std::vector<i32> prev_tokens(context_len);
    for (usize i = 0; i < context_len; ++i) {
        prev_tokens[i] = i % vocab_size;
    }
    
    for (auto _ : state) {
        auto penalized = logits;
        inference::sampling::apply_repetition_penalty(penalized, prev_tokens, 1.2f);
        benchmark::DoNotOptimize(penalized);
    }
    
    state.SetItemsProcessed(state.iterations() * context_len);
}
BENCHMARK(BM_RepetitionPenalty)->RangeMultiplier(4)->Range(64, 4096);

// ============================================================================
// End-to-End Benchmarks
// ============================================================================

static void BM_EndToEndGeneration(benchmark::State& state) {
    usize num_tokens = state.range(0);
    usize vocab_size = 50000;
    
    inference::InferenceEngine engine(42);
    
    auto forward_fn = [vocab_size](const std::vector<i32>& tokens) {
        (void)tokens;
        std::vector<f32> logits(vocab_size, 0.0f);
        // Simple mock that favors tokens based on position
        for (usize i = 0; i < vocab_size; ++i) {
            logits[i] = -static_cast<f32>(i) / 1000.0f;
        }
        logits[42] = 10.0f;  // Always favor token 42
        return logits;
    };
    
    auto config = inference::GenerationConfig::greedy()
        .with_max_tokens(num_tokens);
    
    std::vector<i32> prompt = {1, 2, 3};
    
    for (auto _ : state) {
        auto result = engine.generate(prompt, forward_fn, config);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * num_tokens);
}
BENCHMARK(BM_EndToEndGeneration)->Arg(10)->Arg(50)->Arg(100);

// ============================================================================
// Memory Benchmarks
// ============================================================================

static void BM_VectorAllocation(benchmark::State& state) {
    usize size = state.range(0);
    
    for (auto _ : state) {
        std::vector<f32> vec(size);
        benchmark::DoNotOptimize(vec.data());
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(f32));
}
BENCHMARK(BM_VectorAllocation)->RangeMultiplier(4)->Range(1024, 1024 * 1024);

// ============================================================================
// Main
// ============================================================================

BENCHMARK_MAIN();












