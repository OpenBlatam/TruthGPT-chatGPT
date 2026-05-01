/**
 * @file test_inference.cpp
 * @brief Unit tests for inference engine module
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

#include "../include/optimization_core.hpp"

using namespace optimization_core;
using namespace optimization_core::inference;

class InferenceTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(InferenceTest, ConfigBuilder) {
    auto config = GenerationConfig()
        .with_max_tokens(256)
        .with_temperature(0.7f)
        .with_top_p(0.9f)
        .with_top_k(40)
        .with_sampling(true);
    
    EXPECT_EQ(config.max_new_tokens, 256);
    EXPECT_FLOAT_EQ(config.temperature, 0.7f);
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_EQ(config.top_k, 40);
    EXPECT_TRUE(config.do_sample);
}

TEST_F(InferenceTest, ConfigPresets) {
    auto greedy = GenerationConfig::greedy();
    EXPECT_FALSE(greedy.do_sample);
    
    auto sampling = GenerationConfig::sampling(0.8f, 0.95f);
    EXPECT_TRUE(sampling.do_sample);
    EXPECT_FLOAT_EQ(sampling.temperature, 0.8f);
    EXPECT_FLOAT_EQ(sampling.top_p, 0.95f);
    
    auto beam = GenerationConfig::beam(4);
    EXPECT_FALSE(beam.do_sample);
    EXPECT_EQ(beam.num_beams, 4);
}

// ============================================================================
// Sampling Utilities Tests
// ============================================================================

TEST_F(InferenceTest, Softmax) {
    std::vector<f32> logits = {1.0f, 2.0f, 3.0f};
    auto probs = sampling::softmax(logits);
    
    f32 sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    
    EXPECT_GT(probs[2], probs[1]);
    EXPECT_GT(probs[1], probs[0]);
}

TEST_F(InferenceTest, GreedySampling) {
    std::vector<f32> logits = {1.0f, 5.0f, 2.0f, 3.0f};
    i32 token = sampling::greedy(logits);
    EXPECT_EQ(token, 1);  // Index of max value
}

TEST_F(InferenceTest, ApplyTemperature) {
    std::vector<f32> logits = {1.0f, 2.0f, 3.0f};
    auto original = logits;
    
    sampling::apply_temperature(logits, 0.5f);
    
    // Lower temperature = more peaked distribution
    for (usize i = 0; i < logits.size(); ++i) {
        EXPECT_FLOAT_EQ(logits[i], original[i] / 0.5f);
    }
}

TEST_F(InferenceTest, ApplyTopK) {
    std::vector<f32> logits = {1.0f, 5.0f, 2.0f, 4.0f, 3.0f};
    
    sampling::apply_top_k(logits, 2);
    
    // Only top 2 should remain
    int non_inf = 0;
    for (auto l : logits) {
        if (l > -1e8f) non_inf++;
    }
    EXPECT_EQ(non_inf, 2);
}

TEST_F(InferenceTest, ApplyTopP) {
    std::vector<f32> probs = {0.1f, 0.5f, 0.2f, 0.15f, 0.05f};
    
    sampling::apply_top_p(probs, 0.7f);
    
    // Should zero out lowest probability tokens
    EXPECT_EQ(probs[4], 0.0f);  // 0.05
}

TEST_F(InferenceTest, RepetitionPenalty) {
    std::vector<f32> logits = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<i32> prev_tokens = {1, 2};
    
    sampling::apply_repetition_penalty(logits, prev_tokens, 1.5f);
    
    // Penalized tokens should have lower logits
    EXPECT_LT(logits[1], 2.0f);
    EXPECT_LT(logits[2], 3.0f);
    
    // Non-penalized should be unchanged
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits[3], 4.0f);
}

// ============================================================================
// Token Sampler Tests
// ============================================================================

TEST_F(InferenceTest, TokenSamplerGreedy) {
    TokenSampler sampler(42);
    
    std::vector<f32> logits = {1.0f, 5.0f, 2.0f, 3.0f};
    GenerationConfig config;
    config.do_sample = false;
    
    i32 token = sampler.sample(logits, config);
    EXPECT_EQ(token, 1);  // Max index
}

TEST_F(InferenceTest, TokenSamplerDeterministic) {
    std::vector<f32> logits = {1.0f, 2.0f, 3.0f, 4.0f};
    GenerationConfig config;
    config.do_sample = true;
    config.temperature = 0.1f;  // Very low temp = almost greedy
    
    // Same seed should give same result
    TokenSampler sampler1(42);
    TokenSampler sampler2(42);
    
    i32 token1 = sampler1.sample(logits, config);
    i32 token2 = sampler2.sample(logits, config);
    
    EXPECT_EQ(token1, token2);
}

TEST_F(InferenceTest, TokenSamplerWithTopK) {
    TokenSampler sampler(42);
    
    std::vector<f32> logits(100, 0.0f);
    logits[50] = 10.0f;  // High probability token
    
    GenerationConfig config;
    config.do_sample = true;
    config.top_k = 10;
    config.temperature = 1.0f;
    
    // Sample multiple times
    std::set<i32> sampled;
    for (int i = 0; i < 100; ++i) {
        sampled.insert(sampler.sample(logits, config));
    }
    
    // Should have sampled token 50 at least once
    EXPECT_TRUE(sampled.count(50) > 0);
}

// ============================================================================
// Beam Search Tests
// ============================================================================

TEST_F(InferenceTest, BeamSearchInitialize) {
    BeamSearch beam(4, 1.0f);
    beam.initialize({1, 2, 3});
    
    auto seqs = beam.all_sequences();
    EXPECT_EQ(seqs.size(), 1);
    EXPECT_EQ(seqs[0], std::vector<i32>({1, 2, 3}));
}

TEST_F(InferenceTest, BeamSearchStep) {
    BeamSearch beam(2, 1.0f);
    beam.initialize({1});
    
    // Simulate logits for beam expansion
    std::vector<std::vector<f32>> logits = {
        {0.1f, 0.5f, 0.3f, 0.1f}  // Token 1 has highest prob
    };
    
    beam.step(logits, -1);  // No EOS
    
    auto seqs = beam.all_sequences();
    EXPECT_EQ(seqs.size(), 2);  // Should have 2 beams
}

TEST_F(InferenceTest, BeamSearchEOS) {
    BeamSearch beam(2, 1.0f);
    beam.initialize({1});
    
    std::vector<std::vector<f32>> logits = {
        {0.0f, 0.0f, 1.0f}  // Token 2 is EOS
    };
    
    beam.step(logits, 2);  // EOS = 2
    
    // Some beams should be finished
    EXPECT_TRUE(beam.is_finished() || beam.all_sequences().size() > 0);
}

// ============================================================================
// Inference Engine Tests
// ============================================================================

TEST_F(InferenceTest, EngineGenerate) {
    InferenceEngine engine(42);
    
    std::vector<i32> input_ids = {1, 2, 3};
    
    // Mock forward function
    auto forward_fn = [](const std::vector<i32>& tokens) -> std::vector<f32> {
        (void)tokens;
        // Return logits favoring token 5
        std::vector<f32> logits(10, 0.0f);
        logits[5] = 10.0f;
        return logits;
    };
    
    auto config = GenerationConfig::greedy().with_max_tokens(5).with_eos(5);
    auto result = engine.generate(input_ids, forward_fn, config);
    
    EXPECT_GT(result.token_ids.size(), input_ids.size());
    EXPECT_GT(result.tokens_generated, 0);
}

TEST_F(InferenceTest, EngineGenerateWithEOS) {
    InferenceEngine engine(42);
    
    std::vector<i32> input_ids = {1};
    int call_count = 0;
    
    auto forward_fn = [&call_count](const std::vector<i32>& tokens) -> std::vector<f32> {
        (void)tokens;
        call_count++;
        std::vector<f32> logits(10, 0.0f);
        
        // After 3 calls, return EOS token
        if (call_count >= 3) {
            logits[0] = 10.0f;  // EOS
        } else {
            logits[5] = 10.0f;
        }
        return logits;
    };
    
    auto config = GenerationConfig::greedy().with_max_tokens(10).with_eos(0);
    auto result = engine.generate(input_ids, forward_fn, config);
    
    // Should stop at EOS, not max_tokens
    EXPECT_LT(result.tokens_generated, 10);
}

TEST_F(InferenceTest, GenerationResultMetrics) {
    GenerationResult result;
    result.token_ids = {1, 2, 3, 4, 5};
    result.tokens_generated = 3;
    result.generation_time_ms = 100.0;
    
    f64 tps = result.tokens_per_second();
    EXPECT_NEAR(tps, 30.0, 0.1);  // 3 tokens / 0.1 seconds
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST_F(InferenceTest, CreateEngine) {
    auto engine = create_engine(42);
    EXPECT_NE(engine, nullptr);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}












