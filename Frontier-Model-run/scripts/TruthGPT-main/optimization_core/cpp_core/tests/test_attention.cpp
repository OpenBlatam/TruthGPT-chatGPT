/**
 * @file test_attention.cpp
 * @brief Unit tests for attention module
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <random>

#include "../include/optimization_core.hpp"

using namespace optimization_core;
using namespace optimization_core::attention;

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random generator
        gen_ = std::mt19937(42);
        dist_ = std::normal_distribution<f32>(0.0f, 1.0f);
    }
    
    std::vector<f32> random_tensor(usize size) {
        std::vector<f32> tensor(size);
        for (auto& v : tensor) v = dist_(gen_);
        return tensor;
    }
    
    std::mt19937 gen_;
    std::normal_distribution<f32> dist_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(AttentionTest, ConfigBuilder) {
    auto config = AttentionConfig()
        .with_heads(12)
        .with_head_dim(64)
        .with_flash(128)
        .with_causal()
        .with_dropout(0.1f);
    
    EXPECT_EQ(config.num_heads, 12);
    EXPECT_EQ(config.head_dim, 64);
    EXPECT_EQ(config.block_size, 128);
    EXPECT_TRUE(config.use_flash);
    EXPECT_TRUE(config.use_causal_mask);
    EXPECT_FLOAT_EQ(config.dropout, 0.1f);
    EXPECT_EQ(config.d_model(), 768);
}

TEST_F(AttentionTest, ConfigValidation) {
    AttentionConfig config;
    config.num_heads = 0;
    
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST_F(AttentionTest, ConfigScale) {
    AttentionConfig config;
    config.head_dim = 64;
    
    f32 expected_scale = 1.0f / std::sqrt(64.0f);
    EXPECT_FLOAT_EQ(config.get_scale(), expected_scale);
    
    config.with_scale(0.5f);
    EXPECT_FLOAT_EQ(config.get_scale(), 0.5f);
}

// ============================================================================
// Math Utilities Tests
// ============================================================================

TEST_F(AttentionTest, Softmax) {
    std::vector<f32> logits = {1.0f, 2.0f, 3.0f};
    auto probs = math::softmax(logits);
    
    // Sum should be 1
    f32 sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    
    // Values should be ordered
    EXPECT_GT(probs[2], probs[1]);
    EXPECT_GT(probs[1], probs[0]);
}

TEST_F(AttentionTest, SoftmaxNumericalStability) {
    // Large values that could overflow without proper handling
    std::vector<f32> logits = {1000.0f, 1001.0f, 1002.0f};
    auto probs = math::softmax(logits);
    
    f32 sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    
    // No NaN or Inf
    for (auto p : probs) {
        EXPECT_FALSE(std::isnan(p));
        EXPECT_FALSE(std::isinf(p));
    }
}

// ============================================================================
// Mask Tests
// ============================================================================

TEST_F(AttentionTest, CausalMask) {
    auto m = mask::create_causal(4);
    
    EXPECT_EQ(m.size(), 16);
    
    // Lower triangle should be 0
    EXPECT_FLOAT_EQ(m[0 * 4 + 0], 0.0f);  // (0, 0)
    EXPECT_FLOAT_EQ(m[1 * 4 + 0], 0.0f);  // (1, 0)
    EXPECT_FLOAT_EQ(m[1 * 4 + 1], 0.0f);  // (1, 1)
    
    // Upper triangle should be -inf
    EXPECT_LT(m[0 * 4 + 1], -1e8f);  // (0, 1)
    EXPECT_LT(m[0 * 4 + 2], -1e8f);  // (0, 2)
    EXPECT_LT(m[1 * 4 + 2], -1e8f);  // (1, 2)
}

TEST_F(AttentionTest, PaddingMask) {
    std::vector<usize> lengths = {3, 2, 4};
    auto m = mask::create_padding(lengths, 5);
    
    EXPECT_EQ(m.size(), 15);  // 3 * 5
    
    // Check first sequence (length 3)
    EXPECT_FLOAT_EQ(m[0 * 5 + 2], 0.0f);   // Valid
    EXPECT_LT(m[0 * 5 + 3], -1e8f);        // Padding
    
    // Check second sequence (length 2)
    EXPECT_FLOAT_EQ(m[1 * 5 + 1], 0.0f);   // Valid
    EXPECT_LT(m[1 * 5 + 2], -1e8f);        // Padding
}

// ============================================================================
// Attention Forward Tests
// ============================================================================

#ifdef HAVE_EIGEN

TEST_F(AttentionTest, ScaledDotProductAttention) {
    AttentionConfig config;
    config.num_heads = 1;
    config.head_dim = 8;
    
    ScaledDotProductAttention attn(config);
    
    usize batch = 2;
    usize seq = 4;
    usize dim = 8;
    
    auto q = random_tensor(batch * seq * dim);
    auto k = random_tensor(batch * seq * dim);
    auto v = random_tensor(batch * seq * dim);
    
    auto output = attn.forward(q, k, v, batch, seq, std::nullopt);
    
    EXPECT_EQ(output.size(), batch * seq * dim);
    
    // Check no NaN
    for (auto o : output) {
        EXPECT_FALSE(std::isnan(o));
    }
}

TEST_F(AttentionTest, FlashAttention) {
    AttentionConfig config;
    config.num_heads = 1;
    config.head_dim = 8;
    config.block_size = 2;
    
    FlashAttention attn(config);
    
    usize batch = 1;
    usize seq = 8;
    usize dim = 8;
    
    auto q = random_tensor(batch * seq * dim);
    auto k = random_tensor(batch * seq * dim);
    auto v = random_tensor(batch * seq * dim);
    
    auto output = attn.forward(q, k, v, batch, seq, std::nullopt);
    
    EXPECT_EQ(output.size(), batch * seq * dim);
}

TEST_F(AttentionTest, FlashAttentionCausal) {
    AttentionConfig config;
    config.num_heads = 1;
    config.head_dim = 8;
    config.block_size = 2;
    config.use_causal_mask = true;
    
    FlashAttention attn(config);
    
    usize batch = 1;
    usize seq = 4;
    usize dim = 8;
    
    auto q = random_tensor(batch * seq * dim);
    auto k = random_tensor(batch * seq * dim);
    auto v = random_tensor(batch * seq * dim);
    
    auto output = attn.forward(q, k, v, batch, seq, std::nullopt);
    
    EXPECT_EQ(output.size(), batch * seq * dim);
}

TEST_F(AttentionTest, AttentionStatsCompute) {
    auto stats = AttentionStats::compute(4, 512, 12, 64);
    
    EXPECT_EQ(stats.total_tokens, 4 * 512);
    EXPECT_EQ(stats.attention_computations, 4 * 12 * 512 * 512);
    EXPECT_GT(stats.memory_peak_mb, 0.0);
}

TEST_F(AttentionTest, SparseAttention) {
    AttentionConfig config;
    config.num_heads = 1;
    config.head_dim = 8;
    
    SparseAttention attn(config, 2, 1);  // window=2, global=1
    
    usize batch = 1;
    usize seq = 8;
    usize dim = 8;
    
    auto q = random_tensor(batch * seq * dim);
    auto k = random_tensor(batch * seq * dim);
    auto v = random_tensor(batch * seq * dim);
    
    auto output = attn.forward(q, k, v, batch, seq, std::nullopt);
    
    EXPECT_EQ(output.size(), batch * seq * dim);
}

TEST_F(AttentionTest, CreateAttentionFactory) {
    AttentionConfig config;
    config.use_flash = true;
    
    auto attn = create_attention(config);
    EXPECT_NE(attn, nullptr);
    
    config.use_flash = false;
    attn = create_attention(config);
    EXPECT_NE(attn, nullptr);
}

#endif // HAVE_EIGEN

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}












