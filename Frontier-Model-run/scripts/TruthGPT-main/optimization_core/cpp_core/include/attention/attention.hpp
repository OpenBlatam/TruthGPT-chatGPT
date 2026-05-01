#pragma once

/**
 * @file attention.hpp
 * @brief Refactored attention module with clean interfaces
 * 
 * Features:
 * - Builder pattern for configuration
 * - Strategy pattern for different attention types
 * - Clean separation of concerns
 * - Consistent with Rust implementation patterns
 */

#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include "../common/types.hpp"

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

namespace optimization_core {
namespace attention {

// ============================================================================
// Configuration
// ============================================================================

/**
 * @brief Attention configuration with builder pattern
 */
struct AttentionConfig {
    usize num_heads = 8;
    usize head_dim = 64;
    f32 dropout = 0.0f;
    bool use_flash = true;
    usize block_size = 64;
    bool use_causal_mask = false;
    f32 scale = -1.0f;  // Auto-compute if negative
    
    // Builder methods
    AttentionConfig& with_heads(usize n) { num_heads = n; return *this; }
    AttentionConfig& with_head_dim(usize d) { head_dim = d; return *this; }
    AttentionConfig& with_dropout(f32 p) { dropout = p; return *this; }
    AttentionConfig& with_flash(usize bs) { use_flash = true; block_size = bs; return *this; }
    AttentionConfig& with_causal() { use_causal_mask = true; return *this; }
    AttentionConfig& with_scale(f32 s) { scale = s; return *this; }
    
    f32 get_scale() const {
        return scale > 0 ? scale : 1.0f / std::sqrt(static_cast<f32>(head_dim));
    }
    
    usize d_model() const { return num_heads * head_dim; }
    
    void validate() const {
        if (num_heads == 0) throw std::invalid_argument("num_heads must be > 0");
        if (head_dim == 0) throw std::invalid_argument("head_dim must be > 0");
        if (block_size == 0) throw std::invalid_argument("block_size must be > 0");
        if (dropout < 0 || dropout > 1) throw std::invalid_argument("dropout must be in [0, 1]");
    }
};

// ============================================================================
// Attention Statistics
// ============================================================================

struct AttentionStats {
    usize total_tokens = 0;
    usize attention_computations = 0;
    f64 memory_peak_mb = 0.0;
    f64 compute_time_ms = 0.0;
    
    static AttentionStats compute(usize batch, usize seq_len, 
                                   usize num_heads, usize head_dim) {
        AttentionStats stats;
        stats.total_tokens = batch * seq_len;
        stats.attention_computations = batch * num_heads * seq_len * seq_len;
        stats.memory_peak_mb = static_cast<f64>(batch * num_heads * seq_len * seq_len * 4) 
                              / (1024.0 * 1024.0);
        return stats;
    }
};

// ============================================================================
// Attention Interface
// ============================================================================

/**
 * @brief Abstract interface for attention implementations
 */
class IAttention {
public:
    virtual ~IAttention() = default;
    
    virtual std::vector<f32> forward(
        const std::vector<f32>& query,
        const std::vector<f32>& key,
        const std::vector<f32>& value,
        usize batch_size,
        usize seq_len,
        const std::optional<std::vector<f32>>& mask = std::nullopt
    ) = 0;
    
    virtual AttentionStats get_stats() const = 0;
    virtual void reset_stats() = 0;
};

// ============================================================================
// Math Utilities
// ============================================================================

namespace math {

/**
 * @brief Numerically stable softmax
 */
inline void softmax_inplace(f32* data, usize size) {
    if (size == 0) return;
    
    // Find max
    f32 max_val = data[0];
    for (usize i = 1; i < size; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    
    // Compute exp and sum
    f32 sum = 0.0f;
    for (usize i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    
    // Normalize
    if (sum > 0) {
        f32 inv_sum = 1.0f / sum;
        for (usize i = 0; i < size; ++i) {
            data[i] *= inv_sum;
        }
    }
}

/**
 * @brief Softmax returning new vector
 */
inline std::vector<f32> softmax(const std::vector<f32>& input) {
    std::vector<f32> output = input;
    softmax_inplace(output.data(), output.size());
    return output;
}

} // namespace math

// ============================================================================
// Mask Utilities
// ============================================================================

namespace mask {

/**
 * @brief Create causal mask for autoregressive attention
 */
inline std::vector<f32> create_causal(usize seq_len) {
    std::vector<f32> m(seq_len * seq_len);
    for (usize i = 0; i < seq_len; ++i) {
        for (usize j = 0; j < seq_len; ++j) {
            m[i * seq_len + j] = (j <= i) ? 0.0f : -1e9f;
        }
    }
    return m;
}

/**
 * @brief Create padding mask from lengths
 */
inline std::vector<f32> create_padding(const std::vector<usize>& lengths, usize max_len) {
    usize batch = lengths.size();
    std::vector<f32> m(batch * max_len, 0.0f);
    for (usize b = 0; b < batch; ++b) {
        for (usize j = lengths[b]; j < max_len; ++j) {
            m[b * max_len + j] = -1e9f;
        }
    }
    return m;
}

} // namespace mask

#ifdef HAVE_EIGEN

// ============================================================================
// Eigen-based Implementations
// ============================================================================

using Matrix = Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXf;

/**
 * @brief Standard scaled dot-product attention
 */
class ScaledDotProductAttention : public IAttention {
public:
    explicit ScaledDotProductAttention(AttentionConfig config = {})
        : config_(std::move(config)) {
        config_.validate();
    }
    
    std::vector<f32> forward(
        const std::vector<f32>& query,
        const std::vector<f32>& key,
        const std::vector<f32>& value,
        usize batch_size,
        usize seq_len,
        const std::optional<std::vector<f32>>& mask = std::nullopt
    ) override {
        Timer timer;
        
        usize d_k = config_.head_dim;
        f32 scale = config_.get_scale();
        
        // Map inputs to Eigen matrices
        Eigen::Map<const Matrix> Q(query.data(), batch_size * seq_len, d_k);
        Eigen::Map<const Matrix> K(key.data(), batch_size * seq_len, d_k);
        Eigen::Map<const Matrix> V(value.data(), batch_size * seq_len, d_k);
        
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        Matrix scores = (Q * K.transpose()) * scale;
        
        // Apply mask
        if (mask.has_value()) {
            Eigen::Map<const Matrix> M(mask->data(), scores.rows(), scores.cols());
            scores += M;
        } else if (config_.use_causal_mask) {
            for (i32 i = 0; i < scores.rows(); ++i) {
                for (i32 j = i + 1; j < scores.cols(); ++j) {
                    scores(i, j) = -1e9f;
                }
            }
        }
        
        // Softmax per row
        for (i32 i = 0; i < scores.rows(); ++i) {
            Vector row = scores.row(i);
            f32 max_val = row.maxCoeff();
            row = (row.array() - max_val).exp();
            scores.row(i) = row / row.sum();
        }
        
        // Compute output: scores @ V
        Matrix output = scores * V;
        
        // Update stats
        stats_.total_tokens = batch_size * seq_len;
        stats_.attention_computations = batch_size * seq_len * seq_len;
        stats_.compute_time_ms = timer.elapsed_ms();
        
        return std::vector<f32>(output.data(), output.data() + output.size());
    }
    
    AttentionStats get_stats() const override { return stats_; }
    void reset_stats() override { stats_ = {}; }

private:
    AttentionConfig config_;
    AttentionStats stats_;
};

/**
 * @brief Flash attention with block-wise processing
 */
class FlashAttention : public IAttention {
public:
    explicit FlashAttention(AttentionConfig config = {})
        : config_(std::move(config)) {
        config_.use_flash = true;
        config_.validate();
    }
    
    std::vector<f32> forward(
        const std::vector<f32>& query,
        const std::vector<f32>& key,
        const std::vector<f32>& value,
        usize batch_size,
        usize seq_len,
        const std::optional<std::vector<f32>>& mask = std::nullopt
    ) override {
        Timer timer;
        
        usize d_k = config_.head_dim;
        usize block_size = config_.block_size;
        f32 scale = config_.get_scale();
        
        Eigen::Map<const Matrix> Q(query.data(), seq_len, d_k);
        Eigen::Map<const Matrix> K(key.data(), seq_len, d_k);
        Eigen::Map<const Matrix> V(value.data(), seq_len, d_k);
        
        Matrix output = Matrix::Zero(seq_len, d_k);
        Vector m = Vector::Constant(seq_len, -std::numeric_limits<f32>::infinity());
        Vector l = Vector::Zero(seq_len);
        
        // Process K/V in blocks
        for (usize j = 0; j < seq_len; j += block_size) {
            usize block_len = std::min(block_size, seq_len - j);
            
            Matrix K_j = K.middleRows(j, block_len);
            Matrix V_j = V.middleRows(j, block_len);
            
            // Compute scores for this block
            Matrix S = (Q * K_j.transpose()) * scale;
            
            // Apply causal mask
            if (config_.use_causal_mask) {
                for (usize i = 0; i < static_cast<usize>(S.rows()); ++i) {
                    for (usize k = 0; k < block_len; ++k) {
                        if (j + k > i) {
                            S(i, k) = -std::numeric_limits<f32>::infinity();
                        }
                    }
                }
            }
            
            // Online softmax update
            Vector m_j = S.rowwise().maxCoeff();
            Vector m_new = m.cwiseMax(m_j);
            
            Vector alpha = (m - m_new).array().exp();
            Vector beta = (m_j - m_new).array().exp();
            
            Matrix P = (S.colwise() - m_new).array().exp().matrix();
            
            Vector l_j = P.rowwise().sum();
            Vector l_new = alpha.cwiseProduct(l) + beta.cwiseProduct(l_j);
            
            output = (alpha.asDiagonal() * output) + (P * V_j);
            
            m = m_new;
            l = l_new;
        }
        
        // Final normalization
        for (i32 i = 0; i < output.rows(); ++i) {
            if (l(i) > 0) {
                output.row(i) /= l(i);
            }
        }
        
        stats_.total_tokens = batch_size * seq_len;
        stats_.compute_time_ms = timer.elapsed_ms();
        
        return std::vector<f32>(output.data(), output.data() + output.size());
    }
    
    AttentionStats get_stats() const override { return stats_; }
    void reset_stats() override { stats_ = {}; }

private:
    AttentionConfig config_;
    AttentionStats stats_;
};

/**
 * @brief Sparse attention with local + global pattern
 */
class SparseAttention : public IAttention {
public:
    SparseAttention(AttentionConfig config, usize local_window, usize global_tokens)
        : config_(std::move(config))
        , local_window_(local_window)
        , global_tokens_(global_tokens) {
        config_.validate();
    }
    
    std::vector<f32> forward(
        const std::vector<f32>& query,
        const std::vector<f32>& key,
        const std::vector<f32>& value,
        usize batch_size,
        usize seq_len,
        const std::optional<std::vector<f32>>& mask = std::nullopt
    ) override {
        Timer timer;
        
        usize d_k = config_.head_dim;
        f32 scale = config_.get_scale();
        
        Eigen::Map<const Matrix> Q(query.data(), seq_len, d_k);
        Eigen::Map<const Matrix> K(key.data(), seq_len, d_k);
        Eigen::Map<const Matrix> V(value.data(), seq_len, d_k);
        
        Matrix output = Matrix::Zero(seq_len, d_k);
        
        for (usize i = 0; i < seq_len; ++i) {
            std::vector<f32> weights(seq_len, -1e9f);
            
            for (usize j = 0; j < seq_len; ++j) {
                bool is_local = std::abs(static_cast<i64>(i) - static_cast<i64>(j)) 
                               <= static_cast<i64>(local_window_);
                bool is_global = j < global_tokens_;
                
                if (is_local || is_global) {
                    f32 score = 0.0f;
                    for (usize k = 0; k < d_k; ++k) {
                        score += Q(i, k) * K(j, k);
                    }
                    weights[j] = score * scale;
                }
            }
            
            // Softmax
            math::softmax_inplace(weights.data(), weights.size());
            
            // Weighted sum
            for (usize j = 0; j < seq_len; ++j) {
                for (usize d = 0; d < d_k; ++d) {
                    output(i, d) += weights[j] * V(j, d);
                }
            }
        }
        
        stats_.total_tokens = batch_size * seq_len;
        stats_.compute_time_ms = timer.elapsed_ms();
        
        return std::vector<f32>(output.data(), output.data() + output.size());
    }
    
    AttentionStats get_stats() const override { return stats_; }
    void reset_stats() override { stats_ = {}; }

private:
    AttentionConfig config_;
    usize local_window_;
    usize global_tokens_;
    AttentionStats stats_;
};

#endif // HAVE_EIGEN

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create attention instance based on config
 */
inline std::unique_ptr<IAttention> create_attention(const AttentionConfig& config) {
#ifdef HAVE_EIGEN
    if (config.use_flash) {
        return std::make_unique<FlashAttention>(config);
    }
    return std::make_unique<ScaledDotProductAttention>(config);
#else
    throw std::runtime_error("Eigen not available");
#endif
}

/**
 * @brief Create sparse attention
 */
inline std::unique_ptr<IAttention> create_sparse_attention(
    const AttentionConfig& config,
    usize local_window,
    usize global_tokens = 0
) {
#ifdef HAVE_EIGEN
    return std::make_unique<SparseAttention>(config, local_window, global_tokens);
#else
    throw std::runtime_error("Eigen not available");
#endif
}

} // namespace attention
} // namespace optimization_core












