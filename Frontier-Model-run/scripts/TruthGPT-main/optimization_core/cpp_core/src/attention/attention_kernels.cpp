/**
 * @file attention_kernels.cpp
 * @brief CPU-optimized attention kernels using Eigen and SIMD
 * 
 * Provides high-performance attention computation for CPU inference,
 * with support for multi-threading via TBB/OpenMP.
 */

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "attention/parallel_utils.hpp"
#include "attention/head_utils.hpp"
#include "attention/math_utils.hpp"
#include "attention/mask_utils.hpp"
#include "attention/matrix_utils.hpp"
#include "attention/flash_softmax.hpp"
#include "attention/constants.hpp"

namespace optimization_core {
namespace attention {

/**
 * @brief Softmax with numerical stability (online algorithm)
 */
template<typename T>
void safe_softmax(T* data, int size) {
    // Find max for numerical stability
    T max_val = *std::max_element(data, data + size);
    
    // Compute exp(x - max) and sum
    T sum = 0;
    for (int i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    
    // Normalize
    T inv_sum = T(1) / sum;
    for (int i = 0; i < size; ++i) {
        data[i] *= inv_sum;
    }
}

#ifdef HAVE_EIGEN
/**
 * @brief Vectorized softmax using Eigen
 */
void eigen_softmax(Eigen::Ref<Eigen::VectorXf> vec) {
    float max_val = vec.maxCoeff();
    vec = (vec.array() - max_val).exp();
    vec /= vec.sum();
}

/**
 * @brief Batch softmax for attention scores
 */
void batch_softmax(Eigen::Ref<Eigen::MatrixXf> scores) {
    // Apply softmax to each row
    parallel::parallel_for(0, scores.rows(), [&](int i) {
        Eigen::VectorXf row = scores.row(i);
        eigen_softmax(row);
        scores.row(i) = row;
    });
}

/**
 * @brief Scaled dot-product attention (CPU optimized)
 * 
 * Computes: softmax(QK^T / sqrt(d_k)) V
 * 
 * @param Q Query matrix [seq_q, head_dim]
 * @param K Key matrix [seq_k, head_dim]
 * @param V Value matrix [seq_k, head_dim]
 * @param scale Scaling factor (1/sqrt(head_dim))
 * @param mask Optional attention mask [seq_q, seq_k]
 * @return Output matrix [seq_q, head_dim]
 */
Eigen::MatrixXf scaled_dot_product_attention(
    const Eigen::Ref<const Eigen::MatrixXf>& Q,
    const Eigen::Ref<const Eigen::MatrixXf>& K,
    const Eigen::Ref<const Eigen::MatrixXf>& V,
    float scale,
    const Eigen::Ref<const Eigen::MatrixXf>* mask = nullptr
) {
    // Compute attention scores: QK^T / sqrt(d_k)
    Eigen::MatrixXf scores = (Q * K.transpose()) * scale;
    
    // Apply mask if provided
    if (mask != nullptr) {
        scores = scores.array() + (1.0f - mask->array()) * (-1e9f);
    }
    
    // Apply softmax
    batch_softmax(scores);
    
    // Compute output: scores * V
    return scores * V;
}

/**
 * @brief Multi-head attention with parallel head processing
 */
Eigen::MatrixXf multi_head_attention(
    const Eigen::Ref<const Eigen::MatrixXf>& Q,  // [seq, d_model]
    const Eigen::Ref<const Eigen::MatrixXf>& K,  // [seq, d_model]
    const Eigen::Ref<const Eigen::MatrixXf>& V,  // [seq, d_model]
    const Eigen::Ref<const Eigen::MatrixXf>& Wq, // [d_model, d_model]
    const Eigen::Ref<const Eigen::MatrixXf>& Wk, // [d_model, d_model]
    const Eigen::Ref<const Eigen::MatrixXf>& Wv, // [d_model, d_model]
    const Eigen::Ref<const Eigen::MatrixXf>& Wo, // [d_model, d_model]
    int n_heads,
    const Eigen::Ref<const Eigen::MatrixXf>* mask = nullptr
) {
    int seq_len = Q.rows();
    int d_model = Q.cols();
    int head_dim = d_model / n_heads;
    float scale = math::compute_attention_scale(head_dim);
    
    // Linear projections
    Eigen::MatrixXf Q_proj = Q * Wq;
    Eigen::MatrixXf K_proj = K * Wk;
    Eigen::MatrixXf V_proj = V * Wv;
    
    // Output accumulator
    Eigen::MatrixXf output = matrix::zeros(seq_len, d_model);
    
    // Process each head in parallel
    parallel::parallel_for(0, n_heads, [&](int h) {
        // Extract head slices
        auto slices = head::extract_head_slices(Q_proj, K_proj, V_proj, h, head_dim);
        
        // Compute attention for this head
        Eigen::MatrixXf head_output = scaled_dot_product_attention(
            slices.Q_h, slices.K_h, slices.V_h, scale, mask);
        
        // Write to output (thread-safe: different columns)
        head::write_head_output(output, head_output, h, head_dim);
    });
    
    // Output projection
    return output * Wo;
}

/**
 * @brief Flash Attention block-wise computation
 * 
 * Memory-efficient attention that processes in blocks to avoid
 * materializing the full attention matrix.
 */
Eigen::MatrixXf flash_attention_block(
    const Eigen::Ref<const Eigen::MatrixXf>& Q,  // [seq_q, head_dim]
    const Eigen::Ref<const Eigen::MatrixXf>& K,  // [seq_k, head_dim]
    const Eigen::Ref<const Eigen::MatrixXf>& V,  // [seq_k, head_dim]
    float scale,
    int block_size = 64,
    bool causal = false
) {
    int seq_q = Q.rows();
    int seq_k = K.rows();
    int head_dim = Q.cols();
    
    // Initialize output and running statistics
    Eigen::MatrixXf O = matrix::zeros(seq_q, head_dim);
    flash::OnlineSoftmaxStats stats(seq_q);
    
    // Process K/V in blocks
    for (int j = 0; j < seq_k; j += block_size) {
        int block_len = std::min(block_size, seq_k - j);
        
        // Extract block
        Eigen::MatrixXf K_j = K.middleRows(j, block_len);
        Eigen::MatrixXf V_j = V.middleRows(j, block_len);
        
        // Compute attention scores for this block: Q * K_j^T
        Eigen::MatrixXf S_j = (Q * K_j.transpose()) * scale;
        
        // Apply causal mask if needed
        if (causal) {
            mask::apply_causal_mask_block(S_j, j, seq_q, block_len);
        }
        
        // Online softmax update
        stats = flash::update_online_softmax(S_j, V_j, stats, O);
    }
    
    // Final normalization
    flash::normalize_output(O, stats.l);
    
    return O;
}

/**
 * @brief Grouped Query Attention (GQA) for efficient inference
 * 
 * Uses fewer KV heads than query heads, reducing memory bandwidth.
 */
Eigen::MatrixXf grouped_query_attention(
    const Eigen::Ref<const Eigen::MatrixXf>& Q,  // [seq, n_heads * head_dim]
    const Eigen::Ref<const Eigen::MatrixXf>& K,  // [seq, n_kv_heads * head_dim]
    const Eigen::Ref<const Eigen::MatrixXf>& V,  // [seq, n_kv_heads * head_dim]
    int n_heads,
    int n_kv_heads,
    float scale
) {
    int seq_len = Q.rows();
    int head_dim = Q.cols() / n_heads;
    int heads_per_group = n_heads / n_kv_heads;
    
    Eigen::MatrixXf output = matrix::zeros(seq_len, n_heads * head_dim);
    
    // Process each KV head group
    parallel::parallel_for(0, n_kv_heads, [&](int kv_h) {
        // Get KV for this group
        auto K_h = head::extract_head_slice(K, kv_h, head_dim);
        auto V_h = head::extract_head_slice(V, kv_h, head_dim);
        
        // Process all query heads in this group
        for (int g = 0; g < heads_per_group; ++g) {
            int q_h = kv_h * heads_per_group + g;
            auto Q_h = head::extract_head_slice(Q, q_h, head_dim);
            
            // Compute attention
            Eigen::MatrixXf head_out = scaled_dot_product_attention(
                Q_h, K_h, V_h, scale);
            
            head::write_head_output(output, head_out, q_h, head_dim);
        }
    });
    
    return output;
}

#endif // HAVE_EIGEN

/**
 * @brief SIMD-optimized softmax for float arrays
 */
void simd_softmax(float* data, int size) {
    #if defined(__AVX512F__)
    // AVX-512 implementation
    __m512 max_vec = _mm512_set1_ps(constants::NEGATIVE_INFINITY);
    
    // Find max
    int i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(data + i);
        max_vec = _mm512_max_ps(max_vec, v);
    }
    float max_val = _mm512_reduce_max_ps(max_vec);
    for (; i < size; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    
    // Compute exp and sum
    __m512 sum_vec = _mm512_setzero_ps();
    __m512 max_broadcast = _mm512_set1_ps(max_val);
    
    for (i = 0; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(data + i);
        v = _mm512_sub_ps(v, max_broadcast);
        // Note: Would need to implement or call exp approximation
        // For now, fall back to scalar
    }
    
    // Fall back to scalar for exp
    safe_softmax(data, size);
    
    #elif defined(__AVX2__)
    // AVX2 implementation would go here
    safe_softmax(data, size);
    
    #else
    safe_softmax(data, size);
    #endif
}

} // namespace attention
} // namespace optimization_core





