#pragma once

/**
 * @file mask_utils.hpp
 * @brief Utilities for applying attention masks
 */

#include <limits>

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

namespace optimization_core {
namespace attention {
namespace mask {

#ifdef HAVE_EIGEN

/**
 * @brief Apply causal mask to attention scores block
 * 
 * Sets attention scores to -infinity for positions where
 * key position > query position (causal masking).
 * 
 * @param scores Attention scores matrix [seq_q, block_len]
 * @param block_start Starting index of the block in sequence
 * @param seq_q Query sequence length
 * @param block_len Block length
 */
inline void apply_causal_mask_block(
    Eigen::Ref<Eigen::MatrixXf> scores,
    int block_start,
    int seq_q,
    int block_len
) {
    const float mask_value = -std::numeric_limits<float>::infinity();
    
    for (int i = 0; i < seq_q; ++i) {
        for (int k = 0; k < block_len; ++k) {
            if (block_start + k > i) {
                scores(i, k) = mask_value;
            }
        }
    }
}

/**
 * @brief Apply causal mask to full attention scores matrix
 * 
 * @param scores Attention scores matrix [seq_q, seq_k]
 * @param seq_q Query sequence length
 * @param seq_k Key sequence length
 */
inline void apply_causal_mask(
    Eigen::Ref<Eigen::MatrixXf> scores,
    int seq_q,
    int seq_k
) {
    const float mask_value = -std::numeric_limits<float>::infinity();
    
    for (int i = 0; i < seq_q; ++i) {
        for (int j = i + 1; j < seq_k; ++j) {
            scores(i, j) = mask_value;
        }
    }
}

#endif // HAVE_EIGEN

} // namespace mask
} // namespace attention
} // namespace optimization_core








