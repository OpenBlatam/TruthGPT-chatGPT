#pragma once

/**
 * @file flash_softmax.hpp
 * @brief Flash Attention online softmax update utilities
 */

#include <limits>
#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

namespace optimization_core {
namespace attention {
namespace flash {

#ifdef HAVE_EIGEN

/**
 * @brief Statistics for online softmax computation
 */
struct OnlineSoftmaxStats {
    Eigen::VectorXf m;  // Running maximum
    Eigen::VectorXf l;  // Running normalization factor
    
    /**
     * @brief Initialize statistics for Flash Attention
     * 
     * @param size Sequence length (number of query positions)
     */
    OnlineSoftmaxStats(int size) 
        : m(Eigen::VectorXf::Constant(size, -std::numeric_limits<float>::infinity())),
          l(Eigen::VectorXf::Zero(size)) {}
};

/**
 * @brief Update online softmax statistics and output for a block
 * 
 * Implements the online softmax update from Flash Attention paper.
 * This allows computing softmax in blocks without materializing
 * the full attention matrix.
 * 
 * @param S_j Attention scores for current block [seq_q, block_len]
 * @param V_j Value matrix for current block [block_len, head_dim]
 * @param stats Current online softmax statistics
 * @param O Current output matrix [seq_q, head_dim] (updated in-place)
 * 
 * @return Updated statistics
 */
inline OnlineSoftmaxStats update_online_softmax(
    const Eigen::Ref<const Eigen::MatrixXf>& S_j,
    const Eigen::Ref<const Eigen::MatrixXf>& V_j,
    const OnlineSoftmaxStats& stats,
    Eigen::Ref<Eigen::MatrixXf> O
) {
    int seq_q = S_j.rows();
    
    // Compute max for this block
    Eigen::VectorXf m_j = S_j.rowwise().maxCoeff();
    Eigen::VectorXf m_new = stats.m.cwiseMax(m_j);
    
    // Rescale factors
    Eigen::VectorXf alpha = (stats.m - m_new).array().exp();
    Eigen::VectorXf beta = (m_j - m_new).array().exp();
    
    // Compute probabilities: P_j = exp(S_j - m_new)
    Eigen::MatrixXf P_j = (S_j.colwise() - m_new).array().exp().matrix();
    
    // Update normalization factor
    Eigen::VectorXf l_j = P_j.rowwise().sum();
    Eigen::VectorXf l_new = alpha.cwiseProduct(stats.l) + beta.cwiseProduct(l_j);
    
    // Update output: O = (alpha * O) + (P_j * V_j)
    O = (alpha.asDiagonal() * O) + (P_j * V_j);
    
    // Return updated statistics
    OnlineSoftmaxStats new_stats(seq_q);
    new_stats.m = m_new;
    new_stats.l = l_new;
    return new_stats;
}

/**
 * @brief Normalize output matrix using normalization factors
 * 
 * Safely normalizes each row by its normalization factor, avoiding
 * division by zero.
 * 
 * @param O Output matrix [seq_q, head_dim] (updated in-place)
 * @param l Normalization factors [seq_q]
 */
inline void normalize_output(
    Eigen::Ref<Eigen::MatrixXf> O,
    const Eigen::Ref<const Eigen::VectorXf>& l
) {
    for (int i = 0; i < O.rows(); ++i) {
        if (l(i) > 0) {
            O.row(i) /= l(i);
        }
    }
}

#endif // HAVE_EIGEN

} // namespace flash
} // namespace attention
} // namespace optimization_core








