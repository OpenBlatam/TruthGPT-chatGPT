#pragma once

/**
 * @file head_utils.hpp
 * @brief Utilities for extracting and working with attention head slices
 */

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

namespace optimization_core {
namespace attention {
namespace head {

#ifdef HAVE_EIGEN

/**
 * @brief Structure holding head slices for Q, K, V
 * 
 * Provides convenient access to head slices extracted from
 * projected query, key, and value matrices.
 */
struct HeadSlices {
    Eigen::Ref<Eigen::MatrixXf> Q_h;
    Eigen::Ref<Eigen::MatrixXf> K_h;
    Eigen::Ref<Eigen::MatrixXf> V_h;
    
    /**
     * @brief Construct head slices from projected matrices
     * 
     * @param Q_proj Projected query matrix [seq, d_model]
     * @param K_proj Projected key matrix [seq, d_model]
     * @param V_proj Projected value matrix [seq, d_model]
     * @param head_idx Head index (0-based)
     * @param head_dim Head dimension
     */
    HeadSlices(
        Eigen::MatrixXf& Q_proj,
        Eigen::MatrixXf& K_proj,
        Eigen::MatrixXf& V_proj,
        int head_idx,
        int head_dim
    ) : Q_h(Q_proj.middleCols(head_idx * head_dim, head_dim)),
        K_h(K_proj.middleCols(head_idx * head_dim, head_dim)),
        V_h(V_proj.middleCols(head_idx * head_dim, head_dim)) {}
};

/**
 * @brief Extract head slices from projected matrices
 * 
 * @param Q_proj Projected query matrix
 * @param K_proj Projected key matrix
 * @param V_proj Projected value matrix
 * @param head_idx Head index
 * @param head_dim Head dimension
 * @return HeadSlices structure with references to head slices
 */
inline HeadSlices extract_head_slices(
    Eigen::MatrixXf& Q_proj,
    Eigen::MatrixXf& K_proj,
    Eigen::MatrixXf& V_proj,
    int head_idx,
    int head_dim
) {
    return HeadSlices(Q_proj, K_proj, V_proj, head_idx, head_dim);
}

/**
 * @brief Extract single head slice from matrix
 * 
 * @param matrix Input matrix [seq, d_model]
 * @param head_idx Head index
 * @param head_dim Head dimension
 * @return Reference to head slice [seq, head_dim]
 */
inline Eigen::Ref<Eigen::MatrixXf> extract_head_slice(
    Eigen::MatrixXf& matrix,
    int head_idx,
    int head_dim
) {
    return matrix.middleCols(head_idx * head_dim, head_dim);
}

/**
 * @brief Write head output back to output matrix
 * 
 * @param output Output matrix [seq, d_model]
 * @param head_output Head output [seq, head_dim]
 * @param head_idx Head index
 * @param head_dim Head dimension
 */
inline void write_head_output(
    Eigen::MatrixXf& output,
    const Eigen::Ref<const Eigen::MatrixXf>& head_output,
    int head_idx,
    int head_dim
) {
    output.middleCols(head_idx * head_dim, head_dim) = head_output;
}

#endif // HAVE_EIGEN

} // namespace head
} // namespace attention
} // namespace optimization_core








