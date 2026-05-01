#pragma once

/**
 * @file matrix_utils.hpp
 * @brief Matrix and vector initialization utilities
 */

#include <limits>
#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

namespace optimization_core {
namespace attention {
namespace matrix {

#ifdef HAVE_EIGEN

/**
 * @brief Create zero-initialized matrix
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Zero-initialized matrix
 */
inline Eigen::MatrixXf zeros(int rows, int cols) {
    return Eigen::MatrixXf::Zero(rows, cols);
}

/**
 * @brief Create zero-initialized vector
 * 
 * @param size Vector size
 * @return Zero-initialized vector
 */
inline Eigen::VectorXf zeros(int size) {
    return Eigen::VectorXf::Zero(size);
}

/**
 * @brief Create vector initialized with negative infinity
 * 
 * Useful for Flash Attention max tracking initialization.
 * 
 * @param size Vector size
 * @return Vector initialized with -infinity
 */
inline Eigen::VectorXf negative_infinity(int size) {
    return Eigen::VectorXf::Constant(
        size, 
        -std::numeric_limits<float>::infinity()
    );
}

/**
 * @brief Create matrix initialized with constant value
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @param value Constant value
 * @return Matrix initialized with constant
 */
inline Eigen::MatrixXf constant(int rows, int cols, float value) {
    return Eigen::MatrixXf::Constant(rows, cols, value);
}

#endif // HAVE_EIGEN

} // namespace matrix
} // namespace attention
} // namespace optimization_core








