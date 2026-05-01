#pragma once

/**
 * @file math_utils.hpp
 * @brief Mathematical utilities for attention computation
 */

#include <cmath>

namespace optimization_core {
namespace attention {
namespace math {

/**
 * @brief Compute attention scale factor
 * 
 * The scale factor is 1/sqrt(head_dim) used in scaled dot-product attention
 * to prevent dot products from growing too large.
 * 
 * @param head_dim Head dimension
 * @return Scale factor: 1.0 / sqrt(head_dim)
 */
inline float compute_attention_scale(int head_dim) {
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

/**
 * @brief Compute attention scale factor (template version)
 * 
 * @tparam T Numeric type
 * @param head_dim Head dimension
 * @return Scale factor: 1.0 / sqrt(head_dim)
 */
template<typename T>
inline T compute_attention_scale(T head_dim) {
    return T(1) / std::sqrt(head_dim);
}

} // namespace math
} // namespace attention
} // namespace optimization_core








