#pragma once

/**
 * @file constants.hpp
 * @brief Constants for attention computation
 */

#include <limits>

namespace optimization_core {
namespace attention {
namespace constants {

/**
 * @brief Negative infinity value for masking
 */
constexpr float NEGATIVE_INFINITY = -std::numeric_limits<float>::infinity();

/**
 * @brief Small epsilon for numerical stability
 */
constexpr float EPSILON = 1e-9f;

/**
 * @brief Large negative value for masking (alternative to -inf)
 */
constexpr float MASK_VALUE = -1e9f;

} // namespace constants
} // namespace attention
} // namespace optimization_core








