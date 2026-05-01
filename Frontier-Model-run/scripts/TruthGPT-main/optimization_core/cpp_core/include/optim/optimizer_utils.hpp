#pragma once

/**
 * @file optimizer_utils.hpp
 * @brief Common utilities for optimizers
 */

#include <vector>
#include <stdexcept>
#include <string>

namespace optimization_core {
namespace optim {
namespace utils {

/**
 * @brief Ensure optimizer state is initialized
 * 
 * Checks if state is empty and initializes it if needed.
 * 
 * @param state Optimizer state to check
 * @param size Required size for state
 * @param condition Optional condition (default: always initialize if empty)
 */
template<typename State>
void ensure_state_initialized(
    State& state,
    size_t size,
    bool condition = true
) {
    if (condition && state.m.empty()) {
        state.initialize(size);
    }
}

/**
 * @brief Validate that params and grads have same size
 * 
 * @param params Parameter vector
 * @param grads Gradient vector
 * @throws std::invalid_argument if sizes don't match
 */
template<typename T>
void validate_same_size(
    const std::vector<T>& params,
    const std::vector<T>& grads
) {
    if (params.size() != grads.size()) {
        throw std::invalid_argument(
            "params and grads must have same size. "
            "Got params.size()=" + std::to_string(params.size()) + 
            ", grads.size()=" + std::to_string(grads.size())
        );
    }
}

} // namespace utils
} // namespace optim
} // namespace optimization_core








