#pragma once

/**
 * @file weight_decay.hpp
 * @brief Weight decay application utilities
 */

namespace optimization_core {
namespace optim {
namespace weight_decay {

/**
 * @brief Apply weight decay to parameter (AdamW/Lion style)
 * 
 * Applies: param -= lr * weight_decay * param
 * 
 * This is the standard weight decay used in AdamW and Lion optimizers,
 * where weight decay is applied directly to parameters.
 * 
 * @param param Parameter to update (modified in-place)
 * @param lr Learning rate
 * @param weight_decay Weight decay coefficient
 */
inline void apply_to_param(float& param, float lr, float weight_decay) {
    if (weight_decay > 0) {
        param -= lr * weight_decay * param;
    }
}

/**
 * @brief Apply weight decay to gradient (SGD style)
 * 
 * Applies: grad += weight_decay * param
 * 
 * This is the weight decay used in SGD optimizer,
 * where weight decay is applied to gradients before momentum.
 * 
 * @param grad Gradient to update (modified in-place)
 * @param param Parameter value
 * @param weight_decay Weight decay coefficient
 */
inline void apply_to_gradient(float& grad, float param, float weight_decay) {
    if (weight_decay > 0) {
        grad += weight_decay * param;
    }
}

} // namespace weight_decay
} // namespace optim
} // namespace optimization_core








