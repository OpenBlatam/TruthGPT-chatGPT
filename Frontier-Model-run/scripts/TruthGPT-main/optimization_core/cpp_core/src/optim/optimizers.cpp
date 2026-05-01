/**
 * @file optimizers.cpp
 * @brief High-performance optimizers using ensmallen-style patterns
 * 
 * Provides fast implementations of common optimizers (Adam, AdamW, Lion)
 * with SIMD vectorization and multi-threading support.
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include "attention/parallel_utils.hpp"
#include "optim/optimizer_utils.hpp"
#include "optim/weight_decay.hpp"

namespace optimization_core {
namespace optim {

/**
 * @brief Optimizer state for momentum-based optimizers
 */
struct OptimizerState {
    std::vector<float> m;  // First moment (mean)
    std::vector<float> v;  // Second moment (variance)
    int64_t t = 0;         // Timestep
    
    void initialize(size_t size) {
        m.resize(size, 0.0f);
        v.resize(size, 0.0f);
        t = 0;
    }
    
    void reset() {
        std::fill(m.begin(), m.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        t = 0;
    }
};

/**
 * @brief Adam optimizer configuration
 */
struct AdamConfig {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
    bool amsgrad = false;
};

/**
 * @brief High-performance Adam optimizer
 * 
 * Implements Adam with optional weight decay (AdamW variant).
 * Uses SIMD vectorization when available.
 */
class Adam {
public:
    explicit Adam(const AdamConfig& config = AdamConfig())
        : config_(config) {}
    
    /**
     * @brief Perform single optimization step
     * 
     * @param params Parameter vector to update
     * @param grads Gradient vector
     * @param state Optimizer state (momentum buffers)
     */
    void step(std::vector<float>& params,
              const std::vector<float>& grads,
              OptimizerState& state) {
        
        utils::validate_same_size(params, grads);
        utils::ensure_state_initialized(state, params.size());
        
        state.t++;
        
        // Bias correction factors
        float bias_correction1 = 1.0f - std::pow(config_.beta1, state.t);
        float bias_correction2 = 1.0f - std::pow(config_.beta2, state.t);
        float step_size = config_.lr / bias_correction1;
        
        parallel::parallel_for(0, static_cast<int>(params.size()), [&](int i) {
            adam_update_single(params[i], grads[i], state.m[i], state.v[i],
                              step_size, bias_correction2);
        });
    }
    
#ifdef HAVE_EIGEN
    /**
     * @brief Vectorized step using Eigen
     */
    void step_eigen(Eigen::Ref<Eigen::VectorXf> params,
                    const Eigen::Ref<const Eigen::VectorXf>& grads,
                    OptimizerState& state) {
        
        utils::ensure_state_initialized(state, params.size());
        
        state.t++;
        
        // Map state vectors
        Eigen::Map<Eigen::VectorXf> m(state.m.data(), state.m.size());
        Eigen::Map<Eigen::VectorXf> v(state.v.data(), state.v.size());
        
        // Bias correction
        float bias_correction1 = 1.0f - std::pow(config_.beta1, state.t);
        float bias_correction2 = 1.0f - std::pow(config_.beta2, state.t);
        
        // Update biased first moment estimate
        m = config_.beta1 * m + (1.0f - config_.beta1) * grads;
        
        // Update biased second raw moment estimate
        v = config_.beta2 * v + (1.0f - config_.beta2) * grads.cwiseProduct(grads);
        
        // Compute bias-corrected estimates
        Eigen::VectorXf m_hat = m / bias_correction1;
        Eigen::VectorXf v_hat = v / bias_correction2;
        
        // Apply weight decay (AdamW)
        if (config_.weight_decay > 0) {
            params -= config_.lr * config_.weight_decay * params;
        }
        
        // Update parameters
        params -= config_.lr * m_hat.cwiseQuotient(
            v_hat.cwiseSqrt().array() + config_.eps);
    }
#endif

private:
    AdamConfig config_;
    
    void adam_update_single(float& param, float grad, float& m, float& v,
                           float step_size, float bias_correction2) {
        // Update biased first moment
        m = config_.beta1 * m + (1.0f - config_.beta1) * grad;
        
        // Update biased second moment
        v = config_.beta2 * v + (1.0f - config_.beta2) * grad * grad;
        
        // Compute bias-corrected second moment
        float v_hat = v / bias_correction2;
        
        // Apply weight decay (AdamW style)
        weight_decay::apply_to_param(param, config_.lr, config_.weight_decay);
        
        // Update parameter
        param -= step_size * m / (std::sqrt(v_hat) + config_.eps);
    }
};

/**
 * @brief Lion optimizer configuration
 */
struct LionConfig {
    float lr = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.99f;
    float weight_decay = 0.0f;
};

/**
 * @brief Lion optimizer (Chen et al., 2023)
 * 
 * A simpler optimizer that uses only sign operations,
 * resulting in uniform update magnitudes.
 */
class Lion {
public:
    explicit Lion(const LionConfig& config = LionConfig())
        : config_(config) {}
    
    void step(std::vector<float>& params,
              const std::vector<float>& grads,
              OptimizerState& state) {
        
        utils::ensure_state_initialized(state, params.size());
        
        parallel::parallel_for(0, static_cast<int>(params.size()), [&](int i) {
            lion_update_single(params[i], grads[i], state.m[i]);
        });
    }

private:
    LionConfig config_;
    
    void lion_update_single(float& param, float grad, float& m) {
        // Interpolated momentum for update direction
        float update = config_.beta1 * m + (1.0f - config_.beta1) * grad;
        
        // Apply weight decay
        weight_decay::apply_to_param(param, config_.lr, config_.weight_decay);
        
        // Sign-based update
        param -= config_.lr * sign(update);
        
        // Update momentum for next iteration
        m = config_.beta2 * m + (1.0f - config_.beta2) * grad;
    }
    
    static float sign(float x) {
        if (x > 0) return 1.0f;
        if (x < 0) return -1.0f;
        return 0.0f;
    }
};

/**
 * @brief SGD with momentum configuration
 */
struct SGDConfig {
    float lr = 1e-2f;
    float momentum = 0.9f;
    float weight_decay = 0.0f;
    bool nesterov = false;
};

/**
 * @brief SGD optimizer with momentum
 */
class SGD {
public:
    explicit SGD(const SGDConfig& config = SGDConfig())
        : config_(config) {}
    
    void step(std::vector<float>& params,
              const std::vector<float>& grads,
              OptimizerState& state) {
        
        utils::ensure_state_initialized(state, params.size(), config_.momentum > 0);
        
        parallel::parallel_for(0, static_cast<int>(params.size()), [&](int i) {
            sgd_update_single(params[i], grads[i], 
                             config_.momentum > 0 ? &state.m[i] : nullptr);
        });
    }

private:
    SGDConfig config_;
    
    void sgd_update_single(float& param, float grad, float* momentum_buf) {
        // Apply weight decay
        weight_decay::apply_to_gradient(grad, param, config_.weight_decay);
        
        if (config_.momentum > 0 && momentum_buf != nullptr) {
            // Update momentum buffer
            *momentum_buf = config_.momentum * (*momentum_buf) + grad;
            
            if (config_.nesterov) {
                grad += config_.momentum * (*momentum_buf);
            } else {
                grad = *momentum_buf;
            }
        }
        
        // Update parameter
        param -= config_.lr * grad;
    }
};

/**
 * @brief Learning rate scheduler base class
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(int64_t step, float base_lr) = 0;
};

/**
 * @brief Cosine annealing learning rate scheduler
 */
class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(int64_t total_steps, float min_lr = 0.0f)
        : total_steps_(total_steps), min_lr_(min_lr) {}
    
    float get_lr(int64_t step, float base_lr) override {
        if (step >= total_steps_) {
            return min_lr_;
        }
        float progress = static_cast<float>(step) / total_steps_;
        return min_lr_ + 0.5f * (base_lr - min_lr_) * 
               (1.0f + std::cos(M_PI * progress));
    }

private:
    int64_t total_steps_;
    float min_lr_;
};

/**
 * @brief Linear warmup scheduler
 */
class LinearWarmup : public LRScheduler {
public:
    LinearWarmup(int64_t warmup_steps)
        : warmup_steps_(warmup_steps) {}
    
    float get_lr(int64_t step, float base_lr) override {
        if (step >= warmup_steps_) {
            return base_lr;
        }
        return base_lr * static_cast<float>(step) / warmup_steps_;
    }

private:
    int64_t warmup_steps_;
};

/**
 * @brief Combined warmup + cosine annealing scheduler
 */
class WarmupCosineScheduler : public LRScheduler {
public:
    WarmupCosineScheduler(int64_t warmup_steps, int64_t total_steps, float min_lr = 0.0f)
        : warmup_steps_(warmup_steps), total_steps_(total_steps), min_lr_(min_lr) {}
    
    float get_lr(int64_t step, float base_lr) override {
        if (step < warmup_steps_) {
            // Linear warmup
            return base_lr * static_cast<float>(step) / warmup_steps_;
        }
        
        // Cosine annealing
        int64_t decay_steps = total_steps_ - warmup_steps_;
        int64_t decay_step = step - warmup_steps_;
        
        if (decay_step >= decay_steps) {
            return min_lr_;
        }
        
        float progress = static_cast<float>(decay_step) / decay_steps;
        return min_lr_ + 0.5f * (base_lr - min_lr_) *
               (1.0f + std::cos(M_PI * progress));
    }

private:
    int64_t warmup_steps_;
    int64_t total_steps_;
    float min_lr_;
};

} // namespace optim
} // namespace optimization_core





