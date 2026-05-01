"""
Optimizers

Functions for creating and managing optimizers.

Provides utilities for creating appropriate optimizers with validation
and automatic parameter tuning recommendations.
"""

include("constants.jl")
include("validation.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER CREATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    create_optimizer(learning_rate, optimizer_type=:adam)

Create an optimizer for training.

# Arguments
- `learning_rate`: Learning rate (must be positive)
- `optimizer_type`: Type of optimizer (:adam, :sgd, :rmsprop)

# Returns
- Optimizer instance (Adam, Descent, or RMSProp)

# Throws
- `ArgumentError` if learning_rate is invalid or optimizer_type is unsupported

# Examples
```julia
# Default Adam optimizer
opt = create_optimizer(0.001)

# SGD optimizer
opt = create_optimizer(0.01, :sgd)

# RMSProp optimizer
opt = create_optimizer(0.001, :rmsprop)
```

# Notes
- Adam: Adaptive moment estimation (recommended default)
  - Combines benefits of AdaGrad and RMSProp
  - Good default for most deep learning tasks
  - Automatically adapts learning rate per parameter
- SGD: Stochastic gradient descent (simple, fast)
  - Simple and fast, but may require more tuning
  - Good for convex problems or when memory is limited
- RMSProp: Root mean square propagation (good for RNNs)
  - Good for non-stationary objectives
  - Often used in recurrent neural networks

# Algorithm
1. Validate learning_rate (must be positive, warn if unusually high)
2. Validate optimizer_type
3. Create and return appropriate optimizer instance
"""
function create_optimizer(learning_rate::Float64, optimizer_type::Symbol = OPTIMIZER_ADAM)
    # Validate learning rate
    validate_learning_rate(learning_rate)
    
    # Validate optimizer type
    valid_optimizer_types = [OPTIMIZER_ADAM, OPTIMIZER_SGD, OPTIMIZER_RMSPROP]
    if optimizer_type ∉ valid_optimizer_types
        throw(ArgumentError(
            "Unknown optimizer type: :$optimizer_type. Supported: $valid_optimizer_types"
        ))
    end
    
    # Create appropriate optimizer
    if optimizer_type == OPTIMIZER_ADAM
        return create_adam_optimizer(learning_rate)
    elseif optimizer_type == OPTIMIZER_SGD
        return create_sgd_optimizer(learning_rate)
    else  # OPTIMIZER_RMSPROP
        return create_rmsprop_optimizer(learning_rate)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIFIC OPTIMIZER CREATORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    create_adam_optimizer(learning_rate)

Create Adam optimizer (adaptive moment estimation).

# Arguments
- `learning_rate`: Learning rate

# Returns
- Adam optimizer instance

# Notes
- Combines benefits of AdaGrad and RMSProp
- Automatically adapts learning rate per parameter
- Recommended default for most deep learning tasks
"""
function create_adam_optimizer(learning_rate::Float64)
    return Adam(learning_rate)
end

"""
    create_sgd_optimizer(learning_rate)

Create SGD optimizer (stochastic gradient descent).

# Arguments
- `learning_rate`: Learning rate

# Returns
- Descent optimizer instance (Flux's SGD implementation)

# Notes
- Simple and fast gradient descent
- May require more hyperparameter tuning than Adam
- Good for convex problems or when memory is limited
"""
function create_sgd_optimizer(learning_rate::Float64)
    return Descent(learning_rate)
end

"""
    create_rmsprop_optimizer(learning_rate)

Create RMSProp optimizer (root mean square propagation).

# Arguments
- `learning_rate`: Learning rate

# Returns
- RMSProp optimizer instance

# Notes
- Good for non-stationary objectives
- Often used in recurrent neural networks
- Adapts learning rate based on moving average of squared gradients
"""
function create_rmsprop_optimizer(learning_rate::Float64)
    return RMSProp(learning_rate)
end


