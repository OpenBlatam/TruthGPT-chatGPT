"""
Loss Functions

Functions for creating and managing loss functions.

Provides utilities for creating appropriate loss functions based on task type
(classification vs regression) with proper validation and error handling.
"""

include("constants.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION CREATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    create_loss_function(model, loss_type=:crossentropy)

Create a loss function for the model.

# Arguments
- `model`: Flux model (Chain)
- `loss_type`: Type of loss (:crossentropy, :mse, :mae)

# Returns
- Loss function (x, y) -> loss_value

# Throws
- `ArgumentError` if loss_type is invalid

# Examples
```julia
model = Chain(Dense(784, 128), Dense(128, 10))
loss_fn = create_loss_function(model, :crossentropy)
loss = loss_fn(x_train, y_train)
```

# Notes
- Cross-entropy: For classification tasks (expects logits and one-hot or class indices)
- MSE: For regression tasks (squared error, sensitive to outliers)
- MAE: For regression tasks (absolute error, more robust to outliers)

# Algorithm
1. Validate loss_type
2. Create closure that:
   a. Computes model predictions: model(x)
   b. Applies appropriate loss function
   c. Returns scalar loss value
"""
function create_loss_function(model, loss_type::Symbol = LOSS_CROSSENTROPY)
    # Validate loss type
    valid_loss_types = [LOSS_CROSSENTROPY, LOSS_MSE, LOSS_MAE]
    if loss_type ∉ valid_loss_types
        throw(ArgumentError(
            "Unknown loss type: :$loss_type. Supported: $valid_loss_types"
        ))
    end
    
    # Create appropriate loss function based on type
    if loss_type == LOSS_CROSSENTROPY
        return create_crossentropy_loss(model)
    elseif loss_type == LOSS_MSE
        return create_mse_loss(model)
    else  # LOSS_MAE
        return create_mae_loss(model)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIFIC LOSS FUNCTION CREATORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    create_crossentropy_loss(model)

Create cross-entropy loss function for classification.

# Arguments
- `model`: Flux model

# Returns
- Loss function (x, y) -> crossentropy(model(x), y)

# Notes
- Expects model output to be logits (before softmax)
- y can be one-hot encoded or class indices
- Suitable for multi-class classification tasks
"""
function create_crossentropy_loss(model)
    return (x, y) -> Flux.crossentropy(model(x), y)
end

"""
    create_mse_loss(model)

Create mean squared error loss function for regression.

# Arguments
- `model`: Flux model

# Returns
- Loss function (x, y) -> mse(model(x), y)

# Notes
- Mean squared error: mean((pred - target)²)
- Sensitive to outliers (penalizes large errors more)
- Suitable for regression tasks with normally distributed errors
"""
function create_mse_loss(model)
    return (x, y) -> Flux.mse(model(x), y)
end

"""
    create_mae_loss(model)

Create mean absolute error loss function for regression.

# Arguments
- `model`: Flux model

# Returns
- Loss function (x, y) -> mae(model(x), y)

# Notes
- Mean absolute error: mean(|pred - target|)
- More robust to outliers than MSE
- Suitable for regression tasks with outliers or heavy-tailed distributions
"""
function create_mae_loss(model)
    return (x, y) -> Flux.mae(model(x), y)
end

