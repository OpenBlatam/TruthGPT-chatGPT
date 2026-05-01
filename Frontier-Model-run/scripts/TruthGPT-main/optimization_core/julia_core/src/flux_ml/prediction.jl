"""
Prediction

Functions for making predictions with trained models.
"""

include("constants.jl")
include("device.jl")
include("validation.jl")

"""
    predict(model, x; device=:cpu)

Make predictions with model.

# Arguments
- `model`: Trained Flux model
- `x`: Input data
- `device`: Device to use (:cpu or :gpu, default: :cpu)

# Returns
- Model predictions (moved back to CPU if needed)

# Examples
```julia
model = create_model(784, [128, 64], 10)
X_test = randn(Float32, 784, 100)
predictions = predict(model, X_test)

# GPU prediction
predictions = predict(model, X_test, device=:gpu)
```
"""
function predict(model::Chain, x::Array; device::Symbol = DEVICE_CPU)
    # Validate input before processing
    validate_prediction_input(model, x)
    
    # Setup prediction environment (device, move data/model)
    actual_device, x_device, model_device = setup_prediction(model, x, device)
    
    # Make prediction (no gradient tracking needed for inference)
    predictions = compute_predictions(model_device, x_device)
    
    # Move back to CPU if needed (for compatibility with downstream code)
    return finalize_predictions(predictions, actual_device)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    setup_prediction(model, x, device)

Setup prediction environment by ensuring device availability and moving data/model.

# Arguments
- `model`: Flux model
- `x`: Input data
- `device`: Requested device

# Returns
- Tuple of (actual_device, x_device, model_device)
"""
function setup_prediction(model::Chain, x::Array, device::Symbol)
    # Ensure device is available (fallback to CPU if GPU unavailable)
    actual_device = ensure_device_available(device)
    
    # Move to device (with automatic fallback)
    x_device = move_to_device(x, actual_device)
    model_device = move_to_device(model, actual_device)
    
    return actual_device, x_device, model_device
end

"""
    compute_predictions(model, x)

Compute model predictions.

# Arguments
- `model`: Flux model (on appropriate device)
- `x`: Input data (on appropriate device)

# Returns
- Model predictions

# Notes
- No gradient tracking needed for inference
- Flux automatically handles eval mode for inference
"""
function compute_predictions(model::Chain, x::Array)
    return model(x)
end

"""
    finalize_predictions(predictions, device)

Finalize predictions by moving back to CPU if needed.

# Arguments
- `predictions`: Model predictions
- `device`: Device used for prediction

# Returns
- Predictions (on CPU if GPU was used)

# Notes
- Ensures predictions are always on CPU regardless of training device
- This provides compatibility with downstream code
"""
function finalize_predictions(predictions, device::Symbol)
    if device == DEVICE_GPU
        return predictions |> cpu
    end
    return predictions
end

