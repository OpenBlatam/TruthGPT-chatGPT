"""
FluxML Types

Type definitions for FluxML module.

Provides structured configuration types for training and model management
with comprehensive validation and sensible defaults.
"""

include("constants.jl")
include("validation.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TrainingConfig

Configuration for model training with comprehensive options.

# Fields
- `epochs::Int`: Number of training epochs (must be positive)
- `learning_rate::Float64`: Learning rate for optimizer (must be positive)
- `batch_size::Int`: Batch size for training (must be positive)
- `device::Symbol`: Device to use (:cpu or :gpu)
- `verbose::Bool`: Print training progress
- `loss_type::Symbol`: Type of loss function (:crossentropy, :mse, :mae)
- `optimizer_type::Symbol`: Type of optimizer (:adam, :sgd, :rmsprop)

# Examples
```julia
# Default configuration
config = TrainingConfig()

# Custom configuration
config = TrainingConfig(
    epochs=50,
    learning_rate=0.01,
    batch_size=32,
    device=:gpu,
    verbose=true,
    loss_type=:crossentropy,
    optimizer_type=:adam
)

# Minimal configuration (only specify what differs from defaults)
config = TrainingConfig(epochs=100, learning_rate=0.001)
```

# Notes
- All fields are validated during construction
- Sensible defaults are provided for all parameters
- Device validation includes automatic fallback to CPU if GPU unavailable
- Loss and optimizer types are validated against supported options

# Algorithm
1. Validate all parameters using helper functions
2. Create and return validated TrainingConfig instance
"""
struct TrainingConfig
    epochs::Int
    learning_rate::Float64
    batch_size::Int
    device::Symbol
    verbose::Bool
    loss_type::Symbol
    optimizer_type::Symbol
    
    function TrainingConfig(
        epochs::Int = DEFAULT_EPOCHS,
        learning_rate::Float64 = DEFAULT_LEARNING_RATE,
        batch_size::Int = DEFAULT_BATCH_SIZE,
        device::Symbol = DEVICE_CPU,
        verbose::Bool = true,
        loss_type::Symbol = LOSS_CROSSENTROPY,
        optimizer_type::Symbol = OPTIMIZER_ADAM
    )
        # Validate all parameters using helper functions
        validate_epochs(epochs)
        validate_learning_rate(learning_rate)
        validate_batch_size(batch_size)
        validate_device(device)
        validate_loss_type(loss_type)
        validate_optimizer_type(optimizer_type)
        
        new(epochs, learning_rate, batch_size, device, verbose, loss_type, optimizer_type)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS FOR TrainingConfig
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_epochs(epochs)

Validate number of training epochs.

# Arguments
- `epochs`: Number of epochs to validate

# Throws
- `ArgumentError` if epochs is invalid
"""
function validate_epochs(epochs::Int)
    if epochs <= 0
        throw(ArgumentError("epochs must be positive, got $epochs"))
    end
end

"""
    validate_batch_size(batch_size)

Validate batch size for training.

# Arguments
- `batch_size`: Batch size to validate

# Throws
- `ArgumentError` if batch_size is invalid
"""
function validate_batch_size(batch_size::Int)
    if batch_size <= 0
        throw(ArgumentError("batch_size must be positive, got $batch_size"))
    end
end

"""
    validate_device(device)

Validate device specification.

# Arguments
- `device`: Device symbol to validate (:cpu or :gpu)

# Throws
- `ArgumentError` if device is invalid
"""
function validate_device(device::Symbol)
    valid_devices = [DEVICE_CPU, DEVICE_GPU]
    if device ∉ valid_devices
        throw(ArgumentError("device must be one of $valid_devices, got :$device"))
    end
end

"""
    validate_loss_type(loss_type)

Validate loss function type.

# Arguments
- `loss_type`: Loss type symbol to validate

# Throws
- `ArgumentError` if loss_type is invalid
"""
function validate_loss_type(loss_type::Symbol)
    valid_loss_types = [LOSS_CROSSENTROPY, LOSS_MSE, LOSS_MAE]
    if loss_type ∉ valid_loss_types
        throw(ArgumentError(
            "Invalid loss type: :$loss_type. Supported: $valid_loss_types"
        ))
    end
end

"""
    validate_optimizer_type(optimizer_type)

Validate optimizer type.

# Arguments
- `optimizer_type`: Optimizer type symbol to validate

# Throws
- `ArgumentError` if optimizer_type is invalid
"""
function validate_optimizer_type(optimizer_type::Symbol)
    valid_optimizer_types = [OPTIMIZER_ADAM, OPTIMIZER_SGD, OPTIMIZER_RMSPROP]
    if optimizer_type ∉ valid_optimizer_types
        throw(ArgumentError(
            "Invalid optimizer type: :$optimizer_type. Supported: $valid_optimizer_types"
        ))
    end
end



