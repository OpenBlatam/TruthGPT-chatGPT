"""
Validation Helpers

Functions for validating model architectures, training data, and inputs.

Provides comprehensive validation for all inputs to ensure correct
functionality and helpful error messages.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_model_architecture(input_size, hidden_sizes, output_size)

Validate model architecture parameters.

# Arguments
- `input_size`: Size of input layer (must be positive)
- `hidden_sizes`: Vector of hidden layer sizes (must be non-empty, all positive)
- `output_size`: Size of output layer (must be positive)

# Throws
- `ArgumentError` if any parameter is invalid

# Examples
```julia
validate_model_architecture(784, [128, 64], 10)  # Valid
validate_model_architecture(0, [128], 10)        # Throws: input_size must be positive
validate_model_architecture(784, [], 10)          # Throws: hidden_sizes cannot be empty
```

# Notes
- All sizes must be positive integers
- hidden_sizes cannot be empty
- Validates complete architecture before model creation
"""
function validate_model_architecture(
    input_size::Int,
    hidden_sizes::Vector{Int},
    output_size::Int
)
    # Validate input size
    if input_size <= 0
        throw(ArgumentError("input_size must be positive, got $input_size"))
    end
    
    # Validate hidden sizes
    if isempty(hidden_sizes)
        throw(ArgumentError("hidden_sizes cannot be empty"))
    end
    
    invalid_sizes = [s for s in hidden_sizes if s <= 0]
    if !isempty(invalid_sizes)
        throw(ArgumentError(
            "All hidden layer sizes must be positive. Invalid sizes: $invalid_sizes"
        ))
    end
    
    # Validate output size
    if output_size <= 0
        throw(ArgumentError("output_size must be positive, got $output_size"))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_training_data(train_data)

Validate training data structure and content.

# Arguments
- `train_data`: Tuple of (X, y) training data

# Throws
- `ArgumentError` or `DimensionMismatch` if data is invalid

# Examples
```julia
X = randn(Float32, 784, 1000)  # 784 features, 1000 samples
y = rand(1:10, 1000)           # 1000 labels
validate_training_data((X, y))  # Valid

X = randn(Float32, 784, 1000)
y = rand(1:10, 500)             # Mismatched samples
validate_training_data((X, y))  # Throws: DimensionMismatch
```

# Notes
- Checks that X and y are not empty
- Verifies that number of samples match between X and y
- Validates that data types are numeric
- Uses last dimension for sample count (Flux convention)

# Algorithm
1. Check X is not empty
2. Check y is not empty
3. Validate X is numeric
4. Validate y is numeric
5. Check sample counts match (last dimension)
"""
function validate_training_data(train_data::Tuple{Array, Array})
    x_train, y_train = train_data
    
    # Validate that data is not empty
    if isempty(x_train)
        throw(ArgumentError("Training features cannot be empty"))
    end
    
    if isempty(y_train)
        throw(ArgumentError("Training labels cannot be empty"))
    end
    
    # Check that data types are numeric
    if !(eltype(x_train) <: Number)
        throw(ArgumentError(
            "Training features must be numeric, got $(eltype(x_train))"
        ))
    end
    
    if !(eltype(y_train) <: Number)
        throw(ArgumentError(
            "Training labels must be numeric, got $(eltype(y_train))"
        ))
    end
    
    # Check that data shapes are compatible
    # Get number of samples from last dimension (Flux convention)
    x_samples = size(x_train, ndims(x_train))
    y_samples = size(y_train, ndims(y_train))
    
    if x_samples != y_samples
        throw(DimensionMismatch(
            "Number of samples in X ($x_samples) and y ($y_samples) must match"
        ))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_prediction_input(model, x)

Validate input for prediction.

# Arguments
- `model`: Flux model (Chain)
- `x`: Input data

# Throws
- `ArgumentError` if input is invalid

# Examples
```julia
model = Chain(Dense(784, 128), Dense(128, 10))
x = randn(Float32, 784, 100)
validate_prediction_input(model, x)  # Valid

x = []
validate_prediction_input(model, x)   # Throws: Input data cannot be empty
```

# Notes
- Validates that input is not empty
- Validates that input is numeric
- Model input dimension validation is handled by Flux during forward pass
- This function provides early validation for better error messages
"""
function validate_prediction_input(model::Chain, x::Array)
    # Validate that input is not empty
    if isempty(x)
        throw(ArgumentError("Input data cannot be empty"))
    end
    
    # Validate that input is numeric
    if !(eltype(x) <: Number)
        throw(ArgumentError(
            "Input data must be numeric, got $(eltype(x))"
        ))
    end
    
    # Note: Model input dimension validation would require knowing model architecture
    # This is typically handled by Flux during forward pass with more specific errors
end

"""
    validate_batch_size(batch_size)

Validate batch size parameter.

# Arguments
- `batch_size`: Batch size to validate

# Throws
- `ArgumentError` if batch_size is invalid

# Examples
```julia
validate_batch_size(32)   # Valid
validate_batch_size(0)    # Throws: batch_size must be positive
validate_batch_size(-1)   # Throws: batch_size must be positive
```
"""
function validate_batch_size(batch_size::Int)
    if batch_size <= 0
        throw(ArgumentError("batch_size must be positive, got $batch_size"))
    end
end

"""
    validate_learning_rate(learning_rate)

Validate learning rate parameter.

# Arguments
- `learning_rate`: Learning rate to validate

# Throws
- `ArgumentError` if learning_rate is invalid

# Examples
```julia
validate_learning_rate(0.001)  # Valid
validate_learning_rate(0.0)    # Throws: learning_rate must be positive
validate_learning_rate(-0.001) # Throws: learning_rate must be positive
```

# Notes
- Warns if learning_rate > 1.0 (unusually high)
- Typical learning rates are in [1e-5, 1e-1]
"""
function validate_learning_rate(learning_rate::Float64)
    if learning_rate <= 0.0
        throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
    end
    if !isfinite(learning_rate)
        throw(ArgumentError("learning_rate must be finite, got $learning_rate"))
    end
    if learning_rate > 1.0
        @warn "Learning rate $learning_rate is unusually high. Typical values are in [1e-5, 1e-1]"
    end
end
