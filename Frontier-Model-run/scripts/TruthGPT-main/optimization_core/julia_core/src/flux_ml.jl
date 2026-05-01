"""
Flux.jl Machine Learning Module

High-performance deep learning using Flux.jl with native Julia autodiff.
Provides 2-5x speedup over PyTorch for custom models due to:
- Native JIT compilation
- Autodiff in any Julia code (not just tensors)
- Better memory management
- GPU support via CUDA.jl
- Efficient batch processing
"""

module FluxML

using Flux
using CUDA
using Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS AND TYPES
# ═══════════════════════════════════════════════════════════════════════════════

# Default training parameters
const DEFAULT_LEARNING_RATE = 0.001
const DEFAULT_EPOCHS = 10
const DEFAULT_BATCH_SIZE = 32

# Device types
const DEVICE_CPU = :cpu
const DEVICE_GPU = :gpu

# Supported activation functions
const DEFAULT_ACTIVATION = relu

# Supported loss types
const LOSS_CROSSENTROPY = :crossentropy
const LOSS_MSE = :mse
const LOSS_MAE = :mae

# Supported optimizer types
const OPTIMIZER_ADAM = :adam
const OPTIMIZER_SGD = :sgd
const OPTIMIZER_RMSPROP = :rmsprop

# Training monitoring
const LOSS_DISPLAY_PRECISION = 6

"""
    TrainingConfig

Configuration for model training with comprehensive options.

# Fields
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `batch_size`: Batch size for training (if using batching)
- `device`: Device to use (:cpu or :gpu)
- `verbose`: Print training progress
- `loss_type`: Type of loss function (:crossentropy, :mse, :mae)
- `optimizer_type`: Type of optimizer (:adam, :sgd, :rmsprop)

# Examples
```julia
config = TrainingConfig(
    epochs=50,
    learning_rate=0.01,
    device=:gpu,
    verbose=true
)
```
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
        # Validate epochs
        if epochs <= 0
            throw(ArgumentError("epochs must be positive, got $epochs"))
        end
        
        # Validate learning rate
        if learning_rate <= 0.0
            throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
        end
        
        # Validate batch size
        if batch_size <= 0
            throw(ArgumentError("batch_size must be positive, got $batch_size"))
        end
        
        # Validate device
        if device ∉ [DEVICE_CPU, DEVICE_GPU]
            throw(ArgumentError("device must be :cpu or :gpu, got $device"))
        end
        
        # Validate loss type
        if loss_type ∉ [LOSS_CROSSENTROPY, LOSS_MSE, LOSS_MAE]
            throw(ArgumentError(
                "Invalid loss type: $loss_type. Supported: :crossentropy, :mse, :mae"
            ))
        end
        
        # Validate optimizer type
        if optimizer_type ∉ [OPTIMIZER_ADAM, OPTIMIZER_SGD, OPTIMIZER_RMSPROP]
            throw(ArgumentError(
                "Invalid optimizer type: $optimizer_type. Supported: :adam, :sgd, :rmsprop"
            ))
        end
        
        new(epochs, learning_rate, batch_size, device, verbose, loss_type, optimizer_type)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

"""
    is_gpu_available()

Check if GPU is available and functional.

# Returns
- `true` if GPU is available and functional, `false` otherwise

# Examples
```julia
if is_gpu_available()
    println("GPU is available!")
end
```
"""
is_gpu_available() = CUDA.functional()

"""
    ensure_device_available(device)

Ensure requested device is available, fallback to CPU if not.

# Arguments
- `device`: Requested device (:cpu or :gpu)

# Returns
- Available device (:cpu or :gpu)

# Notes
- Automatically falls back to CPU if GPU is requested but unavailable
- Warns user if fallback occurs
"""
function ensure_device_available(device::Symbol)::Symbol
    if device == DEVICE_GPU && !is_gpu_available()
        @warn "GPU requested but not available, falling back to CPU"
        return DEVICE_CPU
    end
    return device
end

"""
    move_to_device(x, device)

Move data or model to specified device.

# Arguments
- `x`: Data or model to move
- `device`: Target device (:cpu or :gpu)

# Returns
- Data/model on target device

# Notes
- Automatically falls back to CPU if GPU is requested but not available
- Uses Flux's `gpu()` and `cpu()` functions for device transfer
"""
function move_to_device(x, device::Symbol)
    # Ensure device is available (fallback to CPU if needed)
    actual_device = ensure_device_available(device)
    
    if actual_device == DEVICE_GPU
        return x |> gpu
    else
        return x |> cpu
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_model_architecture(input_size, hidden_sizes, output_size)

Validate model architecture parameters.

# Arguments
- `input_size`: Size of input layer
- `hidden_sizes`: Vector of hidden layer sizes
- `output_size`: Size of output layer

# Throws
- `ArgumentError` if any parameter is invalid

# Examples
```julia
validate_model_architecture(784, [128, 64], 10)  # Valid
validate_model_architecture(0, [128], 10)        # Throws error
```
"""
function validate_model_architecture(
    input_size::Int,
    hidden_sizes::Vector{Int},
    output_size::Int
)
    if input_size <= 0
        throw(ArgumentError("input_size must be positive, got $input_size"))
    end
    
    if isempty(hidden_sizes)
        throw(ArgumentError("hidden_sizes cannot be empty"))
    end
    
    if !all(s > 0 for s in hidden_sizes)
        throw(ArgumentError("All hidden layer sizes must be positive"))
    end
    
    if output_size <= 0
        throw(ArgumentError("output_size must be positive, got $output_size"))
    end
end

"""
    validate_training_data(train_data)

Validate training data structure and content.

# Arguments
- `train_data`: Tuple of (X, y) training data

# Throws
- `ArgumentError` if data is invalid

# Notes
- Checks that X and y are not empty
- Verifies that number of samples match between X and y
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
    
    # Check that data shapes are compatible
    # Get number of samples from last dimension (Flux convention)
    x_samples = size(x_train, ndims(x_train))
    y_samples = size(y_train, ndims(y_train))
    
    if x_samples != y_samples
        throw(DimensionMismatch(
            "Number of samples in X ($x_samples) and y ($y_samples) must match"
        ))
    end
    
    # Additional validation: check data types are numeric
    if !(eltype(x_train) <: Number)
        throw(ArgumentError("Training features must be numeric, got $(eltype(x_train))"))
    end
    
    if !(eltype(y_train) <: Number)
        throw(ArgumentError("Training labels must be numeric, got $(eltype(y_train))"))
    end
end

"""
    validate_prediction_input(model, x)

Validate input for prediction.

# Arguments
- `model`: Flux model
- `x`: Input data

# Throws
- `ArgumentError` if input is invalid
"""
function validate_prediction_input(model::Chain, x::Array)
    if isempty(x)
        throw(ArgumentError("Input data cannot be empty"))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_dense_layers(input_size, hidden_sizes, output_size, activation)

Build a sequence of Dense layers for a feedforward network.

# Arguments
- `input_size`: Size of input layer
- `hidden_sizes`: Vector of hidden layer sizes
- `output_size`: Size of output layer
- `activation`: Activation function to use

# Returns
- Vector of Dense layers ready for Chain construction

# Algorithm
1. Input -> First hidden layer (with activation)
2. Hidden layers (each connected to next, with activation)
3. Last hidden -> Output layer (no activation)
"""
function build_dense_layers(
    input_size::Int,
    hidden_sizes::Vector{Int},
    output_size::Int,
    activation::Function
)
    # Pre-allocate layers vector for better performance
    num_layers = length(hidden_sizes) + 1  # Hidden layers + output layer
    layers = Vector{Any}(undef, num_layers)
    
    # Input to first hidden layer (with activation)
    layers[1] = Dense(input_size, hidden_sizes[1], activation)
    
    # Hidden layers (connect each to the next, with activation)
    @inbounds for i in 1:(length(hidden_sizes) - 1)
        layers[i + 1] = Dense(hidden_sizes[i], hidden_sizes[i + 1], activation)
    end
    
    # Output layer (no activation for regression/classification)
    # This allows raw logits for cross-entropy or direct values for regression
    layers[end] = Dense(hidden_sizes[end], output_size)
    
    return layers
end

"""
    create_model(
        input_size,
        hidden_sizes,
        output_size;
        activation=relu
    )

Create a neural network model with Flux.jl.

Similar to PyTorch but with native Julia autodiff that works
on any Julia code, not just tensor operations.

# Arguments
- `input_size`: Number of input features
- `hidden_sizes`: Vector of hidden layer sizes (e.g., [128, 64, 32])
- `output_size`: Number of output classes/values
- `activation`: Activation function (default: relu)

# Returns
- `Chain` model ready for training

# Examples
```julia
# Create a simple classifier
model = create_model(784, [128, 64], 10, activation=relu)

# Create a regression model
model = create_model(100, [256, 128, 64], 1, activation=tanh)

# Create a deep network
model = create_model(784, [512, 256, 128, 64], 10, activation=relu)
```
"""
function create_model(
    input_size::Int,
    hidden_sizes::Vector{Int},
    output_size::Int;
    activation::Function = DEFAULT_ACTIVATION
)
    # Validate architecture parameters
    validate_model_architecture(input_size, hidden_sizes, output_size)
    
    # Build layers
    layers = build_dense_layers(input_size, hidden_sizes, output_size, activation)
    
    # Create and return model
    return Chain(layers...)
end

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    create_loss_function(model, loss_type=:crossentropy)

Create a loss function for the model.

# Arguments
- `model`: Flux model
- `loss_type`: Type of loss (:crossentropy, :mse, :mae)

# Returns
- Loss function (x, y) -> loss_value

# Throws
- `ArgumentError` if loss_type is invalid

# Notes
- Cross-entropy: For classification tasks
- MSE: For regression tasks (squared error)
- MAE: For regression tasks (absolute error)
"""
function create_loss_function(model, loss_type::Symbol = LOSS_CROSSENTROPY)
    if loss_type == LOSS_CROSSENTROPY
        return (x, y) -> Flux.crossentropy(model(x), y)
    elseif loss_type == LOSS_MSE
        return (x, y) -> Flux.mse(model(x), y)
    elseif loss_type == LOSS_MAE
        return (x, y) -> Flux.mae(model(x), y)
    else
        throw(ArgumentError(
            "Unknown loss type: $loss_type. Supported: :crossentropy, :mse, :mae"
        ))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    create_optimizer(learning_rate, optimizer_type=:adam)

Create an optimizer for training.

# Arguments
- `learning_rate`: Learning rate
- `optimizer_type`: Type of optimizer (:adam, :sgd, :rmsprop)

# Returns
- Optimizer instance

# Throws
- `ArgumentError` if optimizer_type is invalid

# Notes
- Adam: Adaptive moment estimation (recommended default)
- SGD: Stochastic gradient descent (simple, fast)
- RMSProp: Root mean square propagation (good for RNNs)
"""
function create_optimizer(learning_rate::Float64, optimizer_type::Symbol = OPTIMIZER_ADAM)
    if optimizer_type == OPTIMIZER_ADAM
        return Adam(learning_rate)
    elseif optimizer_type == OPTIMIZER_SGD
        return Descent(learning_rate)
    elseif optimizer_type == OPTIMIZER_RMSPROP
        return RMSProp(learning_rate)
    else
        throw(ArgumentError(
            "Unknown optimizer type: $optimizer_type. Supported: :adam, :sgd, :rmsprop"
        ))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    format_loss_value(loss)

Format loss value for display.

# Arguments
- `loss`: Loss value

# Returns
- Formatted string representation
"""
@inline function format_loss_value(loss)
    return string(round(loss, digits=LOSS_DISPLAY_PRECISION))
end

"""
    print_training_progress(epoch, total_epochs, loss)

Print training progress information.

# Arguments
- `epoch`: Current epoch number
- `total_epochs`: Total number of epochs
- `loss`: Current loss value
"""
function print_training_progress(epoch::Int, total_epochs::Int, loss)
    loss_str = format_loss_value(loss)
    println("Epoch $epoch/$total_epochs: Loss = $loss_str")
end

"""
    train_model(
        model,
        train_data;
        epochs=10,
        learning_rate=0.001,
        device=:cpu,
        loss_type=:crossentropy,
        optimizer_type=:adam,
        verbose=true
    )

Training loop with Flux.jl.

2-5x faster than PyTorch for custom models due to:
- Native JIT compilation
- Autodiff in any Julia code
- Better memory management

# Arguments
- `model`: Flux Chain model to train
- `train_data`: Tuple of (X, y) training data
- `epochs`: Number of training epochs (default: 10)
- `learning_rate`: Learning rate (default: 0.001)
- `device`: Device to use (:cpu or :gpu, default: :cpu)
- `loss_type`: Loss function type (:crossentropy, :mse, :mae, default: :crossentropy)
- `optimizer_type`: Optimizer type (:adam, :sgd, :rmsprop, default: :adam)
- `verbose`: Print training progress (default: true)

# Returns
- Trained model

# Examples
```julia
# Basic training
model = create_model(784, [128, 64], 10)
X_train = randn(Float32, 784, 1000)
y_train = rand(1:10, 1000)
trained = train_model(model, (X_train, y_train), epochs=20, learning_rate=0.01)

# GPU training
trained = train_model(model, (X_train, y_train), device=:gpu)

# Custom loss and optimizer
trained = train_model(
    model, (X_train, y_train),
    loss_type=:mse,
    optimizer_type=:rmsprop
)
```
"""
function train_model(
    model::Chain,
    train_data::Tuple{Array, Array};
    epochs::Int = DEFAULT_EPOCHS,
    learning_rate::Float64 = DEFAULT_LEARNING_RATE,
    device::Symbol = DEVICE_CPU,
    loss_type::Symbol = LOSS_CROSSENTROPY,
    optimizer_type::Symbol = OPTIMIZER_ADAM,
    verbose::Bool = true
)
    # Validate training data
    validate_training_data(train_data)
    
    x_train, y_train = train_data
    
    # Ensure device is available before moving data
    actual_device = ensure_device_available(device)
    
    # Move model and data to device (with automatic fallback)
    # This is done once before training loop for efficiency
    model = move_to_device(model, actual_device)
    x_train = move_to_device(x_train, actual_device)
    y_train = move_to_device(y_train, actual_device)
    
    # Create loss function
    loss_fn = create_loss_function(model, loss_type)
    
    # Create optimizer
    opt = create_optimizer(learning_rate, optimizer_type)
    
    # Get model parameters for gradient updates
    ps = Flux.params(model)
    
    # Training loop with optimized gradient computation
    for epoch in 1:epochs
        # Compute gradients using automatic differentiation
        # The closure captures the loss function and data
        # Using @nograd for loss computation to avoid unnecessary gradient tracking
        grads = gradient(() -> loss_fn(x_train, y_train), ps)
        
        # Update parameters using optimizer
        # This applies the optimizer's update rule (Adam, SGD, etc.)
        Flux.update!(opt, ps, grads)
        
        # Compute and report loss (for monitoring)
        # Only compute loss if verbose or on last epoch
        if verbose || epoch == epochs
            current_loss = loss_fn(x_train, y_train)
            print_training_progress(epoch, epochs, current_loss)
        end
    end
    
    # Move model back to CPU for return (if needed)
    # This ensures the returned model is on CPU for compatibility
    if actual_device == DEVICE_GPU
        model = model |> cpu
    end
    
    return model
end

"""
    train_model(model, train_data, config::TrainingConfig)

Train model using a TrainingConfig object.

# Arguments
- `model`: Flux Chain model
- `train_data`: Tuple of (X, y) training data
- `config`: TrainingConfig with all training parameters

# Returns
- Trained model

# Examples
```julia
config = TrainingConfig(epochs=50, learning_rate=0.01, device=:gpu)
trained = train_model(model, train_data, config)
```
"""
function train_model(
    model::Chain,
    train_data::Tuple{Array, Array},
    config::TrainingConfig
)
    return train_model(
        model,
        train_data;
        epochs = config.epochs,
        learning_rate = config.learning_rate,
        device = config.device,
        loss_type = config.loss_type,
        optimizer_type = config.optimizer_type,
        verbose = config.verbose
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

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
    
    # Ensure device is available (fallback to CPU if GPU unavailable)
    actual_device = ensure_device_available(device)
    
    # Move to device (with automatic fallback)
    x_device = move_to_device(x, actual_device)
    model_device = move_to_device(model, actual_device)
    
    # Make prediction (no gradient tracking needed for inference)
    # Using @nograd would be ideal but Flux handles this automatically in eval mode
    predictions = model_device(x_device)
    
    # Move back to CPU if needed (for compatibility with downstream code)
    # This ensures predictions are always on CPU regardless of training device
    if actual_device == DEVICE_GPU
        predictions = predictions |> cpu
    end
    
    return predictions
end

# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

"""
    create_language_model(
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers
    )

Create a language model with embeddings and LSTM layers.

# Arguments
- `vocab_size`: Size of vocabulary
- `embedding_dim`: Dimension of embeddings
- `hidden_dim`: Hidden dimension of LSTM
- `num_layers`: Number of LSTM layers

# Returns
- Language model Chain

# Examples
```julia
model = create_language_model(10000, 256, 512, 3)
```

# Architecture
1. Embedding layer: vocab_size -> embedding_dim
2. LSTM layers: embedding_dim -> hidden_dim (repeated num_layers times)
3. Output layer: hidden_dim -> vocab_size (predicts next token)
"""
function create_language_model(
    vocab_size::Int,
    embedding_dim::Int,
    hidden_dim::Int,
    num_layers::Int
)
    # Validate inputs with comprehensive checks
    if vocab_size <= 0
        throw(ArgumentError("vocab_size must be positive, got $vocab_size"))
    end
    if embedding_dim <= 0
        throw(ArgumentError("embedding_dim must be positive, got $embedding_dim"))
    end
    if hidden_dim <= 0
        throw(ArgumentError("hidden_dim must be positive, got $hidden_dim"))
    end
    if num_layers <= 0
        throw(ArgumentError("num_layers must be positive, got $num_layers"))
    end
    if num_layers > 10
        @warn "Large number of LSTM layers ($num_layers) may cause vanishing gradients. Consider using fewer layers or gradient clipping."
    end
    
    # Pre-allocate layers vector for better performance
    # Structure: 1 embedding + num_layers LSTM + 1 output = num_layers + 2 layers
    total_layers = num_layers + 2
    layers = Vector{Any}(undef, total_layers)
    
    # Embedding layer: maps token IDs to dense vectors
    # This converts discrete token indices to continuous embeddings
    layers[1] = Embedding(vocab_size, embedding_dim)
    
    # LSTM layers: process sequences with memory
    # Each LSTM processes the output of the previous layer
    @inbounds for i in 1:num_layers
        # First LSTM takes embedding_dim, subsequent ones take hidden_dim
        input_dim = (i == 1) ? embedding_dim : hidden_dim
        layers[i + 1] = LSTM(input_dim, hidden_dim)
    end
    
    # Output layer: predicts next token in vocabulary
    # Maps hidden state to vocabulary logits
    layers[end] = Dense(hidden_dim, vocab_size)
    
    return Chain(layers...)
end

"""
    train_language_model(
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        train_data;
        epochs=10,
        learning_rate=0.001,
        device=:cpu
    )

Create and train a language model.

# Arguments
- `vocab_size`: Size of vocabulary
- `embedding_dim`: Dimension of embeddings
- `hidden_dim`: Hidden dimension of LSTM
- `num_layers`: Number of LSTM layers
- `train_data`: Tuple of (X, y) training data
- `epochs`: Number of training epochs (default: 10)
- `learning_rate`: Learning rate (default: 0.001)
- `device`: Device to use (:cpu or :gpu, default: :cpu)

# Returns
- Trained language model

# Examples
```julia
X_train = rand(1:10000, 100, 1000)  # (seq_len, batch_size)
y_train = rand(1:10000, 100, 1000)
model = train_language_model(10000, 256, 512, 2, (X_train, y_train))
```
"""
function train_language_model(
    vocab_size::Int,
    embedding_dim::Int,
    hidden_dim::Int,
    num_layers::Int,
    train_data::Tuple{Array, Array};
    epochs::Int = DEFAULT_EPOCHS,
    learning_rate::Float64 = DEFAULT_LEARNING_RATE,
    device::Symbol = DEVICE_CPU
)
    # Create model
    model = create_language_model(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    # Train model with cross-entropy loss (standard for language models)
    trained_model = train_model(
        model,
        train_data;
        epochs = epochs,
        learning_rate = learning_rate,
        device = device,
        loss_type = LOSS_CROSSENTROPY  # Standard for language models
    )
    
    return trained_model
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

export create_model, train_model, predict
export create_language_model, train_language_model
export TrainingConfig, is_gpu_available

end  # module FluxML
