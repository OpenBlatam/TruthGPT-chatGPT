"""
Training

Functions for training neural network models.
"""

include("constants.jl")
include("device.jl")
include("validation.jl")
include("losses.jl")
include("optimizers.jl")
include("models.jl")

"""
    format_loss_value(loss)

Format loss value for display with consistent precision.

# Arguments
- `loss`: Loss value (must be a number)

# Returns
- Formatted string representation

# Examples
```julia
format_loss_value(0.123456)  # Returns "0.123"
```
"""
@inline function format_loss_value(loss)
    # Validate input is numeric
    if !(loss isa Number)
        throw(ArgumentError("loss must be a number, got $(typeof(loss))"))
    end
    
    # Format with consistent precision for display
    return string(round(loss, digits=LOSS_DISPLAY_PRECISION))
end

"""
    print_training_progress(epoch, total_epochs, loss)

Print training progress information with formatted output.

# Arguments
- `epoch`: Current epoch number (must be positive and <= total_epochs)
- `total_epochs`: Total number of epochs (must be positive)
- `loss`: Current loss value (must be a number)

# Examples
```julia
print_training_progress(1, 10, 0.5)  # Prints "Epoch 1/10: Loss = 0.500"
```
"""
function print_training_progress(epoch::Int, total_epochs::Int, loss)
    # Validate inputs
    validate_training_progress_inputs(epoch, total_epochs, loss)
    
    # Format and print progress
    loss_str = format_loss_value(loss)
    println("Epoch $epoch/$total_epochs: Loss = $loss_str")
end

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS FOR TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_training_progress_inputs(epoch, total_epochs, loss)

Validate inputs for training progress printing.

# Arguments
- `epoch`: Current epoch number
- `total_epochs`: Total number of epochs
- `loss`: Current loss value

# Throws
- `ArgumentError` if any input is invalid
"""
function validate_training_progress_inputs(epoch::Int, total_epochs::Int, loss)
    if epoch <= 0
        throw(ArgumentError("epoch must be positive, got $epoch"))
    end
    if total_epochs <= 0
        throw(ArgumentError("total_epochs must be positive, got $total_epochs"))
    end
    if epoch > total_epochs
        throw(ArgumentError("epoch ($epoch) must be <= total_epochs ($total_epochs)"))
    end
    if !(loss isa Number)
        throw(ArgumentError("loss must be a number, got $(typeof(loss))"))
    end
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
    # Validate all training parameters
    validate_training_params(train_data, epochs, learning_rate)
    
    x_train, y_train = train_data
    
    # Setup training environment (device, loss, optimizer)
    model, x_train, y_train, loss_fn, opt, ps = setup_training(
        model, x_train, y_train, device, loss_type, learning_rate, optimizer_type
    )
    
    # Execute training loop
    for epoch in 1:epochs
        loss_val, grads = compute_gradients(loss_fn, x_train, y_train, ps)
        validate_loss_value(loss_val, epoch)
        update_parameters(opt, ps, grads)
        
        if verbose
            print_training_progress(epoch, epochs, loss_val)
        end
    end
    
    # Move model back to CPU if needed
    return finalize_training(model, device)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SETUP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_training_params(train_data, epochs, learning_rate)

Validate training parameters.

# Arguments
- `train_data`: Training data tuple
- `epochs`: Number of epochs
- `learning_rate`: Learning rate

# Throws
- `ArgumentError` if any parameter is invalid
"""
function validate_training_params(
    train_data::Tuple{Array, Array},
    epochs::Int,
    learning_rate::Float64
)
    validate_training_data(train_data)
    validate_epochs(epochs)
    validate_learning_rate(learning_rate)
end

"""
    setup_training(model, x_train, y_train, device, loss_type, learning_rate, optimizer_type)

Setup training environment (device, loss function, optimizer).

# Arguments
- `model`: Flux model
- `x_train`: Training features
- `y_train`: Training labels
- `device`: Device to use
- `loss_type`: Type of loss function
- `learning_rate`: Learning rate
- `optimizer_type`: Type of optimizer

# Returns
- Tuple of (model, x_train, y_train, loss_fn, opt, ps)
"""
function setup_training(
    model::Chain,
    x_train::Array,
    y_train::Array,
    device::Symbol,
    loss_type::Symbol,
    learning_rate::Float64,
    optimizer_type::Symbol
)
    # Ensure device is available (fallback to CPU if GPU unavailable)
    actual_device = ensure_device_available(device)
    
    # Move model and data to device (with automatic fallback)
    model = move_to_device(model, actual_device)
    x_train = move_to_device(x_train, actual_device)
    y_train = move_to_device(y_train, actual_device)
    
    # Create loss function
    loss_fn = create_loss_function(model, loss_type)
    
    # Create optimizer
    opt = create_optimizer(learning_rate, optimizer_type)
    
    # Get model parameters for gradient updates
    ps = Flux.params(model)
    
    return model, x_train, y_train, loss_fn, opt, ps
end

"""
    compute_gradients(loss_fn, x_train, y_train, ps)

Compute loss and gradients using automatic differentiation.

# Arguments
- `loss_fn`: Loss function
- `x_train`: Training features
- `y_train`: Training labels
- `ps`: Model parameters

# Returns
- Tuple of (loss_value, gradients)
"""
function compute_gradients(loss_fn, x_train::Array, y_train::Array, ps)
    return withgradient(() -> loss_fn(x_train, y_train), ps)
end

"""
    validate_loss_value(loss_val, epoch)

Validate loss value (check for NaN or Inf).

# Arguments
- `loss_val`: Loss value
- `epoch`: Current epoch number

# Throws
- `Error` if loss is not finite
"""
function validate_loss_value(loss_val, epoch::Int)
    if !isfinite(loss_val)
        error("Training failed: loss is not finite ($loss_val) at epoch $epoch. "
              "Consider reducing learning_rate or checking data.")
    end
end

"""
    update_parameters(opt, ps, grads)

Update model parameters using optimizer.

# Arguments
- `opt`: Optimizer
- `ps`: Model parameters
- `grads`: Gradients

# Notes
- Modifies parameters in-place
"""
function update_parameters(opt, ps, grads)
    Flux.update!(opt, ps, grads)
end

"""
    finalize_training(model, device)

Finalize training by moving model back to CPU if needed.

# Arguments
- `model`: Trained model
- `device`: Device used for training

# Returns
- Model (on CPU if GPU was used)
"""
function finalize_training(model::Chain, device::Symbol)
    if device == DEVICE_GPU
        return model |> cpu
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
    # Validate model architecture parameters
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
    
    # Validate training data
    validate_training_data(train_data)
    
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

