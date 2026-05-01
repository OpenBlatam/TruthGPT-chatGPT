"""
Model Building

Functions for creating neural network models.
"""

include("constants.jl")
include("validation.jl")

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
    # Total layers: 1 input->hidden + (n-1) hidden->hidden + 1 hidden->output = n+1
    num_layers = length(hidden_sizes) + 1
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
    validate_language_model_params(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    # Build language model layers
    layers = build_language_model_layers(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    return Chain(layers...)
end

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS FOR LANGUAGE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_language_model_params(vocab_size, embedding_dim, hidden_dim, num_layers)

Validate parameters for language model creation.

# Arguments
- `vocab_size`: Size of vocabulary
- `embedding_dim`: Dimension of embeddings
- `hidden_dim`: Hidden dimension of LSTM
- `num_layers`: Number of LSTM layers

# Throws
- `ArgumentError` if any parameter is invalid
"""
function validate_language_model_params(
    vocab_size::Int,
    embedding_dim::Int,
    hidden_dim::Int,
    num_layers::Int
)
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
end

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER BUILDING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_language_model_layers(vocab_size, embedding_dim, hidden_dim, num_layers)

Build layers for a language model.

# Arguments
- `vocab_size`: Size of vocabulary
- `embedding_dim`: Dimension of embeddings
- `hidden_dim`: Hidden dimension of LSTM
- `num_layers`: Number of LSTM layers

# Returns
- Vector of layers ready for Chain construction

# Architecture
1. Embedding layer: vocab_size -> embedding_dim
2. LSTM layers: embedding_dim -> hidden_dim (repeated num_layers times)
3. Output layer: hidden_dim -> vocab_size (predicts next token)
"""
function build_language_model_layers(
    vocab_size::Int,
    embedding_dim::Int,
    hidden_dim::Int,
    num_layers::Int
)
    # Pre-allocate layers vector for better performance
    # Structure: 1 embedding + num_layers LSTM + 1 output = num_layers + 2 layers
    total_layers = num_layers + 2
    layers = Vector{Any}(undef, total_layers)
    
    # Embedding layer: maps token IDs to dense vectors
    layers[1] = create_embedding_layer(vocab_size, embedding_dim)
    
    # LSTM layers: process sequences with memory
    build_lstm_layers!(layers, embedding_dim, hidden_dim, num_layers)
    
    # Output layer: predicts next token in vocabulary
    layers[end] = create_output_layer(hidden_dim, vocab_size)
    
    return layers
end

"""
    create_embedding_layer(vocab_size, embedding_dim)

Create embedding layer for language model.

# Arguments
- `vocab_size`: Size of vocabulary
- `embedding_dim`: Dimension of embeddings

# Returns
- Embedding layer

# Notes
- Converts discrete token indices to continuous embeddings
"""
function create_embedding_layer(vocab_size::Int, embedding_dim::Int)
    return Embedding(vocab_size, embedding_dim)
end

"""
    build_lstm_layers!(layers, embedding_dim, hidden_dim, num_layers)

Build LSTM layers for language model.

# Arguments
- `layers`: Layers vector to modify (in-place)
- `embedding_dim`: Dimension of embeddings
- `hidden_dim`: Hidden dimension of LSTM
- `num_layers`: Number of LSTM layers

# Notes
- Modifies layers vector in-place
- First LSTM takes embedding_dim, subsequent ones take hidden_dim
"""
function build_lstm_layers!(
    layers::Vector{Any},
    embedding_dim::Int,
    hidden_dim::Int,
    num_layers::Int
)
    @inbounds for i in 1:num_layers
        # First LSTM takes embedding_dim, subsequent ones take hidden_dim
        input_dim = (i == 1) ? embedding_dim : hidden_dim
        layers[i + 1] = LSTM(input_dim, hidden_dim)
    end
end

"""
    create_output_layer(hidden_dim, vocab_size)

Create output layer for language model.

# Arguments
- `hidden_dim`: Hidden dimension of LSTM
- `vocab_size`: Size of vocabulary

# Returns
- Dense output layer

# Notes
- Maps hidden state to vocabulary logits
"""
function create_output_layer(hidden_dim::Int, vocab_size::Int)
    return Dense(hidden_dim, vocab_size)
end

