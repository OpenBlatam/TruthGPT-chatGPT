"""
Transformer implementation in Julia for TruthGPT

High-performance transformer layers with:
- Multi-head attention with RoPE
- Feed-forward networks with SwiGLU
- Pre-norm architecture
- Causal masking
- Token generation with top-k/top-p sampling
"""

using LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default configuration values
const DEFAULT_D_MODEL = 768
const DEFAULT_N_HEADS = 12
const DEFAULT_N_LAYERS = 12
const DEFAULT_D_FF = 3072
const DEFAULT_VOCAB_SIZE = 32000
const DEFAULT_MAX_SEQ_LEN = 2048
const DEFAULT_DROPOUT = 0.1f0
const DEFAULT_LAYER_NORM_EPS = 1f-5
const DEFAULT_ROPE_BASE = 10000.0f0

# Weight initialization
const DEFAULT_EMBED_SCALE = 0.02f0
const DEFAULT_LM_HEAD_SCALE = 0.02f0

# Generation defaults
const DEFAULT_MAX_NEW_TOKENS = 100
const DEFAULT_TEMPERATURE = 1.0f0
const DEFAULT_TOP_K = 50
const DEFAULT_TOP_P = 0.9f0
const DEFAULT_EOS_TOKEN_ID = 2

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TransformerConfig

Transformer model configuration.

# Fields
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer layers
- `d_ff`: Feed-forward dimension
- `vocab_size`: Vocabulary size
- `max_seq_len`: Maximum sequence length
- `dropout`: Dropout probability
- `layer_norm_eps`: Layer normalization epsilon
- `use_rope`: Use Rotary Position Embeddings
- `rope_base`: Base frequency for RoPE

# Examples
```julia
config = TransformerConfig(
    d_model=1024,
    n_heads=16,
    n_layers=24
)
```
"""
Base.@kwdef struct TransformerConfig
    d_model::Int = DEFAULT_D_MODEL
    n_heads::Int = DEFAULT_N_HEADS
    n_layers::Int = DEFAULT_N_LAYERS
    d_ff::Int = DEFAULT_D_FF
    vocab_size::Int = DEFAULT_VOCAB_SIZE
    max_seq_len::Int = DEFAULT_MAX_SEQ_LEN
    dropout::Float32 = DEFAULT_DROPOUT
    layer_norm_eps::Float32 = DEFAULT_LAYER_NORM_EPS
    use_rope::Bool = true
    rope_base::Float32 = DEFAULT_ROPE_BASE
    
    function TransformerConfig(
        d_model::Int = DEFAULT_D_MODEL,
        n_heads::Int = DEFAULT_N_HEADS,
        n_layers::Int = DEFAULT_N_LAYERS,
        d_ff::Int = DEFAULT_D_FF,
        vocab_size::Int = DEFAULT_VOCAB_SIZE,
        max_seq_len::Int = DEFAULT_MAX_SEQ_LEN,
        dropout::Float32 = DEFAULT_DROPOUT,
        layer_norm_eps::Float32 = DEFAULT_LAYER_NORM_EPS,
        use_rope::Bool = true,
        rope_base::Float32 = DEFAULT_ROPE_BASE
    )
        # Validate all configuration parameters
        validate_transformer_config(
            d_model, n_heads, n_layers, d_ff, vocab_size, max_seq_len, dropout
        )
        
        new(d_model, n_heads, n_layers, d_ff, vocab_size, max_seq_len,
            dropout, layer_norm_eps, use_rope, rope_base)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_transformer_config(d_model, n_heads, n_layers, d_ff, vocab_size, max_seq_len, dropout)

Validate transformer configuration parameters.

# Arguments
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer layers
- `d_ff`: Feed-forward dimension
- `vocab_size`: Vocabulary size
- `max_seq_len`: Maximum sequence length
- `dropout`: Dropout probability

# Throws
- `ArgumentError` if any parameter is invalid
"""
function validate_transformer_config(
    d_model::Int,
    n_heads::Int,
    n_layers::Int,
    d_ff::Int,
    vocab_size::Int,
    max_seq_len::Int,
    dropout::Float32
)
    validate_d_model(d_model)
    validate_n_heads(n_heads)
    validate_head_divisibility(d_model, n_heads)
    validate_n_layers(n_layers)
    validate_d_ff(d_ff)
    validate_vocab_size(vocab_size)
    validate_max_seq_len(max_seq_len)
    validate_dropout(dropout)
end

"""
    validate_d_model(d_model)

Validate model dimension.

# Arguments
- `d_model`: Model dimension

# Throws
- `ArgumentError` if d_model is invalid
"""
function validate_d_model(d_model::Int)
    if d_model <= 0
        throw(ArgumentError("d_model must be positive, got $d_model"))
    end
end

"""
    validate_n_heads(n_heads)

Validate number of attention heads.

# Arguments
- `n_heads`: Number of attention heads

# Throws
- `ArgumentError` if n_heads is invalid
"""
function validate_n_heads(n_heads::Int)
    if n_heads <= 0
        throw(ArgumentError("n_heads must be positive, got $n_heads"))
    end
end

"""
    validate_head_divisibility(d_model, n_heads)

Validate that d_model is divisible by n_heads.

# Arguments
- `d_model`: Model dimension
- `n_heads`: Number of attention heads

# Throws
- `ArgumentError` if d_model is not divisible by n_heads
"""
function validate_head_divisibility(d_model::Int, n_heads::Int)
    if d_model % n_heads != 0
        throw(ArgumentError("d_model ($d_model) must be divisible by n_heads ($n_heads)"))
    end
end

"""
    validate_n_layers(n_layers)

Validate number of transformer layers.

# Arguments
- `n_layers`: Number of transformer layers

# Throws
- `ArgumentError` if n_layers is invalid
"""
function validate_n_layers(n_layers::Int)
    if n_layers <= 0
        throw(ArgumentError("n_layers must be positive, got $n_layers"))
    end
end

"""
    validate_d_ff(d_ff)

Validate feed-forward dimension.

# Arguments
- `d_ff`: Feed-forward dimension

# Throws
- `ArgumentError` if d_ff is invalid
"""
function validate_d_ff(d_ff::Int)
    if d_ff <= 0
        throw(ArgumentError("d_ff must be positive, got $d_ff"))
    end
end

"""
    validate_vocab_size(vocab_size)

Validate vocabulary size.

# Arguments
- `vocab_size`: Vocabulary size

# Throws
- `ArgumentError` if vocab_size is invalid
"""
function validate_vocab_size(vocab_size::Int)
    if vocab_size <= 0
        throw(ArgumentError("vocab_size must be positive, got $vocab_size"))
    end
end

"""
    validate_max_seq_len(max_seq_len)

Validate maximum sequence length.

# Arguments
- `max_seq_len`: Maximum sequence length

# Throws
- `ArgumentError` if max_seq_len is invalid
"""
function validate_max_seq_len(max_seq_len::Int)
    if max_seq_len <= 0
        throw(ArgumentError("max_seq_len must be positive, got $max_seq_len"))
    end
end

"""
    validate_dropout(dropout)

Validate dropout probability.

# Arguments
- `dropout`: Dropout probability

# Throws
- `ArgumentError` if dropout is invalid
"""
function validate_dropout(dropout::Float32)
    if dropout < 0.0f0 || dropout > 1.0f0
        throw(ArgumentError("dropout must be in [0, 1], got $dropout"))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# ROTARY POSITION EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    precompute_rope_freqs(dim, max_seq_len; base=10000.0)

Precompute RoPE frequency matrices.

# Arguments
- `dim`: Dimension (must be even)
- `max_seq_len`: Maximum sequence length
- `base`: Base frequency (default: 10000.0)

# Returns
- Tuple of (cos_cache, sin_cache) matrices

# Examples
```julia
cos_cache, sin_cache = precompute_rope_freqs(64, 2048)
```
"""
function precompute_rope_freqs(
    dim::Int,
    max_seq_len::Int;
    base::Float32 = DEFAULT_ROPE_BASE
)
    # Validate inputs
    validate_rope_dim(dim)
    validate_max_seq_len(max_seq_len)
    
    # Compute RoPE frequencies
    inv_freq = compute_rope_inv_freqs(dim, base)
    freqs = compute_rope_freqs(inv_freq, max_seq_len)
    
    # Precompute cos and sin caches
    cos_cache = cos.(freqs)
    sin_cache = sin.(freqs)
    
    return cos_cache, sin_cache
end

# ═══════════════════════════════════════════════════════════════════════════════
# ROPE COMPUTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_rope_dim(dim)

Validate dimension for RoPE (must be positive and even).

# Arguments
- `dim`: Dimension to validate

# Throws
- `ArgumentError` if dim is invalid
"""
function validate_rope_dim(dim::Int)
    if dim <= 0 || dim % 2 != 0
        throw(ArgumentError("dim must be positive and even, got $dim"))
    end
end

"""
    compute_rope_inv_freqs(dim, base)

Compute inverse frequencies for RoPE.

# Arguments
- `dim`: Dimension (must be even)
- `base`: Base frequency

# Returns
- Vector of inverse frequencies

# Algorithm
- Computes 1 / (base^(2i/dim)) for i in 0:2:dim-1
"""
function compute_rope_inv_freqs(dim::Int, base::Float32)
    return 1.0f0 ./ (base .^ (Float32.(0:2:dim-1) ./ dim))
end

"""
    compute_rope_freqs(inv_freq, max_seq_len)

Compute frequencies for all positions.

# Arguments
- `inv_freq`: Inverse frequencies vector
- `max_seq_len`: Maximum sequence length

# Returns
- Frequency matrix [max_seq_len, dim/2]

# Algorithm
- Computes positions * inv_freq' for all positions
"""
function compute_rope_freqs(inv_freq::Vector{Float32}, max_seq_len::Int)
    positions = Float32.(0:max_seq_len-1)
    return positions * inv_freq'
end

"""
    apply_rope(q, k, cos_cache, sin_cache; start_pos=0)

Apply rotary position embeddings to Q and K tensors.

# Arguments
- `q`: Query tensor [batch, seq_len, d_head]
- `k`: Key tensor [batch, seq_len, d_head]
- `cos_cache`: Precomputed cosine cache
- `sin_cache`: Precomputed sine cache
- `start_pos`: Starting position (default: 0)

# Returns
- Tuple of (q_rotated, k_rotated)

# Examples
```julia
q_rot, k_rot = apply_rope(q, k, cos_cache, sin_cache, start_pos=10)
```
"""
function apply_rope(
    q::Array{Float32, 3},
    k::Array{Float32, 3},
    cos_cache::Matrix{Float32},
    sin_cache::Matrix{Float32};
    start_pos::Int = 0
)
    batch, seq_len, d_head = size(q)
    
    # Validate inputs
    if size(k) != size(q)
        throw(DimensionMismatch("Q and K must have same shape"))
    end
    if d_head % 2 != 0
        throw(ArgumentError("d_head must be even, got $d_head"))
    end
    
    half_dim = d_head ÷ 2
    q_rotated = similar(q)
    k_rotated = similar(k)
    
    @inbounds for b in 1:batch, s in 1:seq_len
        pos = start_pos + s
        
        # Check bounds
        if pos > size(cos_cache, 1)
            throw(ArgumentError(
                "Position $pos exceeds cache size $(size(cos_cache, 1))"
            ))
        end
        
        # Apply rotation to each pair of dimensions
        for i in 1:half_dim
            cos_θ = cos_cache[pos, i]
            sin_θ = sin_cache[pos, i]
            
            # Rotate Q
            q0, q1 = q[b, s, i], q[b, s, i + half_dim]
            q_rotated[b, s, i] = q0 * cos_θ - q1 * sin_θ
            q_rotated[b, s, i + half_dim] = q0 * sin_θ + q1 * cos_θ
            
            # Rotate K
            k0, k1 = k[b, s, i], k[b, s, i + half_dim]
            k_rotated[b, s, i] = k0 * cos_θ - k1 * sin_θ
            k_rotated[b, s, i + half_dim] = k0 * sin_θ + k1 * cos_θ
        end
    end
    
    return q_rotated, k_rotated
end

# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

"""
    MultiHeadAttention

Multi-head attention layer with RoPE support.

# Fields
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `d_head`: Dimension per head
- `W_q`, `W_k`, `W_v`, `W_o`: Weight matrices
- `cos_cache`, `sin_cache`: RoPE caches
- `use_rope`: Whether to use RoPE
"""
mutable struct MultiHeadAttention
    d_model::Int
    n_heads::Int
    d_head::Int
    
    W_q::Matrix{Float32}
    W_k::Matrix{Float32}
    W_v::Matrix{Float32}
    W_o::Matrix{Float32}
    
    # RoPE caches
    cos_cache::Matrix{Float32}
    sin_cache::Matrix{Float32}
    use_rope::Bool
end

"""
    MultiHeadAttention(config::TransformerConfig)

Create MultiHeadAttention layer from config.

# Arguments
- `config`: TransformerConfig

# Returns
- MultiHeadAttention instance
"""
function MultiHeadAttention(config::TransformerConfig)
    d_head = config.d_model ÷ config.n_heads
    
    # Initialize weights with Xavier initialization
    W_q = xavier_init(config.d_model, config.d_model)
    W_k = xavier_init(config.d_model, config.d_model)
    W_v = xavier_init(config.d_model, config.d_model)
    W_o = xavier_init(config.d_model, config.d_model)
    
    # Precompute RoPE if enabled
    if config.use_rope
        cos_cache, sin_cache = precompute_rope_freqs(
            d_head, config.max_seq_len, base=config.rope_base
        )
    else
        # Dummy caches (1x1 to avoid allocation issues)
        cos_cache = zeros(Float32, 1, 1)
        sin_cache = zeros(Float32, 1, 1)
    end
    
    return MultiHeadAttention(
        config.d_model, config.n_heads, d_head,
        W_q, W_k, W_v, W_o,
        cos_cache, sin_cache, config.use_rope
    )
end

"""
    forward(attn, x; mask=nothing, start_pos=0)

Forward pass through multi-head attention.

# Arguments
- `attn`: MultiHeadAttention layer
- `x`: Input tensor [batch, seq_len, d_model]
- `mask`: Optional attention mask [1, 1, seq_len, seq_len]
- `start_pos`: Starting position for RoPE (default: 0)

# Returns
- Output tensor [batch, seq_len, d_model]

# Examples
```julia
attn = MultiHeadAttention(config)
x = randn(Float32, 2, 128, 768)
output = forward(attn, x)
```
"""
function forward(
    attn::MultiHeadAttention,
    x::Array{Float32, 3};
    mask::Union{Nothing, Array{Float32}} = nothing,
    start_pos::Int = 0
)
    batch, seq_len, d_model = size(x)
    
    # Validate input dimension
    if d_model != attn.d_model
        throw(DimensionMismatch(
            "Input dimension $d_model must match d_model $(attn.d_model)"
        ))
    end
    
    # Linear projections: [batch*seq_len, d_model] * [d_model, d_model]
    x_flat = reshape(x, :, d_model)
    Q = x_flat * attn.W_q'
    K = x_flat * attn.W_k'
    V = x_flat * attn.W_v'
    
    # Reshape to [batch, seq_len, n_heads, d_head]
    Q = reshape(Q, batch, seq_len, attn.n_heads, attn.d_head)
    K = reshape(K, batch, seq_len, attn.n_heads, attn.d_head)
    V = reshape(V, batch, seq_len, attn.n_heads, attn.d_head)
    
    # Transpose to [batch, n_heads, seq_len, d_head]
    Q = permutedims(Q, (1, 3, 2, 4))
    K = permutedims(K, (1, 3, 2, 4))
    V = permutedims(V, (1, 3, 2, 4))
    
    # Apply RoPE per head (in-place would be better, but apply_rope returns new arrays)
    if attn.use_rope
        @inbounds for h in 1:attn.n_heads
            # Extract views for this head across all batches
            q_head = @view Q[:, h, :, :]  # [batch, seq_len, d_head]
            k_head = @view K[:, h, :, :]  # [batch, seq_len, d_head]
            
            # Apply RoPE (returns new arrays)
            q_rotated, k_rotated = apply_rope(
                q_head, k_head, attn.cos_cache, attn.sin_cache, start_pos=start_pos
            )
            
            # Copy back (unfortunately apply_rope doesn't support in-place)
            Q[:, h, :, :] = q_rotated
            K[:, h, :, :] = k_rotated
        end
    end
    
    # Scaled dot-product attention
    # Pre-compute scale factor for efficiency
    scale = 1.0f0 / sqrt(Float32(attn.d_head))
    
    # Compute attention scores using batched matrix multiplication
    # Q: [batch, n_heads, seq_len, d_head]
    # K: [batch, n_heads, seq_len, d_head]
    # scores = Q @ K^T: [batch, n_heads, seq_len, seq_len]
    scores = zeros(Float32, batch, attn.n_heads, seq_len, seq_len)
    
    # Use batched matrix multiplication for better performance
    @inbounds for b in 1:batch, h in 1:attn.n_heads
        # Extract Q and K for this batch and head
        Q_head = @view Q[b, h, :, :]  # [seq_len, d_head]
        K_head = @view K[b, h, :, :]  # [seq_len, d_head]
        
        # Compute scores: Q @ K^T
        scores_head = Q_head * K_head'  # [seq_len, seq_len]
        scores[b, h, :, :] = scores_head .* scale
    end
    
    # Apply mask if provided
    if !isnothing(mask)
        if size(mask) != size(scores)
            throw(DimensionMismatch(
                "Mask size $(size(mask)) must match scores size $(size(scores))"
            ))
        end
        scores = scores .+ mask
    end
    
    # Softmax over last dimension
    attn_weights = softmax(scores, dims=4)
    
    # Apply attention to values using batched matrix multiplication
    # attn_weights: [batch, n_heads, seq_len, seq_len]
    # V: [batch, n_heads, seq_len, d_head]
    # output = attn_weights @ V: [batch, n_heads, seq_len, d_head]
    output = zeros(Float32, batch, attn.n_heads, seq_len, attn.d_head)
    
    @inbounds for b in 1:batch, h in 1:attn.n_heads
        # Extract views for this batch and head
        attn_weights_head = @view attn_weights[b, h, :, :]  # [seq_len, seq_len]
        V_head = @view V[b, h, :, :]  # [seq_len, d_head]
        
        # Compute output: attn_weights @ V
        output_head = attn_weights_head * V_head  # [seq_len, d_head]
        output[b, h, :, :] = output_head
    end
    
    # Transpose back to [batch, seq_len, n_heads, d_head]
    output = permutedims(output, (1, 3, 2, 4))
    
    # Reshape to [batch, seq_len, d_model]
    output = reshape(output, batch, seq_len, attn.d_model)
    
    # Output projection
    output_flat = reshape(output, :, attn.d_model)
    output = output_flat * attn.W_o'
    output = reshape(output, batch, seq_len, attn.d_model)
    
    return output
end

# ═══════════════════════════════════════════════════════════════════════════════
# FEED-FORWARD NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FeedForward

Feed-forward network with SwiGLU activation.

# Fields
- `W_gate`: Gate projection matrix
- `W_up`: Up projection matrix
- `W_down`: Down projection matrix
"""
mutable struct FeedForward
    W_gate::Matrix{Float32}
    W_up::Matrix{Float32}
    W_down::Matrix{Float32}
end

"""
    FeedForward(d_model, d_ff)

Create FeedForward layer.

# Arguments
- `d_model`: Model dimension
- `d_ff`: Feed-forward dimension

# Returns
- FeedForward instance
"""
function FeedForward(d_model::Int, d_ff::Int)
    if d_model <= 0
        throw(ArgumentError("d_model must be positive, got $d_model"))
    end
    if d_ff <= 0
        throw(ArgumentError("d_ff must be positive, got $d_ff"))
    end
    
    return FeedForward(
        xavier_init(d_model, d_ff),
        xavier_init(d_model, d_ff),
        xavier_init(d_ff, d_model)
    )
end

"""
    forward(ff, x)

Forward pass through feed-forward network.

# Arguments
- `ff`: FeedForward layer
- `x`: Input tensor [batch, seq_len, d_model]

# Returns
- Output tensor [batch, seq_len, d_model]

# Examples
```julia
ff = FeedForward(768, 3072)
x = randn(Float32, 2, 128, 768)
output = forward(ff, x)
```
"""
function forward(ff::FeedForward, x::Array{Float32, 3})
    batch, seq_len, d_model = size(x)
    x_flat = reshape(x, :, d_model)
    
    # SwiGLU: swish(gate) * up
    gate = x_flat * ff.W_gate'
    up = x_flat * ff.W_up'
    
    hidden = swish(gate) .* up
    
    output = hidden * ff.W_down'
    return reshape(output, batch, seq_len, d_model)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

"""
    TransformerBlock

Single transformer block with pre-norm architecture.

# Fields
- `attention`: MultiHeadAttention layer
- `ff`: FeedForward layer
- `norm1_weight`, `norm1_bias`: First layer norm parameters
- `norm2_weight`, `norm2_bias`: Second layer norm parameters
- `eps`: Layer norm epsilon
"""
mutable struct TransformerBlock
    attention::MultiHeadAttention
    ff::FeedForward
    norm1_weight::Vector{Float32}
    norm1_bias::Vector{Float32}
    norm2_weight::Vector{Float32}
    norm2_bias::Vector{Float32}
    eps::Float32
end

"""
    TransformerBlock(config::TransformerConfig)

Create TransformerBlock from config.

# Arguments
- `config`: TransformerConfig

# Returns
- TransformerBlock instance
"""
function TransformerBlock(config::TransformerConfig)
    return TransformerBlock(
        MultiHeadAttention(config),
        FeedForward(config.d_model, config.d_ff),
        ones(Float32, config.d_model),      # norm1_weight
        zeros(Float32, config.d_model),      # norm1_bias
        ones(Float32, config.d_model),      # norm2_weight
        zeros(Float32, config.d_model),     # norm2_bias
        config.layer_norm_eps
    )
end

"""
    forward(block, x; mask=nothing, start_pos=0)

Forward pass through transformer block.

Uses pre-norm architecture: norm -> attention -> residual -> norm -> FFN -> residual

# Arguments
- `block`: TransformerBlock
- `x`: Input tensor [batch, seq_len, d_model]
- `mask`: Optional attention mask
- `start_pos`: Starting position for RoPE

# Returns
- Output tensor [batch, seq_len, d_model]
"""
function forward(
    block::TransformerBlock,
    x::Array{Float32, 3};
    mask::Union{Nothing, Array{Float32}} = nothing,
    start_pos::Int = 0
)
    # Pre-norm + attention + residual
    normed = layer_norm(x, block.norm1_weight, block.norm1_bias, ε=block.eps)
    attn_out = forward(block.attention, normed, mask=mask, start_pos=start_pos)
    x = x .+ attn_out
    
    # Pre-norm + FFN + residual
    normed = layer_norm(x, block.norm2_weight, block.norm2_bias, ε=block.eps)
    ff_out = forward(block.ff, normed)
    x = x .+ ff_out
    
    return x
end

# ═══════════════════════════════════════════════════════════════════════════════
# FULL TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Transformer

Full transformer model.

# Fields
- `config`: TransformerConfig
- `embed`: Token embedding matrix [vocab_size, d_model]
- `blocks`: Vector of TransformerBlock layers
- `final_norm_weight`, `final_norm_bias`: Final layer norm parameters
- `lm_head`: Language model head [d_model, vocab_size]
"""
mutable struct Transformer
    config::TransformerConfig
    embed::Matrix{Float32}
    blocks::Vector{TransformerBlock}
    final_norm_weight::Vector{Float32}
    final_norm_bias::Vector{Float32}
    lm_head::Matrix{Float32}
end

"""
    Transformer(config::TransformerConfig)

Create Transformer model from config.

# Arguments
- `config`: TransformerConfig

# Returns
- Transformer instance

# Examples
```julia
config = TransformerConfig(d_model=1024, n_layers=24)
model = Transformer(config)
```
"""
function Transformer(config::TransformerConfig)
    # Initialize token embeddings
    embed = randn(Float32, config.vocab_size, config.d_model) .* DEFAULT_EMBED_SCALE
    
    # Create transformer blocks
    blocks = [TransformerBlock(config) for _ in 1:config.n_layers]
    
    # Final layer norm
    final_norm_weight = ones(Float32, config.d_model)
    final_norm_bias = zeros(Float32, config.d_model)
    
    # Language model head
    lm_head = randn(Float32, config.d_model, config.vocab_size) .* DEFAULT_LM_HEAD_SCALE
    
    return Transformer(
        config, embed, blocks, final_norm_weight, final_norm_bias, lm_head
    )
end

"""
    forward(model, input_ids; start_pos=0)

Forward pass through transformer.

# Arguments
- `model`: Transformer model
- `input_ids`: Input token IDs [batch, seq_len]
- `start_pos`: Starting position for RoPE (default: 0)

# Returns
- Logits [batch, seq_len, vocab_size]

# Examples
```julia
model = Transformer(config)
input_ids = rand(0:config.vocab_size-1, 2, 128)
logits = forward(model, input_ids)
```
"""
function forward(
    model::Transformer,
    input_ids::Array{Int, 2};
    start_pos::Int = 0
)
    batch, seq_len = size(input_ids)
    
    # Validate input IDs
    if any(id -> id < 0 || id >= model.config.vocab_size, input_ids)
        throw(ArgumentError(
            "Input IDs must be in [0, vocab_size-1] = [0, $(model.config.vocab_size-1)]"
        ))
    end
    
    # Token embeddings (Julia is 1-indexed)
    x = model.embed[input_ids .+ 1, :]
    x = reshape(x, batch, seq_len, model.config.d_model)
    
    # Create causal mask
    mask = create_causal_mask(seq_len, start_pos)
    
    # Transformer blocks
    for block in model.blocks
        x = forward(block, x, mask=mask, start_pos=start_pos)
    end
    
    # Final normalization
    x = layer_norm(
        x, model.final_norm_weight, model.final_norm_bias,
        ε=model.config.layer_norm_eps
    )
    
    # Language model head
    x_flat = reshape(x, :, model.config.d_model)
    logits = x_flat * model.lm_head
    logits = reshape(logits, batch, seq_len, model.config.vocab_size)
    
    return logits
end

"""
    create_causal_mask(seq_len, start_pos=0)

Create causal attention mask.

# Arguments
- `seq_len`: Sequence length
- `start_pos`: Starting position (default: 0)

# Returns
- Causal mask [1, 1, seq_len, seq_len] with -Inf for future positions

# Examples
```julia
mask = create_causal_mask(128)
```
"""
function create_causal_mask(seq_len::Int, start_pos::Int = 0)
    if seq_len <= 0
        throw(ArgumentError("seq_len must be positive, got $seq_len"))
    end
    
    mask = fill(-Inf32, seq_len, seq_len)
    
    # Allow attention to current and past positions
    @inbounds for i in 1:seq_len
        for j in 1:i
            mask[i, j] = 0.0f0
        end
    end
    
    return reshape(mask, 1, 1, seq_len, seq_len)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generate(model, input_ids; max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9)

Generate tokens using the transformer model.

# Arguments
- `model`: Transformer model
- `input_ids`: Initial token IDs (vector)
- `max_new_tokens`: Maximum number of tokens to generate (default: 100)
- `temperature`: Sampling temperature (default: 1.0)
- `top_k`: Top-k sampling parameter (default: 50)
- `top_p`: Top-p (nucleus) sampling parameter (default: 0.9)

# Returns
- Vector of generated token IDs (including input)

# Examples
```julia
model = Transformer(config)
input_ids = [1, 2, 3]
tokens = generate(model, input_ids, max_new_tokens=50, temperature=0.8)
```
"""
function generate(
    model::Transformer,
    input_ids::Vector{Int};
    max_new_tokens::Int = DEFAULT_MAX_NEW_TOKENS,
    temperature::Float32 = DEFAULT_TEMPERATURE,
    top_k::Int = DEFAULT_TOP_K,
    top_p::Float32 = DEFAULT_TOP_P
)
    # Validate inputs
    if isempty(input_ids)
        throw(ArgumentError("input_ids cannot be empty"))
    end
    if max_new_tokens <= 0
        throw(ArgumentError("max_new_tokens must be positive, got $max_new_tokens"))
    end
    if temperature <= 0.0f0
        throw(ArgumentError("temperature must be positive, got $temperature"))
    end
    if top_k <= 0
        throw(ArgumentError("top_k must be positive, got $top_k"))
    end
    if top_p <= 0.0f0 || top_p > 1.0f0
        throw(ArgumentError("top_p must be in (0, 1], got $top_p"))
    end
    
    tokens = copy(input_ids)
    
    for _ in 1:max_new_tokens
        # Get logits for last position
        input_batch = reshape(tokens, 1, :)
        logits = forward(model, input_batch)
        next_token_logits = logits[1, end, :]
        
        # Apply temperature
        if temperature != 1.0f0
            next_token_logits ./= temperature
        end
        
        # Sample with top-k and top-p
        probs = softmax(next_token_logits)
        next_token = sample_top_k_top_p(probs, top_k, top_p)
        
        push!(tokens, next_token)
        
        # Check for EOS token
        if next_token == DEFAULT_EOS_TOKEN_ID
            break
        end
    end
    
    return tokens
end

"""
    sample_top_k_top_p(probs, top_k, top_p)

Sample from distribution with top-k and top-p filtering.

# Arguments
- `probs`: Probability distribution (vector)
- `top_k`: Number of top tokens to consider
- `top_p`: Cumulative probability threshold

# Returns
- Sampled token index (0-indexed)

# Examples
```julia
probs = softmax(logits)
token = sample_top_k_top_p(probs, 50, 0.9)
```
"""
function sample_top_k_top_p(probs::Vector{Float32}, top_k::Int, top_p::Float32)
    # Validate inputs
    if isempty(probs)
        throw(ArgumentError("probs cannot be empty"))
    end
    if top_k <= 0
        throw(ArgumentError("top_k must be positive, got $top_k"))
    end
    if top_p <= 0.0f0 || top_p > 1.0f0
        throw(ArgumentError("top_p must be in (0, 1], got $top_p"))
    end
    
    # Step 1: Top-k filtering - get indices of top-k probabilities
    sorted_indices = sortperm(probs, rev=true)
    k_limit = min(top_k, length(probs))
    top_k_indices = sorted_indices[1:k_limit]
    
    # Step 2: Top-p (nucleus) filtering - find cutoff based on cumulative probability
    cumsum_probs = 0.0f0
    cutoff_idx = length(top_k_indices)
    
    @inbounds for (i, idx) in enumerate(top_k_indices)
        cumsum_probs += probs[idx]
        if cumsum_probs >= top_p
            cutoff_idx = i
            break
        end
    end
    
    # Step 3: Filter and renormalize probabilities
    filtered_indices = top_k_indices[1:cutoff_idx]
    filtered_probs = probs[filtered_indices]
    
    # Renormalize to ensure probabilities sum to 1
    prob_sum = sum(filtered_probs)
    if prob_sum > 0.0f0
        filtered_probs ./= prob_sum
    else
        # Fallback: uniform distribution if all probabilities are zero
        filtered_probs = fill(1.0f0 / length(filtered_indices), length(filtered_indices))
    end
    
    # Step 4: Sample from filtered distribution using cumulative distribution
    r = rand(Float32)
    cumsum = 0.0f0
    
    @inbounds for (prob, idx) in zip(filtered_probs, filtered_indices)
        cumsum += prob
        if r <= cumsum
            return idx - 1  # Convert back to 0-indexed (Julia uses 1-indexed)
        end
    end
    
    # Fallback: return last token if numerical issues occur
    return filtered_indices[end] - 1
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

export TransformerConfig, Transformer, MultiHeadAttention, FeedForward
export TransformerBlock, generate, create_causal_mask
export precompute_rope_freqs, apply_rope
