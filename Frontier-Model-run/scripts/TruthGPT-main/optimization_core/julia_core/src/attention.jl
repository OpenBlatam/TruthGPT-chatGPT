"""
High-Performance Attention Mechanisms

Optimized implementations using Julia's native parallelism and SIMD.
Supports standard attention, Flash Attention, and Rotary Position Embeddings (RoPE).
"""

using LinearAlgebra
using LoopVectorization

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default configuration values
const DEFAULT_NUM_HEADS = 12
const DEFAULT_HEAD_DIM = 64
const DEFAULT_DROPOUT = 0.0f0
const DEFAULT_BLOCK_SIZE = 64
const DEFAULT_ROPE_BASE = 10000.0f0

# Numerical constants
const NEGATIVE_INFINITY = -Inf
const DEFAULT_XAVIER_SCALE = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AttentionConfig

Configuration for attention computation.

# Fields
- `num_heads`: Number of attention heads
- `head_dim`: Dimension of each attention head
- `dropout`: Dropout probability (0.0 = no dropout)
- `use_flash`: Use Flash Attention for memory efficiency
- `block_size`: Block size for Flash Attention
- `use_causal`: Use causal masking (for autoregressive models)

# Examples
```julia
config = AttentionConfig(
    num_heads=8,
    head_dim=64,
    use_flash=true,
    use_causal=true
)
```
"""
struct AttentionConfig
    num_heads::Int
    head_dim::Int
    dropout::Float32
    use_flash::Bool
    block_size::Int
    use_causal::Bool
    
    function AttentionConfig(
        num_heads::Int = DEFAULT_NUM_HEADS,
        head_dim::Int = DEFAULT_HEAD_DIM,
        dropout::Float32 = DEFAULT_DROPOUT,
        use_flash::Bool = true,
        block_size::Int = DEFAULT_BLOCK_SIZE,
        use_causal::Bool = true
    )
        # Validate all parameters
        validate_attention_config(num_heads, head_dim, dropout, block_size)
        
        new(num_heads, head_dim, dropout, use_flash, block_size, use_causal)
    end
end

"""
    AttentionConfig(; kwargs...)

Create AttentionConfig with keyword arguments.
"""
function AttentionConfig(;
    num_heads::Int = DEFAULT_NUM_HEADS,
    head_dim::Int = DEFAULT_HEAD_DIM,
    dropout::Float32 = DEFAULT_DROPOUT,
    use_flash::Bool = true,
    block_size::Int = DEFAULT_BLOCK_SIZE,
    use_causal::Bool = true
)
    AttentionConfig(num_heads, head_dim, dropout, use_flash, block_size, use_causal)
end

"""
    d_model(config::AttentionConfig)

Get total model dimension (num_heads * head_dim).

# Arguments
- `config`: AttentionConfig

# Returns
- Total model dimension
"""
d_model(config::AttentionConfig) = config.num_heads * config.head_dim

"""
    scale(config::AttentionConfig)

Get attention scale factor (1 / sqrt(head_dim)).

# Arguments
- `config`: AttentionConfig

# Returns
- Scale factor as Float32
"""
scale(config::AttentionConfig) = Float32(1.0 / sqrt(config.head_dim))

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_attention_config(num_heads, head_dim, dropout, block_size)

Validate attention configuration parameters.

# Arguments
- `num_heads`: Number of attention heads
- `head_dim`: Dimension of each attention head
- `dropout`: Dropout probability
- `block_size`: Block size for Flash Attention

# Throws
- `ArgumentError` if any parameter is invalid
"""
function validate_attention_config(
    num_heads::Int,
    head_dim::Int,
    dropout::Float32,
    block_size::Int
)
    validate_num_heads(num_heads)
    validate_head_dim(head_dim)
    validate_attention_dropout(dropout)
    validate_block_size(block_size)
end

"""
    validate_num_heads(num_heads)

Validate number of attention heads.

# Arguments
- `num_heads`: Number of attention heads

# Throws
- `ArgumentError` if num_heads is invalid
"""
function validate_num_heads(num_heads::Int)
    if num_heads <= 0
        throw(ArgumentError("num_heads must be positive, got $num_heads"))
    end
end

"""
    validate_head_dim(head_dim)

Validate head dimension.

# Arguments
- `head_dim`: Dimension of each attention head

# Throws
- `ArgumentError` if head_dim is invalid
"""
function validate_head_dim(head_dim::Int)
    if head_dim <= 0
        throw(ArgumentError("head_dim must be positive, got $head_dim"))
    end
end

"""
    validate_attention_dropout(dropout)

Validate dropout probability for attention.

# Arguments
- `dropout`: Dropout probability

# Throws
- `ArgumentError` if dropout is invalid
"""
function validate_attention_dropout(dropout::Float32)
    if dropout < 0.0 || dropout > 1.0
        throw(ArgumentError("dropout must be in [0, 1], got $dropout"))
    end
end

"""
    validate_block_size(block_size)

Validate block size for Flash Attention.

# Arguments
- `block_size`: Block size

# Throws
- `ArgumentError` if block_size is invalid
"""
function validate_block_size(block_size::Int)
    if block_size <= 0
        throw(ArgumentError("block_size must be positive, got $block_size"))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_attention_inputs(Q, K, V)

Validate that Q, K, V tensors have compatible shapes.

# Arguments
- `Q`: Query tensor
- `K`: Key tensor
- `V`: Value tensor

# Throws
- `ArgumentError` if shapes are incompatible
"""
function validate_attention_inputs(Q, K, V)
    if ndims(Q) != 4 || ndims(K) != 4 || ndims(V) != 4
        throw(ArgumentError("Q, K, V must be 4D tensors [batch, heads, seq, head_dim]"))
    end
    
    batch_q, heads_q, seq_q, head_dim_q = size(Q)
    batch_k, heads_k, seq_k, head_dim_k = size(K)
    batch_v, heads_v, seq_v, head_dim_v = size(V)
    
    if batch_q != batch_k || batch_q != batch_v
        throw(DimensionMismatch("Batch sizes must match: Q=$batch_q, K=$batch_k, V=$batch_v"))
    end
    if heads_q != heads_k || heads_q != heads_v
        throw(DimensionMismatch("Head counts must match: Q=$heads_q, K=$heads_k, V=$heads_v"))
    end
    if head_dim_q != head_dim_k || head_dim_q != head_dim_v
        throw(DimensionMismatch("Head dimensions must match: Q=$head_dim_q, K=$head_dim_k, V=$head_dim_v"))
    end
    if seq_k != seq_v
        throw(DimensionMismatch("Key and value sequence lengths must match: K=$seq_k, V=$seq_v"))
    end
end

"""
    compute_qk_scores(Q, K, scale)

Compute Q @ K^T attention scores with scaling.

# Arguments
- `Q`: Query tensor [batch, heads, seq_q, head_dim]
- `K`: Key tensor [batch, heads, seq_k, head_dim]
- `scale`: Scale factor (typically 1/sqrt(head_dim))

# Returns
- Attention scores [batch, heads, seq_q, seq_k]
"""
function compute_qk_scores(Q::Array{T, 4}, K::Array{T, 4}, scale::T) where T
    batch, heads, seq_q, head_dim = size(Q)
    _, _, seq_k, _ = size(K)
    
    attn_scores = zeros(T, batch, heads, seq_q, seq_k)
    
    @turbo for b in 1:batch, h in 1:heads, i in 1:seq_q, j in 1:seq_k
        sum_qk = zero(T)
        for d in 1:head_dim
            sum_qk += Q[b, h, i, d] * K[b, h, j, d]
        end
        attn_scores[b, h, i, j] = sum_qk * scale
    end
    
    return attn_scores
end

"""
    compute_attention_output(attn_weights, V)

Compute attention output from weights and values.

# Arguments
- `attn_weights`: Attention weights [batch, heads, seq_q, seq_k]
- `V`: Value tensor [batch, heads, seq_k, head_dim]

# Returns
- Output tensor [batch, heads, seq_q, head_dim]
"""
function compute_attention_output(attn_weights::Array{T, 4}, V::Array{T, 4}) where T
    batch, heads, seq_q, seq_k = size(attn_weights)
    _, _, _, head_dim = size(V)
    
    output = zeros(T, batch, heads, seq_q, head_dim)
    
    @turbo for b in 1:batch, h in 1:heads, i in 1:seq_q, d in 1:head_dim
        sum_v = zero(T)
        for j in 1:seq_k
            sum_v += attn_weights[b, h, i, j] * V[b, h, j, d]
        end
        output[b, h, i, d] = sum_v
    end
    
    return output
end

# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    attention_forward(Q, K, V, config; mask=nothing)

Compute scaled dot-product attention.

# Arguments
- `Q`: Query tensor [batch, heads, seq_q, head_dim]
- `K`: Key tensor [batch, heads, seq_k, head_dim]
- `V`: Value tensor [batch, heads, seq_k, head_dim]
- `config`: AttentionConfig
- `mask`: Optional attention mask (added to scores before softmax)

# Returns
- Output tensor [batch, heads, seq_q, head_dim]

# Examples
```julia
Q = randn(Float32, 2, 8, 128, 64)
K = randn(Float32, 2, 8, 128, 64)
V = randn(Float32, 2, 8, 128, 64)
config = AttentionConfig(num_heads=8, head_dim=64)
output = attention_forward(Q, K, V, config)
```
"""
function attention_forward(
    Q::Array{T, 4},
    K::Array{T, 4},
    V::Array{T, 4},
    config::AttentionConfig;
    mask::Union{Nothing, AbstractArray} = nothing
) where T <: AbstractFloat
    # Validate inputs
    validate_attention_inputs(Q, K, V)
    
    # Compute attention scores: Q @ K^T / sqrt(head_dim)
    attn_scores = compute_qk_scores(Q, K, scale(config))
    
    # Apply causal mask if needed
    if config.use_causal
        apply_causal_mask!(attn_scores)
    end
    
    # Apply optional mask
    if !isnothing(mask)
        if size(mask) != size(attn_scores)
            throw(DimensionMismatch("Mask size $(size(mask)) must match scores size $(size(attn_scores))"))
        end
        attn_scores .+= mask
    end
    
    # Compute attention weights with softmax
    attn_weights = softmax_4d(attn_scores)
    
    # Apply dropout if specified (only during training)
    if config.dropout > 0.0f0
        attn_weights = apply_dropout(attn_weights, config.dropout, training=true)
    end
    
    # Compute output: attn_weights @ V
    output = compute_attention_output(attn_weights, V)
    
    return output
end

"""
    apply_causal_mask!(scores)

Apply causal mask in-place to attention scores.

Sets scores[i, j] = -Inf for j > i to prevent attending to future positions.

# Arguments
- `scores`: Attention scores [batch, heads, seq_q, seq_k] (modified in-place)
"""
function apply_causal_mask!(scores::Array{T, 4}) where T
    _, _, seq_q, seq_k = size(scores)
    
    @inbounds for i in 1:seq_q, j in (i+1):seq_k
        scores[:, :, i, j] .= T(NEGATIVE_INFINITY)
    end
end

"""
    softmax_4d(x)

Compute softmax over the last dimension of a 4D tensor.

Uses numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))

# Arguments
- `x`: Input tensor [batch, heads, seq, seq]

# Returns
- Softmax output [batch, heads, seq, seq]
"""
function softmax_4d(x::Array{T, 4}) where T
    # Find max along last dimension for numerical stability
    max_x = maximum(x, dims=4)
    
    # Compute exp(x - max) for numerical stability
    exp_x = exp.(x .- max_x)
    
    # Normalize by sum
    exp_x ./ sum(exp_x, dims=4)
end

"""
    apply_dropout(x, dropout_rate; rng=Random.GLOBAL_RNG, training=true)

Apply dropout to attention weights during training.

Randomly sets elements to zero with probability `dropout_rate` and scales
remaining elements by 1 / (1 - dropout_rate) to maintain expected value.

# Arguments
- `x`: Input tensor [batch, heads, seq_q, seq_k]
- `dropout_rate`: Dropout probability (0.0 = no dropout, 1.0 = all zeros)
- `rng`: Random number generator (default: GLOBAL_RNG)
- `training`: If false, returns input unchanged (default: true)

# Returns
- Tensor with dropout applied (same shape as input)

# Examples
```julia
weights = randn(Float32, 2, 8, 128, 128)
dropped = apply_dropout(weights, 0.1f0, training=true)
```
"""
function apply_dropout(
    x::Array{T, 4},
    dropout_rate::Float32;
    rng::AbstractRNG=Random.GLOBAL_RNG,
    training::Bool=true
) where T
    # No dropout if rate is zero or not training
    if dropout_rate <= 0.0f0 || !training
        return x
    end
    
    if dropout_rate >= 1.0f0
        return zeros(T, size(x))
    end
    
    # Create dropout mask: 1 with probability (1 - dropout_rate), 0 otherwise
    mask = rand(rng, T, size(x)) .> dropout_rate
    
    # Scale by 1 / (1 - dropout_rate) to maintain expected value
    scale = T(1.0 / (1.0 - dropout_rate))
    
    return x .* mask .* scale
end

# ═══════════════════════════════════════════════════════════════════════════════
# FLASH ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    flash_attention(Q, K, V, config)

Memory-efficient Flash Attention implementation.

Uses block-wise computation to reduce memory from O(N²) to O(N).
This is critical for long sequences where standard attention is infeasible.

# Arguments
- `Q`: Query tensor [batch, heads, seq_q, head_dim]
- `K`: Key tensor [batch, heads, seq_k, head_dim]
- `V`: Value tensor [batch, heads, seq_k, head_dim]
- `config`: AttentionConfig with block_size

# Returns
- Output tensor [batch, heads, seq_q, head_dim]

# Examples
```julia
Q = randn(Float32, 2, 8, 2048, 64)
K = randn(Float32, 2, 8, 2048, 64)
V = randn(Float32, 2, 8, 2048, 64)
config = AttentionConfig(block_size=64, use_flash=true)
output = flash_attention(Q, K, V, config)
```
"""
function flash_attention(
    Q::Array{T, 4},
    K::Array{T, 4},
    V::Array{T, 4},
    config::AttentionConfig
) where T <: AbstractFloat
    # Validate inputs
    validate_attention_inputs(Q, K, V)
    
    batch, heads, seq_q, head_dim = size(Q)
    _, _, seq_k, _ = size(K)
    
    block_size = config.block_size
    s = scale(config)
    
    # Initialize output and running statistics
    output = zeros(T, batch, heads, seq_q, head_dim)
    l = zeros(T, batch, heads, seq_q)  # Running sum of exp values
    m = fill(T(NEGATIVE_INFINITY), batch, heads, seq_q)  # Running max
    
    # Process K/V in blocks
    num_blocks_k = cld(seq_k, block_size)
    
    for bk in 1:num_blocks_k
        # Extract K/V block
        k_start = (bk - 1) * block_size + 1
        k_end = min(bk * block_size, seq_k)
        k_size = k_end - k_start + 1
        
        K_block = @view K[:, :, k_start:k_end, :]
        V_block = @view V[:, :, k_start:k_end, :]
        
        # Compute attention scores for this block
        attn_block = compute_attention_block(Q, K_block, s)
        
        # Apply causal mask if needed
        if config.use_causal
            mask_causal_block!(attn_block, k_start)
        end
        
        # Update running max
        m_new = maximum(attn_block, dims=4)[:, :, :, 1]
        m_combined = max.(m, m_new)
        
        # Compute exponentials with numerical stability
        exp_old = exp.(m .- m_combined)
        exp_new = exp.(attn_block .- reshape(m_combined, batch, heads, seq_q, 1))
        
        # Update running sum of exponentials
        l_new = sum(exp_new, dims=4)[:, :, :, 1]
        l .= l .* exp_old .+ l_new
        
        # Update output incrementally
        for d in 1:head_dim
            output[:, :, :, d] .= output[:, :, :, d] .* exp_old
            for j in 1:k_size
                output[:, :, :, d] .+= exp_new[:, :, :, j] .* V_block[:, :, j, d]
            end
        end
        
        # Update running max
        m .= m_combined
    end
    
    # Normalize output by sum of exponentials
    for d in 1:head_dim
        output[:, :, :, d] ./= l
    end
    
    return output
end

"""
    compute_attention_block(Q, K_block, scale)

Compute attention scores for a block of keys.

# Arguments
- `Q`: Query tensor [batch, heads, seq_q, head_dim]
- `K_block`: Key block [batch, heads, k_size, head_dim]
- `scale`: Scale factor

# Returns
- Attention scores [batch, heads, seq_q, k_size]
"""
function compute_attention_block(Q::Array{T, 4}, K_block::Array{T, 4}, scale::T) where T
    batch, heads, seq_q, head_dim = size(Q)
    _, _, k_size, _ = size(K_block)
    
    attn = zeros(T, batch, heads, seq_q, k_size)
    
    @turbo for b in 1:batch, h in 1:heads, i in 1:seq_q, j in 1:k_size
        sum_qk = zero(T)
        for d in 1:head_dim
            sum_qk += Q[b, h, i, d] * K_block[b, h, j, d]
        end
        attn[b, h, i, j] = sum_qk * scale
    end
    
    return attn
end

"""
    mask_causal_block!(attn, k_start)

Apply causal mask to an attention block in-place.

# Arguments
- `attn`: Attention scores block [batch, heads, seq_q, k_size] (modified in-place)
- `k_start`: Starting position of this block in the full sequence
"""
function mask_causal_block!(attn::Array{T, 4}, k_start::Int) where T
    batch, heads, seq_q, k_size = size(attn)
    
    @inbounds for i in 1:seq_q
        for j in 1:k_size
            global_j = k_start + j - 1
            if global_j > i
                attn[:, :, i, j] .= T(NEGATIVE_INFINITY)
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    MultiHeadAttention{T}

Multi-head attention layer with learnable projections.

# Fields
- `config`: AttentionConfig
- `Wq`: Query projection matrix [d_model, d_model]
- `Wk`: Key projection matrix [d_model, d_model]
- `Wv`: Value projection matrix [d_model, d_model]
- `Wo`: Output projection matrix [d_model, d_model]
"""
mutable struct MultiHeadAttention{T}
    config::AttentionConfig
    Wq::Matrix{T}
    Wk::Matrix{T}
    Wv::Matrix{T}
    Wo::Matrix{T}
end

"""
    MultiHeadAttention(config; T::Type=Float32)

Create a MultiHeadAttention layer with Xavier initialization.

# Arguments
- `config`: AttentionConfig
- `T`: Element type (default: Float32)

# Returns
- MultiHeadAttention instance
"""
function MultiHeadAttention(config::AttentionConfig; T::Type = Float32)
    d = d_model(config)
    scale_factor = T(sqrt(DEFAULT_XAVIER_SCALE / d))
    
    # Initialize with Xavier/Glorot initialization
    Wq = randn(T, d, d) * scale_factor
    Wk = randn(T, d, d) * scale_factor
    Wv = randn(T, d, d) * scale_factor
    Wo = randn(T, d, d) * scale_factor
    
    return MultiHeadAttention(config, Wq, Wk, Wv, Wo)
end

"""
    forward(mha, x; kv_cache=nothing)

Forward pass through multi-head attention.

# Arguments
- `mha`: MultiHeadAttention layer
- `x`: Input tensor [batch, seq_len, d_model]
- `kv_cache`: Optional KV cache (not yet implemented)

# Returns
- Output tensor [batch, seq_len, d_model]

# Examples
```julia
config = AttentionConfig(num_heads=8, head_dim=64)
mha = MultiHeadAttention(config)
x = randn(Float32, 2, 128, 512)
output = forward(mha, x)
```
"""
function forward(mha::MultiHeadAttention, x::Array{T, 3}; kv_cache = nothing) where T
    batch, seq_len, d = size(x)
    config = mha.config
    
    # Validate input dimension
    if d != d_model(config)
        throw(DimensionMismatch(
            "Input dimension $d must match d_model $(d_model(config))"
        ))
    end
    
    # Flatten batch and sequence for matrix multiplication
    x_flat = reshape(x, batch * seq_len, d)
    
    # Project to Q, K, V
    Q = x_flat * mha.Wq
    K = x_flat * mha.Wk
    V = x_flat * mha.Wv
    
    # Reshape to [batch, seq_len, heads, head_dim]
    Q = reshape(Q, batch, seq_len, config.num_heads, config.head_dim)
    K = reshape(K, batch, seq_len, config.num_heads, config.head_dim)
    V = reshape(V, batch, seq_len, config.num_heads, config.head_dim)
    
    # Permute to [batch, heads, seq_len, head_dim]
    Q = permutedims(Q, (1, 3, 2, 4))
    K = permutedims(K, (1, 3, 2, 4))
    V = permutedims(V, (1, 3, 2, 4))
    
    # Apply attention
    if config.use_flash
        attn_out = flash_attention(Q, K, V, config)
    else
        attn_out = attention_forward(Q, K, V, config)
    end
    
    # Permute back to [batch, seq_len, heads, head_dim]
    attn_out = permutedims(attn_out, (1, 3, 2, 4))
    
    # Flatten and project output
    attn_out = reshape(attn_out, batch * seq_len, d)
    output = attn_out * mha.Wo
    
    # Reshape to original shape
    return reshape(output, batch, seq_len, d)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ROTARY POSITION EMBEDDINGS (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RoPE{T}

Rotary Position Embeddings (RoPE) for relative position encoding.

# Fields
- `dim`: Dimension of embeddings
- `max_seq_len`: Maximum sequence length
- `cos_cache`: Cached cosine values [max_seq_len, half_dim]
- `sin_cache`: Cached sine values [max_seq_len, half_dim]
"""
struct RoPE{T}
    dim::Int
    max_seq_len::Int
    cos_cache::Matrix{T}
    sin_cache::Matrix{T}
end

"""
    RoPE(dim, max_seq_len; T::Type=Float32, base=10000.0)

Create RoPE with precomputed rotation matrices.

# Arguments
- `dim`: Dimension of embeddings (must be even)
- `max_seq_len`: Maximum sequence length
- `T`: Element type (default: Float32)
- `base`: Base frequency for position encoding (default: 10000.0)

# Returns
- RoPE instance with cached cos/sin values

# Examples
```julia
rope = RoPE(64, 2048)
```
"""
function RoPE(
    dim::Int,
    max_seq_len::Int;
    T::Type = Float32,
    base::T = T(DEFAULT_ROPE_BASE)
)
    if dim <= 0 || dim % 2 != 0
        throw(ArgumentError("dim must be positive and even, got $dim"))
    end
    if max_seq_len <= 0
        throw(ArgumentError("max_seq_len must be positive, got $max_seq_len"))
    end
    
    half_dim = dim ÷ 2
    
    # Compute inverse frequencies
    inv_freq = T[1.0 / (base^(2i / dim)) for i in 0:(half_dim-1)]
    
    # Precompute cos and sin caches
    cos_cache = zeros(T, max_seq_len, half_dim)
    sin_cache = zeros(T, max_seq_len, half_dim)
    
    for pos in 1:max_seq_len
        for (i, freq) in enumerate(inv_freq)
            angle = (pos - 1) * freq
            cos_cache[pos, i] = cos(angle)
            sin_cache[pos, i] = sin(angle)
        end
    end
    
    return RoPE(dim, max_seq_len, cos_cache, sin_cache)
end

"""
    apply_rope!(q, k, rope, start_pos=1)

Apply rotary position embeddings in-place to Q and K tensors.

# Arguments
- `q`: Query tensor [batch, heads, seq_len, head_dim] (modified in-place)
- `k`: Key tensor [batch, heads, seq_len, head_dim] (modified in-place)
- `rope`: RoPE instance
- `start_pos`: Starting position in sequence (default: 1)

# Examples
```julia
rope = RoPE(64, 2048)
Q = randn(Float32, 2, 8, 128, 64)
K = randn(Float32, 2, 8, 128, 64)
apply_rope!(Q, K, rope)
```
"""
function apply_rope!(
    q::Array{T, 4},
    k::Array{T, 4},
    rope::RoPE,
    start_pos::Int = 1
) where T
    batch, heads, seq_len, head_dim = size(q)
    
    # Validate dimensions
    if head_dim != rope.dim
        throw(DimensionMismatch(
            "head_dim $head_dim must match rope.dim $(rope.dim)"
        ))
    end
    if size(k) != size(q)
        throw(DimensionMismatch("Q and K must have same shape"))
    end
    
    half_dim = head_dim ÷ 2
    
    @inbounds for b in 1:batch, h in 1:heads, s in 1:seq_len
        pos = start_pos + s - 1
        
        # Check bounds
        if pos > rope.max_seq_len
            throw(ArgumentError(
                "Position $pos exceeds max_seq_len $(rope.max_seq_len)"
            ))
        end
        
        # Apply rotation to each pair of dimensions
        for i in 1:half_dim
            cos_val = rope.cos_cache[pos, i]
            sin_val = rope.sin_cache[pos, i]
            
            # Rotate Q
            q0 = q[b, h, s, i]
            q1 = q[b, h, s, i + half_dim]
            q[b, h, s, i] = q0 * cos_val - q1 * sin_val
            q[b, h, s, i + half_dim] = q0 * sin_val + q1 * cos_val
            
            # Rotate K
            k0 = k[b, h, s, i]
            k1 = k[b, h, s, i + half_dim]
            k[b, h, s, i] = k0 * cos_val - k1 * sin_val
            k[b, h, s, i + half_dim] = k0 * sin_val + k1 * cos_val
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

export AttentionConfig, d_model, scale
export attention_forward, flash_attention
export MultiHeadAttention, forward
export RoPE, apply_rope!
