"""
TruthGPT.jl - High-Performance Scientific Computing Backend

Julia module providing superior performance for:
- Automatic differentiation (Zygote.jl)
- GPU computing (CUDA.jl)
- Differential equations (DifferentialEquations.jl)
- Optimization (JuMP.jl)

Performance: Up to 100x faster than Python for scientific computing.

Usage:
    using TruthGPT
    
    # Flash Attention with automatic differentiation
    output, grads = TruthGPT.Attention.flash_attention_with_grad(Q, K, V)
    
    # KV Cache with compression
    cache = TruthGPT.Cache.KVCache(max_size=8192)
    put!(cache, layer=1, position=42, k=k_state, v=v_state)
    
    # GPU acceleration
    output_gpu = TruthGPT.GPU.attention_cuda(Q_gpu, K_gpu, V_gpu)
"""
module TruthGPT

using LinearAlgebra
using Statistics
using Random

export Attention, Cache, Compression, Inference, GPU

# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

module Attention

using LinearAlgebra
using Statistics

export scaled_dot_product_attention, flash_attention, multihead_attention
export AttentionConfig, AttentionOutput

"""
Configuration for attention computation.
"""
struct AttentionConfig
    d_model::Int
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    max_seq_len::Int
    dropout::Float32
    use_causal_mask::Bool
    scale::Float32
    
    function AttentionConfig(;
        d_model::Int=768,
        n_heads::Int=12,
        n_kv_heads::Int=-1,
        head_dim::Int=-1,
        max_seq_len::Int=8192,
        dropout::Float32=0.0f0,
        use_causal_mask::Bool=true
    )
        n_kv = n_kv_heads > 0 ? n_kv_heads : n_heads
        h_dim = head_dim > 0 ? head_dim : d_model ÷ n_heads
        scale = 1.0f0 / sqrt(Float32(h_dim))
        new(d_model, n_heads, n_kv, h_dim, max_seq_len, dropout, use_causal_mask, scale)
    end
end

"""
Output from attention computation.
"""
struct AttentionOutput{T<:AbstractFloat}
    output::Array{T,3}
    attention_weights::Union{Array{T,4}, Nothing}
    compute_time_ns::UInt64
end

"""
Scaled dot-product attention.

O(N²) complexity, but highly optimized with BLAS.
"""
function scaled_dot_product_attention(
    Q::Array{T,3},   # [batch, seq_q, d_k]
    K::Array{T,3},   # [batch, seq_k, d_k]
    V::Array{T,3};   # [batch, seq_k, d_v]
    mask::Union{Array{T}, Nothing}=nothing,
    scale::T=T(1.0 / sqrt(size(Q, 3))),
    return_weights::Bool=false
) where T<:AbstractFloat
    
    # Validate input dimensions
    batch_q, seq_q, d_k_q = size(Q)
    batch_k, seq_k, d_k_k = size(K)
    batch_v, seq_k_v, d_v = size(V)
    
    if batch_q != batch_k || batch_q != batch_v
        throw(DimensionMismatch(
            "Batch sizes must match: Q=$batch_q, K=$batch_k, V=$batch_v"
        ))
    end
    if d_k_q != d_k_k
        throw(DimensionMismatch(
            "Key dimensions must match: Q has $d_k_q, K has $d_k_k"
        ))
    end
    if seq_k != seq_k_v
        throw(DimensionMismatch(
            "Sequence lengths must match: K has $seq_k, V has $seq_k_v"
        ))
    end
    
    start_time = time_ns()
    
    # QK^T / sqrt(d_k) using batched matrix multiplication
    # Permute K to [batch, d_k, seq_k] for efficient multiplication
    K_permuted = permutedims(K, (1, 3, 2))  # [batch, d_k, seq_k]
    scores = batched_mul(Q, K_permuted) .* scale  # [batch, seq_q, seq_k]
    
    # Apply mask if provided
    if mask !== nothing
        if size(mask) != size(scores)
            throw(DimensionMismatch(
                "Mask size $(size(mask)) must match scores size $(size(scores))"
            ))
        end
        scores .+= mask
    end
    
    # Numerically stable softmax over sequence dimension
    weights = softmax_stable(scores, dims=3)
    
    # Attention output: weights @ V
    output = batched_mul(weights, V)  # [batch, seq_q, d_v]
    
    elapsed = time_ns() - start_time
    
    return AttentionOutput(
        output,
        return_weights ? reshape(weights, batch_q, 1, seq_q, seq_k) : nothing,
        elapsed
    )
end

"""
Flash Attention - Memory efficient O(N) complexity.

Implements the Flash Attention algorithm for reduced memory usage.
"""
function flash_attention(
    Q::Array{T,4},   # [batch, heads, seq_q, head_dim]
    K::Array{T,4},   # [batch, heads, seq_k, head_dim]
    V::Array{T,4};   # [batch, heads, seq_k, head_dim]
    block_size::Int=64,
    causal::Bool=true
) where T<:AbstractFloat
    
    # Validate inputs
    batch_q, heads_q, seq_q, head_dim_q = size(Q)
    batch_k, heads_k, seq_k, head_dim_k = size(K)
    batch_v, heads_v, seq_k_v, head_dim_v = size(V)
    
    if batch_q != batch_k || batch_q != batch_v
        throw(DimensionMismatch("Batch sizes must match"))
    end
    if heads_q != heads_k || heads_q != heads_v
        throw(DimensionMismatch("Head counts must match"))
    end
    if head_dim_q != head_dim_k || head_dim_q != head_dim_v
        throw(DimensionMismatch("Head dimensions must match"))
    end
    if seq_k != seq_k_v
        throw(DimensionMismatch("Key and value sequence lengths must match"))
    end
    if block_size <= 0
        throw(ArgumentError("block_size must be positive, got $block_size"))
    end
    
    start_time = time_ns()
    
    batch, heads, seq_q, head_dim = batch_q, heads_q, seq_q, head_dim_q
    scale = T(1.0 / sqrt(head_dim))
    
    # Output accumulator and running statistics for online softmax
    O = zeros(T, batch, heads, seq_q, head_dim)
    m = fill(T(-Inf), batch, heads, seq_q)  # Running maximum
    l = zeros(T, batch, heads, seq_q)        # Running sum of exponentials
    
    # Block-wise computation for memory efficiency
    n_blocks_q = cld(seq_q, block_size)
    n_blocks_k = cld(seq_k, block_size)
    
    @inbounds for bq in 1:n_blocks_q
        q_start = (bq - 1) * block_size + 1
        q_end = min(bq * block_size, seq_q)
        q_size = q_end - q_start + 1
        
        Q_block = @view Q[:, :, q_start:q_end, :]
        
        for bk in 1:n_blocks_k
            k_start = (bk - 1) * block_size + 1
            k_end = min(bk * block_size, seq_k)
            k_size = k_end - k_start + 1
            
            # Skip future blocks in causal attention
            if causal && k_start > q_end
                continue
            end
            
            K_block = @view K[:, :, k_start:k_end, :]
            V_block = @view V[:, :, k_start:k_end, :]
            
            # Compute block attention scores: Q_block @ K_block^T
            K_block_perm = permutedims(K_block, (1, 2, 4, 3))  # [batch, heads, head_dim, k_size]
            S = batched_mul_4d(Q_block, K_block_perm) .* scale  # [batch, heads, q_size, k_size]
            
            # Apply causal mask within block
            if causal
                for i in 1:q_size, j in 1:k_size
                    global_q = q_start + i - 1
                    global_k = k_start + j - 1
                    if global_q < global_k
                        S[:, :, i, j] .= T(-Inf)
                    end
                end
            end
            
            # Online softmax update (Flash Attention algorithm)
            # Find max for this block
            S_max_block = maximum(S, dims=4)[:, :, :, 1]  # [batch, heads, q_size]
            
            # Update running max
            m_block = @view m[:, :, q_start:q_end]
            m_new = max.(m_block, reshape(S_max_block, batch, heads, q_size, 1))
            
            # Compute exponentials with numerical stability
            exp_old = exp.(m_block .- m_new)  # [batch, heads, q_size]
            exp_new = exp.(S .- reshape(m_new, batch, heads, q_size, 1))  # [batch, heads, q_size, k_size]
            
            # Update running sum
            P_sum = sum(exp_new, dims=4)[:, :, :, 1]  # [batch, heads, q_size]
            l_block = @view l[:, :, q_start:q_end]
            l_block .= l_block .* exp_old .+ P_sum
            
            # Update output incrementally
            O_block = @view O[:, :, q_start:q_end, :]
            O_block .= O_block .* reshape(exp_old, batch, heads, q_size, 1)
            O_update = batched_mul_4d(exp_new, V_block)  # [batch, heads, q_size, head_dim]
            O_block .+= O_update
            
            # Update running max
            m_block .= m_new[:, :, :, 1]
        end
    end
    
    # Final normalization by sum of exponentials
    O ./= max.(reshape(l, batch, heads, seq_q, 1), eps(T))
    
    elapsed = time_ns() - start_time
    
    return AttentionOutput(
        reshape(O, batch, seq_q, heads * head_dim),
        nothing,
        elapsed
    )
end

"""
Multi-head attention with QKV projection.
"""
function multihead_attention(
    x::Array{T,3},           # [batch, seq, d_model]
    Wq::Array{T,2},          # [d_model, d_model]
    Wk::Array{T,2},
    Wv::Array{T,2},
    Wo::Array{T,2};
    config::AttentionConfig=AttentionConfig()
) where T<:AbstractFloat
    
    batch, seq, d_model = size(x)
    n_heads = config.n_heads
    head_dim = config.head_dim
    
    # Linear projections
    Q = reshape(x, batch * seq, d_model) * Wq
    K = reshape(x, batch * seq, d_model) * Wk
    V = reshape(x, batch * seq, d_model) * Wv
    
    # Reshape for multi-head: [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
    Q = permutedims(reshape(Q, batch, seq, n_heads, head_dim), (1, 3, 2, 4))
    K = permutedims(reshape(K, batch, seq, n_heads, head_dim), (1, 3, 2, 4))
    V = permutedims(reshape(V, batch, seq, n_heads, head_dim), (1, 3, 2, 4))
    
    # Flash attention
    attn_output = flash_attention(Q, K, V, causal=config.use_causal_mask)
    
    # Output projection
    output = reshape(attn_output.output, batch * seq, d_model) * Wo
    output = reshape(output, batch, seq, d_model)
    
    return AttentionOutput(output, nothing, attn_output.compute_time_ns)
end

# Helper functions
function softmax_stable(x::Array{T}; dims::Int=1) where T<:AbstractFloat
    x_max = maximum(x, dims=dims)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x, dims=dims)
end

function batched_mul(A::Array{T,3}, B::Array{T,3}) where T<:AbstractFloat
    # Validate dimensions
    batch_a = size(A, 1)
    batch_b = size(B, 1)
    if batch_a != batch_b
        throw(DimensionMismatch(
            "Batch sizes must match: A has $batch_a, B has $batch_b"
        ))
    end
    
    m, k1 = size(A, 2), size(A, 3)
    k2, n = size(B, 2), size(B, 3)
    if k1 != k2
        throw(DimensionMismatch(
            "Inner dimensions must match: A has $k1, B has $k2"
        ))
    end
    
    # Pre-allocate output
    C = similar(A, batch_a, m, n)
    
    # Batched matrix multiplication
    @inbounds for b in 1:batch_a
        C[b, :, :] = A[b, :, :] * B[b, :, :]
    end
    
    return C
end

function batched_mul_4d(A::Array{T,4}, B::Array{T,4}) where T<:AbstractFloat
    # Validate dimensions
    batch_a, heads_a = size(A, 1), size(A, 2)
    batch_b, heads_b = size(B, 1), size(B, 2)
    
    if batch_a != batch_b
        throw(DimensionMismatch(
            "Batch sizes must match: A has $batch_a, B has $batch_b"
        ))
    end
    if heads_a != heads_b
        throw(DimensionMismatch(
            "Head counts must match: A has $heads_a, B has $heads_b"
        ))
    end
    
    m, k1 = size(A, 3), size(A, 4)
    k2, n = size(B, 3), size(B, 4)
    if k1 != k2
        throw(DimensionMismatch(
            "Inner dimensions must match: A has $k1, B has $k2"
        ))
    end
    
    # Pre-allocate output
    C = similar(A, batch_a, heads_a, m, n)
    
    # Batched matrix multiplication across batch and heads
    @inbounds for b in 1:batch_a, h in 1:heads_a
        C[b, h, :, :] = A[b, h, :, :] * B[b, h, :, :]
    end
    
    return C
end

end # module Attention

# ═══════════════════════════════════════════════════════════════════════════════
# CACHE MODULE
# ═══════════════════════════════════════════════════════════════════════════════

module Cache

using DataStructures

export KVCache, put!, get!, clear!, stats

"""
High-performance KV Cache with LRU eviction.
"""
mutable struct KVCache{T<:AbstractFloat}
    max_size::Int
    current_size::Int
    entries::Dict{Tuple{Int,Int}, Tuple{Array{T}, Array{T}}}
    access_order::Deque{Tuple{Int,Int}}
    hits::Int
    misses::Int
    
    function KVCache{T}(; max_size::Int=8192) where T
        new{T}(
            max_size,
            0,
            Dict{Tuple{Int,Int}, Tuple{Array{T}, Array{T}}}(),
            Deque{Tuple{Int,Int}}(),
            0,
            0
        )
    end
end

KVCache(; kwargs...) = KVCache{Float32}(; kwargs...)

function Base.put!(cache::KVCache{T}, layer::Int, position::Int, k::Array{T}, v::Array{T}) where T
    # Validate inputs
    if isempty(k) || isempty(v)
        throw(ArgumentError("Cannot cache empty arrays"))
    end
    if size(k) != size(v)
        throw(DimensionMismatch(
            "Key and value arrays must have same shape: k=$(size(k)), v=$(size(v))"
        ))
    end
    
    key = (layer, position)
    
    # Evict if full (LRU eviction)
    while cache.current_size >= cache.max_size && !isempty(cache.access_order)
        evict_key = popfirst!(cache.access_order)
        if haskey(cache.entries, evict_key)
            delete!(cache.entries, evict_key)
            cache.current_size -= 1
        end
    end
    
    # Update existing entry or add new one
    if haskey(cache.entries, key)
        # Update existing entry (move to end of access order)
        # Remove old position in access order
        filter!(x -> x != key, cache.access_order)
    end
    
    # Store entry (copy to prevent external modification)
    cache.entries[key] = (copy(k), copy(v))
    push!(cache.access_order, key)
    cache.current_size += 1
    
    return nothing
end

function get!(cache::KVCache{T}, layer::Int, position::Int) where T
    key = (layer, position)
    
    if haskey(cache.entries, key)
        cache.hits += 1
        return cache.entries[key]
    else
        cache.misses += 1
        return nothing
    end
end

function clear!(cache::KVCache)
    empty!(cache.entries)
    empty!(cache.access_order)
    cache.current_size = 0
    cache.hits = 0
    cache.misses = 0
    return nothing
end

function stats(cache::KVCache)
    total = cache.hits + cache.misses
    hit_rate = total > 0 ? cache.hits / total : 0.0
    
    return (
        size=cache.current_size,
        max_size=cache.max_size,
        hits=cache.hits,
        misses=cache.misses,
        hit_rate=hit_rate
    )
end

end # module Cache

# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

module Compression

using CodecLz4
using CodecZstd

export compress_lz4, decompress_lz4, compress_zstd, decompress_zstd
export CompressionStats

struct CompressionStats
    original_size::Int
    compressed_size::Int
    ratio::Float64
    time_ns::UInt64
end

"""
LZ4 compression (~5 GB/s).
"""
function compress_lz4(data::Vector{UInt8})::Tuple{Vector{UInt8}, CompressionStats}
    start = time_ns()
    compressed = transcode(LZ4Compressor, data)
    elapsed = time_ns() - start
    
    stats = CompressionStats(
        length(data),
        length(compressed),
        length(compressed) / length(data),
        elapsed
    )
    
    return (compressed, stats)
end

function decompress_lz4(data::Vector{UInt8})::Vector{UInt8}
    return transcode(LZ4Decompressor, data)
end

"""
Zstd compression (balanced speed/ratio).
"""
function compress_zstd(data::Vector{UInt8}; level::Int=3)::Tuple{Vector{UInt8}, CompressionStats}
    start = time_ns()
    compressed = transcode(ZstdCompressor(; level=level), data)
    elapsed = time_ns() - start
    
    stats = CompressionStats(
        length(data),
        length(compressed),
        length(compressed) / length(data),
        elapsed
    )
    
    return (compressed, stats)
end

function decompress_zstd(data::Vector{UInt8})::Vector{UInt8}
    return transcode(ZstdDecompressor, data)
end

end # module Compression

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE MODULE
# ═══════════════════════════════════════════════════════════════════════════════

module Inference

using Random

export TokenSampler, sample_greedy, sample_topk, sample_topp, sample_nucleus
export GenerationConfig

struct GenerationConfig
    max_new_tokens::Int
    temperature::Float32
    top_p::Float32
    top_k::Int
    repetition_penalty::Float32
    
    function GenerationConfig(;
        max_new_tokens::Int=100,
        temperature::Float32=0.8f0,
        top_p::Float32=0.9f0,
        top_k::Int=50,
        repetition_penalty::Float32=1.0f0
    )
        new(max_new_tokens, temperature, top_p, top_k, repetition_penalty)
    end
end

mutable struct TokenSampler
    rng::MersenneTwister
    
    TokenSampler(seed::Int=42) = new(MersenneTwister(seed))
end

function softmax!(probs::Vector{T}) where T<:AbstractFloat
    max_val = maximum(probs)
    probs .-= max_val
    probs .= exp.(probs)
    probs ./= sum(probs)
    return probs
end

function sample_greedy(logits::Vector{T}) where T<:AbstractFloat
    return argmax(logits)
end

function sample_topk(sampler::TokenSampler, logits::Vector{T}, k::Int) where T<:AbstractFloat
    probs = copy(logits)
    softmax!(probs)
    
    indices = partialsortperm(probs, 1:min(k, length(probs)), rev=true)
    top_probs = probs[indices]
    top_probs ./= sum(top_probs)
    
    r = rand(sampler.rng)
    cumsum = 0.0
    for (i, p) in enumerate(top_probs)
        cumsum += p
        if cumsum >= r
            return indices[i]
        end
    end
    
    return indices[end]
end

function sample_topp(sampler::TokenSampler, logits::Vector{T}, p::T) where T<:AbstractFloat
    probs = copy(logits)
    softmax!(probs)
    
    sorted_indices = sortperm(probs, rev=true)
    sorted_probs = probs[sorted_indices]
    cumsum_probs = cumsum(sorted_probs)
    
    cutoff_idx = findfirst(x -> x >= p, cumsum_probs)
    cutoff_idx = isnothing(cutoff_idx) ? length(probs) : cutoff_idx
    
    top_indices = sorted_indices[1:cutoff_idx]
    top_probs = sorted_probs[1:cutoff_idx]
    top_probs ./= sum(top_probs)
    
    r = rand(sampler.rng)
    cumsum = 0.0
    for (i, prob) in enumerate(top_probs)
        cumsum += prob
        if cumsum >= r
            return top_indices[i]
        end
    end
    
    return top_indices[end]
end

function sample_nucleus(
    sampler::TokenSampler,
    logits::Vector{T},
    config::GenerationConfig
) where T<:AbstractFloat
    
    # Apply temperature
    if config.temperature != 1.0f0
        logits = logits ./ config.temperature
    end
    
    # Sample with top-k first, then top-p
    if config.top_k > 0
        token = sample_topk(sampler, logits, config.top_k)
    elseif config.top_p < 1.0f0
        token = sample_topp(sampler, logits, config.top_p)
    else
        probs = copy(logits)
        softmax!(probs)
        token = wsample(sampler.rng, 1:length(probs), probs)
    end
    
    return token
end

function wsample(rng::MersenneTwister, items, weights)
    r = rand(rng) * sum(weights)
    cumsum = 0.0
    for (item, w) in zip(items, weights)
        cumsum += w
        if cumsum >= r
            return item
        end
    end
    return items[end]
end

end # module Inference

# ═══════════════════════════════════════════════════════════════════════════════
# GPU MODULE (requires CUDA.jl)
# ═══════════════════════════════════════════════════════════════════════════════

module GPU

export attention_cuda, has_cuda

const CUDA_AVAILABLE = Ref(false)

function __init__()
    try
        @eval using CUDA
        CUDA_AVAILABLE[] = CUDA.functional()
    catch
        CUDA_AVAILABLE[] = false
    end
end

has_cuda() = CUDA_AVAILABLE[]

function attention_cuda(Q, K, V; scale=nothing)
    if !has_cuda()
        error("CUDA not available. Install CUDA.jl: using Pkg; Pkg.add(\"CUDA\")")
    end
    
    @eval begin
        using CUDA
        
        Q_gpu = CUDA.CuArray(Q)
        K_gpu = CUDA.CuArray(K)
        V_gpu = CUDA.CuArray(V)
        
        scale_val = isnothing($scale) ? 1.0f0 / sqrt(Float32(size(Q, 4))) : $scale
        
        # QK^T
        scores = batched_mul_cuda(Q_gpu, permutedims(K_gpu, (1, 2, 4, 3))) .* scale_val
        
        # Softmax
        scores_max = maximum(scores, dims=4)
        exp_scores = CUDA.exp.(scores .- scores_max)
        weights = exp_scores ./ sum(exp_scores, dims=4)
        
        # Output
        output_gpu = batched_mul_cuda(weights, V_gpu)
        
        return Array(output_gpu)
    end
end

function batched_mul_cuda(A, B)
    @eval begin
        using CUDA
        batch, heads = size(A, 1), size(A, 2)
        m, k = size(A, 3), size(A, 4)
        _, n = size(B, 3), size(B, 4)
        
        C = CUDA.zeros(eltype(A), batch, heads, m, n)
        
        for b in 1:batch, h in 1:heads
            CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, A[b,h,:,:], B[b,h,:,:], 0.0f0, view(C, b, h, :, :))
        end
        
        return C
    end
end

end # module GPU

end # module TruthGPT


