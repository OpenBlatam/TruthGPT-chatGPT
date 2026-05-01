"""
Utility functions for TruthGPT Julia Core

High-performance utilities for:
- Timing and profiling
- Memory management
- Data type conversion (Float32, Float16, BFloat16)
- Parallel computation
- Random number generation
- Numerical operations (softmax, activations, normalization)
"""

using Base.Threads
using Random
using Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Time conversion constants
const NANOSECONDS_PER_MICROSECOND = 1_000
const NANOSECONDS_PER_MILLISECOND = 1_000_000
const NANOSECONDS_PER_SECOND = 1_000_000_000

# Memory conversion constants
const BYTES_PER_KB = 1024
const BYTES_PER_MB = 1024^2
const BYTES_PER_GB = 1024^3
const BYTES_PER_TB = 1024^4

# Numerical stability constants
const DEFAULT_EPSILON = 1e-5f0
const GELU_APPROX_CONST = 0.044715f0
const SQRT_2_OVER_PI = sqrt(2.0f0 / Float32(π))

# Default benchmark parameters
const DEFAULT_BENCHMARK_ITERATIONS = 100
const DEFAULT_BENCHMARK_WARMUP = 10
const DEFAULT_CHUNK_SIZE = 1000

# Weight initialization constants
const XAVIER_SCALE_FACTOR = 2.0f0
const HE_SCALE_FACTOR = 2.0f0

# ═══════════════════════════════════════════════════════════════════════════════
# TIMING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @timed_block name expr

Execute expression and print timing information.

# Arguments
- `name`: Name/description of the operation
- `expr`: Expression to time

# Returns
- Result of the expression

# Examples
```julia
result = @timed_block "matrix_mult" begin
    A * B
end
# Output: [matrix_mult] 1.23ms
```
"""
macro timed_block(name, expr)
    quote
        local start_time = time_ns()
        local result = $(esc(expr))
        local elapsed_ns = time_ns() - start_time
        local elapsed_ms = elapsed_ns / NANOSECONDS_PER_MILLISECOND
        @info "[$($name)] $(round(elapsed_ms, digits=2))ms"
        result
    end
end

"""
    benchmark(f, args...; iterations=100, warmup=10)

Benchmark a function with warmup iterations to ensure JIT compilation.

# Arguments
- `f`: Function to benchmark
- `args...`: Arguments to pass to function
- `iterations`: Number of benchmark iterations (default: 100)
- `warmup`: Number of warmup iterations (default: 10)

# Returns
- Named tuple with `mean`, `std`, `min`, `max` times in nanoseconds

# Examples
```julia
result = benchmark(softmax, randn(Float32, 100, 1000), iterations=50)
println("Mean time: ", format_time(result.mean))
println("Std dev: ", format_time(result.std))
```

# Performance
- Warmup phase ensures JIT compilation is complete
- Uses high-resolution nanosecond timing
- Returns statistical summary of execution times
"""
function benchmark(
    f::Function,
    args...;
    iterations::Int = DEFAULT_BENCHMARK_ITERATIONS,
    warmup::Int = DEFAULT_BENCHMARK_WARMUP
)
    # Validate inputs
    if iterations <= 0
        throw(ArgumentError("iterations must be positive, got $iterations"))
    end
    if warmup < 0
        throw(ArgumentError("warmup must be non-negative, got $warmup"))
    end
    
    # Warmup phase: ensure JIT compilation
    for _ in 1:warmup
        f(args...)
    end
    
    # Benchmark phase: measure execution times
    times = Vector{Float64}(undef, iterations)
    for i in 1:iterations
        start = time_ns()
        f(args...)
        times[i] = Float64(time_ns() - start)
    end
    
    return (
        mean = mean(times),
        std = std(times),
        min = minimum(times),
        max = maximum(times)
    )
end

"""
    format_time(ns)

Format nanoseconds as human-readable string with appropriate units.

# Arguments
- `ns`: Time in nanoseconds

# Returns
- Formatted string with appropriate unit (ns, μs, ms, s)

# Examples
```julia
format_time(1_500_000)  # "1.5ms"
format_time(500)        # "500.0ns"
format_time(2_500_000_000)  # "2.5s"
```
"""
function format_time(ns::Number)
    if ns < 0
        throw(ArgumentError("Time cannot be negative, got $ns"))
    end
    
    if ns < NANOSECONDS_PER_MICROSECOND
        return "$(round(ns, digits=1))ns"
    elseif ns < NANOSECONDS_PER_MILLISECOND
        return "$(round(ns / NANOSECONDS_PER_MICROSECOND, digits=2))μs"
    elseif ns < NANOSECONDS_PER_SECOND
        return "$(round(ns / NANOSECONDS_PER_MILLISECOND, digits=2))ms"
    else
        return "$(round(ns / NANOSECONDS_PER_SECOND, digits=2))s"
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    format_bytes(bytes)

Format bytes as human-readable string with appropriate units.

# Arguments
- `bytes`: Size in bytes

# Returns
- Formatted string with appropriate unit (B, KB, MB, GB, TB)

# Examples
```julia
format_bytes(1_500_000)  # "1.43 MB"
format_bytes(1024)        # "1.0 KB"
format_bytes(1_073_741_824)  # "1.0 GB"
```
"""
function format_bytes(bytes::Integer)
    if bytes < 0
        throw(ArgumentError("bytes must be non-negative, got $bytes"))
    end
    
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 1
    value = Float64(bytes)
    
    # Convert to appropriate unit
    while value >= BYTES_PER_KB && idx < length(units)
        value /= BYTES_PER_KB
        idx += 1
    end
    
    return "$(round(value, digits=2)) $(units[idx])"
end

"""
    memory_info()

Get current memory usage information from garbage collector.

# Returns
- Named tuple with:
  - `allocated`: Total bytes allocated
  - `total_time_ns`: Total GC time in nanoseconds
  - `num_gc`: Number of GC pauses

# Examples
```julia
info = memory_info()
println("Allocated: ", format_bytes(info.allocated))
```
"""
function memory_info()
    gc_stats = Base.gc_num()
    return (
        allocated = gc_stats.allocd,
        total_time_ns = gc_stats.total_time,
        num_gc = gc_stats.pause
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    to_float32(data)

Convert any numeric array to Float32.

# Arguments
- `data`: Numeric array of any type

# Returns
- Array converted to Float32

# Examples
```julia
x = randn(Float64, 10, 10)
x_f32 = to_float32(x)
```
"""
function to_float32(data::AbstractArray{T}) where T <: Number
    return convert(Array{Float32}, data)
end

"""
    to_float16(data)

Convert any numeric array to Float16.

# Arguments
- `data`: Numeric array of any type

# Returns
- Array converted to Float16

# Examples
```julia
x = randn(Float32, 10, 10)
x_f16 = to_float16(x)
```

# Notes
- Float16 provides 2x memory savings but reduced precision
- Useful for inference when precision is less critical
"""
function to_float16(data::AbstractArray{T}) where T <: Number
    return convert(Array{Float16}, data)
end

"""
    to_bfloat16(data)

Convert Float32 array to BFloat16 representation (stored as UInt16).

BFloat16 (Brain Float) keeps the same exponent range as Float32
but reduces mantissa precision, making it ideal for training.

# Arguments
- `data`: Float32 array

# Returns
- UInt16 array containing BFloat16 values

# Examples
```julia
x = randn(Float32, 10, 10)
x_bf16 = to_bfloat16(x)
```

# Algorithm
- BFloat16 keeps upper 16 bits of Float32 (1 sign + 8 exponent + 7 mantissa)
- Lower 16 bits (mantissa) are discarded
"""
function to_bfloat16(data::AbstractArray{Float32})
    result = similar(data, UInt16)
    
    @inbounds @simd for i in eachindex(data)
        # BF16: keep upper 16 bits of Float32
        bits = reinterpret(UInt32, data[i])
        result[i] = UInt16(bits >> 16)
    end
    
    return result
end

"""
    from_bfloat16(data)

Convert from BFloat16 representation to Float32.

# Arguments
- `data`: UInt16 array containing BFloat16 values

# Returns
- Float32 array

# Examples
```julia
x_bf16 = to_bfloat16(randn(Float32, 10, 10))
x_f32 = from_bfloat16(x_bf16)
```

# Algorithm
- Extends BFloat16 to Float32 by adding 16 zero bits to mantissa
- Preserves sign and exponent, fills mantissa with zeros
"""
function from_bfloat16(data::AbstractArray{UInt16})
    result = similar(data, Float32)
    
    @inbounds @simd for i in eachindex(data)
        # Extend to Float32 by adding 16 zero bits
        bits = UInt32(data[i]) << 16
        result[i] = reinterpret(Float32, bits)
    end
    
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    parallel_map(f, data; chunk_size=1000)

Apply function f to data in parallel chunks using multiple threads.

# Arguments
- `f`: Function to apply to each element
- `data`: Vector of data to process
- `chunk_size`: Number of elements per thread chunk (default: 1000)

# Returns
- Vector of results (same length as input)

# Examples
```julia
data = [1, 2, 3, 4, 5]
results = parallel_map(x -> x^2, data)
# Returns: [1, 4, 9, 16, 25]
```

# Performance
- Uses @threads for parallel execution
- Chunking balances load across threads
- Efficient for CPU-bound operations
"""
function parallel_map(f::Function, data::AbstractVector; chunk_size::Int = DEFAULT_CHUNK_SIZE)
    n = length(data)
    
    if n == 0
        return similar(data)
    end
    
    if chunk_size <= 0
        throw(ArgumentError("chunk_size must be positive, got $chunk_size"))
    end
    
    # Infer return type from first element
    result_type = typeof(f(data[1]))
    results = Vector{result_type}(undef, n)
    
    # Process in chunks to balance load across threads
    @threads for i in 1:chunk_size:n
        chunk_end = min(i + chunk_size - 1, n)
        @inbounds for j in i:chunk_end
            results[j] = f(data[j])
        end
    end
    
    return results
end

"""
    parallel_reduce(f, op, data; chunk_size=1000)

Apply function f and reduce with operator op in parallel.

# Arguments
- `f`: Function to apply to each element before reduction
- `op`: Binary reduction operator (must be associative)
- `data`: Vector of data to process
- `chunk_size`: Number of elements per thread chunk (default: 1000)

# Returns
- Reduced result

# Examples
```julia
data = [1, 2, 3, 4, 5]
sum_squares = parallel_reduce(x -> x^2, +, data)
# Returns: 55 (1² + 2² + 3² + 4² + 5²)
```

# Performance
- Parallel reduction across chunks
- Final sequential reduction of partial results
- Efficient for large datasets
"""
function parallel_reduce(
    f::Function,
    op::Function,
    data::AbstractVector;
    chunk_size::Int = DEFAULT_CHUNK_SIZE
)
    n = length(data)
    
    if n == 0
        throw(ArgumentError("Cannot reduce empty vector"))
    end
    
    if chunk_size <= 0
        throw(ArgumentError("chunk_size must be positive, got $chunk_size"))
    end
    
    nchunks = cld(n, chunk_size)
    result_type = typeof(f(data[1]))
    partial_results = Vector{result_type}(undef, nchunks)
    
    # Process chunks in parallel
    @threads for chunk_idx in 1:nchunks
        start_idx = (chunk_idx - 1) * chunk_size + 1
        end_idx = min(chunk_idx * chunk_size, n)
        
        # Initialize accumulator with first element
        acc = f(data[start_idx])
        
        # Reduce remaining elements in chunk
        @inbounds for i in (start_idx + 1):end_idx
            acc = op(acc, f(data[i]))
        end
        
        partial_results[chunk_idx] = acc
    end
    
    # Final reduction across chunks (sequential)
    return reduce(op, partial_results)
end

# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    random_normal(dims...; μ=0.0f0, σ=1.0f0)

Create array of normally distributed random numbers.

# Arguments
- `dims...`: Dimensions of output array
- `μ`: Mean (default: 0.0)
- `σ`: Standard deviation (default: 1.0)

# Returns
- Array of Float32 random numbers from N(μ, σ²)

# Examples
```julia
x = random_normal(10, 10, μ=0.0f0, σ=1.0f0)
```
"""
function random_normal(dims...; μ::Float32 = 0.0f0, σ::Float32 = 1.0f0)
    if σ < 0.0f0
        throw(ArgumentError("Standard deviation σ must be non-negative, got $σ"))
    end
    
    return randn(Float32, dims...) .* σ .+ μ
end

"""
    random_uniform(dims...; low=0.0f0, high=1.0f0)

Create array of uniformly distributed random numbers.

# Arguments
- `dims...`: Dimensions of output array
- `low`: Lower bound (default: 0.0)
- `high`: Upper bound (default: 1.0)

# Returns
- Array of Float32 random numbers from Uniform(low, high)

# Examples
```julia
x = random_uniform(10, 10, low=-1.0f0, high=1.0f0)
```

# Throws
- `ArgumentError` if low >= high
"""
function random_uniform(dims...; low::Float32 = 0.0f0, high::Float32 = 1.0f0)
    if low >= high
        throw(ArgumentError("low ($low) must be < high ($high)"))
    end
    
    return rand(Float32, dims...) .* (high - low) .+ low
end

"""
    xavier_init(fan_in, fan_out)

Xavier/Glorot initialization for neural network weights.

Suitable for tanh and sigmoid activations. Weights are sampled from
N(0, sqrt(2 / (fan_in + fan_out))).

# Arguments
- `fan_in`: Number of input features
- `fan_out`: Number of output features

# Returns
- Weight matrix [fan_out, fan_in] with Xavier initialization

# Examples
```julia
W = xavier_init(784, 128)  # 128×784 weight matrix
```

# References
Glorot & Bengio (2010) "Understanding the difficulty of training deep feedforward neural networks"
"""
function xavier_init(fan_in::Int, fan_out::Int)
    if fan_in <= 0
        throw(ArgumentError("fan_in must be positive, got $fan_in"))
    end
    if fan_out <= 0
        throw(ArgumentError("fan_out must be positive, got $fan_out"))
    end
    
    # Xavier/Glorot: σ = sqrt(2 / (fan_in + fan_out))
    σ = sqrt(XAVIER_SCALE_FACTOR / (fan_in + fan_out))
    return randn(Float32, fan_out, fan_in) .* σ
end

"""
    he_init(fan_in, fan_out)

He/Kaiming initialization for neural network weights.

Suitable for ReLU and similar activations. Weights are sampled from
N(0, sqrt(2 / fan_in)).

# Arguments
- `fan_in`: Number of input features
- `fan_out`: Number of output features

# Returns
- Weight matrix [fan_out, fan_in] with He initialization

# Examples
```julia
W = he_init(784, 128)  # 128×784 weight matrix
```

# References
He et al. (2015) "Delving Deep into Rectifiers: Surpassing Human-Level Performance"
"""
function he_init(fan_in::Int, fan_out::Int)
    if fan_in <= 0
        throw(ArgumentError("fan_in must be positive, got $fan_in"))
    end
    if fan_out <= 0
        throw(ArgumentError("fan_out must be positive, got $fan_out"))
    end
    
    # He/Kaiming: σ = sqrt(2 / fan_in)
    σ = sqrt(HE_SCALE_FACTOR / fan_in)
    return randn(Float32, fan_out, fan_in) .* σ
end

# ═══════════════════════════════════════════════════════════════════════════════
# NUMERICAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    softmax(x; dims=1)

Compute numerically stable softmax along specified dimension.

# Arguments
- `x`: Input array
- `dims`: Dimension along which to compute softmax (default: 1)

# Returns
- Softmax probabilities (same shape as input, sums to 1 along dims)

# Examples
```julia
logits = randn(Float32, 10, 32)
probs = softmax(logits, dims=1)  # Softmax over classes
```

# Algorithm
1. Subtract max for numerical stability: x - max(x)
2. Compute exp: exp(x - max)
3. Normalize: exp / sum(exp)
"""
function softmax(x::AbstractArray{T}; dims=1) where T
    # Numerically stable: subtract max before exp to prevent overflow
    max_x = maximum(x, dims=dims)
    exp_x = exp.(x .- max_x)
    
    # Normalize by sum (add eps to prevent division by zero)
    return exp_x ./ (sum(exp_x, dims=dims) .+ eps(T))
end

"""
    log_softmax(x; dims=1)

Compute numerically stable log-softmax along specified dimension.

# Arguments
- `x`: Input array
- `dims`: Dimension along which to compute log-softmax (default: 1)

# Returns
- Log-softmax values (same shape as input)

# Examples
```julia
logits = randn(Float32, 10, 32)
log_probs = log_softmax(logits, dims=1)
```

# Algorithm
log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
"""
function log_softmax(x::AbstractArray{T}; dims=1) where T
    # Numerically stable log-softmax using log-sum-exp trick
    max_x = maximum(x, dims=dims)
    shifted = x .- max_x
    log_sum_exp = log.(sum(exp.(shifted), dims=dims) .+ eps(T))
    return shifted .- log_sum_exp
end

"""
    gelu(x)

Gaussian Error Linear Unit (GELU) activation function.

GELU is a smooth, differentiable activation function that has been shown
to work well in transformer models (BERT, GPT, etc.).

# Arguments
- `x`: Input array

# Returns
- GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

# Examples
```julia
x = randn(Float32, 10, 10)
y = gelu(x)
```

# References
Hendrycks & Gimpel (2016) "Gaussian Error Linear Units (GELUs)"

# Notes
- Uses fast approximation: tanh-based rather than erf-based
- Pre-computes constants for efficiency
"""
function gelu(x)
    # Fast GELU approximation using tanh
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    return 0.5f0 .* x .* (1.0f0 .+ tanh.(SQRT_2_OVER_PI .* (x .+ GELU_APPROX_CONST .* x.^3)))
end

"""
    swish(x)

SiLU/Swish activation function.

Swish is a self-gated activation: x * sigmoid(x).
Has been shown to work well in deep networks.

# Arguments
- `x`: Input array

# Returns
- Swish activation: x * sigmoid(x)

# Examples
```julia
x = randn(Float32, 10, 10)
y = swish(x)
```

# References
Ramachandran et al. (2017) "Searching for Activation Functions"
"""
function swish(x)
    return x .* sigmoid.(x)
end

"""
    sigmoid(x)

Sigmoid activation function.

# Arguments
- `x`: Input array

# Returns
- Sigmoid activation: 1 / (1 + exp(-x))

# Examples
```julia
x = randn(Float32, 10, 10)
y = sigmoid(x)
```

# Notes
- Clamps input to prevent overflow in exp(-x)
- Numerically stable implementation
"""
function sigmoid(x)
    # Clamp to prevent overflow
    x_clamped = clamp.(x, -500.0f0, 500.0f0)
    return 1.0f0 ./ (1.0f0 .+ exp.(-x_clamped))
end

"""
    layer_norm(x, γ, β; ε=1e-5)

Layer normalization with learnable scale (γ) and shift (β) parameters.

Normalizes along the last dimension: (x - μ) / sqrt(σ² + ε) * γ + β

# Arguments
- `x`: Input array
- `γ`: Scale parameter (must broadcast with normalized x)
- `β`: Shift parameter (must broadcast with normalized x)
- `ε`: Small constant for numerical stability (default: 1e-5)

# Returns
- Normalized array (same shape as input)

# Examples
```julia
x = randn(Float32, 10, 32)
γ = ones(Float32, 1, 32)
β = zeros(Float32, 1, 32)
y = layer_norm(x, γ, β)
```

# References
Ba et al. (2016) "Layer Normalization"
"""
function layer_norm(x::AbstractArray{T}, γ, β; ε::T = T(DEFAULT_EPSILON)) where T
    # Compute mean and variance along last dimension
    μ = mean(x, dims=ndims(x))
    σ² = var(x, dims=ndims(x), corrected=false)
    
    # Normalize: (x - μ) / sqrt(σ² + ε) * γ + β
    normalized = (x .- μ) ./ sqrt.(σ² .+ ε)
    return γ .* normalized .+ β
end

"""
    rms_norm(x, γ; ε=1e-5)

Root Mean Square (RMS) normalization.

Simpler than layer normalization (no mean subtraction, no shift parameter).
Used in models like GPT-3 and LLaMA for efficiency.

# Arguments
- `x`: Input array
- `γ`: Scale parameter (must broadcast with normalized x)
- `ε`: Small constant for numerical stability (default: 1e-5)

# Returns
- Normalized array (same shape as input)

# Examples
```julia
x = randn(Float32, 10, 32)
γ = ones(Float32, 1, 32)
y = rms_norm(x, γ)
```

# Algorithm
RMS = sqrt(mean(x²) + ε)
Normalized = x / RMS * γ

# References
Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
"""
function rms_norm(x::AbstractArray{T}, γ; ε::T = T(DEFAULT_EPSILON)) where T
    # Compute RMS along last dimension
    rms = sqrt.(mean(x.^2, dims=ndims(x)) .+ ε)
    
    # Normalize: x / rms * γ
    return γ .* x ./ rms
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

export @timed_block, benchmark, format_time, format_bytes, memory_info
export to_float32, to_float16, to_bfloat16, from_bfloat16
export parallel_map, parallel_reduce
export random_normal, random_uniform, xavier_init, he_init
export softmax, log_softmax, gelu, swish, sigmoid, layer_norm, rms_norm
