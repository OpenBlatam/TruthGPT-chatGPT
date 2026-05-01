"""
Quantized Operations

Operations on quantized tensors (matrix multiplication, dot product, etc.).

Provides high-performance operations using integer arithmetic with
proper scaling to maintain numerical accuracy.
"""

include("types.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_matmul_dimensions(a, b, m, n, k)

Validate dimensions for matrix multiplication.

# Arguments
- `a`: First quantized tensor
- `b`: Second quantized tensor
- `m`: Number of rows in A
- `n`: Number of columns in B
- `k`: Number of columns in A / rows in B

# Throws
- `ArgumentError` if dimensions are invalid
"""
function validate_matmul_dimensions(
    a::QuantizedTensor,
    b::QuantizedTensor,
    m::Int, n::Int, k::Int
)
    if length(a.data) != m * k
        throw(ArgumentError(
            "Tensor A dimensions don't match: expected $(m * k) elements, "
            "got $(length(a.data))"
        ))
    end
    if length(b.data) != k * n
        throw(ArgumentError(
            "Tensor B dimensions don't match: expected $(k * n) elements, "
            "got $(length(b.data))"
        ))
    end
    if m <= 0 || n <= 0 || k <= 0
        throw(ArgumentError(
            "Matrix dimensions must be positive: m=$m, n=$n, k=$k"
        ))
    end
end

"""
    compute_scale_product(scale_a, scale_b, T)

Compute combined scale factor for quantized operations.

# Arguments
- `scale_a`: Scale factor for tensor A
- `scale_b`: Scale factor for tensor B
- `T`: Float type

# Returns
- Combined scale factor (scale_a * scale_b)

# Notes
- Used when multiplying two quantized tensors
- Result must be scaled by this factor to get correct float output
"""
@inline function compute_scale_product(scale_a::T, scale_b::T) where T
    return scale_a * scale_b
end

# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    matmul_int8(a::QuantizedTensor, b::QuantizedTensor, m, n, k)

INT8 matrix multiplication with float32 output.

Performs efficient matrix multiplication using integer arithmetic,
then scales the result to get the correct float output.

# Arguments
- `a`: Quantized tensor A (shape: m × k)
- `b`: Quantized tensor B (shape: k × n)
- `m`: Number of rows in A
- `n`: Number of columns in B
- `k`: Number of columns in A / rows in B

# Returns
- Float matrix result (shape: m × n)

# Examples
```julia
a = quantize_int8(randn(Float32, 10, 20))
b = quantize_int8(randn(Float32, 20, 15))
result = matmul_int8(a, b, 10, 15, 20)
```

# Algorithm
1. Validate dimensions
2. Reshape quantized data to matrices
3. For each output element (i, j):
   a. Compute dot product: sum(A[i, :] * B[:, j]) using Int32
   b. Scale result: accumulator * scale_a * scale_b
4. Return float matrix

# Performance
- Uses @inbounds for faster indexing
- Uses @simd for vectorization of inner loop
- Integer arithmetic is faster than float
- Accumulator uses Int32 to prevent overflow

# Notes
- Result is scaled by (scale_a * scale_b) to get correct float output
- Uses Int32 accumulator to handle large intermediate values
"""
function matmul_int8(
    a::QuantizedTensor{T},
    b::QuantizedTensor{T},
    m::Int, n::Int, k::Int
) where T
    # Validate dimensions
    validate_matmul_dimensions(a, b, m, n, k)
    
    # Setup for matrix multiplication
    scale_product = compute_scale_product(a.params.scale, b.params.scale)
    a_matrix, b_matrix = reshape_matrices(a, b, m, n, k)
    
    # Perform matrix multiplication
    result = compute_matmul(a_matrix, b_matrix, m, n, k, scale_product, T)
    
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX MULTIPLICATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    reshape_matrices(a, b, m, n, k)

Reshape quantized tensors to matrices for multiplication.

# Arguments
- `a`: First quantized tensor
- `b`: Second quantized tensor
- `m`: Number of rows in A
- `n`: Number of columns in B
- `k`: Number of columns in A / rows in B

# Returns
- Tuple of (a_matrix, b_matrix)
"""
function reshape_matrices(
    a::QuantizedTensor{T},
    b::QuantizedTensor{T},
    m::Int, n::Int, k::Int
) where T
    a_matrix = reshape(a.data, m, k)
    b_matrix = reshape(b.data, k, n)
    return a_matrix, b_matrix
end

"""
    compute_matmul(a_matrix, b_matrix, m, n, k, scale_product, T)

Compute matrix multiplication with integer arithmetic.

# Arguments
- `a_matrix`: First matrix (Int8)
- `b_matrix`: Second matrix (Int8)
- `m`: Number of rows in A
- `n`: Number of columns in B
- `k`: Number of columns in A / rows in B
- `scale_product`: Combined scale factor
- `T`: Float type

# Returns
- Result matrix (Float)

# Notes
- Uses Int32 accumulator to prevent overflow
- SIMD-optimized inner loop
"""
function compute_matmul(
    a_matrix::Array{Int8, 2},
    b_matrix::Array{Int8, 2},
    m::Int, n::Int, k::Int,
    scale_product::T,
    ::Type{T}
) where T
    result = zeros(T, m, n)
    
    # Matrix multiplication with integer arithmetic
    @inbounds for i in 1:m, j in 1:n
        accumulator = compute_dot_product(a_matrix, b_matrix, i, j, k)
        result[i, j] = T(accumulator) * scale_product
    end
    
    return result
end

"""
    compute_dot_product(a_matrix, b_matrix, i, j, k)

Compute dot product for matrix multiplication element.

# Arguments
- `a_matrix`: First matrix
- `b_matrix`: Second matrix
- `i`: Row index in result
- `j`: Column index in result
- `k`: Inner dimension

# Returns
- Dot product as Int32

# Notes
- SIMD-optimized for performance
"""
function compute_dot_product(
    a_matrix::Array{Int8, 2},
    b_matrix::Array{Int8, 2},
    i::Int, j::Int, k::Int
)
    accumulator = Int32(0)
    
    # SIMD-optimized inner loop for dot product
    @simd for l in 1:k
        accumulator += Int32(a_matrix[i, l]) * Int32(b_matrix[l, j])
    end
    
    return accumulator
end

# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    dot_int8(a::QuantizedTensor, b::QuantizedTensor)

INT8 dot product (inner product) of two quantized vectors.

Performs efficient dot product using integer arithmetic,
then scales the result to get the correct float output.

# Arguments
- `a`: First quantized vector
- `b`: Second quantized vector

# Returns
- Float scalar result

# Examples
```julia
a = quantize_int8(randn(Float32, 100))
b = quantize_int8(randn(Float32, 100))
result = dot_int8(a, b)
```

# Algorithm
1. Validate vectors have same length
2. Compute dot product: sum(a[i] * b[i]) using Int32
3. Scale result: accumulator * scale_a * scale_b
4. Return float scalar

# Performance
- Uses @inbounds and @simd for vectorization
- Integer arithmetic is faster than float
- Accumulator uses Int32 to prevent overflow

# Notes
- Vectors must have the same length
- Result is scaled by (scale_a * scale_b) to get correct float output
"""
function dot_int8(a::QuantizedTensor{T}, b::QuantizedTensor{T}) where T
    # Validate vectors have same length
    validate_dot_product_inputs(a, b)
    
    # Handle empty vectors
    if length(a.data) == 0
        return zero(T)
    end
    
    # Compute dot product
    accumulator = compute_vector_dot_product(a.data, b.data)
    
    # Scale result by both quantization scales
    scale_product = compute_scale_product(a.params.scale, b.params.scale)
    return T(accumulator) * scale_product
end

# ═══════════════════════════════════════════════════════════════════════════════
# DOT PRODUCT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_dot_product_inputs(a, b)

Validate inputs for dot product operation.

# Arguments
- `a`: First quantized vector
- `b`: Second quantized vector

# Throws
- `ArgumentError` if vectors have different lengths
"""
function validate_dot_product_inputs(
    a::QuantizedTensor{T},
    b::QuantizedTensor{T}
) where T
    len_a = length(a.data)
    len_b = length(b.data)
    
    if len_a != len_b
        throw(ArgumentError(
            "Vectors must have same length: got $len_a vs $len_b"
        ))
    end
end

"""
    compute_vector_dot_product(a_data, b_data)

Compute dot product of two quantized vectors.

# Arguments
- `a_data`: First vector data (Int8)
- `b_data`: Second vector data (Int8)

# Returns
- Dot product as Int32

# Notes
- SIMD-optimized for performance
- Uses Int32 accumulator to prevent overflow
"""
function compute_vector_dot_product(
    a_data::Array{Int8},
    b_data::Array{Int8}
)
    accumulator = Int32(0)
    
    # SIMD-optimized dot product
    @inbounds @simd for i in eachindex(a_data)
        accumulator += Int32(a_data[i]) * Int32(b_data[i])
    end
    
    return accumulator
end
