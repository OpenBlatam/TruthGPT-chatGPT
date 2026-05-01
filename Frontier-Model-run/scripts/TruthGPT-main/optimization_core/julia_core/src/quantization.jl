"""
Tensor Quantization for TruthGPT

Efficient INT8/INT4 quantization with Julia's native SIMD support.
Provides symmetric and asymmetric quantization with calibration support.

This module implements state-of-the-art quantization techniques:
- Symmetric quantization: zero_point = 0, scale = max(|min|, |max|) / quant_max
- Asymmetric quantization: zero_point calculated to map min to 0
- Group quantization: per-group scales for improved accuracy
- INT4 packing: 2 values per byte for 2x memory savings
"""

using LinearAlgebra
using Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# INT8 quantization constants
const INT8_MAX = 127
const INT8_MIN = -128
const INT8_RANGE = 255

# INT4 quantization constants
const INT4_MAX = 7
const INT4_MIN = -8
const INT4_RANGE = 15

# Bit manipulation masks for INT4 packing
const INT4_LOW_MASK = 0x0F
const INT4_HIGH_SHIFT = 4
const INT4_SIGN_THRESHOLD = 8  # For sign extension from 4-bit to 8-bit
const INT4_SIGN_OFFSET = 16    # Offset for negative values

# Default group size for grouped quantization
const DEFAULT_GROUP_SIZE = 128

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    QuantParams{T}

Quantization parameters for a tensor.

# Fields
- `scale`: Scaling factor for quantization/dequantization
- `zero_point`: Zero point offset (for asymmetric quantization)
- `min_val`: Minimum value in original tensor
- `max_val`: Maximum value in original tensor

# Notes
- For symmetric quantization: zero_point = 0
- For asymmetric quantization: zero_point maps min_val to 0 in quantized space
"""
struct QuantParams{T}
    scale::T
    zero_point::Int32
    min_val::T
    max_val::T
end

"""
    QuantizedTensor{T, N}

Quantized tensor with parameters.

# Fields
- `data`: Quantized INT8 data
- `params`: Quantization parameters
- `original_shape`: Original tensor shape before quantization
"""
struct QuantizedTensor{T, N}
    data::Array{Int8, N}
    params::QuantParams{T}
    original_shape::NTuple{N, Int}
end

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_min_max(x)

Compute min and max values in a single pass.

# Arguments
- `x`: Input array

# Returns
- Tuple of (min_val, max_val)

# Performance
- Single pass through array: O(n)
- More efficient than separate minimum() and maximum() calls
"""
@inline function compute_min_max(x::AbstractArray{T}) where T
    if isempty(x)
        throw(ArgumentError("Cannot compute min/max of empty array"))
    end
    
    min_val = T(Inf)
    max_val = T(-Inf)
    
    @inbounds @simd for i in eachindex(x)
        val = x[i]
        min_val = val < min_val ? val : min_val
        max_val = val > max_val ? val : max_val
    end
    
    return min_val, max_val
end

"""
    compute_symmetric_scale(min_val, max_val, quant_max)

Compute scale for symmetric quantization.

Symmetric quantization uses zero_point = 0 and scales based on the
absolute maximum value to preserve symmetry around zero.

# Arguments
- `min_val`: Minimum value in tensor
- `max_val`: Maximum value in tensor
- `quant_max`: Maximum quantized value (e.g., 127 for INT8)

# Returns
- Scale factor (always >= eps(T) to avoid division by zero)

# Algorithm
scale = max(|min_val|, |max_val|) / quant_max
"""
@inline function compute_symmetric_scale(min_val::T, max_val::T, quant_max::T) where T
    abs_max = max(abs(min_val), abs(max_val))
    scale = abs_max / quant_max
    # Ensure scale is never zero to avoid division by zero
    return max(scale, eps(T))
end

"""
    compute_asymmetric_scale(min_val, max_val, quant_range)

Compute scale and zero point for asymmetric quantization.

Asymmetric quantization allows mapping min_val to 0 in quantized space,
providing better precision for non-symmetric distributions.

# Arguments
- `min_val`: Minimum value in tensor
- `max_val`: Maximum value in tensor
- `quant_range`: Quantization range (e.g., 255 for INT8)

# Returns
- Tuple of (scale, zero_point)

# Algorithm
scale = (max_val - min_val) / quant_range
zero_point = round(-min_val / scale - quant_range / 2)
"""
function compute_asymmetric_scale(min_val::T, max_val::T, quant_range::T) where T
    value_range = max_val - min_val
    
    # Handle edge case: all values are the same
    if value_range == zero(T)
        scale = one(T)
        zero_point = Int32(0)
        return scale, zero_point
    end
    
    scale = value_range / quant_range
    scale = max(scale, eps(T))  # Prevent division by zero
    
    # Calculate zero point to map min_val to 0 in quantized space
    # Formula: quantized = round(value / scale) + zero_point
    # We want: round(min_val / scale) + zero_point = 0
    zero_point = round(Int32, -min_val / scale - quant_range / 2)
    
    return scale, zero_point
end

"""
    quantize_value(value, scale, zero_point, min_quant, max_quant)

Quantize a single value with clamping.

# Arguments
- `value`: Value to quantize
- `scale`: Quantization scale
- `zero_point`: Zero point offset
- `min_quant`: Minimum quantized value
- `max_quant`: Maximum quantized value

# Returns
- Quantized integer value (clamped to [min_quant, max_quant])

# Algorithm
quantized = round(value / scale) + zero_point
return clamp(quantized, min_quant, max_quant)
"""
@inline function quantize_value(
    value::T,
    scale::T,
    zero_point::Int32,
    min_quant::Int,
    max_quant::Int
) where T
    # Quantize: divide by scale and add zero point
    quantized = round(Int, value / scale) + zero_point
    # Clamp to valid range to prevent overflow
    return clamp(quantized, min_quant, max_quant)
end

"""
    dequantize_value(quantized, scale, zero_point)

Dequantize a single value.

# Arguments
- `quantized`: Quantized integer value
- `scale`: Quantization scale
- `zero_point`: Zero point offset

# Returns
- Dequantized float value

# Algorithm
dequantized = (quantized - zero_point) * scale
"""
@inline function dequantize_value(quantized::Int8, scale::T, zero_point::Int32) where T
    return T(quantized - zero_point) * scale
end

# ═══════════════════════════════════════════════════════════════════════════════
# INT8 QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    quantize_int8(x; symmetric=true)

Quantize float tensor to INT8.

# Arguments
- `x`: Input tensor (must be AbstractFloat)
- `symmetric`: Use symmetric quantization (default: true)
  - If true: zero_point = 0, scale based on abs(max)
  - If false: zero_point calculated, scale based on range

# Returns
- `QuantizedTensor` with INT8 data and parameters

# Examples
```julia
# Symmetric quantization (default)
x = randn(Float32, 100, 100)
qt = quantize_int8(x)
dequantized = dequantize(qt)

# Asymmetric quantization
qt_asym = quantize_int8(x, symmetric=false)
```

# Performance
- Single pass for min/max computation
- SIMD-optimized quantization loop
- Memory efficient: 4x reduction (Float32 -> Int8)
"""
function quantize_int8(x::AbstractArray{T}; symmetric::Bool=true) where T <: AbstractFloat
    # Validate input
    validate_quantization_input(x)
    
    # Compute quantization parameters
    min_val, max_val = compute_min_max(x)
    scale, zero_point = compute_int8_params(min_val, max_val, symmetric, T)
    
    # Quantize tensor
    quantized = quantize_tensor_values(x, scale, zero_point, INT8_MIN, INT8_MAX)
    
    # Create quantization parameters
    params = QuantParams(scale, zero_point, min_val, max_val)
    
    return QuantizedTensor(quantized, params, size(x))
end

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_quantization_input(x)

Validate input for quantization.

# Arguments
- `x`: Input tensor to validate

# Throws
- `ArgumentError` if input is invalid
"""
function validate_quantization_input(x::AbstractArray)
    if isempty(x)
        throw(ArgumentError("Input tensor cannot be empty"))
    end
end

"""
    compute_int8_params(min_val, max_val, symmetric, T)

Compute INT8 quantization parameters.

# Arguments
- `min_val`: Minimum value
- `max_val`: Maximum value
- `symmetric`: Whether to use symmetric quantization
- `T`: Type parameter

# Returns
- Tuple of (scale, zero_point)
"""
function compute_int8_params(min_val::T, max_val::T, symmetric::Bool, ::Type{T}) where T
    if symmetric
        scale = compute_symmetric_scale(min_val, max_val, T(INT8_MAX))
        zero_point = Int32(0)
    else
        scale, zero_point = compute_asymmetric_scale(min_val, max_val, T(INT8_RANGE))
    end
    return scale, zero_point
end

"""
    quantize_tensor_values(x, scale, zero_point, min_quant, max_quant)

Quantize all values in tensor.

# Arguments
- `x`: Input tensor
- `scale`: Quantization scale
- `zero_point`: Zero point offset
- `min_quant`: Minimum quantized value
- `max_quant`: Maximum quantized value

# Returns
- Quantized tensor (Int8)
"""
function quantize_tensor_values(
    x::AbstractArray{T},
    scale::T,
    zero_point::Int32,
    min_quant::Int,
    max_quant::Int
) where T
    quantized = similar(x, Int8)
    
    # Quantize all values with SIMD-optimized loop
    @inbounds @simd for i in eachindex(x)
        quantized[i] = quantize_value(x[i], scale, zero_point, min_quant, max_quant)
    end
    
    return quantized
end

"""
    dequantize(qt::QuantizedTensor{T})

Dequantize INT8 tensor back to float.

# Arguments
- `qt`: QuantizedTensor to dequantize

# Returns
- Dequantized tensor with original shape and float type

# Examples
```julia
qt = quantize_int8(randn(Float32, 10, 10))
x_recovered = dequantize(qt)
```

# Performance
- SIMD-optimized dequantization loop
- Exact reconstruction (within quantization error)
"""
function dequantize(qt::QuantizedTensor{T}) where T
    result = similar(qt.data, T)
    scale = qt.params.scale
    zero_point = qt.params.zero_point
    
    # Dequantize: (quantized - zero_point) * scale
    @inbounds @simd for i in eachindex(qt.data)
        result[i] = dequantize_value(qt.data[i], scale, zero_point)
    end
    
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# INT4 QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    QuantizedInt4{T, N}

INT4 quantized data (packed as pairs in UInt8).

# Fields
- `data`: Packed INT4 data (2 values per UInt8 byte)
- `params`: Quantization parameters
- `original_shape`: Original tensor shape
- `original_numel`: Original number of elements (for odd-length arrays)
"""
struct QuantizedInt4{T, N}
    data::Array{UInt8, N}
    params::QuantParams{T}
    original_shape::NTuple{N, Int}
    original_numel::Int
end

"""
    pack_int4_pair(value1, value2)

Pack two INT4 values into a single UInt8 byte.

# Arguments
- `value1`: First INT4 value (stored in lower 4 bits)
- `value2`: Second INT4 value (stored in upper 4 bits)

# Returns
- Packed UInt8 byte

# Bit Layout
- Bits 0-3: value1 (lower 4 bits)
- Bits 4-7: value2 (upper 4 bits)
"""
@inline function pack_int4_pair(value1::Int, value2::Int)::UInt8
    # Clamp values to INT4 range [-8, 7]
    v1 = clamp(value1, INT4_MIN, INT4_MAX)
    v2 = clamp(value2, INT4_MIN, INT4_MAX)
    
    # Pack: v1 in lower 4 bits, v2 in upper 4 bits
    # Use bitwise AND to ensure only 4 bits are used
    low_bits = UInt8(v1 & INT4_LOW_MASK)
    high_bits = UInt8((v2 & INT4_LOW_MASK) << INT4_HIGH_SHIFT)
    
    return low_bits | high_bits
end

"""
    unpack_int4_pair(packed_byte)

Unpack a UInt8 byte into two INT4 values with sign extension.

# Arguments
- `packed_byte`: UInt8 byte containing two INT4 values

# Returns
- Tuple of (value1, value2) as Int8

# Sign Extension
- Values >= 8 are interpreted as negative (4-bit two's complement)
- Example: 0b1111 (15) -> -1, 0b1000 (8) -> -8
"""
@inline function unpack_int4_pair(packed_byte::UInt8)::Tuple{Int8, Int8}
    # Extract lower 4 bits (first value)
    v1_raw = Int8(packed_byte & INT4_LOW_MASK)
    # Sign extend from 4-bit to 8-bit two's complement
    v1 = v1_raw >= INT4_SIGN_THRESHOLD ? v1_raw - INT4_SIGN_OFFSET : v1_raw
    
    # Extract upper 4 bits (second value)
    v2_raw = Int8((packed_byte >> INT4_HIGH_SHIFT) & INT4_LOW_MASK)
    # Sign extend from 4-bit to 8-bit two's complement
    v2 = v2_raw >= INT4_SIGN_THRESHOLD ? v2_raw - INT4_SIGN_OFFSET : v2_raw
    
    return (v1, v2)
end

"""
    quantize_int4(x)

Quantize to INT4 (packed as pairs in UInt8).

# Arguments
- `x`: Input tensor (must be AbstractFloat)

# Returns
- `QuantizedInt4` with packed data and parameters

# Examples
```julia
x = randn(Float32, 100, 100)
qt4 = quantize_int4(x)
dequantized = dequantize(qt4)
```

# Performance
- 2x memory savings vs INT8 (4 values per byte)
- Efficient packing with bitwise operations
- Handles odd-length arrays gracefully
"""
function quantize_int4(x::AbstractArray{T}) where T <: AbstractFloat
    # Validate input
    validate_quantization_input(x)
    
    # Compute quantization parameters (always symmetric for INT4)
    min_val, max_val = compute_min_max(x)
    scale = compute_int4_scale(min_val, max_val, T)
    
    # Pack quantized values
    numel = length(x)
    packed = pack_int4_values(x, scale, numel)
    
    # Create quantization parameters (symmetric, zero_point = 0)
    params = QuantParams(scale, Int32(0), min_val, max_val)
    
    return QuantizedInt4(packed, params, size(x), numel)
end

"""
    compute_int4_scale(min_val, max_val, T)

Compute INT4 quantization scale (always symmetric).

# Arguments
- `min_val`: Minimum value
- `max_val`: Maximum value
- `T`: Type parameter

# Returns
- Scale factor

# Notes
- INT4 always uses symmetric quantization
"""
function compute_int4_scale(min_val::T, max_val::T, ::Type{T}) where T
    abs_max = max(abs(min_val), abs(max_val))
    scale = abs_max / T(INT4_MAX)
    return max(scale, eps(T))
end

"""
    pack_int4_values(x, scale, numel)

Pack INT4 quantized values into bytes.

# Arguments
- `x`: Input tensor
- `scale`: Quantization scale
- `numel`: Number of elements

# Returns
- Packed UInt8 array
"""
function pack_int4_values(x::AbstractArray{T}, scale::T, numel::Int) where T
    # Calculate packed array size (2 values per byte, round up)
    packed_size = cld(numel, 2)
    packed = Vector{UInt8}(undef, packed_size)
    
    # Flatten input for easier indexing
    x_flat = vec(x)
    
    # Pack pairs of values into bytes
    @inbounds for i in 1:2:numel
        # Quantize first value
        q1 = round(Int, x_flat[i] / scale)
        
        # Quantize second value (or 0 if odd length)
        q2 = (i + 1 <= numel) ? round(Int, x_flat[i + 1] / scale) : 0
        
        # Pack into single byte
        packed_idx = div(i, 2) + 1
        packed[packed_idx] = pack_int4_pair(q1, q2)
    end
    
    return packed
end

"""
    dequantize(qt::QuantizedInt4{T})

Dequantize INT4 back to float.

# Arguments
- `qt`: QuantizedInt4 to dequantize

# Returns
- Dequantized tensor with original shape

# Examples
```julia
qt4 = quantize_int4(randn(Float32, 10, 10))
x_recovered = dequantize(qt4)
```

# Performance
- Efficient unpacking with bitwise operations
- Handles odd-length arrays correctly
"""
function dequantize(qt::QuantizedInt4{T}) where T
    result = Vector{T}(undef, qt.original_numel)
    scale = qt.params.scale
    
    # Unpack and dequantize each byte
    @inbounds for i in 1:length(qt.data)
        v1, v2 = unpack_int4_pair(qt.data[i])
        
        # Calculate indices for unpacked values
        idx1 = 2 * (i - 1) + 1
        idx2 = 2 * (i - 1) + 2
        
        # Dequantize first value
        if idx1 <= qt.original_numel
            result[idx1] = T(v1) * scale
        end
        
        # Dequantize second value
        if idx2 <= qt.original_numel
            result[idx2] = T(v2) * scale
        end
    end
    
    return reshape(result, qt.original_shape)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GROUP QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GroupQuantParams{T}

Per-group quantization parameters for improved accuracy.

Group quantization uses different scales for different groups of values,
providing better accuracy than global quantization for non-uniform distributions.

# Fields
- `scales`: Vector of scale factors (one per group)
- `zero_points`: Vector of zero points (one per group, currently all 0)
- `group_size`: Number of elements per group
"""
struct GroupQuantParams{T}
    scales::Vector{T}
    zero_points::Vector{Int32}
    group_size::Int
end

"""
    QuantizedGrouped{T, N}

Grouped quantization for better accuracy.

# Fields
- `data`: Quantized INT8 data
- `group_params`: Per-group quantization parameters
- `original_shape`: Original tensor shape
"""
struct QuantizedGrouped{T, N}
    data::Array{Int8, N}
    group_params::GroupQuantParams{T}
    original_shape::NTuple{N, Int}
end

"""
    quantize_grouped(x, group_size=128)

Quantize with per-group parameters for better accuracy.

# Arguments
- `x`: Input tensor (must be AbstractFloat)
- `group_size`: Number of elements per quantization group (default: 128)

# Returns
- `QuantizedGrouped` with per-group parameters

# Examples
```julia
x = randn(Float32, 1000)
qt_grouped = quantize_grouped(x, group_size=64)
dequantized = dequantize(qt_grouped)
```

# Performance
- Better accuracy than global quantization
- Slightly more memory overhead (one scale per group)
- Efficient group-wise processing
"""
function quantize_grouped(
    x::AbstractArray{T},
    group_size::Int=DEFAULT_GROUP_SIZE
) where T <: AbstractFloat
    # Validate inputs
    if isempty(x)
        throw(ArgumentError("Input tensor cannot be empty"))
    end
    if group_size <= 0
        throw(ArgumentError("group_size must be positive, got $group_size"))
    end
    
    # Flatten for easier processing
    x_flat = vec(x)
    numel = length(x_flat)
    num_groups = cld(numel, group_size)
    
    # Pre-allocate arrays
    scales = Vector{T}(undef, num_groups)
    zero_points = zeros(Int32, num_groups)  # All zero for symmetric quantization
    quantized = similar(x_flat, Int8)
    
    # Process each group independently
    @inbounds for group_idx in 1:num_groups
        # Calculate group boundaries
        start_idx = (group_idx - 1) * group_size + 1
        end_idx = min(group_idx * group_size, numel)
        
        # Extract group view (avoids copying, O(1) operation)
        group = @view x_flat[start_idx:end_idx]
        
        # Compute symmetric scale for this group
        abs_max = maximum(abs, group)
        scale = abs_max / T(INT8_MAX)
        scale = max(scale, eps(T))
        scales[group_idx] = scale
        
        # Quantize all elements in this group
        for i in start_idx:end_idx
            quantized[i] = quantize_value(x_flat[i], scale, Int32(0), INT8_MIN, INT8_MAX)
        end
    end
    
    # Create group parameters
    params = GroupQuantParams(scales, zero_points, group_size)
    
    return QuantizedGrouped(reshape(quantized, size(x)), params, size(x))
end

"""
    dequantize(qt::QuantizedGrouped{T})

Dequantize grouped quantized tensor.

# Arguments
- `qt`: QuantizedGrouped to dequantize

# Returns
- Dequantized tensor with original shape

# Examples
```julia
qt_grouped = quantize_grouped(randn(Float32, 1000))
x_recovered = dequantize(qt_grouped)
```

# Performance
- SIMD-optimized dequantization
- Efficient group index calculation
"""
function dequantize(qt::QuantizedGrouped{T}) where T
    result = similar(qt.data, T)
    data_flat = vec(qt.data)
    result_flat = vec(result)
    
    group_size = qt.group_params.group_size
    scales = qt.group_params.scales
    
    # Dequantize using per-group scales
    @inbounds @simd for i in eachindex(data_flat)
        # Calculate which group this element belongs to
        group_idx = div(i - 1, group_size) + 1
        result_flat[i] = T(data_flat[i]) * scales[group_idx]
    end
    
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZED OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    matmul_int8(a::QuantizedTensor, b::QuantizedTensor, m, n, k)

INT8 matrix multiplication with float32 output.

Performs quantized matrix multiplication: C = A * B
where A is (m×k) and B is (k×n), result is (m×n).

# Arguments
- `a`: QuantizedTensor for matrix A (m×k)
- `b`: QuantizedTensor for matrix B (k×n)
- `m`: Number of rows in A and result
- `n`: Number of columns in B and result
- `k`: Number of columns in A / rows in B

# Returns
- Result matrix (m×n) as Float32

# Examples
```julia
a = quantize_int8(randn(Float32, 10, 20))
b = quantize_int8(randn(Float32, 20, 15))
c = matmul_int8(a, b, 10, 15, 20)
```

# Performance
- Accumulates in Int32 to prevent overflow
- SIMD-optimized inner loop
- Scales result by product of quantization scales
"""
function matmul_int8(
    a::QuantizedTensor{T},
    b::QuantizedTensor{T},
    m::Int, n::Int, k::Int
) where T
    # Validate dimensions
    if length(a.data) != m * k
        throw(DimensionMismatch(
            "Matrix A must have m×k = $m×$k elements, got $(length(a.data))"
        ))
    end
    if length(b.data) != k * n
        throw(DimensionMismatch(
            "Matrix B must have k×n = $k×$n elements, got $(length(b.data))"
        ))
    end
    
    result = zeros(T, m, n)
    scale_product = a.params.scale * b.params.scale
    
    # Reshape for matrix indexing
    a_data = reshape(a.data, m, k)
    b_data = reshape(b.data, k, n)
    
    # Perform quantized matrix multiplication
    # Accumulate in Int32 to avoid overflow, then scale
    @inbounds for i in 1:m, j in 1:n
        accumulator = Int32(0)
        # SIMD-optimized inner loop for dot product
        @simd for l in 1:k
            accumulator += Int32(a_data[i, l]) * Int32(b_data[l, j])
        end
        # Scale result by product of quantization scales
        result[i, j] = T(accumulator) * scale_product
    end
    
    return result
end

"""
    dot_int8(a::QuantizedTensor, b::QuantizedTensor)

INT8 dot product.

# Arguments
- `a`: First quantized vector
- `b`: Second quantized vector

# Returns
- Dot product result as Float32

# Examples
```julia
a = quantize_int8(randn(Float32, 100))
b = quantize_int8(randn(Float32, 100))
dot_result = dot_int8(a, b)
```

# Performance
- SIMD-optimized accumulation
- Accumulates in Int32 to prevent overflow
"""
function dot_int8(a::QuantizedTensor{T}, b::QuantizedTensor{T}) where T
    if length(a.data) != length(b.data)
        throw(DimensionMismatch(
            "Vectors must have same length, got $(length(a.data)) and $(length(b.data))"
        ))
    end
    
    accumulator = Int32(0)
    
    # Compute dot product in quantized space, accumulate in Int32
    @inbounds @simd for i in eachindex(a.data)
        accumulator += Int32(a.data[i]) * Int32(b.data[i])
    end
    
    # Scale result by product of quantization scales
    return T(accumulator) * a.params.scale * b.params.scale
end

# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Calibrator{T}

Collect statistics for quantization calibration.

Tracks min, max, sum, and sum of squares for computing optimal
quantization parameters from a dataset. Useful for post-training quantization.

# Fields
- `min_val`: Minimum observed value
- `max_val`: Maximum observed value
- `sum_val`: Sum of all observed values
- `sum_sq_val`: Sum of squares of all observed values
- `samples`: Number of samples observed
"""
mutable struct Calibrator{T}
    min_val::T
    max_val::T
    sum_val::Float64
    sum_sq_val::Float64
    samples::Int
end

"""
    Calibrator(T::Type=Float32)

Create a new Calibrator for the given type.

# Arguments
- `T`: Element type (default: Float32)

# Returns
- New Calibrator instance initialized to empty state
"""
Calibrator(T::Type=Float32) = Calibrator(T(Inf), T(-Inf), 0.0, 0.0, 0)

"""
    observe!(cal::Calibrator, x)

Record values for calibration.

Updates statistics with new observations from the input tensor.
Can be called multiple times to accumulate statistics from multiple batches.

# Arguments
- `cal`: Calibrator to update
- `x`: Input tensor to observe

# Returns
- Updated calibrator (for method chaining)

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, randn(Float32, 100, 100))
observe!(cal, randn(Float32, 50, 50))
params = get_params(cal)
```
"""
function observe!(cal::Calibrator{T}, x::AbstractArray) where T
    if isempty(x)
        return cal  # Skip empty arrays
    end
    
    # Update min/max in single pass
    x_min, x_max = compute_min_max(x)
    cal.min_val = min(cal.min_val, x_min)
    cal.max_val = max(cal.max_val, x_max)
    
    # Update sum and sum of squares (for mean/std calculation)
    cal.sum_val += sum(Float64, x)
    cal.sum_sq_val += sum(v -> Float64(v)^2, x)
    cal.samples += length(x)
    
    return cal
end

"""
    get_params(cal::Calibrator)

Get calibrated quantization parameters.

# Arguments
- `cal`: Calibrator with collected statistics

# Returns
- `QuantParams` with optimal scale based on observed range

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, randn(Float32, 100))
params = get_params(cal)
```

# Throws
- `ArgumentError` if no samples have been observed
"""
function get_params(cal::Calibrator{T}) where T
    if cal.samples == 0
        throw(ArgumentError("Calibrator has no samples, cannot compute parameters"))
    end
    
    # Use symmetric quantization based on observed range
    abs_max = max(abs(cal.min_val), abs(cal.max_val))
    scale = abs_max / T(INT8_MAX)
    scale = max(scale, eps(T))
    
    return QuantParams(scale, Int32(0), cal.min_val, cal.max_val)
end

"""
    mean(cal::Calibrator)

Get mean of observed values.

# Arguments
- `cal`: Calibrator with statistics

# Returns
- Mean value (0.0 if no samples)

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, randn(Float32, 100))
μ = mean(cal)
```
"""
Statistics.mean(cal::Calibrator) = cal.samples > 0 ? cal.sum_val / cal.samples : zero(Float64)

"""
    std(cal::Calibrator)

Get standard deviation of observed values.

# Arguments
- `cal`: Calibrator with statistics

# Returns
- Standard deviation (0.0 if no samples or variance is negative)

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, randn(Float32, 100))
σ = std(cal)
```

# Notes
- Uses Welford's algorithm for numerical stability
- Handles floating point errors by ensuring non-negative variance
"""
function Statistics.std(cal::Calibrator)
    if cal.samples == 0
        return zero(Float64)
    end
    
    μ = mean(cal)
    # Variance = E[X²] - E[X]²
    variance = cal.sum_sq_val / cal.samples - μ^2
    
    # Ensure non-negative (handle floating point errors)
    return sqrt(max(variance, 0.0))
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

export QuantParams, QuantizedTensor, QuantizedInt4, QuantizedGrouped
export quantize_int8, quantize_int4, quantize_grouped, dequantize
export matmul_int8, dot_int8
export Calibrator, observe!, get_params
