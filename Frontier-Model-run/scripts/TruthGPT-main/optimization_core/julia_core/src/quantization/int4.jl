"""
INT4 Quantization

Functions for quantizing and dequantizing tensors to/from INT4 (packed in UInt8).

Provides 2x memory savings compared to INT8 by packing two INT4 values
(4 bits each) into a single UInt8 byte.
"""

include("constants.jl")
include("types.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Bit manipulation constants for INT4 packing
const INT4_LOW_MASK = 0x0F      # Lower 4 bits mask
const INT4_HIGH_SHIFT = 4       # Shift for upper 4 bits
const INT4_SIGN_THRESHOLD = 8   # Threshold for sign extension (4-bit signed: -8 to 7)
const INT4_SIGN_OFFSET = 16     # Offset for sign extension

# ═══════════════════════════════════════════════════════════════════════════════
# PACKING/UNPACKING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pack_int4_pair(value1, value2)

Pack two INT4 values into a single UInt8 byte.

# Arguments
- `value1`: First INT4 value (stored in lower 4 bits)
- `value2`: Second INT4 value (stored in upper 4 bits)

# Returns
- Packed UInt8 byte

# Algorithm
- Clamp values to INT4 range [-8, 7]
- Pack: v1 in lower 4 bits, v2 in upper 4 bits
- Result: (v2 << 4) | v1

# Performance
- @inline for inlining in hot loops
- Efficient bit manipulation
"""
@inline function pack_int4_pair(value1::Int, value2::Int)::UInt8
    # Clamp values to INT4 range
    v1 = clamp(value1, INT4_MIN, INT4_MAX)
    v2 = clamp(value2, INT4_MIN, INT4_MAX)
    
    # Pack: v1 in lower 4 bits, v2 in upper 4 bits
    low_bits = UInt8(v1 & INT4_LOW_MASK)
    high_bits = UInt8((v2 & INT4_LOW_MASK) << INT4_HIGH_SHIFT)
    
    return low_bits | high_bits
end

"""
    unpack_int4_pair(packed_byte)

Unpack a UInt8 byte into two INT4 values.

# Arguments
- `packed_byte`: UInt8 byte containing two INT4 values

# Returns
- Tuple of (value1, value2) as Int8

# Algorithm
- Extract lower 4 bits (first value)
- Extract upper 4 bits (second value)
- Sign extend: if value >= 8, subtract 16 (for negative values)

# Performance
- @inline for inlining in hot loops
- Efficient sign extension
"""
@inline function unpack_int4_pair(packed_byte::UInt8)::Tuple{Int8, Int8}
    # Extract lower 4 bits (first value) and sign extend
    v1 = Int8(packed_byte & INT4_LOW_MASK)
    if v1 >= INT4_SIGN_THRESHOLD
        v1 -= INT4_SIGN_OFFSET
    end
    
    # Extract upper 4 bits (second value) and sign extend
    v2 = Int8((packed_byte >> INT4_HIGH_SHIFT) & INT4_LOW_MASK)
    if v2 >= INT4_SIGN_THRESHOLD
        v2 -= INT4_SIGN_OFFSET
    end
    
    return (v1, v2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_int4_scale(min_val, max_val, T)

Compute scale for symmetric INT4 quantization.

# Arguments
- `min_val`: Minimum value in tensor
- `max_val`: Maximum value in tensor
- `T`: Float type

# Returns
- Scale factor for symmetric quantization

# Algorithm
scale = max(|min|, |max|) / INT4_MAX
"""
@inline function compute_int4_scale(min_val::T, max_val::T) where T <: AbstractFloat
    abs_max = max(abs(min_val), abs(max_val))
    return abs_max / T(INT4_MAX)
end

"""
    quantize_int4(x)

Quantize tensor to INT4 (packed as pairs in UInt8 for 2x memory savings).

# Arguments
- `x`: Input float tensor

# Returns
- `QuantizedInt4` with packed UInt8 data and parameters

# Examples
```julia
x = randn(Float32, 10, 10)
qt = quantize_int4(x)
dequantized = dequantize(qt)
```

# Notes
- Each UInt8 byte stores two INT4 values (4 bits each)
- Uses symmetric quantization only (zero_point = 0)
- For odd-length arrays, the last value is padded with zero

# Algorithm
1. Compute min/max and scale
2. Quantize each value to INT4 range [-8, 7]
3. Pack pairs of values into UInt8 bytes
4. Store packed data and quantization parameters

# Performance
- Uses @inbounds for faster indexing
- Efficient packing with bit operations
- Handles odd-length arrays correctly
"""
function quantize_int4(x::AbstractArray{T}) where T <: AbstractFloat
    # Validate input
    validate_quantization_input(x)
    
    # Compute quantization parameters
    min_val, max_val = compute_value_range(x)
    scale = compute_int4_quantization_scale(min_val, max_val, T)
    
    # Pack quantized values
    numel = length(x)
    packed = pack_int4_tensor(x, scale, numel)
    
    # Create and return quantized tensor
    return create_quantized_int4(packed, scale, min_val, max_val, x, numel, T)
end

# ═══════════════════════════════════════════════════════════════════════════════
# INT4 QUANTIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_int4_quantization_scale(min_val, max_val, T)

Compute scale for INT4 quantization.

# Arguments
- `min_val`: Minimum value
- `max_val`: Maximum value
- `T`: Type parameter

# Returns
- Scale factor

# Notes
- Handles edge case where all values are the same
- Always uses symmetric quantization for INT4
"""
function compute_int4_quantization_scale(min_val::T, max_val::T, ::Type{T}) where T <: AbstractFloat
    # Handle edge case: all values are the same
    if min_val == max_val
        return one(T)
    else
        # Symmetric quantization for INT4
        scale = compute_int4_scale(min_val, max_val)
        return max(scale, eps(T))  # Ensure scale is positive
    end
end

"""
    pack_int4_tensor(x, scale, numel)

Pack INT4 quantized values into UInt8 bytes.

# Arguments
- `x`: Input tensor
- `scale`: Quantization scale
- `numel`: Number of elements

# Returns
- Packed UInt8 array
"""
function pack_int4_tensor(x::AbstractArray{T}, scale::T, numel::Int) where T
    # Calculate packed array size (2 INT4 values per UInt8)
    packed_size = cld(numel, 2)
    packed = zeros(UInt8, packed_size)
    
    # Flatten input for easier processing
    x_flat = vec(x)
    
    # Pack pairs of quantized values into UInt8 bytes
    @inbounds for i in 1:2:numel
        # Quantize and pack pair
        quantized_val1 = quantize_to_int4(x_flat[i], scale)
        quantized_val2 = (i + 1 <= numel) ? quantize_to_int4(x_flat[i + 1], scale) : 0
        
        # Pack into UInt8 using helper function
        packed_byte = pack_int4_pair(quantized_val1, quantized_val2)
        packed[div(i, 2) + 1] = packed_byte
    end
    
    return packed
end

"""
    quantize_to_int4(value, scale)

Quantize a single value to INT4 range.

# Arguments
- `value`: Float value
- `scale`: Quantization scale

# Returns
- Quantized INT4 value (clamped to [-8, 7])
"""
@inline function quantize_to_int4(value::T, scale::T) where T
    return clamp(round(Int, value / scale), INT4_MIN, INT4_MAX)
end

"""
    create_quantized_int4(packed, scale, min_val, max_val, x, numel, T)

Create QuantizedInt4 tensor from packed data.

# Arguments
- `packed`: Packed UInt8 data
- `scale`: Quantization scale
- `min_val`: Minimum value
- `max_val`: Maximum value
- `x`: Original tensor (for shape)
- `numel`: Number of elements
- `T`: Type parameter

# Returns
- QuantizedInt4 tensor
"""
function create_quantized_int4(
    packed::Vector{UInt8},
    scale::T,
    min_val::T,
    max_val::T,
    x::AbstractArray{T},
    numel::Int,
    ::Type{T}
) where T
    # Create quantization parameters (symmetric, so zero_point is 0)
    params = QuantParams{T}(scale, Int32(0), min_val, max_val)
    return QuantizedInt4{T, ndims(x)}(packed, params, size(x), numel)
end

"""
    dequantize(qt::QuantizedInt4{T})

Dequantize INT4 tensor back to float representation.

# Arguments
- `qt`: Quantized INT4 tensor

# Returns
- Reconstructed float tensor with original shape

# Examples
```julia
qt = quantize_int4(randn(Float32, 5, 5))
x_reconstructed = dequantize(qt)
```

# Notes
- Unpacks pairs of INT4 values from each UInt8 byte
- Handles sign extension for negative values (INT4 range: -8 to 7)
- Correctly handles odd-length arrays

# Algorithm
1. For each packed byte:
   a. Unpack into two INT4 values
   b. Sign extend if needed
   c. Dequantize: value * scale
2. Reshape to original shape

# Performance
- Uses @inbounds for faster indexing
- Efficient unpacking with helper function
- Handles edge cases (odd-length arrays)
"""
function dequantize(qt::QuantizedInt4{T}) where T
    # Allocate result array
    result = Vector{T}(undef, qt.original_numel)
    
    # Dequantize all packed bytes
    dequantize_packed_bytes!(result, qt.data, qt.params.scale, qt.original_numel, T)
    
    return reshape(result, qt.original_shape)
end

"""
    dequantize_packed_bytes!(result, packed_data, scale, numel, T)

Dequantize packed INT4 bytes into float array.

# Arguments
- `result`: Result array (modified in-place)
- `packed_data`: Packed UInt8 data
- `scale`: Quantization scale
- `numel`: Number of elements
- `T`: Type parameter

# Notes
- Modifies result array in-place
"""
function dequantize_packed_bytes!(
    result::Vector{T},
    packed_data::Array{UInt8},
    scale::T,
    numel::Int,
    ::Type{T}
) where T
    @inbounds for i in 1:length(packed_data)
        byte = packed_data[i]
        
        # Unpack byte into two INT4 values
        val1, val2 = unpack_int4_pair(byte)
        
        # Dequantize and store first value
        idx1 = 2 * (i - 1) + 1
        if idx1 <= numel
            result[idx1] = T(val1) * scale
        end
        
        # Dequantize and store second value
        idx2 = 2 * (i - 1) + 2
        if idx2 <= numel
            result[idx2] = T(val2) * scale
        end
    end
end
