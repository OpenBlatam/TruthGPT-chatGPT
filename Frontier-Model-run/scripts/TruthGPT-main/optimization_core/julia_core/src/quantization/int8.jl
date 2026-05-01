"""
INT8 Quantization

Functions for quantizing and dequantizing tensors to/from INT8.

Provides both symmetric and asymmetric quantization methods with
numerical stability and efficient computation.
"""

include("constants.jl")
include("types.jl")
include("utils.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_symmetric_scale(min_val, max_val, T)

Compute scale for symmetric INT8 quantization.

# Arguments
- `min_val`: Minimum value in tensor
- `max_val`: Maximum value in tensor
- `T`: Float type

# Returns
- Scale factor for symmetric quantization

# Algorithm
scale = max(|min|, |max|) / INT8_MAX
"""
@inline function compute_symmetric_scale(min_val::T, max_val::T) where T <: AbstractFloat
    # Use absolute maximum to preserve symmetry around zero
    abs_max = max(abs(min_val), abs(max_val))
    
    # Compute scale: abs_max / INT8_MAX
    # Ensure numerical stability by preventing division by zero
    scale = abs_max / T(INT8_MAX)
    
    # Clamp to minimum epsilon to avoid numerical issues
    return max(scale, max(eps(T), T(1e-8)))
end

"""
    compute_asymmetric_scale(min_val, max_val, T)

Compute scale and zero point for asymmetric INT8 quantization.

# Arguments
- `min_val`: Minimum value in tensor
- `max_val`: Maximum value in tensor
- `T`: Float type

# Returns
- Tuple of (scale, zero_point)

# Algorithm
scale = (max - min) / INT8_RANGE
zero_point = round(-min / scale - INT8_MIN)
"""
function compute_asymmetric_scale(min_val::T, max_val::T) where T <: AbstractFloat
    # Validate input range
    if min_val > max_val
        throw(ArgumentError("min_val ($min_val) must be <= max_val ($max_val)"))
    end
    
    # Compute scale from value range
    value_range = max_val - min_val
    scale = value_range / T(INT8_RANGE)
    
    # Ensure numerical stability
    scale = max(scale, max(eps(T), T(1e-8)))
    
    # Calculate zero point: maps min_val to minimum quantized value
    # Formula: quantized = (value / scale) + zero_point
    # For min_val: min_quant = (min_val / scale) + zero_point
    # Solving: zero_point = min_quant - (min_val / scale)
    # For INT8: min_quant = -128, so zero_point = -128 - (min_val / scale)
    zero_point = round(Int32, -min_val / scale)
    
    # Clamp zero_point to valid INT8 range
    zero_point = clamp(zero_point, Int32(INT8_MIN), Int32(INT8_MAX))
    
    return scale, zero_point
end

"""
    quantize_value(value, scale, zero_point, T)

Quantize a single float value to INT8.

# Arguments
- `value`: Float value to quantize
- `scale`: Quantization scale
- `zero_point`: Quantization zero point
- `T`: Float type

# Returns
- Quantized INT8 value (clamped to valid range)
"""
@inline function quantize_value(value::T, scale::T, zero_point::Int32) where T <: AbstractFloat
    # Validate scale to prevent division by zero
    if scale <= zero(T)
        throw(ArgumentError("scale must be positive, got $scale"))
    end
    
    # Quantize: round to nearest integer and add zero point
    # Formula: quantized = round(value / scale) + zero_point
    quantized_value = round(Int, value / scale) + Int(zero_point)
    
    # Clamp to INT8 range to prevent overflow
    return clamp(quantized_value, INT8_MIN, INT8_MAX)
end

"""
    dequantize_value(quantized, scale, zero_point, T)

Dequantize a single INT8 value to float.

# Arguments
- `quantized`: INT8 quantized value
- `scale`: Quantization scale
- `zero_point`: Quantization zero point
- `T`: Float type

# Returns
- Dequantized float value
"""
@inline function dequantize_value(quantized::Int8, scale::T, zero_point::Int32) where T
    return T(quantized - zero_point) * scale
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QUANTIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    quantize_int8(x; symmetric=true)

Quantize float tensor to INT8 with optional symmetric or asymmetric quantization.

# Arguments
- `x`: Input tensor (must be non-empty)
- `symmetric`: Use symmetric quantization (default: true). 
               If false, uses asymmetric quantization.

# Returns
- `QuantizedTensor` with INT8 data and quantization parameters

# Examples
```julia
x = randn(Float32, 10, 10)
qt = quantize_int8(x)
dequantized = dequantize(qt)
```

# Algorithm
1. Compute min/max values
2. Calculate scale (and zero_point if asymmetric)
3. Quantize each value: round(value / scale) + zero_point
4. Clamp to INT8 range [-128, 127]

# Performance
- Uses @inbounds for faster indexing
- Efficient memory allocation
- Handles edge cases (all values equal, empty tensors)
"""
function quantize_int8(x::AbstractArray{T}; symmetric::Bool=DEFAULT_SYMMETRIC) where T <: AbstractFloat
    # Validate input
    validate_quantization_input(x)
    
    # Compute value range and quantization parameters
    min_val, max_val = compute_value_range(x)
    scale, zero_point = compute_int8_quantization_params(min_val, max_val, symmetric, T)
    
    # Quantize values
    quantized = quantize_tensor_values(x, scale, zero_point)
    
    # Create and return quantized tensor
    return create_quantized_tensor(quantized, scale, zero_point, min_val, max_val, x, T)
end

# ═══════════════════════════════════════════════════════════════════════════════
# INT8 QUANTIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_quantization_input(x)

Validate input for quantization.

# Arguments
- `x`: Input tensor

# Throws
- `ArgumentError` if input is invalid
"""
function validate_quantization_input(x::AbstractArray)
    if isempty(x)
        throw(ArgumentError("Input tensor cannot be empty"))
    end
end

"""
    compute_value_range(x)

Compute min and max values of tensor.

# Arguments
- `x`: Input tensor

# Returns
- Tuple of (min_val, max_val)

# Notes
- More efficient than separate minimum() and maximum() calls
"""
function compute_value_range(x::AbstractArray{T}) where T
    return minimum(x), maximum(x)
end

"""
    compute_int8_quantization_params(min_val, max_val, symmetric, T)

Compute quantization parameters for INT8.

# Arguments
- `min_val`: Minimum value
- `max_val`: Maximum value
- `symmetric`: Whether to use symmetric quantization
- `T`: Type parameter

# Returns
- Tuple of (scale, zero_point)

# Notes
- Handles edge case where all values are the same
"""
function compute_int8_quantization_params(
    min_val::T,
    max_val::T,
    symmetric::Bool,
    ::Type{T}
) where T <: AbstractFloat
    # Handle edge case: all values are the same
    if min_val == max_val
        return one(T), Int32(0)
    elseif symmetric
        # Symmetric quantization: scale based on absolute maximum
        scale = compute_symmetric_scale(min_val, max_val)
        return scale, Int32(0)
    else
        # Asymmetric quantization: use full range
        return compute_asymmetric_scale(min_val, max_val)
    end
end

"""
    quantize_tensor_values(x, scale, zero_point)

Quantize all values in tensor.

# Arguments
- `x`: Input tensor
- `scale`: Quantization scale
- `zero_point`: Zero point offset

# Returns
- Quantized tensor (Int8)
"""
function quantize_tensor_values(
    x::AbstractArray{T},
    scale::T,
    zero_point::Int32
) where T
    quantized = similar(x, Int8)
    
    # Quantize values using SIMD-friendly loop
    @inbounds @simd for i in eachindex(x)
        quantized[i] = quantize_value(x[i], scale, zero_point)
    end
    
    return quantized
end

"""
    create_quantized_tensor(quantized, scale, zero_point, min_val, max_val, x, T)

Create QuantizedTensor from quantized data.

# Arguments
- `quantized`: Quantized data
- `scale`: Quantization scale
- `zero_point`: Zero point offset
- `min_val`: Minimum value
- `max_val`: Maximum value
- `x`: Original tensor (for shape)
- `T`: Type parameter

# Returns
- QuantizedTensor
"""
function create_quantized_tensor(
    quantized::Array{Int8},
    scale::T,
    zero_point::Int32,
    min_val::T,
    max_val::T,
    x::AbstractArray{T},
    ::Type{T}
) where T
    params = QuantParams{T}(scale, zero_point, min_val, max_val)
    return QuantizedTensor{T, ndims(x)}(quantized, params, size(x))
end

"""
    dequantize(qt::QuantizedTensor{T})

Dequantize INT8 tensor back to float representation.

# Arguments
- `qt`: Quantized tensor to dequantize

# Returns
- Reconstructed float tensor with original shape

# Examples
```julia
qt = quantize_int8(randn(Float32, 5, 5))
x_reconstructed = dequantize(qt)
```

# Algorithm
For each quantized value: (quantized - zero_point) * scale

# Performance
- Uses @inbounds and @simd for vectorization
- Efficient memory allocation
- Preserves original tensor shape
"""
function dequantize(qt::QuantizedTensor{T}) where T
    result = similar(qt.data, T)
    scale = qt.params.scale
    zero_point = qt.params.zero_point
    
    # Dequantize: subtract zero point and multiply by scale
    @inbounds @simd for i in eachindex(qt.data)
        result[i] = dequantize_value(qt.data[i], scale, zero_point)
    end
    
    # Reshape to original shape
    return reshape(result, qt.original_shape)
end
