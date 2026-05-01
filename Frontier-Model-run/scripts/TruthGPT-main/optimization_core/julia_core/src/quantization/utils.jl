"""
Quantization Utilities

Helper functions for quantization operations.

Provides reusable functions for computing scales, zero points, and
quantizing individual values across different quantization schemes.
"""

include("constants.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# SCALE COMPUTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_symmetric_scale(min_val, max_val, quant_max)

Compute scale for symmetric quantization.

# Arguments
- `min_val`: Minimum value in tensor
- `max_val`: Maximum value in tensor
- `quant_max`: Maximum quantized value (e.g., 127 for INT8, 7 for INT4)

# Returns
- Scale factor for symmetric quantization

# Examples
```julia
scale = compute_symmetric_scale(-10.0f0, 10.0f0, 127.0f0)
# Returns: 10.0 / 127.0 ≈ 0.0787
```

# Algorithm
1. Compute abs_max = max(|min_val|, |max_val|)
2. Compute scale = abs_max / quant_max
3. Ensure scale >= eps(T) for numerical stability

# Notes
- Symmetric quantization uses zero_point = 0
- Scale is based on the absolute maximum value
- Ensures all values can be represented in quantized range
"""
function compute_symmetric_scale(min_val::T, max_val::T, quant_max::T) where T
    abs_max = max(abs(min_val), abs(max_val))
    scale = abs_max / quant_max
    return max(scale, eps(T))  # Ensure numerical stability
end

"""
    compute_asymmetric_scale(min_val, max_val, quant_range)

Compute scale and zero point for asymmetric quantization.

# Arguments
- `min_val`: Minimum value in tensor
- `max_val`: Maximum value in tensor
- `quant_range`: Quantization range (e.g., 255 for INT8, 15 for INT4)

# Returns
- Tuple of (scale, zero_point)

# Examples
```julia
scale, zp = compute_asymmetric_scale(-5.0f0, 10.0f0, 255.0f0)
# Returns: (scale ≈ 0.0588, zero_point ≈ 85)
```

# Algorithm
1. Compute value_range = max_val - min_val
2. Compute scale = value_range / quant_range
3. Ensure scale >= eps(T) for numerical stability
4. Compute zero_point = round(-min_val / scale - quant_range / 2)
5. Return (scale, zero_point)

# Notes
- Asymmetric quantization uses full quantized range
- Zero point maps min_val to the minimum quantized value
- Better for distributions not centered around zero
"""
function compute_asymmetric_scale(min_val::T, max_val::T, quant_range::T) where T
    value_range = max_val - min_val
    
    # Handle edge case: all values are the same
    if value_range == 0
        return one(T), Int32(0)
    end
    
    scale = value_range / quant_range
    scale = max(scale, eps(T))  # Ensure numerical stability
    
    # Calculate zero point to map min_val to minimum quantized value
    # Formula: zero_point = round(-min_val / scale - quant_range / 2)
    zero_point = round(Int32, -min_val / scale - quant_range / 2)
    
    return scale, zero_point
end

# ═══════════════════════════════════════════════════════════════════════════════
# VALUE QUANTIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    quantize_value(value, scale, zero_point, min_quant, max_quant)

Quantize a single value with clamping.

# Arguments
- `value`: Value to quantize
- `scale`: Quantization scale
- `zero_point`: Zero point offset
- `min_quant`: Minimum quantized value (e.g., -128 for INT8, -8 for INT4)
- `max_quant`: Maximum quantized value (e.g., 127 for INT8, 7 for INT4)

# Returns
- Quantized integer value (clamped to valid range)

# Examples
```julia
quantized = quantize_value(5.0f0, 0.1f0, 0, -128, 127)
# Returns: clamp(round(5.0 / 0.1) + 0, -128, 127) = 50
```

# Algorithm
1. Compute quantized = round(value / scale) + zero_point
2. Clamp to [min_quant, max_quant] to prevent overflow
3. Return clamped value

# Performance
- @inline for inlining in hot loops
- Efficient rounding and clamping
"""
@inline function quantize_value(
    value::T,
    scale::T,
    zero_point::Int32,
    min_quant::Int,
    max_quant::Int
) where T
    quantized = round(Int, value / scale) + zero_point
    return clamp(quantized, min_quant, max_quant)
end

"""
    dequantize_value(quantized, scale, zero_point, T)

Dequantize a single quantized value.

# Arguments
- `quantized`: Quantized integer value
- `scale`: Quantization scale
- `zero_point`: Zero point offset
- `T`: Float type for output

# Returns
- Dequantized float value

# Examples
```julia
dequantized = dequantize_value(50, 0.1f0, 0, Float32)
# Returns: (50 - 0) * 0.1 = 5.0f0
```

# Algorithm
dequantized = (quantized - zero_point) * scale

# Performance
- @inline for inlining in hot loops
- Efficient type conversion
"""
@inline function dequantize_value(
    quantized::Integer,
    scale::T,
    zero_point::Int32
) where T
    return T(quantized - zero_point) * scale
end

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_quantization_range(min_val, max_val, T)

Validate that min_val <= max_val.

# Arguments
- `min_val`: Minimum value
- `max_val`: Maximum value
- `T`: Type for error message

# Throws
- `ArgumentError` if min_val > max_val
"""
function validate_quantization_range(min_val::T, max_val::T) where T
    if min_val > max_val
        throw(ArgumentError(
            "min_val ($min_val) must be <= max_val ($max_val)"
        ))
    end
end

"""
    validate_positive_scale(scale, T)

Validate that scale is positive.

# Arguments
- `scale`: Scale factor
- `T`: Type for error message

# Throws
- `ArgumentError` if scale <= 0
"""
function validate_positive_scale(scale::T) where T
    if scale <= 0
        throw(ArgumentError("scale must be positive, got $scale"))
    end
end
