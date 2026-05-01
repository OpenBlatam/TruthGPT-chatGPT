"""
Quantization Types

Type definitions for quantization parameters and quantized tensors.

Provides type-safe structures for representing quantized data and
their associated parameters for accurate dequantization.
"""

include("constants.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    QuantParams{T}

Quantization parameters for a tensor.

Stores the scale factor, zero point, and value range needed to
quantize and dequantize tensors accurately.

# Fields
- `scale`: Scaling factor for quantization/dequantization
- `zero_point`: Zero point offset (for asymmetric quantization, typically 0 for symmetric)
- `min_val`: Minimum value in original tensor
- `max_val`: Maximum value in original tensor

# Examples
```julia
params = QuantParams{Float32}(0.1f0, 0, -10.0f0, 10.0f0)
```

# Notes
- For symmetric quantization: zero_point = 0
- For asymmetric quantization: zero_point maps min_val to minimum quantized value
- Scale must be positive for numerical stability
"""
struct QuantParams{T}
    scale::T
    zero_point::Int32
    min_val::T
    max_val::T
    
    function QuantParams{T}(
        scale::T,
        zero_point::Int32,
        min_val::T,
        max_val::T
    ) where T
        # Validate parameters
        scale > zero(T) || throw(ArgumentError(
            "Scale must be positive, got $scale"
        ))
        min_val <= max_val || throw(ArgumentError(
            "min_val ($min_val) must be <= max_val ($max_val)"
        ))
        
        new(scale, zero_point, min_val, max_val)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZED TENSOR TYPES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    QuantizedTensor{T, N}

Quantized tensor with parameters for dequantization.

Stores INT8 quantized data along with quantization parameters
and original shape for accurate reconstruction.

# Fields
- `data`: Quantized INT8 data array [shape]
- `params`: Quantization parameters
- `original_shape`: Original tensor shape before quantization

# Examples
```julia
x = randn(Float32, 10, 20)
qt = quantize_int8(x)
# qt.data: Array{Int8, 2} with shape (10, 20)
# qt.original_shape: (10, 20)
```

# Notes
- Data length must match product of original_shape
- Type parameter T is the original float type (Float32, Float64, etc.)
- Type parameter N is the number of dimensions
"""
struct QuantizedTensor{T, N}
    data::Array{Int8, N}
    params::QuantParams{T}
    original_shape::NTuple{N, Int}
    
    function QuantizedTensor{T, N}(
        data::Array{Int8, N},
        params::QuantParams{T},
        shape::NTuple{N, Int}
    ) where {T, N}
        expected_length = prod(shape)
        actual_length = length(data)
        
        actual_length == expected_length || throw(ArgumentError(
            "Data length ($actual_length) doesn't match shape product ($expected_length)"
        ))
        
        new(data, params, shape)
    end
end

"""
    QuantizedInt4{T, N}

INT4 quantized data (packed as pairs in UInt8 for memory efficiency).

Stores INT4 quantized data packed into UInt8 bytes (2 values per byte)
for 2x memory savings compared to INT8.

# Fields
- `data`: Packed UInt8 array (each byte contains two INT4 values)
- `params`: Quantization parameters (symmetric, zero_point = 0)
- `original_shape`: Original tensor shape
- `original_numel`: Original number of elements (needed for odd-length arrays)

# Examples
```julia
x = randn(Float32, 10, 20)
qt = quantize_int4(x)
# qt.data: Array{UInt8, 2} with shape (10, 10) - half the size!
# qt.original_numel: 200
```

# Notes
- Each UInt8 byte stores two INT4 values (4 bits each)
- For odd-length arrays, last value is padded with 0
- Uses symmetric quantization only (zero_point = 0)
- Packed size = ceil(original_numel / 2)
"""
struct QuantizedInt4{T, N}
    data::Array{UInt8, N}
    params::QuantParams{T}
    original_shape::NTuple{N, Int}
    original_numel::Int
    
    function QuantizedInt4{T, N}(
        data::Array{UInt8, N},
        params::QuantParams{T},
        shape::NTuple{N, Int},
        numel::Int
    ) where {T, N}
        expected_packed_size = cld(numel, 2)
        actual_packed_size = length(data)
        
        actual_packed_size == expected_packed_size || throw(ArgumentError(
            "Packed data size ($actual_packed_size) doesn't match expected size ($expected_packed_size) "
            "for $numel elements"
        ))
        
        prod(shape) == numel || throw(ArgumentError(
            "Shape product ($(prod(shape))) doesn't match numel ($numel)"
        ))
        
        new(data, params, shape, numel)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# GROUP QUANTIZATION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GroupQuantParams{T}

Per-group quantization parameters for improved accuracy.

Stores separate scale factors and zero points for each group of elements,
allowing better quantization accuracy for tensors with varying value distributions.

# Fields
- `scales`: Vector of scale factors (one per group)
- `zero_points`: Vector of zero points (one per group, typically all 0 for symmetric)
- `group_size`: Number of elements per group

# Examples
```julia
scales = [0.1f0, 0.2f0, 0.15f0]  # 3 groups
zero_points = zeros(Int32, 3)
params = GroupQuantParams{Float32}(scales, zero_points, 128)
```

# Notes
- scales and zero_points must have the same length
- All scales must be positive
- group_size must be positive
- Typically uses symmetric quantization (zero_points all 0)
"""
struct GroupQuantParams{T}
    scales::Vector{T}
    zero_points::Vector{Int32}
    group_size::Int
    
    function GroupQuantParams{T}(
        scales::Vector{T},
        zero_points::Vector{Int32},
        group_size::Int
    ) where T
        # Validate lengths match
        length(scales) == length(zero_points) || throw(ArgumentError(
            "scales (length $(length(scales))) and zero_points (length $(length(zero_points))) "
            "must have same length"
        ))
        
        # Validate group_size
        group_size > 0 || throw(ArgumentError(
            "group_size must be positive, got $group_size"
        ))
        
        # Validate all scales are positive
        all(s -> s > zero(T), scales) || throw(ArgumentError(
            "All scales must be positive. Found non-positive values: "
            * string([s for s in scales if s <= zero(T)])
        ))
        
        new(scales, zero_points, group_size)
    end
end

"""
    QuantizedGrouped{T, N}

Grouped quantization for better accuracy (different scale per group).

Stores INT8 quantized data with per-group quantization parameters,
providing better accuracy than single-scale quantization for tensors
with varying value distributions.

# Fields
- `data`: Quantized INT8 data [original_shape]
- `group_params`: Per-group quantization parameters
- `original_shape`: Original tensor shape

# Examples
```julia
x = randn(Float32, 1000)
qt = quantize_grouped(x, group_size=128)
# qt.data: Array{Int8, 1} with shape (1000,)
# qt.group_params.scales: Vector with ceil(1000/128) = 8 scales
```

# Notes
- Each group has its own scale factor for better accuracy
- Data length must match product of original_shape
- Group size determines memory overhead (more groups = more scales)
- Trade-off: larger groups = less memory, smaller groups = better accuracy
"""
struct QuantizedGrouped{T, N}
    data::Array{Int8, N}
    group_params::GroupQuantParams{T}
    original_shape::NTuple{N, Int}
    
    function QuantizedGrouped{T, N}(
        data::Array{Int8, N},
        params::GroupQuantParams{T},
        shape::NTuple{N, Int}
    ) where {T, N}
        expected_length = prod(shape)
        actual_length = length(data)
        
        actual_length == expected_length || throw(ArgumentError(
            "Data length ($actual_length) doesn't match shape product ($expected_length)"
        ))
        
        new(data, params, shape)
    end
end
