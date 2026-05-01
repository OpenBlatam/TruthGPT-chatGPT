"""
Group Quantization

Functions for per-group quantization with improved accuracy.

Provides better quantization accuracy by computing separate scale factors
for each group of elements, rather than a single scale for the entire tensor.
"""

include("constants.jl")
include("types.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_group_scale(group_view, T)

Compute scale factor for a single group using symmetric quantization.

# Arguments
- `group_view`: View of group elements
- `T`: Float type

# Returns
- Scale factor for the group

# Algorithm
scale = max(|values|) / INT8_MAX
"""
@inline function compute_group_scale(group_view::AbstractArray{T}) where T <: AbstractFloat
    abs_max = maximum(abs, group_view)
    scale = abs_max / T(INT8_MAX)
    return max(scale, eps(T))
end

"""
    quantize_group_elements(x_flat, start_idx, end_idx, scale)

Quantize elements in a group.

# Arguments
- `x_flat`: Flattened input array
- `start_idx`: Start index of group
- `end_idx`: End index of group (inclusive)
- `scale`: Scale factor for the group

# Returns
- Vector of quantized Int8 values

# Performance
- Uses @inbounds for faster indexing
- Efficient quantization with clamping
"""
function quantize_group_elements(
    x_flat::AbstractArray{T},
    start_idx::Int,
    end_idx::Int,
    scale::T
) where T <: AbstractFloat
    quantized = Vector{Int8}(undef, end_idx - start_idx + 1)
    
    @inbounds for (local_idx, global_idx) in enumerate(start_idx:end_idx)
        quantized_value = round(Int, x_flat[global_idx] / scale)
        quantized[local_idx] = clamp(quantized_value, INT8_MIN, INT8_MAX)
    end
    
    return quantized
end

"""
    get_group_indices(group_idx, group_size, numel)

Get start and end indices for a quantization group.

# Arguments
- `group_idx`: Group index (1-based)
- `group_size`: Size of each group
- `numel`: Total number of elements

# Returns
- Tuple of (start_idx, end_idx) where end_idx is inclusive
"""
@inline function get_group_indices(group_idx::Int, group_size::Int, numel::Int)
    start_idx = (group_idx - 1) * group_size + 1
    end_idx = min(group_idx * group_size, numel)
    return start_idx, end_idx
end

"""
    get_group_index(element_idx, group_size)

Get the group index for a given element index.

# Arguments
- `element_idx`: Element index (1-based)
- `group_size`: Size of each group

# Returns
- Group index (1-based)
"""
@inline function get_group_index(element_idx::Int, group_size::Int)
    return div(element_idx - 1, group_size) + 1
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QUANTIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    quantize_grouped(x, group_size=128)

Quantize with per-group parameters for better accuracy.

# Arguments
- `x`: Input float tensor
- `group_size`: Number of elements per quantization group (default: 128)

# Returns
- `QuantizedGrouped` tensor with per-group quantization parameters

# Examples
```julia
x = randn(Float32, 1000)
qt = quantize_grouped(x, group_size=128)
dequantized = dequantize(qt)
```

# Notes
- Each group has its own scale factor, improving accuracy for tensors with
  varying value distributions
- Larger group sizes reduce memory overhead but may reduce accuracy
- Smaller group sizes improve accuracy but increase memory overhead

# Algorithm
1. Flatten input tensor
2. Divide into groups of size `group_size`
3. For each group:
   a. Compute group-specific scale (symmetric quantization)
   b. Quantize all elements in group using group scale
4. Store quantized data and per-group scales

# Performance
- Uses @inbounds and @view for efficient memory access
- Pre-allocates arrays for better performance
- Efficient group-wise processing
"""
function quantize_grouped(
    x::AbstractArray{T},
    group_size::Int = DEFAULT_GROUP_SIZE
) where T <: AbstractFloat
    # Validate input
    validate_grouped_quantization_input(x, group_size)
    
    # Setup for grouped quantization
    x_flat = vec(x)
    numel = length(x_flat)
    num_groups = compute_num_groups(numel, group_size)
    
    # Pre-allocate arrays
    scales, zero_points, quantized = allocate_grouped_arrays(T, num_groups, numel)
    
    # Quantize each group independently
    quantize_all_groups!(x_flat, scales, quantized, num_groups, group_size, numel)
    
    # Create and return quantized tensor
    return create_quantized_grouped(quantized, scales, zero_points, group_size, x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GROUPED QUANTIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_grouped_quantization_input(x, group_size)

Validate input for grouped quantization.

# Arguments
- `x`: Input tensor
- `group_size`: Group size

# Throws
- `ArgumentError` if input is invalid
"""
function validate_grouped_quantization_input(x::AbstractArray, group_size::Int)
    if isempty(x)
        throw(ArgumentError("Input tensor cannot be empty"))
    end
    if group_size <= 0
        throw(ArgumentError("group_size must be positive, got $group_size"))
    end
end

"""
    compute_num_groups(numel, group_size)

Compute number of groups needed.

# Arguments
- `numel`: Total number of elements
- `group_size`: Size of each group

# Returns
- Number of groups (rounded up)
"""
function compute_num_groups(numel::Int, group_size::Int)
    return cld(numel, group_size)
end

"""
    allocate_grouped_arrays(T, num_groups, numel)

Allocate arrays for grouped quantization.

# Arguments
- `T`: Float type
- `num_groups`: Number of groups
- `numel`: Total number of elements

# Returns
- Tuple of (scales, zero_points, quantized)
"""
function allocate_grouped_arrays(::Type{T}, num_groups::Int, numel::Int) where T
    scales = Vector{T}(undef, num_groups)
    zero_points = zeros(Int32, num_groups)  # Symmetric quantization uses zero_point = 0
    quantized = Vector{Int8}(undef, numel)
    return scales, zero_points, quantized
end

"""
    quantize_all_groups!(x_flat, scales, quantized, num_groups, group_size, numel)

Quantize all groups in the tensor.

# Arguments
- `x_flat`: Flattened input array
- `scales`: Scales array (modified in-place)
- `quantized`: Quantized array (modified in-place)
- `num_groups`: Number of groups
- `group_size`: Size of each group
- `numel`: Total number of elements

# Notes
- Modifies scales and quantized arrays in-place
"""
function quantize_all_groups!(
    x_flat::AbstractArray{T},
    scales::Vector{T},
    quantized::Vector{Int8},
    num_groups::Int,
    group_size::Int,
    numel::Int
) where T
    @inbounds for group_idx in 1:num_groups
        # Get group indices
        start_idx, end_idx = get_group_indices(group_idx, group_size, numel)
        
        # Create view of group elements (avoids copying)
        group_view = @view x_flat[start_idx:end_idx]
        
        # Compute group-specific scale (symmetric quantization)
        scale = compute_group_scale(group_view)
        scales[group_idx] = scale
        
        # Quantize all elements in this group
        quantize_group_range!(x_flat, quantized, start_idx, end_idx, scale)
    end
end

"""
    quantize_group_range!(x_flat, quantized, start_idx, end_idx, scale)

Quantize a range of elements in a group.

# Arguments
- `x_flat`: Flattened input array
- `quantized`: Quantized array (modified in-place)
- `start_idx`: Start index of group
- `end_idx`: End index of group (inclusive)
- `scale`: Scale factor for the group

# Notes
- Modifies quantized array in-place
"""
function quantize_group_range!(
    x_flat::AbstractArray{T},
    quantized::Vector{Int8},
    start_idx::Int,
    end_idx::Int,
    scale::T
) where T
    @inbounds for global_idx in start_idx:end_idx
        quantized_value = round(Int, x_flat[global_idx] / scale)
        quantized[global_idx] = clamp(quantized_value, INT8_MIN, INT8_MAX)
    end
end

"""
    create_quantized_grouped(quantized, scales, zero_points, group_size, x)

Create QuantizedGrouped tensor from quantized data.

# Arguments
- `quantized`: Quantized data
- `scales`: Per-group scales
- `zero_points`: Per-group zero points
- `group_size`: Group size
- `x`: Original tensor (for shape)

# Returns
- QuantizedGrouped tensor
"""
function create_quantized_grouped(
    quantized::Vector{Int8},
    scales::Vector{T},
    zero_points::Vector{Int32},
    group_size::Int,
    x::AbstractArray{T}
) where T
    params = GroupQuantParams{T}(scales, zero_points, group_size)
    return QuantizedGrouped{T, ndims(x)}(
        reshape(quantized, size(x)),
        params,
        size(x)
    )
end

"""
    dequantize(qt::QuantizedGrouped{T})

Dequantize grouped quantized tensor.

# Arguments
- `qt`: Quantized grouped tensor

# Returns
- Reconstructed float tensor with original shape

# Examples
```julia
qt = quantize_grouped(randn(Float32, 1000))
x_reconstructed = dequantize(qt)
```

# Notes
- Uses per-group scale factors for accurate reconstruction
- Each element is dequantized using its group's scale

# Algorithm
1. Flatten quantized data
2. For each element:
   a. Determine which group it belongs to
   b. Dequantize using that group's scale: value * scale[group_idx]
3. Reshape to original shape

# Performance
- Uses @inbounds for faster indexing
- Efficient group index calculation
- Minimal memory allocations
"""
function dequantize(qt::QuantizedGrouped{T}) where T
    result = similar(qt.data, T)
    data_flat = vec(qt.data)
    result_flat = vec(result)
    
    group_size = qt.group_params.group_size
    scales = qt.group_params.scales
    
    # Dequantize using group-specific scales
    @inbounds for i in eachindex(data_flat)
        group_idx = get_group_index(i, group_size)
        result_flat[i] = T(data_flat[i]) * scales[group_idx]
    end
    
    return result
end
