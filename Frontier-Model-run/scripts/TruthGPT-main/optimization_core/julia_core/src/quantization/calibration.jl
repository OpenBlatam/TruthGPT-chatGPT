"""
Quantization Calibration

Statistics collection and parameter calibration for optimal quantization.

Provides tools for collecting statistics from observed data and computing
optimal quantization parameters based on those statistics.
"""

using Statistics

include("constants.jl")
include("types.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATOR STRUCT
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Calibrator{T}

Collect statistics for quantization calibration from observed data.

# Fields
- `min_val`: Minimum observed value
- `max_val`: Maximum observed value
- `sum_val`: Sum of all observed values (for mean calculation)
- `sum_sq_val`: Sum of squares (for variance calculation)
- `samples`: Number of samples observed

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, randn(Float32, 100))
params = get_params(cal)
```

# Notes
- Accumulates statistics across multiple `observe!` calls
- Can compute mean and standard deviation from accumulated statistics
- Used to determine optimal quantization parameters
"""
mutable struct Calibrator{T}
    min_val::T
    max_val::T
    sum_val::Float64
    sum_sq_val::Float64
    samples::Int
    
    function Calibrator{T}() where T
        new(T(Inf), T(-Inf), 0.0, 0.0, 0)
    end
end

# Convenience constructor
Calibrator(T::Type = Float32) = Calibrator{T}()

# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    observe!(cal::Calibrator, x)

Record values from a tensor for calibration statistics.

# Arguments
- `cal`: Calibrator to update
- `x`: Tensor of values to observe

# Returns
- Updated calibrator (modified in-place)

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, randn(Float32, 100))
observe!(cal, randn(Float32, 200))  # Accumulates statistics
```

# Notes
- Updates min/max, sum, and sum of squares
- Accumulates statistics across multiple calls
- Skips empty arrays (no-op)

# Algorithm
1. Update min_val = min(min_val, min(x))
2. Update max_val = max(max_val, max(x))
3. Accumulate sum_val += sum(x)
4. Accumulate sum_sq_val += sum(x²)
5. Increment samples += length(x)

# Performance
- Efficient accumulation using single pass
- Handles empty arrays gracefully
"""
function observe!(cal::Calibrator{T}, x::AbstractArray) where T
    # Skip empty arrays
    isempty(x) && return cal
    
    # Update statistics
    update_min_max!(cal, x, T)
    accumulate_statistics!(cal, x)
    
    return cal
end

# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    update_min_max!(cal, x, T)

Update min and max values in calibrator.

# Arguments
- `cal`: Calibrator to update
- `x`: Tensor of values
- `T`: Type parameter

# Notes
- Modifies calibrator in-place
"""
function update_min_max!(cal::Calibrator{T}, x::AbstractArray, ::Type{T}) where T
    cal.min_val = min(cal.min_val, T(minimum(x)))
    cal.max_val = max(cal.max_val, T(maximum(x)))
end

"""
    accumulate_statistics!(cal, x)

Accumulate sum and sum of squares statistics.

# Arguments
- `cal`: Calibrator to update
- `x`: Tensor of values

# Notes
- Modifies calibrator in-place
- Uses Float64 for precision
"""
function accumulate_statistics!(cal::Calibrator{T}, x::AbstractArray) where T
    # Accumulate statistics (use Float64 for precision)
    cal.sum_val += sum(Float64, x)
    cal.sum_sq_val += sum(v -> Float64(v)^2, x)
    cal.samples += length(x)
end

"""
    get_params(cal::Calibrator)

Get calibrated quantization parameters from observed statistics.

# Arguments
- `cal`: Calibrator with observed statistics

# Returns
- `QuantParams` suitable for symmetric quantization

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, randn(Float32, 1000))
params = get_params(cal)
```

# Notes
- Requires at least one sample to be observed
- Uses symmetric quantization based on absolute maximum
- Scale is computed as: max(|min|, |max|) / INT8_MAX

# Algorithm
1. Validate that samples > 0
2. Compute abs_max = max(|min_val|, |max_val|)
3. Compute scale = abs_max / INT8_MAX
4. Ensure scale >= eps(T) for numerical stability
5. Return QuantParams with scale, zero_point=0, min_val, max_val

# Throws
- `ArgumentError` if no samples have been observed
"""
function get_params(cal::Calibrator{T}) where T
    # Validate that samples have been observed
    validate_calibrator_samples(cal)
    
    # Compute quantization parameters
    scale = compute_calibrated_scale(cal, T)
    
    return QuantParams{T}(scale, Int32(0), cal.min_val, cal.max_val)
end

"""
    validate_calibrator_samples(cal)

Validate that calibrator has observed samples.

# Arguments
- `cal`: Calibrator to validate

# Throws
- `ArgumentError` if no samples have been observed
"""
function validate_calibrator_samples(cal::Calibrator)
    if cal.samples <= 0
        throw(ArgumentError("Calibrator has no samples. Call observe! first."))
    end
end

"""
    compute_calibrated_scale(cal, T)

Compute scale factor from calibrated statistics.

# Arguments
- `cal`: Calibrator with statistics
- `T`: Type parameter

# Returns
- Scale factor for symmetric quantization

# Notes
- Uses absolute maximum value for symmetric quantization
- Ensures scale >= eps(T) for numerical stability
"""
function compute_calibrated_scale(cal::Calibrator{T}, ::Type{T}) where T
    # Compute scale based on absolute maximum value (symmetric quantization)
    abs_max = max(abs(cal.min_val), abs(cal.max_val))
    scale = abs_max / T(INT8_MAX)
    return max(scale, eps(T))  # Ensure numerical stability
end

# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    mean(cal::Calibrator)

Get mean of observed values.

# Arguments
- `cal`: Calibrator with statistics

# Returns
- Mean value (Float64), or 0.0 if no samples

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, [1.0, 2.0, 3.0])
μ = mean(cal)  # Returns 2.0
```

# Algorithm
mean = sum_val / samples
"""
Statistics.mean(cal::Calibrator) = cal.samples > 0 ? cal.sum_val / cal.samples : 0.0

"""
    std(cal::Calibrator)

Get standard deviation of observed values.

# Arguments
- `cal`: Calibrator with statistics

# Returns
- Standard deviation (Float64), or 0.0 if no samples

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, [1.0, 2.0, 3.0])
σ = std(cal)
```

# Notes
- Uses population standard deviation formula
- Ensures non-negative variance

# Algorithm
1. Compute mean: μ = sum_val / samples
2. Compute variance: σ² = sum_sq_val / samples - μ²
3. Return sqrt(max(σ², 0.0))
"""
function Statistics.std(cal::Calibrator)
    cal.samples > 0 || return 0.0
    
    μ = mean(cal)
    variance = cal.sum_sq_val / cal.samples - μ^2
    
    # Ensure non-negative variance (numerical stability)
    return sqrt(max(variance, 0.0))
end

"""
    reset!(cal::Calibrator)

Reset calibrator to initial state.

# Arguments
- `cal`: Calibrator to reset

# Returns
- Reset calibrator (modified in-place)

# Examples
```julia
cal = Calibrator(Float32)
observe!(cal, randn(Float32, 100))
reset!(cal)  # Clear all statistics
```
"""
function reset!(cal::Calibrator{T}) where T
    cal.min_val = T(Inf)
    cal.max_val = T(-Inf)
    cal.sum_val = 0.0
    cal.sum_sq_val = 0.0
    cal.samples = 0
    return cal
end
