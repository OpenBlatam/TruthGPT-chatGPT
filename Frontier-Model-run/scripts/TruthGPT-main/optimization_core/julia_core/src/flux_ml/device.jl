"""
Device Management

Functions for managing CPU/GPU device operations.

Provides utilities for checking GPU availability, ensuring device compatibility,
and moving data/models between CPU and GPU devices.
"""

using CUDA

include("constants.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE AVAILABILITY
# ═══════════════════════════════════════════════════════════════════════════════

"""
    is_gpu_available()

Check if GPU is available and functional.

# Returns
- `true` if GPU is available and functional, `false` otherwise

# Examples
```julia
if is_gpu_available()
    println("GPU is available!")
    device = :gpu
else
    println("Using CPU")
    device = :cpu
end
```

# Notes
- Uses CUDA.functional() to check GPU availability
- Returns false if CUDA is not installed or GPU is not accessible
"""
is_gpu_available() = CUDA.functional()

"""
    ensure_device_available(device)

Ensure requested device is available, fallback to CPU if not.

# Arguments
- `device`: Requested device (:cpu or :gpu)

# Returns
- Available device (:cpu or :gpu)

# Examples
```julia
device = ensure_device_available(:gpu)
# Returns :gpu if available, :cpu otherwise (with warning)
```

# Notes
- Automatically falls back to CPU if GPU is requested but unavailable
- Warns user if fallback occurs
- Always returns a valid device (never throws)

# Algorithm
1. If device == :gpu and GPU not available:
   a. Log warning
   b. Return :cpu
2. Otherwise, return requested device
"""
function ensure_device_available(device::Symbol)::Symbol
    # Check if GPU is requested but not available
    if device == DEVICE_GPU && !is_gpu_available()
        return handle_gpu_unavailable()
    end
    
    # Validate device is one of the supported types
    if !is_valid_device(device)
        return handle_invalid_device(device)
    end
    
    return device
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    is_valid_device(device)

Check if device is a valid device type.

# Arguments
- `device`: Device symbol to validate

# Returns
- `true` if device is valid, `false` otherwise
"""
function is_valid_device(device::Symbol)::Bool
    return device ∈ [DEVICE_CPU, DEVICE_GPU]
end

"""
    handle_gpu_unavailable()

Handle case when GPU is requested but not available.

# Returns
- `:cpu` (with warning)

# Notes
- Logs warning about GPU fallback
"""
function handle_gpu_unavailable()::Symbol
    @warn "GPU requested but not available, falling back to CPU"
    return DEVICE_CPU
end

"""
    handle_invalid_device(device)

Handle case when device is invalid.

# Arguments
- `device`: Invalid device symbol

# Returns
- `:cpu` (with warning)

# Notes
- Logs warning about invalid device
"""
function handle_invalid_device(device::Symbol)::Symbol
    @warn "Unknown device :$device, falling back to CPU"
    return DEVICE_CPU
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE TRANSFER
# ═══════════════════════════════════════════════════════════════════════════════

"""
    move_to_device(x, device)

Move data or model to specified device.

# Arguments
- `x`: Data or model to move (Array, Flux model, etc.)
- `device`: Target device (:cpu or :gpu)

# Returns
- Data/model on target device

# Examples
```julia
x = randn(Float32, 10, 10)
x_gpu = move_to_device(x, :gpu)  # Moves to GPU if available
x_cpu = move_to_device(x_gpu, :cpu)  # Moves back to CPU
```

# Notes
- Automatically falls back to CPU if GPU is requested but not available
- Uses Flux's `gpu()` and `cpu()` functions for device transfer
- Works with arrays, models, and other Flux-compatible types

# Algorithm
1. Ensure device is available (fallback to CPU if needed)
2. If device == :gpu: apply `gpu()` function
3. If device == :cpu: apply `cpu()` function
4. Return transferred data/model
"""
function move_to_device(x, device::Symbol)
    # Ensure device is available (fallback to CPU if needed)
    actual_device = ensure_device_available(device)
    
    if actual_device == DEVICE_GPU
        return x |> gpu
    else
        return x |> cpu
    end
end

"""
    get_device(x)

Get the device of data or model.

# Arguments
- `x`: Data or model to check

# Returns
- Device symbol (:cpu or :gpu)

# Examples
```julia
x = randn(Float32, 10, 10) |> gpu
device = get_device(x)  # Returns :gpu
```

# Notes
- Checks if data is on GPU using CUDA.isgpu()
- Returns :gpu if on GPU, :cpu otherwise
"""
function get_device(x)::Symbol
    if CUDA.isgpu(x)
        return DEVICE_GPU
    else
        return DEVICE_CPU
    end
end
