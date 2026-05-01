"""
TruthGPT Julia Core
==================

High-performance scientific computing module for TruthGPT.

## Features
- JuMP-based mathematical optimization
- Flux.jl deep learning with automatic differentiation
- DifferentialEquations.jl for ODEs/PDEs
- CUDA.jl GPU acceleration
- Zero-copy Python interoperability via PyCall

## Performance
- 10-100x faster than NumPy for numerical operations
- Native autodiff on any Julia code
- Seamless GPU kernels

## Usage
```julia
using TruthGPTCore

# Optimize model hyperparameters
result = optimize_hyperparams(loss_fn, params, bounds)

# Train with automatic differentiation
model = create_transformer(768, 12, 4096)
train!(model, data, optimizer=AdamW(lr=1e-4))
```
"""
module TruthGPTCore

using LinearAlgebra
using Statistics
using Random
using JSON3

export optimize_hyperparams, create_transformer, attention_forward
export quantize_tensor, dequantize_tensor
export kv_cache_get, kv_cache_put

include("optimization.jl")
include("attention.jl")
include("quantization/quantization.jl")
include("cache.jl")
include("utils.jl")

const VERSION = "0.1.0"

function __init__()
    @info "TruthGPTCore v$VERSION initialized"
end

end # module


