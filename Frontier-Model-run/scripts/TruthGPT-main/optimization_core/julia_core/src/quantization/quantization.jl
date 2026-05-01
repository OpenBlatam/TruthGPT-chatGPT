"""
Quantization Module

Main quantization module providing unified interface for all quantization operations.

This module exports all quantization types, functions, and utilities.
"""

# Include all submodules
include("constants.jl")
include("types.jl")
include("utils.jl")
include("int8.jl")
include("int4.jl")
include("grouped.jl")
include("operations.jl")
include("calibration.jl")

# Re-export all public types and functions
export QuantParams, QuantizedTensor, QuantizedInt4, QuantizedGrouped, GroupQuantParams
export quantize_int8, quantize_int4, quantize_grouped, dequantize
export matmul_int8, dot_int8
export Calibrator, observe!, get_params

# Re-export constants for convenience
export INT8_MIN, INT8_MAX, INT8_RANGE
export INT4_MIN, INT4_MAX, INT4_RANGE
export DEFAULT_GROUP_SIZE, DEFAULT_SYMMETRIC











