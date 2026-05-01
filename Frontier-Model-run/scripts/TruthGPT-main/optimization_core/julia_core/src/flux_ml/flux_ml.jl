"""
FluxML Module

Main FluxML module providing unified interface for all machine learning operations.

This module exports all types, functions, and utilities for Flux-based ML.
"""

using Flux
using CUDA
using Statistics

# Include all submodules
include("constants.jl")
include("types.jl")
include("device.jl")
include("validation.jl")
include("models.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")
include("prediction.jl")

# Re-export all public types and functions
export create_model, train_model, predict
export create_language_model, train_language_model
export TrainingConfig, is_gpu_available

# Re-export constants for convenience
export DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
export DEVICE_CPU, DEVICE_GPU
export LOSS_CROSSENTROPY, LOSS_MSE, LOSS_MAE
export OPTIMIZER_ADAM, OPTIMIZER_SGD, OPTIMIZER_RMSPROP











