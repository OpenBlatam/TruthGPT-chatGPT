"""
FluxML Constants

Default values and configuration constants for FluxML module.

Provides centralized constants for default parameters, device types,
activation functions, loss types, and optimizer types.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TRAINING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Default learning rate (commonly used starting point)
const DEFAULT_LEARNING_RATE = 0.001

# Default number of training epochs
const DEFAULT_EPOCHS = 10

# Default batch size for training
const DEFAULT_BATCH_SIZE = 32

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

# CPU device identifier
const DEVICE_CPU = :cpu

# GPU device identifier
const DEVICE_GPU = :gpu

# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Default activation function (ReLU)
const DEFAULT_ACTIVATION = relu

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

# Cross-entropy loss (for classification)
const LOSS_CROSSENTROPY = :crossentropy

# Mean Squared Error (for regression)
const LOSS_MSE = :mse

# Mean Absolute Error (for regression)
const LOSS_MAE = :mae

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER TYPES
# ═══════════════════════════════════════════════════════════════════════════════

# Adam optimizer (adaptive learning rate)
const OPTIMIZER_ADAM = :adam

# Stochastic Gradient Descent (basic optimizer)
const OPTIMIZER_SGD = :sgd

# RMSprop optimizer (adaptive learning rate)
const OPTIMIZER_RMSPROP = :rmsprop

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

# Number of decimal places for loss display
const LOSS_DISPLAY_PRECISION = 6
