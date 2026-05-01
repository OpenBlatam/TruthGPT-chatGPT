"""
Quantization Constants

Defines all constants used for quantization operations.
"""

# INT8 quantization constants
const INT8_MAX = 127
const INT8_MIN = -128
const INT8_RANGE = 255

# INT4 quantization constants
const INT4_MAX = 7
const INT4_MIN = -8
const INT4_RANGE = 15

# Bit manipulation masks for INT4 packing
const INT4_LOW_MASK = 0x0F
const INT4_HIGH_SHIFT = 4

# Default group size for grouped quantization
const DEFAULT_GROUP_SIZE = 128

# Default quantization mode
const DEFAULT_SYMMETRIC = true











