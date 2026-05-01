"""
Common math utilities for optimization_core.

Provides reusable mathematical and numerical functions.
"""

import math
from typing import List, Optional, Tuple, Union

from .types import Number


# ════════════════════════════════════════════════════════════════════════════════
# NUMBER CLAMPING AND RANGES
# ════════════════════════════════════════════════════════════════════════════════

def clamp(value: Number, min_value: Number, max_value: Number) -> Number:
    """
    Clamp value between min and max bounds.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
    
    Returns:
        Clamped value (guaranteed to be in [min_value, max_value])
    
    Raises:
        ValueError: If min_value > max_value
    
    Examples:
        >>> clamp(15, 0, 10)
        10
        >>> clamp(-5, 0, 10)
        0
        >>> clamp(5, 0, 10)
        5
    """
    # Validate bounds
    if min_value > max_value:
        raise ValueError(
            f"min_value ({min_value}) must be <= max_value ({max_value})"
        )
    
    return max(min_value, min(value, max_value))


def in_range(value: Number, min_value: Number, max_value: Number, inclusive: bool = True) -> bool:
    """
    Check if value is in range.
    
    Args:
        value: Value to check
        min_value: Minimum value
        max_value: Maximum value
        inclusive: Whether range is inclusive
    
    Returns:
        True if value is in range
    
    Example:
        >>> in_range(5, 0, 10)
        True
        >>> in_range(10, 0, 10, inclusive=False)
        False
    """
    if inclusive:
        return min_value <= value <= max_value
    return min_value < value < max_value


# ════════════════════════════════════════════════════════════════════════════════
# ROUNDING AND PRECISION
# ════════════════════════════════════════════════════════════════════════════════

def round_to(value: Number, decimals: int = 2) -> float:
    """
    Round value to specified decimal places.
    
    Args:
        value: Value to round
        decimals: Number of decimal places
    
    Returns:
        Rounded value
    
    Example:
        >>> round_to(3.14159, 2)
        3.14
    """
    multiplier = 10 ** decimals
    return round(value * multiplier) / multiplier


def ceil_to(value: Number, decimals: int = 2) -> float:
    """
    Ceil value to specified decimal places.
    
    Args:
        value: Value to ceil
        decimals: Number of decimal places
    
    Returns:
        Ceiled value
    
    Example:
        >>> ceil_to(3.14159, 2)
        3.15
    """
    multiplier = 10 ** decimals
    return math.ceil(value * multiplier) / multiplier


def floor_to(value: Number, decimals: int = 2) -> float:
    """
    Floor value to specified decimal places.
    
    Args:
        value: Value to floor
        decimals: Number of decimal places
    
    Returns:
        Floored value
    
    Example:
        >>> floor_to(3.14159, 2)
        3.14
    """
    multiplier = 10 ** decimals
    return math.floor(value * multiplier) / multiplier


# ════════════════════════════════════════════════════════════════════════════════
# PERCENTAGES AND RATIOS
# ════════════════════════════════════════════════════════════════════════════════

def percentage(part: Number, total: Number) -> float:
    """
    Calculate percentage.
    
    Args:
        part: Part value
        total: Total value
    
    Returns:
        Percentage (0-100)
    
    Example:
        >>> percentage(25, 100)
        25.0
    """
    if total == 0:
        return 0.0
    return (part / total) * 100.0


def percentage_of(value: Number, percentage: Number) -> float:
    """
    Calculate percentage of value.
    
    Args:
        value: Base value
        percentage: Percentage (0-100)
    
    Returns:
        Percentage of value
    
    Example:
        >>> percentage_of(100, 25)
        25.0
    """
    return (value * percentage) / 100.0


def ratio(numerator: Number, denominator: Number) -> float:
    """
    Calculate ratio.
    
    Args:
        numerator: Numerator
        denominator: Denominator
    
    Returns:
        Ratio (0.0 if denominator is 0)
    
    Example:
        >>> ratio(3, 4)
        0.75
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ════════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════════════════

def mean(values: List[Number]) -> float:
    """
    Calculate arithmetic mean (average) of values.
    
    Args:
        values: List of numeric values (must be non-empty for meaningful result)
    
    Returns:
        Mean value (0.0 if list is empty)
    
    Raises:
        TypeError: If values contains non-numeric types
    
    Examples:
        >>> mean([1, 2, 3, 4, 5])
        3.0
        >>> mean([10.5, 20.5, 30.5])
        20.5
        >>> mean([])
        0.0
    """
    if not values:
        return 0.0
    
    # Validate that all values are numeric
    if not all(isinstance(v, (int, float)) for v in values):
        raise TypeError("All values must be numeric (int or float)")
    
    return sum(values) / len(values)


def median(values: List[Number]) -> float:
    """
    Calculate median.
    
    Args:
        values: List of values
    
    Returns:
        Median value
    
    Example:
        >>> median([1, 2, 3, 4, 5])
        3.0
        >>> median([1, 2, 3, 4])
        2.5
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0
    return float(sorted_values[n // 2])


def std_dev(values: List[Number], sample: bool = False) -> float:
    """
    Calculate standard deviation (population or sample).
    
    Args:
        values: List of numeric values
        sample: If True, use sample standard deviation (Bessel's correction, n-1).
                If False, use population standard deviation (n).
    
    Returns:
        Standard deviation (0.0 if insufficient data)
    
    Raises:
        TypeError: If values contains non-numeric types
    
    Examples:
        >>> std_dev([1, 2, 3, 4, 5])  # Population std dev
        1.4142135623730951
        >>> std_dev([1, 2, 3, 4, 5], sample=True)  # Sample std dev
        1.5811388300841898
    """
    # Need at least 2 values for meaningful standard deviation
    if not values or len(values) < 2:
        return 0.0
    
    # Validate that all values are numeric
    if not all(isinstance(v, (int, float)) for v in values):
        raise TypeError("All values must be numeric (int or float)")
    
    # Calculate mean
    avg = mean(values)
    
    # Calculate variance: sum of squared differences from mean
    variance = sum((x - avg) ** 2 for x in values)
    
    # Use appropriate denominator (n-1 for sample, n for population)
    n = len(values) - 1 if sample else len(values)
    if n <= 0:
        return 0.0
    
    return math.sqrt(variance / n)


# ════════════════════════════════════════════════════════════════════════════════
# NUMBER VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

def is_positive(value: Number) -> bool:
    """
    Check if value is positive.
    
    Args:
        value: Value to check
    
    Returns:
        True if positive
    
    Example:
        >>> is_positive(5)
        True
        >>> is_positive(-5)
        False
    """
    return value > 0


def is_non_negative(value: Number) -> bool:
    """
    Check if value is non-negative.
    
    Args:
        value: Value to check
    
    Returns:
        True if non-negative
    
    Example:
        >>> is_non_negative(0)
        True
        >>> is_non_negative(-1)
        False
    """
    return value >= 0


def is_finite(value: Number) -> bool:
    """
    Check if value is finite (not inf or nan).
    
    Args:
        value: Value to check
    
    Returns:
        True if finite
    
    Example:
        >>> is_finite(5.0)
        True
        >>> is_finite(float('inf'))
        False
    """
    return math.isfinite(value)


# ════════════════════════════════════════════════════════════════════════════════
# INTERPOLATION
# ════════════════════════════════════════════════════════════════════════════════

def lerp(start: Number, end: Number, t: Number) -> float:
    """
    Linear interpolation between start and end.
    
    Args:
        start: Start value
        end: End value
        t: Interpolation factor (0-1)
    
    Returns:
        Interpolated value
    
    Example:
        >>> lerp(0, 10, 0.5)
        5.0
    """
    t = clamp(t, 0.0, 1.0)
    return start + (end - start) * t


def map_range(
    value: Number,
    from_min: Number,
    from_max: Number,
    to_min: Number,
    to_max: Number
) -> float:
    """
    Map value from one range to another using linear transformation.
    
    Args:
        value: Value to map
        from_min: Source range minimum
        from_max: Source range maximum
        to_min: Target range minimum
        to_max: Target range maximum
    
    Returns:
        Mapped value in target range
    
    Raises:
        ValueError: If source range is invalid (from_min == from_max)
    
    Examples:
        >>> map_range(5, 0, 10, 0, 100)  # Maps 5 from [0,10] to [0,100]
        50.0
        >>> map_range(0.5, 0, 1, -1, 1)  # Maps 0.5 from [0,1] to [-1,1]
        0.0
    """
    # Validate source range
    if from_max == from_min:
        raise ValueError(
            f"Source range is invalid: from_min ({from_min}) == from_max ({from_max})"
        )
    
    # Normalize value to [0, 1] in source range
    normalized = (value - from_min) / (from_max - from_min)
    
    # Map normalized value to target range
    return float(to_min + (to_max - to_min) * normalized)


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Clamping
    "clamp",
    "in_range",
    # Rounding
    "round_to",
    "ceil_to",
    "floor_to",
    # Percentages
    "percentage",
    "percentage_of",
    "ratio",
    # Statistics
    "mean",
    "median",
    "std_dev",
    # Validation
    "is_positive",
    "is_non_negative",
    "is_finite",
    # Interpolation
    "lerp",
    "map_range",
]



