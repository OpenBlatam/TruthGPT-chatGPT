"""
Validation utilities for polyglot_core.

Provides input validation, shape checking, and type validation.
"""

from typing import Any, Optional, Tuple, List, Union
import numpy as np


class ValidationError(Exception):
    """Validation error."""
    pass


def validate_tensor(
    tensor: Any,
    name: str = "tensor",
    dtype: Optional[type] = None,
    shape: Optional[Tuple[int, ...]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False
) -> np.ndarray:
    """
    Validate tensor.
    
    Args:
        tensor: Input tensor
        name: Tensor name for error messages
        dtype: Expected dtype
        shape: Expected shape (None for any dimension)
        min_value: Minimum value
        max_value: Maximum value
        allow_none: Allow None values
        
    Returns:
        Validated numpy array
        
    Raises:
        ValidationError: If validation fails
    """
    if tensor is None:
        if allow_none:
            return None
        raise ValidationError(f"{name} cannot be None")
    
    # Convert to numpy array
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = np.asarray(tensor)
        except Exception as e:
            raise ValidationError(f"{name} cannot be converted to numpy array: {e}")
    
    # Check dtype
    if dtype is not None and tensor.dtype != dtype:
        try:
            tensor = tensor.astype(dtype)
        except Exception as e:
            raise ValidationError(f"{name} cannot be cast to {dtype}: {e}")
    
    # Check shape
    if shape is not None:
        if len(tensor.shape) != len(shape):
            raise ValidationError(
                f"{name} has wrong number of dimensions: "
                f"expected {len(shape)}, got {len(tensor.shape)}"
            )
        
        for i, (expected, actual) in enumerate(zip(shape, tensor.shape)):
            if expected is not None and expected != actual:
                raise ValidationError(
                    f"{name} has wrong shape at dimension {i}: "
                    f"expected {expected}, got {actual}"
                )
    
    # Check values
    if min_value is not None:
        if np.any(tensor < min_value):
            raise ValidationError(f"{name} contains values below {min_value}")
    
    if max_value is not None:
        if np.any(tensor > max_value):
            raise ValidationError(f"{name} contains values above {max_value}")
    
    return tensor


def validate_attention_inputs(
    query: Any,
    key: Any,
    value: Any,
    batch_size: int,
    seq_len: int,
    d_model: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate attention inputs.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        
    Returns:
        Tuple of validated tensors
        
    Raises:
        ValidationError: If validation fails
    """
    query = validate_tensor(
        query,
        name="query",
        shape=(batch_size * seq_len, d_model)
    )
    
    key = validate_tensor(
        key,
        name="key",
        shape=(batch_size * seq_len, d_model)
    )
    
    value = validate_tensor(
        value,
        name="value",
        shape=(batch_size * seq_len, d_model)
    )
    
    return query, key, value


def validate_cache_key(
    layer: int,
    position: int,
    tag: str = ""
) -> Tuple[int, int, str]:
    """
    Validate cache key.
    
    Args:
        layer: Layer index
        position: Position index
        tag: Optional tag
        
    Returns:
        Validated (layer, position, tag)
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(layer, int) or layer < 0:
        raise ValidationError(f"layer must be non-negative integer, got {layer}")
    
    if not isinstance(position, int) or position < 0:
        raise ValidationError(f"position must be non-negative integer, got {position}")
    
    if not isinstance(tag, str):
        raise ValidationError(f"tag must be string, got {type(tag)}")
    
    return layer, position, tag


def validate_config(config: Any, config_type: type, name: str = "config"):
    """
    Validate configuration object.
    
    Args:
        config: Configuration object
        config_type: Expected type
        name: Config name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, config_type):
        raise ValidationError(
            f"{name} must be instance of {config_type.__name__}, "
            f"got {type(config).__name__}"
        )


def validate_backend(backend: Any, available_backends: List[str]):
    """
    Validate backend selection.
    
    Args:
        backend: Backend to validate
        available_backends: List of available backends
        
    Raises:
        ValidationError: If backend not available
    """
    backend_str = str(backend) if hasattr(backend, 'name') else str(backend)
    
    if backend_str not in available_backends:
        raise ValidationError(
            f"Backend {backend_str} not available. "
            f"Available: {', '.join(available_backends)}"
        )


def validate_range(
    value: float,
    name: str,
    min_value: float,
    max_value: float
) -> float:
    """
    Validate value is in range.
    
    Args:
        value: Value to validate
        name: Value name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value out of range
    """
    if not (min_value <= value <= max_value):
        raise ValidationError(
            f"{name} must be between {min_value} and {max_value}, got {value}"
        )
    
    return value


def validate_positive(value: Union[int, float], name: str) -> Union[int, float]:
    """
    Validate value is positive.
    
    Args:
        value: Value to validate
        name: Value name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value not positive
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    
    return value


def validate_non_negative(value: Union[int, float], name: str) -> Union[int, float]:
    """
    Validate value is non-negative.
    
    Args:
        value: Value to validate
        name: Value name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value negative
    """
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")
    
    return value













