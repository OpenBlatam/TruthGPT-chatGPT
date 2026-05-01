"""
Shared validation utilities for the entire optimization_core module.

Provides common validation functions that can be used across all modules.
"""
import logging
from typing import Any, Optional, Union, List, Dict, Set, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_not_none(value: Any, name: str) -> None:
    """
    Validate that value is not None.
    
    Args:
        value: Value to validate
        name: Name of parameter (for error messages)
    
    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def validate_not_empty(value: Any, name: str) -> None:
    """
    Validate that value is not empty.
    
    Works with strings, lists, dicts, sets, etc.
    
    Args:
        value: Value to validate
        name: Name of parameter (for error messages)
    
    Raises:
        ValueError: If value is empty
    """
    if not value:
        raise ValueError(f"{name} cannot be empty")


def validate_type(value: Any, expected_type: type, name: str) -> None:
    """
    Validate that value is of expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        name: Name of parameter (for error messages)
    
    Raises:
        TypeError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def validate_in_range(
    value: Union[int, float],
    name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True
) -> None:
    """
    Validate that value is within a range.
    
    Args:
        value: Value to validate
        name: Name of parameter
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        inclusive_min: Whether min is inclusive
        inclusive_max: Whether max is inclusive
    
    Raises:
        ValueError: If value is out of range
    """
    if min_value is not None:
        if inclusive_min:
            if value < min_value:
                raise ValueError(f"{name} must be >= {min_value}, got {value}")
        else:
            if value <= min_value:
                raise ValueError(f"{name} must be > {min_value}, got {value}")
    
    if max_value is not None:
        if inclusive_max:
            if value > max_value:
                raise ValueError(f"{name} must be <= {max_value}, got {value}")
        else:
            if value >= max_value:
                raise ValueError(f"{name} must be < {max_value}, got {value}")


def validate_one_of(
    value: Any,
    name: str,
    allowed_values: Union[List, Set, tuple]
) -> None:
    """
    Validate that value is one of allowed values.
    
    Args:
        value: Value to validate
        name: Name of parameter
        allowed_values: List/set/tuple of allowed values
    
    Raises:
        ValueError: If value is not in allowed values
    """
    if value not in allowed_values:
        raise ValueError(
            f"{name} must be one of {allowed_values}, got {value}"
        )


def validate_path_exists(
    path: Union[str, Path],
    name: str = "path",
    must_exist: bool = True
) -> Path:
    """
    Validate that path exists.
    
    Args:
        path: Path to validate
        name: Name of parameter
        must_exist: Whether path must exist
    
    Returns:
        Path object
    
    Raises:
        FileNotFoundError: If path doesn't exist and must_exist=True
        ValueError: If path is invalid
    """
    if not path:
        raise ValueError(f"{name} cannot be empty")
    
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    
    return path_obj


def validate_callable(value: Any, name: str) -> None:
    """
    Validate that value is callable.
    
    Args:
        value: Value to validate
        name: Name of parameter
    
    Raises:
        TypeError: If value is not callable
    """
    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value).__name__}")


def validate_dict_keys(
    value: Dict,
    name: str,
    required_keys: Optional[List[str]] = None,
    allowed_keys: Optional[List[str]] = None
) -> None:
    """
    Validate dictionary keys.
    
    Args:
        value: Dictionary to validate
        name: Name of parameter
        required_keys: List of required keys
        allowed_keys: List of allowed keys (all keys must be in this list)
    
    Raises:
        ValueError: If keys are invalid
    """
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary")
    
    if required_keys:
        missing = [key for key in required_keys if key not in value]
        if missing:
            raise ValueError(
                f"{name} missing required keys: {missing}. "
                f"Available keys: {list(value.keys())}"
            )
    
    if allowed_keys:
        invalid = [key for key in value.keys() if key not in allowed_keys]
        if invalid:
            raise ValueError(
                f"{name} contains invalid keys: {invalid}. "
                f"Allowed keys: {allowed_keys}"
            )


def validate_list_items(
    value: List,
    name: str,
    item_validator: Optional[Callable] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> None:
    """
    Validate list items.
    
    Args:
        value: List to validate
        name: Name of parameter
        item_validator: Optional validator function for each item
        min_length: Minimum list length
        max_length: Maximum list length
    
    Raises:
        ValueError: If list is invalid
    """
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    
    if min_length is not None and len(value) < min_length:
        raise ValueError(
            f"{name} must have at least {min_length} items, "
            f"got {len(value)}"
        )
    
    if max_length is not None and len(value) > max_length:
        raise ValueError(
            f"{name} must have at most {max_length} items, "
            f"got {len(value)}"
        )
    
    if item_validator:
        for i, item in enumerate(value):
            try:
                item_validator(item)
            except Exception as e:
                raise ValueError(
                    f"{name}[{i}] validation failed: {e}"
                ) from e













