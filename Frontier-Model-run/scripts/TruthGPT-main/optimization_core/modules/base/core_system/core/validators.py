"""
Common validation utilities for optimization_core.

Provides reusable validation functions to ensure consistency
across all modules (inference, data, training, etc.).
"""
import logging
from typing import List, Optional, Union, Set, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass


def validate_non_empty_string(
    value: str,
    name: str
) -> None:
    """
    Validate non-empty string.
    
    Args:
        value: Value to validate
        name: Name of parameter (for error messages)
    
    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value)}")
    
    if not value.strip():
        raise ValidationError(f"{name} cannot be empty or whitespace")


def validate_path(
    path: Union[str, Path],
    name: str = "path",
    must_exist: bool = False,
    allowed_extensions: Optional[List[str]] = None,
    must_be_file: bool = False,
    must_be_dir: bool = False
) -> Path:
    """
    Validate file or directory path.
    
    Args:
        path: Path to validate
        name: Name of parameter (for error messages)
        must_exist: Whether path must exist
        allowed_extensions: List of allowed file extensions (e.g., ['.parquet', '.csv'])
        must_be_file: Whether path must be a file
        must_be_dir: Whether path must be a directory
    
    Returns:
        Path object
    
    Raises:
        ValidationError: If path is invalid
        FileNotFoundError: If path doesn't exist and must_exist=True
    """
    if not path:
        raise ValidationError(f"{name} cannot be empty")
    
    if not isinstance(path, (str, Path)):
        raise ValidationError(f"{name} must be str or Path, got {type(path)}")
    
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    
    if must_exist:
        if must_be_file and not path_obj.is_file():
            raise ValidationError(f"{name} must be a file: {path}")
        if must_be_dir and not path_obj.is_dir():
            raise ValidationError(f"{name} must be a directory: {path}")
    
    if allowed_extensions:
        suffix = path_obj.suffix.lower()
        if suffix not in allowed_extensions:
            raise ValidationError(
                f"{name} extension {suffix} not allowed. "
                f"Allowed: {allowed_extensions}"
            )
    
    return path_obj


def validate_model_path(
    model_path: Union[str, Path],
    must_exist: bool = False
) -> Path:
    """
    Validate model path.
    
    Args:
        model_path: Path to model
        must_exist: Whether path must exist
    
    Returns:
        Path object
    
    Raises:
        ValidationError: If path is invalid
        FileNotFoundError: If path doesn't exist and must_exist=True
    """
    return validate_path(
        model_path,
        name="model_path",
        must_exist=must_exist,
        must_be_file=False
    )


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Validate file path.
    
    Args:
        file_path: Path to file
        must_exist: Whether file must exist
        allowed_extensions: List of allowed file extensions
    
    Returns:
        Path object
    
    Raises:
        ValidationError: If path is invalid
        FileNotFoundError: If file doesn't exist and must_exist=True
    """
    return validate_path(
        file_path,
        name="file_path",
        must_exist=must_exist,
        allowed_extensions=allowed_extensions,
        must_be_file=True
    )


def validate_positive_number(
    value: Union[int, float],
    name: str,
    min_value: Union[int, float] = 0,
    max_value: Optional[Union[int, float]] = None
) -> None:
    """
    Validate positive number.
    
    Args:
        value: Value to validate
        name: Name of parameter
        min_value: Minimum allowed value
        max_value: Maximum allowed value (optional)
    
    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    
    if value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")


def validate_positive_int(
    value: int,
    name: str,
    min_value: int = 1,
    max_value: Optional[int] = None
) -> None:
    """
    Validate positive integer.
    
    Args:
        value: Value to validate
        name: Name of parameter (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value (optional)
    
    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value)}")
    
    validate_positive_number(value, name, min_value, max_value)


def validate_float_range(
    value: float,
    name: str,
    min_value: float,
    max_value: float,
    inclusive_min: bool = True,
    inclusive_max: bool = True
) -> None:
    """
    Validate float in range with optional inclusive/exclusive bounds.
    
    Args:
        value: Value to validate (can be int or float)
        name: Name of parameter (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        inclusive_min: Whether min bound is inclusive (default: True)
        inclusive_max: Whether max bound is inclusive (default: True)
    
    Raises:
        ValidationError: If value is invalid or out of range
    
    Examples:
        >>> validate_float_range(5.0, "temperature", 0.0, 10.0)  # Valid
        >>> validate_float_range(0.0, "ratio", 0.0, 1.0, inclusive_min=False)  # Invalid
    """
    # Validate input type
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    
    # Validate range bounds
    if min_value > max_value:
        raise ValidationError(
            f"Invalid range: min_value ({min_value}) must be <= max_value ({max_value})"
        )
    
    # Check if value is within range
    min_check = value >= min_value if inclusive_min else value > min_value
    max_check = value <= max_value if inclusive_max else value < max_value
    
    if not (min_check and max_check):
        min_op = ">=" if inclusive_min else ">"
        max_op = "<=" if inclusive_max else "<"
        raise ValidationError(
            f"{name} must be {min_op} {min_value} and {max_op} {max_value}, "
            f"got {value}"
        )


def validate_generation_params(
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    min_tokens: int = 1,
    max_tokens_limit: Optional[int] = None
) -> None:
    """
    Validate generation parameters.
    
    Args:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (optional)
        repetition_penalty: Repetition penalty (optional)
        min_tokens: Minimum allowed tokens
        max_tokens_limit: Maximum allowed tokens (optional)
    
    Raises:
        ValidationError: If any parameter is invalid
    """
    validate_positive_int(
        max_tokens,
        "max_tokens",
        min_value=min_tokens,
        max_value=max_tokens_limit
    )
    
    if temperature <= 0:
        raise ValidationError(f"temperature must be > 0, got {temperature}")
    
    validate_float_range(top_p, "top_p", 0.0, 1.0)
    
    if top_k is not None and top_k < -1:
        raise ValidationError(f"top_k must be >= -1, got {top_k}")
    
    if repetition_penalty is not None and repetition_penalty < 1.0:
        raise ValidationError(
            f"repetition_penalty must be >= 1.0, got {repetition_penalty}"
        )


def validate_sampling_params(
    temperature: float,
    top_p: float,
    top_k: Optional[int] = None
) -> None:
    """
    Validate sampling parameters.
    
    Args:
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (optional)
    
    Raises:
        ValidationError: If any parameter is invalid
    """
    if temperature <= 0:
        raise ValidationError(f"temperature must be > 0, got {temperature}")
    
    validate_float_range(top_p, "top_p", 0.0, 1.0)
    
    if top_k is not None and top_k < -1:
        raise ValidationError(f"top_k must be >= -1, got {top_k}")


def validate_batch_size(
    batch_size: int,
    max_batch_size: Optional[int] = None,
    min_batch_size: int = 1
) -> int:
    """
    Validate and normalize batch size with automatic truncation.
    
    Args:
        batch_size: Requested batch size
        max_batch_size: Maximum allowed batch size (optional)
        min_batch_size: Minimum allowed batch size (default: 1)
    
    Returns:
        Normalized batch size (clamped to valid range)
    
    Raises:
        ValidationError: If batch size is invalid
    
    Examples:
        >>> validate_batch_size(32, max_batch_size=64)  # Returns 32
        >>> validate_batch_size(100, max_batch_size=64)  # Returns 64 (truncated)
    """
    # Validate minimum batch size
    validate_positive_int(batch_size, "batch_size", min_value=min_batch_size)
    
    # Validate max_batch_size if provided
    if max_batch_size is not None:
        if max_batch_size < min_batch_size:
            raise ValidationError(
                f"max_batch_size ({max_batch_size}) must be >= min_batch_size ({min_batch_size})"
            )
        
        # Truncate if exceeds maximum
        if batch_size > max_batch_size:
            logger.warning(
                f"Batch size {batch_size} exceeds max_batch_size {max_batch_size}. "
                f"Truncating to {max_batch_size}."
            )
            return max_batch_size
    
    return batch_size


def validate_precision(
    precision: str,
    valid_precisions: Set[str]
) -> None:
    """
    Validate precision parameter.
    
    Args:
        precision: Precision value
        valid_precisions: Set of valid precision values
    
    Raises:
        ValidationError: If precision is invalid
    """
    if precision not in valid_precisions:
        raise ValidationError(
            f"precision must be one of {valid_precisions}, got {precision}"
        )


def validate_quantization(
    quantization: Optional[str],
    valid_quantizations: Set[str]
) -> None:
    """
    Validate quantization parameter.
    
    Args:
        quantization: Quantization value or None
        valid_quantizations: Set of valid quantization values
    
    Raises:
        ValidationError: If quantization is invalid
    """
    if quantization is not None and quantization not in valid_quantizations:
        raise ValidationError(
            f"quantization must be one of {valid_quantizations} or None, "
            f"got {quantization}"
        )


def validate_dataframe_schema(
    schema: Dict[str, Any],
    required_columns: List[str],
    column_name: str = "DataFrame"
) -> None:
    """
    Validate DataFrame schema has required columns.
    
    Args:
        schema: DataFrame schema (dict of column_name: dtype)
        required_columns: List of required column names
        column_name: Name for error messages
    
    Raises:
        ValidationError: If required columns are missing
    """
    if not required_columns:
        return
    
    missing_cols = [col for col in required_columns if col not in schema]
    if missing_cols:
        raise ValidationError(
            f"{column_name} missing required columns: {missing_cols}. "
            f"Available columns: {list(schema.keys())}"
        )


def validate_column_exists(
    schema: Dict[str, Any],
    column: str,
    column_name: str = "column"
) -> None:
    """
    Validate column exists in schema.
    
    Args:
        schema: DataFrame schema
        column: Column name to check
        column_name: Name for error messages
    
    Raises:
        ValidationError: If column doesn't exist
    """
    if column not in schema:
        raise ValidationError(
            f"{column_name} '{column}' not found in schema. "
            f"Available columns: {list(schema.keys())}"
        )



