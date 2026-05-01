"""
Polars Processor Helpers
========================

Helper functions for Polars data processing operations.
"""

import logging
from typing import List, Union, Optional, Any
from pathlib import Path

from optimization_core.core.file_utils import validate_file_path
from optimization_core.core.validators import validate_path

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

logger = logging.getLogger(__name__)


def validate_polars_available():
    """Check if Polars is available."""
    if not POLARS_AVAILABLE:
        raise ImportError(
            "Polars is not installed. Install with: pip install polars"
        )


def normalize_paths(paths: Union[str, Path, List[str], List[Path]]) -> List[Path]:
    """
    Normalize file paths to list of Path objects.
    
    Uses core.validators.validate_path for consistency.
    
    Args:
        paths: Single path or list of paths
    
    Returns:
        List of Path objects
    
    Raises:
        ValueError: If paths is empty
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    
    if not paths:
        raise ValueError("paths cannot be empty")
    
    return [validate_path(p) for p in paths]


def validate_file_exists(path: Path, extension: Optional[str] = None):
    """
    Validate that file exists and optionally has correct extension.
    
    Uses core.file_utils.validate_file_path for consistency.
    
    Args:
        path: Path to validate
        extension: Expected file extension (e.g., '.parquet')
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If extension doesn't match
    """
    validate_file_path(
        path,
        must_exist=True,
        allowed_extensions=[extension] if extension else None
    )


def detect_dataframe_type(df: Union[pl.DataFrame, pl.LazyFrame]) -> str:
    """
    Detect if DataFrame is eager or lazy.
    
    Args:
        df: DataFrame or LazyFrame
    
    Returns:
        "eager" or "lazy"
    """
    if isinstance(df, pl.LazyFrame):
        return "lazy"
    return "eager"


def ensure_lazy(df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.LazyFrame:
    """
    Convert DataFrame to LazyFrame if needed.
    
    Args:
        df: DataFrame or LazyFrame
    
    Returns:
        LazyFrame
    """
    if isinstance(df, pl.LazyFrame):
        return df
    return df.lazy()


def ensure_eager(df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
    """
    Convert LazyFrame to DataFrame if needed.
    
    Args:
        df: DataFrame or LazyFrame
    
    Returns:
        DataFrame
    """
    if isinstance(df, pl.DataFrame):
        return df
    return df.collect()


def get_numeric_columns(df: Union[pl.DataFrame, pl.LazyFrame]) -> List[str]:
    """
    Get list of numeric column names.
    
    Args:
        df: DataFrame or LazyFrame
    
    Returns:
        List of numeric column names
    """
    if isinstance(df, pl.LazyFrame):
        # For LazyFrame, we need to collect schema
        schema = df.schema
    else:
        schema = df.schema
    
    numeric_types = (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    )
    
    return [
        col for col, dtype in schema.items()
        if isinstance(dtype, numeric_types)
    ]


def log_dataframe_info(
    df: Union[pl.DataFrame, pl.LazyFrame],
    operation: str,
    logger_instance: Optional[logging.Logger] = None
):
    """
    Log DataFrame information.
    
    Args:
        df: DataFrame or LazyFrame
        operation: Operation name
        logger_instance: Logger instance (optional)
    """
    log = logger_instance or logger
    
    if isinstance(df, pl.LazyFrame):
        # For LazyFrame, we can only get schema
        log.debug(f"{operation}: LazyFrame with {len(df.schema)} columns")
    else:
        log.debug(
            f"{operation}: DataFrame shape={df.shape}, "
            f"columns={len(df.columns)}, memory={df.estimated_size()}"
        )


