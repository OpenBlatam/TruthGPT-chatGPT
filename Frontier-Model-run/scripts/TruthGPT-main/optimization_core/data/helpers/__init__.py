"""
Data Processor Helper Modules
==============================

Helper functions for data processing operations.
"""

from .polars_helpers import (
    validate_polars_available,
    normalize_paths,
    validate_file_exists,
    detect_dataframe_type,
    ensure_lazy,
    ensure_eager,
    get_numeric_columns,
    log_dataframe_info,
)

__all__ = [
    "validate_polars_available",
    "normalize_paths",
    "validate_file_exists",
    "detect_dataframe_type",
    "ensure_lazy",
    "ensure_eager",
    "get_numeric_columns",
    "log_dataframe_info",
]





