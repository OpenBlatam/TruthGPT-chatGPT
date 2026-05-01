"""
File utilities for data processing.

This module re-exports common file utilities from modules.base.core_system.core.file_utils
for backward compatibility.
"""
from optimization_core.core.file_utils import (
    detect_file_format,
    validate_file_format,
    ensure_output_directory,
    get_file_size,
    get_file_size_mb,
    list_files,
    get_file_info,
    safe_remove,
    safe_rename,
    get_temp_path,
    SUPPORTED_FORMATS,
)

__all__ = [
    "detect_file_format",
    "validate_file_format",
    "ensure_output_directory",
    "get_file_size",
    "get_file_size_mb",
    "list_files",
    "get_file_info",
    "safe_remove",
    "safe_rename",
    "get_temp_path",
    "SUPPORTED_FORMATS",
]
    """
    Detect file format from extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        Format name (parquet, csv, jsonl, etc.)
    
    Raises:
        ValueError: If format cannot be detected
    """
    path_obj = Path(file_path)
    suffix = path_obj.suffix.lower()
    
    if suffix in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[suffix]
    
    raise ValueError(
        f"Unsupported file format: {suffix}. "
        f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
    )


def validate_file_format(
    file_path: Union[str, Path],
    allowed_formats: Optional[Set[str]] = None
) -> str:
    """
    Validate and return file format.
    
    Args:
        file_path: Path to file
        allowed_formats: Set of allowed format names (e.g., {'parquet', 'csv'})
    
    Returns:
        Format name
    
    Raises:
        ValueError: If format is invalid or not allowed
    """
    format_name = detect_file_format(file_path)
    
    if allowed_formats and format_name not in allowed_formats:
        raise ValueError(
            f"File format '{format_name}' not allowed. "
            f"Allowed formats: {allowed_formats}"
        )
    
    return format_name


def ensure_output_directory(output_path: Union[str, Path]) -> Path:
    """
    Ensure output directory exists, create if needed.
    
    Args:
        output_path: Path to output file
    
    Returns:
        Path object
    
    Raises:
        OSError: If directory cannot be created
    """
    path_obj = Path(output_path)
    directory = path_obj.parent
    
    if directory and not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise
    
    return path_obj


