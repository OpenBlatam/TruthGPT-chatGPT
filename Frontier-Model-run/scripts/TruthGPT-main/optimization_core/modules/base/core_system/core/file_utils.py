"""
Common file utilities for optimization_core.

Provides reusable file handling functions shared across all modules.
"""
import logging
from typing import Optional, Set, Union, List
from pathlib import Path
import os

from .validators import validate_path

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    '.parquet': 'parquet',
    '.csv': 'csv',
    '.jsonl': 'jsonl',
    '.json': 'json',
    '.tsv': 'tsv',
    '.txt': 'txt',
    '.pkl': 'pkl',
    '.pickle': 'pickle',
    '.h5': 'h5',
    '.hdf5': 'hdf5',
    '.npz': 'npz',
    '.npy': 'npy',
}


def detect_file_format(file_path: Union[str, Path]) -> str:
    """
    Detect file format from extension.
    
    Args:
        file_path: Path to file (must be str or Path)
    
    Returns:
        Format name (parquet, csv, jsonl, etc.)
    
    Raises:
        TypeError: If file_path is not str or Path
        ValueError: If format cannot be detected or file_path is empty
    
    Examples:
        >>> detect_file_format("data.csv")
        'csv'
        >>> detect_file_format(Path("data.parquet"))
        'parquet'
    """
    # Validate input
    if not isinstance(file_path, (str, Path)):
        raise TypeError(f"file_path must be str or Path, got {type(file_path).__name__}")
    if not file_path:
        raise ValueError("file_path cannot be empty")
    
    path_obj = Path(file_path)
    suffix = path_obj.suffix.lower()
    
    # Check if suffix is in supported formats
    if suffix in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[suffix]
    
    # Raise error with helpful message
    raise ValueError(
        f"Unsupported file format: {suffix}. "
        f"Supported formats: {sorted(SUPPORTED_FORMATS.keys())}"
    )


def validate_file_format(
    file_path: Union[str, Path],
    allowed_formats: Optional[Set[str]] = None
) -> str:
    """
    Validate file format and return format name.
    
    Args:
        file_path: Path to file
        allowed_formats: Set of allowed format extensions (e.g., {'.parquet', '.csv'})
    
    Returns:
        Format name
    
    Raises:
        ValueError: If format is not allowed
    """
    format_name = detect_file_format(file_path)
    
    if allowed_formats:
        path_obj = Path(file_path)
        suffix = path_obj.suffix.lower()
        if suffix not in allowed_formats:
            raise ValueError(
                f"File format {suffix} not allowed. "
                f"Allowed formats: {allowed_formats}"
            )
    
    return format_name


def ensure_output_directory(file_path: Union[str, Path]) -> Path:
    """
    Ensure output directory exists, creating it if necessary.
    
    Args:
        file_path: Path to output file
    
    Returns:
        Path object to output directory
    """
    path_obj = Path(file_path)
    output_dir = path_obj.parent
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")
    
    return output_dir


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in bytes
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = validate_path(file_path, must_exist=True, must_be_file=True)
    return path_obj.stat().st_size


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in MB
    """
    return get_file_size(file_path) / (1024 * 1024)


def list_files(
    directory: Union[str, Path],
    pattern: Optional[str] = None,
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory.
    
    Args:
        directory: Directory path
        pattern: Glob pattern (e.g., "*.parquet")
        recursive: Search recursively
    
    Returns:
        List of file paths
    """
    dir_path = validate_path(directory, must_exist=True, must_be_dir=True)
    
    if pattern:
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
    else:
        if recursive:
            files = [f for f in dir_path.rglob("*") if f.is_file()]
        else:
            files = [f for f in dir_path.iterdir() if f.is_file()]
    
    return sorted(files)


def get_file_info(file_path: Union[str, Path]) -> dict:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
    
    Returns:
        Dictionary with file info
    """
    path_obj = validate_path(file_path, must_exist=True, must_be_file=True)
    stat = path_obj.stat()
    
    return {
        "path": str(path_obj),
        "name": path_obj.name,
        "stem": path_obj.stem,
        "suffix": path_obj.suffix,
        "format": detect_file_format(path_obj),
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "exists": True,
        "is_file": path_obj.is_file(),
        "is_dir": path_obj.is_dir(),
    }


def safe_remove(file_path: Union[str, Path], missing_ok: bool = True) -> bool:
    """
    Safely remove a file.
    
    Args:
        file_path: Path to file
        missing_ok: Don't raise error if file doesn't exist
    
    Returns:
        True if removed, False otherwise
    """
    try:
        path_obj = Path(file_path)
        if path_obj.exists() and path_obj.is_file():
            path_obj.unlink()
            logger.debug(f"Removed file: {file_path}")
            return True
        elif not missing_ok:
            raise FileNotFoundError(f"File not found: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Failed to remove file {file_path}: {e}")
        if not missing_ok:
            raise
        return False


def safe_rename(
    source: Union[str, Path],
    target: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """
    Safely rename/move a file.
    
    Args:
        source: Source file path
        target: Target file path
        overwrite: Overwrite target if it exists
    
    Returns:
        True if renamed successfully
    """
    try:
        source_path = validate_path(source, must_exist=True, must_be_file=True)
        target_path = Path(target)
        
        if target_path.exists():
            if overwrite:
                target_path.unlink()
            else:
                raise FileExistsError(f"Target file exists: {target}")
        
        ensure_output_directory(target_path)
        source_path.rename(target_path)
        logger.debug(f"Renamed {source} to {target}")
        return True
    except Exception as e:
        logger.error(f"Failed to rename {source} to {target}: {e}")
        raise


def get_temp_path(
    prefix: str = "tmp",
    suffix: str = ".tmp",
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    Get a temporary file path.
    
    Args:
        prefix: Filename prefix
        suffix: Filename suffix
        directory: Directory for temp file (uses system temp if None)
    
    Returns:
        Path to temporary file (file is not created)
    """
    import tempfile
    
    if directory:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
    else:
        dir_path = Path(tempfile.gettempdir())
    
    import uuid
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    return dir_path / filename













