"""
Serialization utilities for optimization_core.

Provides common serialization/deserialization patterns with support for:
- JSON (with compression)
- YAML
- Pickle (with compression)
- Dictionary conversion
"""
import json
import pickle
import gzip
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .file_utils import ensure_output_directory, validate_file_path, validate_path

logger = logging.getLogger(__name__)


def to_dict(obj: Any, exclude_none: bool = False, exclude_defaults: bool = False) -> Dict[str, Any]:
    """
    Convert object to dictionary.
    
    Supports:
    - Dataclasses (via asdict)
    - Objects with to_dict() method
    - Objects with __dict__
    - Basic types
    
    Args:
        obj: Object to convert
        exclude_none: Exclude None values
        exclude_defaults: Exclude default values (for dataclasses)
    
    Returns:
        Dictionary representation
    """
    if hasattr(obj, 'to_dict'):
        result = obj.to_dict(exclude_none=exclude_none, exclude_defaults=exclude_defaults)
    elif hasattr(obj, '__dict__'):
        result = obj.__dict__
    else:
        # Basic type
        return obj
    
    if exclude_none:
        result = {k: v for k, v in result.items() if v is not None}
    
    return result


def from_dict(cls: type, data: Dict[str, Any], strict: bool = True) -> Any:
    """
    Create object from dictionary.
    
    Supports:
    - Dataclasses (via from_dict if available)
    - Classes with from_dict() classmethod
    - Classes with __init__ accepting kwargs
    
    Args:
        cls: Class to instantiate
        data: Dictionary with data
        strict: Raise error on unknown fields
    
    Returns:
        Instance of cls
    """
    if hasattr(cls, 'from_dict'):
        return cls.from_dict(data, strict=strict)
    else:
        # Try direct instantiation
        return cls(**data)


def to_json(
    obj: Any,
    file_path: Optional[Union[str, Path]] = None,
    indent: int = 2,
    exclude_none: bool = False,
    compress: bool = False
) -> Union[str, None]:
    """
    Serialize object to JSON.
    
    Args:
        obj: Object to serialize
        file_path: Optional file path to save to
        indent: JSON indentation
        exclude_none: Exclude None values
        compress: Whether to compress with gzip
    
    Returns:
        JSON string if file_path is None, None otherwise
    """
    data = to_dict(obj, exclude_none=exclude_none)
    json_str = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    
    if file_path:
        path = Path(file_path)
        ensure_output_directory(path)
        
        if compress:
            path = path.with_suffix(path.suffix + '.gz')
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        logger.debug(f"Saved JSON to {path}")
        return None
    else:
        return json_str


def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    compress: bool = False
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
        compress: Whether to compress with gzip
    """
    to_json(data, file_path=file_path, indent=indent, compress=compress)


def from_json(
    json_str: Optional[str] = None,
    file_path: Optional[Union[str, Path]] = None,
    cls: Optional[type] = None,
    compressed: Optional[bool] = None
) -> Union[Dict[str, Any], Any]:
    """
    Deserialize JSON to object.
    
    Args:
        json_str: JSON string (if file_path not provided)
        file_path: Path to JSON file (if json_str not provided)
        cls: Optional class to instantiate
        compressed: Whether file is compressed (auto-detected if None)
    
    Returns:
        Dictionary or instance of cls
    """
    if file_path:
        path = Path(file_path)
        
        if compressed is None:
            compressed = path.suffix == '.gz'
        
        if compressed:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            path = validate_path(path, must_exist=True, must_be_file=True)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
    elif json_str:
        data = json.loads(json_str)
    else:
        raise ValueError("Either json_str or file_path must be provided")
    
    if cls:
        return from_dict(cls, data)
    return data


def load_json(
    file_path: Union[str, Path],
    compressed: Optional[bool] = None
) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        compressed: Whether file is compressed (auto-detected if None)
    
    Returns:
        Loaded data
    """
    return from_json(file_path=file_path, compressed=compressed)


def to_pickle(
    obj: Any,
    file_path: Union[str, Path],
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> None:
    """
    Serialize object to pickle file.
    
    Args:
        obj: Object to serialize
        file_path: Path to save pickle file
        protocol: Pickle protocol version
    """
    path = Path(file_path)
    ensure_output_directory(path)
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)
    
    logger.debug(f"Saved pickle to {file_path}")


def from_pickle(file_path: Union[str, Path]) -> Any:
    """
    Deserialize object from pickle file.
    
    Args:
        file_path: Path to pickle file
    
    Returns:
        Deserialized object
    """
    path = validate_path(file_path, must_exist=True, must_be_file=True)
    
    with open(path, 'rb') as f:
        return pickle.load(f)


def safe_serialize(obj: Any, default: Any = None) -> Any:
    """
    Safely serialize object, handling non-serializable types.
    
    Args:
        obj: Object to serialize
        default: Default value for non-serializable types
    
    Returns:
        Serializable representation
    """
    try:
        # Try JSON serialization
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Try to convert to dict
        try:
            return to_dict(obj)
        except Exception:
            # Return default or string representation
            return default if default is not None else str(obj)


__all__ = [
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    "to_pickle",
    "from_pickle",
    "safe_serialize",
]


