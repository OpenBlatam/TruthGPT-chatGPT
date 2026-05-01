"""
Serialization utilities for optimization_core.

Provides utilities for serializing and deserializing data, models, and configurations.
"""
import logging
import json
import pickle
from typing import Any, Dict, Optional, Union
from pathlib import Path
import gzip

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_str = json.dumps(data, indent=indent, default=str)
    
    if compress:
        file_path = file_path.with_suffix(file_path.suffix + '.gz')
        with gzip.open(file_path, 'wt') as f:
            f.write(json_str)
    else:
        with open(file_path, 'w') as f:
            f.write(json_str)
    
    logger.debug(f"Saved JSON to {file_path}")


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
    file_path = Path(file_path)
    
    if compressed is None:
        compressed = file_path.suffix == '.gz'
    
    if compressed:
        with gzip.open(file_path, 'rt') as f:
            return json.load(f)
    else:
        with open(file_path, 'r') as f:
            return json.load(f)


def save_yaml(
    data: Any,
    file_path: Union[str, Path]
) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is not installed. Install with: pip install pyyaml")
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    logger.debug(f"Saved YAML to {file_path}")


def load_yaml(
    file_path: Union[str, Path]
) -> Any:
    """
    Load data from YAML file.
    
    Args:
        file_path: Path to YAML file
    
    Returns:
        Loaded data
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is not installed. Install with: pip install pyyaml")
    
    file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_pickle(
    data: Any,
    file_path: Union[str, Path],
    compress: bool = False
) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        compress: Whether to compress with gzip
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        file_path = file_path.with_suffix(file_path.suffix + '.gz')
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    logger.debug(f"Saved pickle to {file_path}")


def load_pickle(
    file_path: Union[str, Path],
    compressed: Optional[bool] = None
) -> Any:
    """
    Load data from pickle file.
    
    Args:
        file_path: Path to pickle file
        compressed: Whether file is compressed (auto-detected if None)
    
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if compressed is None:
        compressed = file_path.suffix == '.gz'
    
    if compressed:
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def serialize_to_dict(
    obj: Any,
    include_private: bool = False
) -> Dict[str, Any]:
    """
    Serialize object to dictionary.
    
    Uses core.serialization.to_dict for consistency.
    
    Args:
        obj: Object to serialize
        include_private: Whether to include private attributes
    
    Returns:
        Dictionary representation
    """
    from optimization_core.core.serialization import to_dict
    result = to_dict(obj, exclude_none=False)
    
    if not include_private:
        result = {k: v for k, v in result.items() if not k.startswith('_')}
    
    return result


