"""
Configuration utilities for optimization_core.

Provides utilities for loading, saving, merging, and validating configuration files.
Consolidates functionality from utils/config_utils.py and related modules.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path

from .serialization import to_json, from_json
from .file_utils import detect_file_format, validate_path
from .validators import ValidationError

# Try to import YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING AND SAVING
# ════════════════════════════════════════════════════════════════════════════════

def load_config(
    config_path: Union[str, Path],
    format: Optional[str] = None,
    required: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Supports JSON and YAML formats with auto-detection.
    
    Args:
        config_path: Path to configuration file
        format: File format ('json' or 'yaml'). Auto-detected if None
        required: Whether file must exist (default: True)
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist and required=True
        ValueError: If format is unsupported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        if required:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        logger.warning(f"Configuration file not found: {config_path}, returning empty dict")
        return {}
    
    # Auto-detect format
    if format is None:
        format = detect_file_format(config_path)
        if format not in ('json', 'yaml'):
            raise ValueError(f"Unsupported format for {config_path}. Supported: json, yaml")
    
    # Load configuration
    try:
        if format == 'json':
            return from_json(file_path=config_path)
        elif format in ('yaml', 'yml'):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is not installed. Install with: pip install pyyaml")
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config(
    config: Dict[str, Any],
    config_path: Union[str, Path],
    format: Optional[str] = None,
    indent: int = 2
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        format: File format ('json' or 'yaml'). Auto-detected from extension if None
        indent: Indentation for JSON/YAML (default: 2)
    
    Raises:
        ValueError: If format is unsupported
    """
    config_path = Path(config_path)
    
    # Auto-detect format from extension
    if format is None:
        format = detect_file_format(config_path)
        if format not in ('json', 'yaml'):
            format = 'json'  # Default to JSON
    
    try:
        if format == 'json':
            to_json(config, file_path=config_path, indent=indent)
        elif format in ('yaml', 'yml'):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is not installed. Install with: pip install pyyaml")
            from .file_utils import ensure_output_directory
            ensure_output_directory(config_path)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG MERGING
# ════════════════════════════════════════════════════════════════════════════════

def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
    deep: bool = True
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        deep: Whether to perform deep merge (default: True)
    
    Returns:
        Merged configuration
    
    Example:
        >>> base = {'a': 1, 'b': {'c': 2}}
        >>> override = {'b': {'d': 3}}
        >>> merge_configs(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    if not deep:
        return {**base_config, **override_config}
    
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value, deep=True)
        else:
            result[key] = value
    
    return result


def merge_multiple_configs(
    *configs: Dict[str, Any],
    deep: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge (later ones override earlier ones)
        deep: Whether to perform deep merge (default: True)
    
    Returns:
        Merged configuration
    
    Example:
        >>> config1 = {'a': 1}
        >>> config2 = {'b': 2}
        >>> config3 = {'a': 3}
        >>> merge_multiple_configs(config1, config2, config3)
        {'a': 3, 'b': 2}
    """
    if not configs:
        return {}
    
    result = configs[0].copy()
    for config in configs[1:]:
        result = merge_configs(result, config, deep=deep)
    
    return result


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

def validate_config(
    config: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    schema: Optional[Dict[str, Type]] = None,
    allow_extra: bool = True
) -> None:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        schema: Optional schema mapping keys to expected types
        allow_extra: Whether to allow extra keys not in schema
    
    Raises:
        ValidationError: If configuration is invalid
    
    Example:
        >>> validate_config(
        ...     {'model': 'gpt', 'batch_size': 32},
        ...     required_keys=['model'],
        ...     schema={'batch_size': int}
        ... )
    """
    errors = []
    
    # Check required keys
    if required_keys:
        missing = [key for key in required_keys if key not in config]
        if missing:
            errors.append(f"Missing required keys: {missing}")
    
    # Validate schema
    if schema:
        for key, expected_type in schema.items():
            if key in config:
                if not isinstance(config[key], expected_type):
                    errors.append(
                        f"Key '{key}' must be of type {expected_type.__name__}, "
                        f"got {type(config[key]).__name__}"
                    )
            elif not allow_extra:
                # Key in schema but not in config - only error if not optional
                pass
    
    if errors:
        raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG VALUE ACCESS
# ════════════════════════════════════════════════════════════════════════════════

def get_config_value(
    config: Dict[str, Any],
    key: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """
    Get configuration value with optional default.
    
    Supports dot notation for nested keys (e.g., "model.name").
    
    Args:
        config: Configuration dictionary
        key: Key to get (supports dot notation, e.g., "model.name")
        default: Default value if key not found
        required: Whether key is required (raises error if not found)
    
    Returns:
        Configuration value
    
    Raises:
        ValueError: If key is required but not found
    
    Example:
        >>> config = {'model': {'name': 'gpt', 'size': 'large'}}
        >>> get_config_value(config, 'model.name')
        'gpt'
        >>> get_config_value(config, 'model.temperature', default=1.0)
        1.0
    """
    # Support dot notation
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            if required:
                raise ValueError(f"Required configuration key not found: {key}")
            return default
    
    return value


def set_config_value(
    config: Dict[str, Any],
    key: str,
    value: Any,
    create_nested: bool = True
) -> None:
    """
    Set configuration value.
    
    Supports dot notation for nested keys (e.g., "model.name").
    
    Args:
        config: Configuration dictionary (modified in place)
        key: Key to set (supports dot notation)
        value: Value to set
        create_nested: Whether to create nested dicts if they don't exist
    
    Example:
        >>> config = {}
        >>> set_config_value(config, 'model.name', 'gpt')
        >>> config
        {'model': {'name': 'gpt'}}
    """
    keys = key.split('.')
    current = config
    
    # Navigate/create nested structure
    for k in keys[:-1]:
        if k not in current:
            if create_nested:
                current[k] = {}
            else:
                raise KeyError(f"Key path '{key}' cannot be created (missing: {k})")
        elif not isinstance(current[k], dict):
            raise ValueError(f"Cannot set nested key '{key}' (conflict at '{k}')")
        
        current = current[k]
    
    # Set final value
    current[keys[-1]] = value


def has_config_key(
    config: Dict[str, Any],
    key: str
) -> bool:
    """
    Check if configuration has a key.
    
    Supports dot notation for nested keys.
    
    Args:
        config: Configuration dictionary
        key: Key to check (supports dot notation)
    
    Returns:
        True if key exists, False otherwise
    
    Example:
        >>> config = {'model': {'name': 'gpt'}}
        >>> has_config_key(config, 'model.name')
        True
        >>> has_config_key(config, 'model.size')
        False
    """
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return False
    
    return True


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG TRANSFORMATION
# ════════════════════════════════════════════════════════════════════════════════

def flatten_config(
    config: Dict[str, Any],
    separator: str = '.',
    prefix: str = ''
) -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary.
    
    Args:
        config: Nested configuration dictionary
        separator: Separator for nested keys (default: '.')
        prefix: Prefix for keys (used internally for recursion)
    
    Returns:
        Flattened configuration dictionary
    
    Example:
        >>> config = {'model': {'name': 'gpt', 'size': 'large'}}
        >>> flatten_config(config)
        {'model.name': 'gpt', 'model.size': 'large'}
    """
    result = {}
    
    for key, value in config.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_config(value, separator=separator, prefix=full_key))
        else:
            result[full_key] = value
    
    return result


def unflatten_config(
    flat_config: Dict[str, Any],
    separator: str = '.'
) -> Dict[str, Any]:
    """
    Unflatten configuration dictionary with dot notation.
    
    Args:
        flat_config: Flat configuration dictionary
        separator: Separator for nested keys (default: '.')
    
    Returns:
        Nested configuration dictionary
    
    Example:
        >>> flat = {'model.name': 'gpt', 'model.size': 'large'}
        >>> unflatten_config(flat)
        {'model': {'name': 'gpt', 'size': 'large'}}
    """
    result = {}
    
    for key, value in flat_config.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Loading and saving
    'load_config',
    'save_config',
    # Merging
    'merge_configs',
    'merge_multiple_configs',
    # Validation
    'validate_config',
    # Value access
    'get_config_value',
    'set_config_value',
    'has_config_key',
    # Transformation
    'flatten_config',
    'unflatten_config',
]


