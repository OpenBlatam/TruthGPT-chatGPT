"""
Configuration utilities for the entire optimization_core module.

Provides utilities for loading, validating, and managing configuration.

This module re-exports functionality from modules.base.core_system.core.config_utils for backward compatibility.
New code should import directly from optimization_core.core.config_utils.
"""

# Re-export from core module for backward compatibility
try:
    from optimization_core.core.config_utils import (
        load_config,
        save_config,
        merge_configs,
        merge_multiple_configs,
        validate_config,
        get_config_value,
        set_config_value,
        has_config_key,
        flatten_config,
        unflatten_config,
    )
except ImportError:
    # Fallback implementation if core module not available
    import logging
    import json
    import yaml
    from typing import Dict, Any, Optional, Union
    from pathlib import Path
    
    logger = logging.getLogger(__name__)


def load_config(
    config_path: Union[str, Path],
    format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Supports JSON and YAML formats.
    
    Args:
        config_path: Path to configuration file
        format: File format ('json' or 'yaml'). Auto-detected if None
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If format is unsupported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Auto-detect format
    if format is None:
        suffix = config_path.suffix.lower()
        if suffix in ('.json',):
            format = 'json'
        elif suffix in ('.yaml', '.yml'):
            format = 'yaml'
        else:
            raise ValueError(f"Unable to detect format for {config_path}")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            if format == 'json':
                return json.load(f)
            elif format == 'yaml':
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config(
    config: Dict[str, Any],
    config_path: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        format: File format ('json' or 'yaml')
    
    Raises:
        ValueError: If format is unsupported
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            if format == 'json':
                json.dump(config, f, indent=2)
            elif format == 'yaml':
                yaml.dump(config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


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
        deep: Whether to perform deep merge
    
    Returns:
        Merged configuration
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


def validate_config(
    config: Dict[str, Any],
    required_keys: Optional[list] = None,
    schema: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        schema: Optional schema for validation
    
    Raises:
        ValueError: If configuration is invalid
    """
    if required_keys:
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(
                f"Configuration missing required keys: {missing}"
            )
    
    if schema:
        # Basic schema validation
        for key, expected_type in schema.items():
            if key in config:
                if not isinstance(config[key], expected_type):
                    raise ValueError(
                        f"Configuration key '{key}' must be of type "
                        f"{expected_type.__name__}, got {type(config[key]).__name__}"
                    )


def get_config_value(
    config: Dict[str, Any],
    key: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """
    Get configuration value with optional default.
    
    Args:
        config: Configuration dictionary
        key: Key to get (supports dot notation, e.g., "model.name")
        default: Default value if key not found
        required: Whether key is required
    
    Returns:
        Configuration value
    
    Raises:
        ValueError: If key is required but not found
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
    """Set configuration value with dot notation support."""
    keys = key.split('.')
    current = config
    
    for k in keys[:-1]:
        if k not in current:
            if create_nested:
                current[k] = {}
            else:
                raise KeyError(f"Key path '{key}' cannot be created (missing: {k})")
        elif not isinstance(current[k], dict):
            raise ValueError(f"Cannot set nested key '{key}' (conflict at '{k}')")
        current = current[k]
    
    current[keys[-1]] = value


def has_config_key(
    config: Dict[str, Any],
    key: str
) -> bool:
    """Check if configuration has a key (supports dot notation)."""
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return False
    
    return True


def merge_multiple_configs(
    *configs: Dict[str, Any],
    deep: bool = True
) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    if not configs:
        return {}
    
    result = configs[0].copy()
    for config in configs[1:]:
        result = merge_configs(result, config, deep=deep)
    
    return result


def flatten_config(
    config: Dict[str, Any],
    separator: str = '.',
    prefix: str = ''
) -> Dict[str, Any]:
    """Flatten nested configuration dictionary."""
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
    """Unflatten configuration dictionary with dot notation."""
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


__all__ = [
    'load_config',
    'save_config',
    'merge_configs',
    'merge_multiple_configs',
    'validate_config',
    'get_config_value',
    'set_config_value',
    'has_config_key',
    'flatten_config',
    'unflatten_config',
]


