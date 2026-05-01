"""
Common environment variable utilities for optimization_core.

Provides reusable functions for loading and parsing environment variables.
"""

import os
import logging
from typing import Any, Optional, Union, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    logger.debug("python-dotenv not available, using basic env loading")


# ════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLE GETTERS
# ════════════════════════════════════════════════════════════════════════════════

def get_env(
    key: str,
    default: Optional[Any] = None,
    required: bool = False,
    var_type: type = str
) -> Optional[Any]:
    """
    Get environment variable with type conversion.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: Raise error if not found
        var_type: Type to convert to (str, int, float, bool)
    
    Returns:
        Environment variable value (converted to var_type)
    
    Raises:
        ValueError: If required and not found
    
    Example:
        >>> port = get_env("PORT", default=8000, var_type=int)
        >>> debug = get_env("DEBUG", default=False, var_type=bool)
    """
    value = os.getenv(key, default)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' not set")
        return default
    
    return _convert_type(value, var_type)


def get_env_str(
    key: str,
    default: Optional[str] = None,
    required: bool = False
) -> Optional[str]:
    """
    Get string environment variable.
    
    Args:
        key: Environment variable name
        default: Default value
        required: Raise error if not found
    
    Returns:
        String value
    """
    return get_env(key, default, required, str)


def get_env_int(
    key: str,
    default: Optional[int] = None,
    required: bool = False
) -> Optional[int]:
    """
    Get integer environment variable.
    
    Args:
        key: Environment variable name
        default: Default value
        required: Raise error if not found
    
    Returns:
        Integer value
    
    Raises:
        ValueError: If value cannot be converted to int
    """
    return get_env(key, default, required, int)


def get_env_float(
    key: str,
    default: Optional[float] = None,
    required: bool = False
) -> Optional[float]:
    """
    Get float environment variable.
    
    Args:
        key: Environment variable name
        default: Default value
        required: Raise error if not found
    
    Returns:
        Float value
    
    Raises:
        ValueError: If value cannot be converted to float
    """
    return get_env(key, default, required, float)


def get_env_bool(
    key: str,
    default: bool = False,
    required: bool = False
) -> bool:
    """
    Get boolean environment variable.
    
    Supports: true, false, 1, 0, yes, no, on, off (case insensitive)
    
    Args:
        key: Environment variable name
        default: Default value
        required: Raise error if not found
    
    Returns:
        Boolean value
    """
    value = os.getenv(key)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' not set")
        return default
    
    return _parse_bool(value)


def get_env_list(
    key: str,
    default: Optional[list] = None,
    separator: str = ",",
    required: bool = False
) -> list:
    """
    Get list environment variable (comma-separated by default).
    
    Args:
        key: Environment variable name
        default: Default value
        separator: Separator character
        required: Raise error if not found
    
    Returns:
        List of strings
    """
    value = os.getenv(key)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' not set")
        return default or []
    
    return [item.strip() for item in value.split(separator) if item.strip()]


# ════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT FILE LOADING
# ════════════════════════════════════════════════════════════════════════════════

def load_env_file(env_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file (defaults to .env in current directory)
    
    Returns:
        Dictionary of environment variables loaded
    
    Example:
        >>> load_env_file(".env")
        {'DATABASE_URL': 'postgresql://...', 'API_KEY': '...'}
    """
    if env_path is None:
        env_path = Path(".env")
    else:
        env_path = Path(env_path)
    
    if not env_path.exists():
        logger.debug(f".env file not found at {env_path}")
        return {}
    
    if HAS_DOTENV:
        # Use python-dotenv if available
        load_dotenv(env_path, override=False)
        return {}
    else:
        # Basic .env file parsing
        env_vars = {}
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        env_vars[key] = value
                        os.environ[key] = value  # Set in environment
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")
        
        return env_vars


# ════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT DETECTION
# ════════════════════════════════════════════════════════════════════════════════

def get_environment() -> str:
    """
    Get current environment (development, staging, production, testing).
    
    Returns:
        Environment name (lowercase)
    
    Example:
        >>> env = get_environment()
        'development'
    """
    env_var = os.getenv("ENVIRONMENT", "").lower()
    
    if env_var in ["prod", "production"]:
        return "production"
    elif env_var in ["stage", "staging"]:
        return "staging"
    elif env_var in ["test", "testing"]:
        return "testing"
    else:
        return "development"


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment() == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return get_environment() == "development"


def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_environment() == "testing"


# ════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def _convert_type(value: Any, var_type: type) -> Any:
    """
    Convert value to specified type.
    
    Args:
        value: Value to convert
        var_type: Target type
    
    Returns:
        Converted value
    
    Raises:
        ValueError: If conversion fails
    """
    if var_type == bool:
        return _parse_bool(value)
    elif var_type == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert '{value}' to int")
    elif var_type == float:
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert '{value}' to float")
    elif var_type == str:
        return str(value)
    else:
        return var_type(value)


def _parse_bool(value: Union[str, bool]) -> bool:
    """
    Parse boolean value from string.
    
    Supports: true, false, 1, 0, yes, no, on, off (case insensitive)
    
    Args:
        value: Value to parse
    
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    
    value_str = str(value).lower().strip()
    
    if value_str in ("true", "1", "yes", "on", "enabled"):
        return True
    elif value_str in ("false", "0", "no", "off", "disabled"):
        return False
    else:
        raise ValueError(f"Cannot convert '{value}' to bool")


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Getters
    "get_env",
    "get_env_str",
    "get_env_int",
    "get_env_float",
    "get_env_bool",
    "get_env_list",
    # File loading
    "load_env_file",
    # Environment detection
    "get_environment",
    "is_production",
    "is_development",
    "is_testing",
]













