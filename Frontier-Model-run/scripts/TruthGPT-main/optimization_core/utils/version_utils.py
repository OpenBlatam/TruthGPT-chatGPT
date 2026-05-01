"""
Version utilities for optimization_core.

Provides utilities for version management and compatibility checking.
"""
import logging
from typing import Dict, Optional, Tuple
from packaging import version

logger = logging.getLogger(__name__)

# Current version
__version__ = "1.0.0"


def get_version() -> str:
    """Get current version."""
    return __version__


def parse_version(version_string: str) -> version.Version:
    """
    Parse version string.
    
    Args:
        version_string: Version string
    
    Returns:
        Version object
    """
    return version.parse(version_string)


def check_version_compatibility(
    required_version: str,
    current_version: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Check if current version is compatible with required version.
    
    Args:
        required_version: Required version (e.g., ">=1.0.0")
        current_version: Current version (defaults to module version)
    
    Returns:
        Tuple of (is_compatible, message)
    """
    if current_version is None:
        current_version = get_version()
    
    try:
        req_spec = version.SpecifierSet(required_version)
        is_compatible = req_spec.contains(current_version)
        
        if is_compatible:
            message = f"Version {current_version} is compatible with {required_version}"
        else:
            message = f"Version {current_version} is NOT compatible with {required_version}"
        
        return is_compatible, message
    except Exception as e:
        logger.error(f"Error checking version compatibility: {e}")
        return False, f"Error: {e}"


def get_version_info() -> Dict[str, str]:
    """
    Get version information.
    
    Returns:
        Dictionary with version info
    """
    return {
        "version": get_version(),
        "module": "optimization_core",
    }


def format_version(version_string: str) -> str:
    """
    Format version string consistently.
    
    Args:
        version_string: Version string
    
    Returns:
        Formatted version string
    """
    try:
        v = version.parse(version_string)
        return str(v)
    except Exception:
        return version_string













