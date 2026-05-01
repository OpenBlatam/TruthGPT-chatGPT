"""
Version management for polyglot_core.

Provides version information and compatibility checking.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class Version:
    """Version information."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """
        Parse version string.
        
        Args:
            version_str: Version string (e.g., "1.2.3", "1.2.3-alpha", "1.2.3+build")
            
        Returns:
            Version object
        """
        # Pattern: major.minor.patch[-prerelease][+build]
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([\w\.]+))?(?:\+([\w\.]+))?$'
        match = re.match(pattern, version_str)
        
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")
        
        major, minor, patch, prerelease, build = match.groups()
        
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=prerelease,
            build=build
        )
    
    def compare(self, other: "Version") -> int:
        """
        Compare versions.
        
        Args:
            other: Other version to compare
            
        Returns:
            -1 if self < other, 0 if equal, 1 if self > other
        """
        # Compare major, minor, patch
        if self.major != other.major:
            return 1 if self.major > other.major else -1
        if self.minor != other.minor:
            return 1 if self.minor > other.minor else -1
        if self.patch != other.patch:
            return 1 if self.patch > other.patch else -1
        
        # Compare prerelease (None is greater than any prerelease)
        if self.prerelease is None and other.prerelease is not None:
            return 1
        if self.prerelease is not None and other.prerelease is None:
            return -1
        if self.prerelease != other.prerelease:
            return 1 if self.prerelease > other.prerelease else -1
        
        return 0
    
    def __lt__(self, other: "Version") -> bool:
        return self.compare(other) < 0
    
    def __le__(self, other: "Version") -> bool:
        return self.compare(other) <= 0
    
    def __eq__(self, other: "Version") -> bool:
        return self.compare(other) == 0
    
    def __ge__(self, other: "Version") -> bool:
        return self.compare(other) >= 0
    
    def __gt__(self, other: "Version") -> bool:
        return self.compare(other) > 0


def get_version() -> str:
    """Get current polyglot_core version."""
    from . import __version__
    return __version__


def get_version_info() -> Dict[str, any]:
    """
    Get detailed version information.
    
    Returns:
        Dictionary with version information
    """
    version_str = get_version()
    version = Version.parse(version_str)
    
    return {
        'version': str(version),
        'major': version.major,
        'minor': version.minor,
        'patch': version.patch,
        'prerelease': version.prerelease,
        'build': version.build
    }


def check_compatibility(required_version: str, current_version: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check version compatibility.
    
    Args:
        required_version: Required version (e.g., ">=1.2.0")
        current_version: Current version (default: get_version())
        
    Returns:
        Tuple of (is_compatible, message)
    """
    if current_version is None:
        current_version = get_version()
    
    try:
        current = Version.parse(current_version)
        
        # Parse requirement (simple: >=, <=, ==, >, <)
        pattern = r'^(>=|<=|==|>|<)?\s*(\d+\.\d+\.\d+(?:-[\w\.]+)?(?:\+[\w\.]+)?)$'
        match = re.match(pattern, required_version.strip())
        
        if not match:
            return False, f"Invalid version requirement: {required_version}"
        
        op, version_str = match.groups()
        op = op or "=="
        required = Version.parse(version_str)
        
        if op == ">=":
            compatible = current >= required
        elif op == "<=":
            compatible = current <= required
        elif op == ">":
            compatible = current > required
        elif op == "<":
            compatible = current < required
        else:  # ==
            compatible = current == required
        
        if compatible:
            return True, f"Version {current_version} is compatible with {required_version}"
        else:
            return False, f"Version {current_version} is not compatible with {required_version}"
    
    except Exception as e:
        return False, f"Error checking compatibility: {e}"













