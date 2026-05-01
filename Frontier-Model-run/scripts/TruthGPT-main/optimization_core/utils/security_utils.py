"""
Security utilities for optimization_core.

Provides utilities for security and validation.
"""
import logging
import hashlib
import secrets
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def hash_string(
    text: str,
    algorithm: str = "sha256"
) -> str:
    """
    Hash a string.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (sha256, sha512, md5)
    
    Returns:
        Hashed string
    """
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    return hash_func(text.encode()).hexdigest()


def generate_token(
    length: int = 32
) -> str:
    """
    Generate a secure random token.
    
    Args:
        length: Token length
    
    Returns:
        Random token
    """
    return secrets.token_urlsafe(length)


def validate_file_hash(
    file_path: Path,
    expected_hash: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Validate file hash.
    
    Args:
        file_path: Path to file
        expected_hash: Expected hash
        algorithm: Hash algorithm
    
    Returns:
        True if hash matches
    """
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    
    with open(file_path, 'rb') as f:
        file_hash = hash_func(f.read()).hexdigest()
    
    return file_hash == expected_hash


def sanitize_path(
    path: str,
    base_path: Optional[Path] = None
) -> Path:
    """
    Sanitize file path to prevent directory traversal.
    
    Args:
        path: Path to sanitize
        base_path: Base path to resolve against
    
    Returns:
        Sanitized Path
    """
    path_obj = Path(path)
    
    # Resolve against base path if provided
    if base_path:
        resolved = (base_path / path_obj).resolve()
        # Ensure it's still within base path
        if not str(resolved).startswith(str(base_path.resolve())):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved
    
    return path_obj.resolve()


class SecureConfig:
    """Secure configuration manager."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize secure config.
        
        Args:
            secret_key: Secret key for encryption (generates if None)
        """
        self.secret_key = secret_key or generate_token()
        self._config: Dict[str, Any] = {}
    
    def set(
        self,
        key: str,
        value: Any,
        encrypt: bool = False
    ):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            encrypt: Whether to encrypt value
        """
        if encrypt:
            # Simple encryption (in practice, use proper encryption)
            value = hash_string(f"{value}{self.secret_key}")
        
        self._config[key] = value
    
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
        
        Returns:
            Configuration value
        """
        return self._config.get(key, default)













