"""
Security utilities for polyglot_core.

Provides security features like encryption, secrets management, and access control.
"""

from typing import Optional, Dict, Any
import hashlib
import hmac
import secrets
import base64


class SecurityManager:
    """
    Security manager for polyglot_core.
    
    Provides encryption, hashing, and secrets management.
    """
    
    @staticmethod
    def hash_data(data: bytes, algorithm: str = "sha256") -> str:
        """
        Hash data.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, sha512, md5)
            
        Returns:
            Hexadecimal hash string
        """
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Base64 encoded token
        """
        token = secrets.token_bytes(length)
        return base64.urlsafe_b64encode(token).decode('utf-8')
    
    @staticmethod
    def generate_secret_key() -> str:
        """Generate secret key for HMAC."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def hmac_sign(data: bytes, secret: str) -> str:
        """
        Sign data with HMAC.
        
        Args:
            data: Data to sign
            secret: Secret key
            
        Returns:
            HMAC signature
        """
        signature = hmac.new(
            secret.encode('utf-8'),
            data,
            hashlib.sha256
        ).hexdigest()
        return signature
    
    @staticmethod
    def verify_hmac(data: bytes, signature: str, secret: str) -> bool:
        """
        Verify HMAC signature.
        
        Args:
            data: Original data
            signature: HMAC signature
            secret: Secret key
            
        Returns:
            True if signature is valid
        """
        expected = SecurityManager.hmac_sign(data, secret)
        return hmac.compare_digest(expected, signature)
    
    @staticmethod
    def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
        """
        Mask sensitive data.
        
        Args:
            data: Data to mask
            visible_chars: Number of visible characters at start/end
            
        Returns:
            Masked string
        """
        if len(data) <= visible_chars * 2:
            return "*" * len(data)
        
        return data[:visible_chars] + "*" * (len(data) - visible_chars * 2) + data[-visible_chars:]


class SecretsManager:
    """
    Secrets manager for polyglot_core.
    
    Manages sensitive configuration and secrets.
    """
    
    def __init__(self):
        self._secrets: Dict[str, str] = {}
        self._encrypted: Dict[str, bool] = {}
    
    def set_secret(self, key: str, value: str, encrypted: bool = False):
        """
        Set secret value.
        
        Args:
            key: Secret key
            value: Secret value
            encrypted: Whether value is already encrypted
        """
        self._secrets[key] = value
        self._encrypted[key] = encrypted
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret value.
        
        Args:
            key: Secret key
            default: Default value if not found
            
        Returns:
            Secret value or default
        """
        return self._secrets.get(key, default)
    
    def has_secret(self, key: str) -> bool:
        """Check if secret exists."""
        return key in self._secrets
    
    def remove_secret(self, key: str):
        """Remove secret."""
        self._secrets.pop(key, None)
        self._encrypted.pop(key, None)
    
    def list_secrets(self) -> list:
        """List all secret keys (not values)."""
        return list(self._secrets.keys())


# Global managers
_global_security_manager = SecurityManager()
_global_secrets_manager = SecretsManager()


def get_security_manager() -> SecurityManager:
    """Get global security manager."""
    return _global_security_manager


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager."""
    return _global_secrets_manager













