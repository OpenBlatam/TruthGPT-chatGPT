"""
Common encoding and hashing utilities for optimization_core.

Provides reusable functions for encoding, decoding, and hashing.
"""

import base64
import hashlib
import hmac
import secrets
from typing import Optional, Union

from .types import OptionalStr


# ════════════════════════════════════════════════════════════════════════════════
# BASE64 ENCODING/DECODING
# ════════════════════════════════════════════════════════════════════════════════

def encode_base64(data: Union[str, bytes]) -> str:
    """
    Encode data to base64 string.
    
    Args:
        data: Data to encode (string or bytes)
    
    Returns:
        Base64 encoded string
    
    Example:
        >>> encode_base64("hello")
        'aGVsbG8='
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('ascii')


def decode_base64(encoded: str) -> bytes:
    """
    Decode base64 string to bytes.
    
    Args:
        encoded: Base64 encoded string
    
    Returns:
        Decoded bytes
    
    Example:
        >>> decode_base64("aGVsbG8=")
        b'hello'
    """
    return base64.b64decode(encoded)


def encode_base64_urlsafe(data: Union[str, bytes]) -> str:
    """
    Encode data to URL-safe base64 string.
    
    Args:
        data: Data to encode (string or bytes)
    
    Returns:
        URL-safe base64 encoded string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.urlsafe_b64encode(data).decode('ascii')


def decode_base64_urlsafe(encoded: str) -> bytes:
    """
    Decode URL-safe base64 string to bytes.
    
    Args:
        encoded: URL-safe base64 encoded string
    
    Returns:
        Decoded bytes
    """
    return base64.urlsafe_b64decode(encoded)


# ════════════════════════════════════════════════════════════════════════════════
# HEX ENCODING/DECODING
# ════════════════════════════════════════════════════════════════════════════════

def encode_hex(data: Union[str, bytes]) -> str:
    """
    Encode data to hexadecimal string.
    
    Args:
        data: Data to encode (string or bytes)
    
    Returns:
        Hexadecimal string
    
    Example:
        >>> encode_hex("hello")
        '68656c6c6f'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return data.hex()


def decode_hex(encoded: str) -> bytes:
    """
    Decode hexadecimal string to bytes.
    
    Args:
        encoded: Hexadecimal string
    
    Returns:
        Decoded bytes
    
    Example:
        >>> decode_hex("68656c6c6f")
        b'hello'
    """
    return bytes.fromhex(encoded)


# ════════════════════════════════════════════════════════════════════════════════
# HASHING
# ════════════════════════════════════════════════════════════════════════════════

def hash_data(
    data: Union[str, bytes],
    algorithm: str = "sha256"
) -> str:
    """
    Hash data using specified algorithm.
    
    Args:
        data: Data to hash (string or bytes)
        algorithm: Hash algorithm (md5, sha1, sha224, sha256, sha384, sha512, blake2b)
    
    Returns:
        Hexadecimal hash string
    
    Example:
        >>> hash_data("hello", "sha256")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data)
    return hash_obj.hexdigest()


def hash_file(
    file_path: Union[str, bytes],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    Hash file using specified algorithm.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm
        chunk_size: Chunk size for reading file
    
    Returns:
        Hexadecimal hash string
    
    Example:
        >>> file_hash = hash_file("data.txt", "sha256")
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


# ════════════════════════════════════════════════════════════════════════════════
# HMAC
# ════════════════════════════════════════════════════════════════════════════════

def create_hmac(
    data: Union[str, bytes],
    key: Union[str, bytes],
    algorithm: str = "sha256"
) -> str:
    """
    Create HMAC for data verification.
    
    Args:
        data: Data to create HMAC for
        key: Secret key
        algorithm: Hash algorithm for HMAC
    
    Returns:
        Hexadecimal HMAC string
    
    Example:
        >>> hmac_value = create_hmac("message", "secret_key", "sha256")
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    if isinstance(key, str):
        key = key.encode('utf-8')
    
    return hmac.new(key, data, hashlib.sha256 if algorithm == "sha256" else hashlib.sha512).hexdigest()


def verify_hmac(
    data: Union[str, bytes],
    key: Union[str, bytes],
    expected_hmac: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Verify HMAC for data integrity.
    
    Args:
        data: Data to verify
        key: Secret key
        expected_hmac: Expected HMAC value
        algorithm: Hash algorithm for HMAC
    
    Returns:
        True if HMAC matches
    
    Example:
        >>> is_valid = verify_hmac("message", "secret_key", hmac_value)
    """
    calculated_hmac = create_hmac(data, key, algorithm)
    return hmac.compare_digest(calculated_hmac, expected_hmac)


# ════════════════════════════════════════════════════════════════════════════════
# RANDOM GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def generate_random_bytes(length: int = 32) -> bytes:
    """
    Generate cryptographically secure random bytes.
    
    Args:
        length: Length of random bytes
    
    Returns:
        Random bytes
    
    Example:
        >>> random_bytes = generate_random_bytes(32)
    """
    return secrets.token_bytes(length)


def generate_random_hex(length: int = 32) -> str:
    """
    Generate cryptographically secure random hex string.
    
    Args:
        length: Length of hex string (bytes, so hex will be 2x length)
    
    Returns:
        Random hex string
    
    Example:
        >>> random_hex = generate_random_hex(16)  # 32 hex chars
    """
    return secrets.token_hex(length)


def generate_random_urlsafe(length: int = 32) -> str:
    """
    Generate cryptographically secure random URL-safe string.
    
    Args:
        length: Length of string
    
    Returns:
        Random URL-safe string
    
    Example:
        >>> random_token = generate_random_urlsafe(32)
    """
    return secrets.token_urlsafe(length)


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Base64
    "encode_base64",
    "decode_base64",
    "encode_base64_urlsafe",
    "decode_base64_urlsafe",
    # Hex
    "encode_hex",
    "decode_hex",
    # Hashing
    "hash_data",
    "hash_file",
    # HMAC
    "create_hmac",
    "verify_hmac",
    # Random
    "generate_random_bytes",
    "generate_random_hex",
    "generate_random_urlsafe",
]













