"""
Common URL utilities for optimization_core.

Provides reusable functions for URL manipulation and parsing.
"""

import logging
from typing import Dict, Optional, Union
from urllib.parse import urlparse, urljoin, urlencode, parse_qs, quote, unquote

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# URL PARSING
# ════════════════════════════════════════════════════════════════════════════════

def parse_url(url: str) -> Dict[str, Optional[str]]:
    """
    Parse URL into components.
    
    Args:
        url: URL to parse
    
    Returns:
        Dictionary with URL components (scheme, netloc, path, params, query, fragment)
    
    Example:
        >>> parse_url("https://example.com/path?key=value")
        {'scheme': 'https', 'netloc': 'example.com', 'path': '/path', ...}
    """
    try:
        parsed = urlparse(url)
        return {
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'path': parsed.path,
            'params': parsed.params,
            'query': parsed.query,
            'fragment': parsed.fragment,
            'username': parsed.username,
            'password': parsed.password,
            'hostname': parsed.hostname,
            'port': str(parsed.port) if parsed.port else None,
        }
    except Exception as e:
        logger.error(f"Error parsing URL '{url}': {e}")
        return {}


def get_domain(url: str) -> Optional[str]:
    """
    Get domain from URL.
    
    Args:
        url: URL to parse
    
    Returns:
        Domain name or None if invalid
    
    Example:
        >>> get_domain("https://example.com/path")
        'example.com'
    """
    try:
        parsed = urlparse(url)
        return parsed.hostname
    except Exception:
        return None


def get_path(url: str) -> Optional[str]:
    """
    Get path from URL.
    
    Args:
        url: URL to parse
    
    Returns:
        Path component or None if invalid
    
    Example:
        >>> get_path("https://example.com/path/to/resource")
        '/path/to/resource'
    """
    try:
        parsed = urlparse(url)
        return parsed.path
    except Exception:
        return None


def get_query_params(url: str) -> Dict[str, list]:
    """
    Get query parameters from URL.
    
    Args:
        url: URL to parse
    
    Returns:
        Dictionary of query parameters (values are lists)
    
    Example:
        >>> get_query_params("https://example.com?key1=value1&key2=value2")
        {'key1': ['value1'], 'key2': ['value2']}
    """
    try:
        parsed = urlparse(url)
        return parse_qs(parsed.query)
    except Exception:
        return {}


def get_query_param(url: str, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get single query parameter from URL.
    
    Args:
        url: URL to parse
        key: Parameter key
        default: Default value if not found
    
    Returns:
        Parameter value or default
    
    Example:
        >>> get_query_param("https://example.com?page=1", "page")
        '1'
    """
    params = get_query_params(url)
    values = params.get(key, [])
    return values[0] if values else default


# ════════════════════════════════════════════════════════════════════════════════
# URL BUILDING
# ════════════════════════════════════════════════════════════════════════════════

def build_url(
    base: str,
    path: Optional[str] = None,
    params: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    fragment: Optional[str] = None
) -> str:
    """
    Build URL from components.
    
    Args:
        base: Base URL
        path: Path to append
        params: Query parameters
        fragment: URL fragment
    
    Returns:
        Complete URL
    
    Example:
        >>> build_url("https://example.com", "/api", {"key": "value"})
        'https://example.com/api?key=value'
    """
    url = base
    
    if path:
        url = urljoin(url.rstrip('/') + '/', path.lstrip('/'))
    
    if params:
        query_string = urlencode(params, doseq=True)
        url = f"{url}?{query_string}" if "?" not in url else f"{url}&{query_string}"
    
    if fragment:
        url = f"{url}#{fragment}"
    
    return url


def join_url(base: str, *paths: str) -> str:
    """
    Join URL components.
    
    Args:
        base: Base URL
        *paths: Path components to join
    
    Returns:
        Joined URL
    
    Example:
        >>> join_url("https://example.com", "api", "v1", "users")
        'https://example.com/api/v1/users'
    """
    url = base.rstrip('/')
    for path in paths:
        url = urljoin(url + '/', path.lstrip('/'))
    return url


# ════════════════════════════════════════════════════════════════════════════════
# URL ENCODING/DECODING
# ════════════════════════════════════════════════════════════════════════════════

def encode_url(url: str, safe: str = "/") -> str:
    """
    URL encode string.
    
    Args:
        url: URL to encode
        safe: Characters that should not be encoded
    
    Returns:
        Encoded URL
    
    Example:
        >>> encode_url("hello world")
        'hello%20world'
    """
    return quote(url, safe=safe)


def decode_url(encoded_url: str) -> str:
    """
    URL decode string.
    
    Args:
        encoded_url: Encoded URL
    
    Returns:
        Decoded URL
    
    Example:
        >>> decode_url("hello%20world")
        'hello world'
    """
    return unquote(encoded_url)


# ════════════════════════════════════════════════════════════════════════════════
# URL VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

def is_valid_url(url: str) -> bool:
    """
    Check if string is a valid URL.
    
    Args:
        url: String to check
    
    Returns:
        True if valid URL
    
    Example:
        >>> is_valid_url("https://example.com")
        True
        >>> is_valid_url("not a url")
        False
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """
    Normalize URL (remove trailing slash, lowercase scheme/host).
    
    Args:
        url: URL to normalize
    
    Returns:
        Normalized URL
    
    Example:
        >>> normalize_url("HTTPS://EXAMPLE.COM/path/")
        'https://example.com/path'
    """
    try:
        parsed = urlparse(url)
        normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
        
        # Normalize path
        path = parsed.path.rstrip('/')
        if path:
            normalized += path
        
        # Add query and fragment
        if parsed.query:
            normalized += f"?{parsed.query}"
        if parsed.fragment:
            normalized += f"#{parsed.fragment}"
        
        return normalized
    except Exception:
        return url


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Parsing
    "parse_url",
    "get_domain",
    "get_path",
    "get_query_params",
    "get_query_param",
    # Building
    "build_url",
    "join_url",
    # Encoding
    "encode_url",
    "decode_url",
    # Validation
    "is_valid_url",
    "normalize_url",
]













