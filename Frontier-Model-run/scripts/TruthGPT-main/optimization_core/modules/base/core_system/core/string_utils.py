"""
Common string utilities for optimization_core.

Provides reusable string manipulation and text processing functions.
"""

import re
import unicodedata
import hashlib
import secrets
import string
from typing import List, Optional, Set

from .types import OptionalStr


# ════════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING AND NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════════

def clean_text(text: str, normalize_unicode: bool = True) -> str:
    """
    Clean and normalize text (removes extra whitespace, normalizes unicode).
    
    Args:
        text: Text to clean (must be a string)
        normalize_unicode: Whether to normalize unicode characters (default: True)
    
    Returns:
        Cleaned text (empty string if text is empty or None)
    
    Raises:
        TypeError: If text is not a string
    
    Examples:
        >>> clean_text("  Hello   World  ")
        'Hello World'
        >>> clean_text("")
        ''
    """
    # Validate input
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    
    # Handle empty string efficiently
    if not text:
        return ""
    
    # Normalize unicode (decomposes characters, useful for removing accents)
    if normalize_unicode:
        text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace (collapses multiple spaces/tabs/newlines to single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Text to normalize
    
    Returns:
        Text with normalized whitespace
    
    Example:
        >>> normalize_whitespace("hello\\t\\nworld")
        'hello world'
    """
    return re.sub(r'\s+', ' ', text).strip()


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text (preserves text content).
    
    Args:
        text: Text with HTML tags (must be a string)
    
    Returns:
        Text without HTML tags (empty string if text is empty)
    
    Raises:
        TypeError: If text is not a string
    
    Examples:
        >>> remove_html_tags("<p>Hello</p>")
        'Hello'
        >>> remove_html_tags("<div>Text</div>")
        'Text'
        >>> remove_html_tags("")
        ''
    """
    # Validate input
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    
    # Handle empty string efficiently
    if not text:
        return ""
    
    # Remove all HTML tags (matches <...> patterns)
    return re.sub(r'<[^>]+>', '', text)


# ════════════════════════════════════════════════════════════════════════════════
# TEXT TRANSFORMATION
# ════════════════════════════════════════════════════════════════════════════════

def slugify(text: str, separator: str = "-", max_length: Optional[int] = None) -> str:
    """
    Convert text to URL-friendly slug (lowercase, alphanumeric, separated).
    
    Args:
        text: Text to slugify (must be a string)
        separator: Separator character (default: "-")
        max_length: Maximum length (None for no limit, must be positive if provided)
    
    Returns:
        Slug string (empty string if text is empty)
    
    Raises:
        TypeError: If text is not a string or separator is not a string
        ValueError: If max_length is provided and <= 0
    
    Examples:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("Test Text", separator="_")
        'test_text'
        >>> slugify("")
        ''
    """
    # Validate inputs
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not isinstance(separator, str):
        raise TypeError(f"separator must be a string, got {type(separator).__name__}")
    if max_length is not None and max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    
    # Handle empty string efficiently
    if not text:
        return ""
    
    # Normalize unicode (decomposes characters for better handling)
    text = unicodedata.normalize('NFKD', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and underscores with separator
    text = re.sub(r'[\s_]+', separator, text)
    
    # Remove special characters, keep alphanumeric and separator
    text = re.sub(rf'[^\w{re.escape(separator)}]', '', text)
    
    # Remove multiple consecutive separators
    text = re.sub(rf'{re.escape(separator)}+', separator, text)
    
    # Remove leading/trailing separators
    text = text.strip(separator)
    
    # Truncate if needed (preserve separator position)
    if max_length is not None and len(text) > max_length:
        text = text[:max_length].rstrip(separator)
    
    return text


def truncate(text: str, max_length: int, suffix: str = "...", preserve_words: bool = False) -> str:
    """
    Truncate text to max length with suffix (optionally preserving word boundaries).
    
    Args:
        text: Text to truncate (must be a string)
        max_length: Maximum length (must be positive and >= len(suffix))
        suffix: Suffix to append (default: "...")
        preserve_words: If True, don't cut words (truncates at word boundary)
    
    Returns:
        Truncated text (original text if within max_length)
    
    Raises:
        TypeError: If text or suffix is not a string
        ValueError: If max_length <= 0 or max_length < len(suffix)
    
    Examples:
        >>> truncate("Hello world", 8)
        'Hello...'
        >>> truncate("Hello world", 8, preserve_words=True)
        'Hello...'
        >>> truncate("Short", 10)
        'Short'
    """
    # Validate inputs
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not isinstance(suffix, str):
        raise TypeError(f"suffix must be a string, got {type(suffix).__name__}")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if max_length < len(suffix):
        raise ValueError(f"max_length ({max_length}) must be >= len(suffix) ({len(suffix)})")
    
    # Handle text that fits within max_length
    if len(text) <= max_length:
        return text
    
    # Calculate available space for text (excluding suffix)
    available_length = max_length - len(suffix)
    
    if preserve_words:
        # Try to preserve word boundaries (don't cut words in the middle)
        truncated = text[:available_length]
        last_space = truncated.rfind(' ')
        # Only use word boundary if it's not too far from the end (at least 50% of available)
        if last_space > available_length * 0.5:
            truncated = truncated[:last_space]
        return truncated + suffix
    else:
        # Simple truncation at exact position
        return text[:available_length] + suffix


def camel_to_snake(text: str) -> str:
    """
    Convert camelCase to snake_case.
    
    Args:
        text: Text in camelCase
    
    Returns:
        Text in snake_case
    
    Example:
        >>> camel_to_snake("HelloWorld")
        'hello_world'
    """
    # Insert underscore before uppercase letters
    text = re.sub(r'(?<!^)(?=[A-Z])', '_', text)
    return text.lower()


def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
    """
    Convert snake_case to camelCase.
    
    Args:
        text: Text in snake_case
        capitalize_first: Whether to capitalize first letter
    
    Returns:
        Text in camelCase
    
    Example:
        >>> snake_to_camel("hello_world")
        'helloWorld'
        >>> snake_to_camel("hello_world", capitalize_first=True)
        'HelloWorld'
    """
    components = text.split('_')
    if capitalize_first:
        return ''.join(x.capitalize() for x in components)
    return components[0] + ''.join(x.capitalize() for x in components[1:])


# ════════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_words(text: str, min_length: int = 1) -> List[str]:
    """
    Extract words from text.
    
    Args:
        text: Text to extract words from
        min_length: Minimum word length
    
    Returns:
        List of words
    
    Example:
        >>> extract_words("Hello world!")
        ['hello', 'world']
    """
    if not text:
        return []
    
    # Clean text first
    text = clean_text(text)
    
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    
    # Filter by minimum length
    return [w for w in words if len(w) >= min_length]


def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from text.
    
    Args:
        text: Text to extract hashtags from
    
    Returns:
        List of hashtags (without #)
    
    Example:
        >>> extract_hashtags("Hello #world #python")
        ['world', 'python']
    """
    hashtags = re.findall(r'#(\w+)', text)
    return hashtags


def extract_mentions(text: str) -> List[str]:
    """
    Extract mentions from text.
    
    Args:
        text: Text to extract mentions from
    
    Returns:
        List of mentions (without @)
    
    Example:
        >>> extract_mentions("Hello @user1 @user2")
        ['user1', 'user2']
    """
    mentions = re.findall(r'@(\w+)', text)
    return mentions


def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Text to extract emails from
    
    Returns:
        List of email addresses
    
    Example:
        >>> extract_emails("Contact: user@example.com")
        ['user@example.com']
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text to extract URLs from
    
    Returns:
        List of URLs
    
    Example:
        >>> extract_urls("Visit https://example.com")
        ['https://example.com']
    """
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(pattern, text)


# ════════════════════════════════════════════════════════════════════════════════
# STRING GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def generate_random_string(length: int = 32, include_symbols: bool = False) -> str:
    """
    Generate cryptographically secure random string.
    
    Args:
        length: Length of string (must be positive)
        include_symbols: Whether to include symbols (default: False)
    
    Returns:
        Random string (empty string if length is 0)
    
    Raises:
        ValueError: If length < 0
    
    Examples:
        >>> len(generate_random_string(16))
        16
        >>> generate_random_string(0)
        ''
    """
    # Validate input
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    
    # Handle zero length efficiently
    if length == 0:
        return ""
    
    # Build character set (alphanumeric, optionally with symbols)
    characters = string.ascii_letters + string.digits
    if include_symbols:
        characters += "!@#$%^&*"
    
    # Generate random string using cryptographically secure random choice
    return ''.join(secrets.choice(characters) for _ in range(length))


def generate_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of text using specified algorithm.
    
    Args:
        text: Text to hash (must be a string)
        algorithm: Hash algorithm (md5, sha1, sha256, sha512, default: "sha256")
    
    Returns:
        Hexadecimal hash string (empty string if text is empty)
    
    Raises:
        TypeError: If text is not a string
        ValueError: If algorithm is not supported
    
    Examples:
        >>> len(generate_hash("hello"))
        64  # SHA256 produces 64 hex characters
        >>> generate_hash("")
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    # Validate inputs
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    
    # Supported hash algorithms
    supported_algorithms = {'md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512'}
    if algorithm.lower() not in supported_algorithms:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Supported: {sorted(supported_algorithms)}"
        )
    
    # Generate hash (handles empty string correctly)
    try:
        hash_obj = hashlib.new(algorithm.lower())
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()
    except ValueError as e:
        raise ValueError(f"Failed to create hash with algorithm '{algorithm}': {e}")


# ════════════════════════════════════════════════════════════════════════════════
# STRING VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

def is_email(text: str) -> bool:
    """
    Check if text is a valid email address.
    
    Args:
        text: Text to check
    
    Returns:
        True if valid email
    
    Example:
        >>> is_email("user@example.com")
        True
    """
    pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
    return bool(re.match(pattern, text))


def is_url(text: str) -> bool:
    """
    Check if text is a valid URL.
    
    Args:
        text: Text to check
    
    Returns:
        True if valid URL
    
    Example:
        >>> is_url("https://example.com")
        True
    """
    pattern = r'^http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$'
    return bool(re.match(pattern, text))


def sanitize_filename(filename: str, max_length: int = 255, replacement: str = "_") -> str:
    """
    Sanitize filename for filesystem (removes invalid characters, preserves extension).
    
    Args:
        filename: Filename to sanitize (must be a string)
        max_length: Maximum length (default: 255, must be positive)
        replacement: Character to replace invalid chars with (default: "_", must be single char)
    
    Returns:
        Sanitized filename (empty string if filename is empty after sanitization)
    
    Raises:
        TypeError: If filename or replacement is not a string
        ValueError: If max_length <= 0 or replacement is not a single character
    
    Examples:
        >>> sanitize_filename("my file.txt")
        'my_file.txt'
        >>> sanitize_filename("file<>name.txt")
        'file__name.txt'
    """
    # Validate inputs
    if not isinstance(filename, str):
        raise TypeError(f"filename must be a string, got {type(filename).__name__}")
    if not isinstance(replacement, str):
        raise TypeError(f"replacement must be a string, got {type(replacement).__name__}")
    if len(replacement) != 1:
        raise ValueError(f"replacement must be a single character, got '{replacement}'")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    
    # Handle empty filename
    if not filename:
        return ""
    
    # Remove invalid filesystem characters (Windows and Unix)
    filename = re.sub(r'[<>:"/\\|?*]', replacement, filename)
    
    # Remove leading/trailing dots and spaces (invalid in some filesystems)
    filename = filename.strip('. ')
    
    # Handle case where filename becomes empty after sanitization
    if not filename:
        return "file"  # Default safe filename
    
    # Truncate if needed (try to preserve extension)
    if len(filename) > max_length:
        if '.' in filename:
            # Preserve extension if possible
            name, ext = filename.rsplit('.', 1)
            max_name_length = max_length - len(ext) - 1
            if max_name_length > 0:
                filename = name[:max_name_length] + '.' + ext
            else:
                # Extension too long, truncate entire filename
                filename = filename[:max_length]
        else:
            # No extension, simple truncation
            filename = filename[:max_length]
    
    return filename


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Cleaning
    "clean_text",
    "normalize_whitespace",
    "remove_html_tags",
    # Transformation
    "slugify",
    "truncate",
    "camel_to_snake",
    "snake_to_camel",
    # Extraction
    "extract_words",
    "extract_hashtags",
    "extract_mentions",
    "extract_emails",
    "extract_urls",
    # Generation
    "generate_random_string",
    "generate_hash",
    # Validation
    "is_email",
    "is_url",
    "sanitize_filename",
]




