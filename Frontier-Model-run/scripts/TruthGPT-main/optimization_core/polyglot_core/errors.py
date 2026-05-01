"""
Custom exceptions for polyglot_core.

Provides specific exception types for better error handling.
"""


class PolyglotError(Exception):
    """Base exception for polyglot_core."""
    pass


class BackendError(PolyglotError):
    """Error related to backend operations."""
    pass


class BackendNotAvailableError(BackendError):
    """Backend is not available."""
    pass


class BackendSelectionError(BackendError):
    """Error selecting backend."""
    pass


class CacheError(PolyglotError):
    """Error related to cache operations."""
    pass


class CacheFullError(CacheError):
    """Cache is full."""
    pass


class CacheKeyError(CacheError):
    """Invalid cache key."""
    pass


class AttentionError(PolyglotError):
    """Error related to attention operations."""
    pass


class AttentionShapeError(AttentionError):
    """Invalid attention input shape."""
    pass


class CompressionError(PolyglotError):
    """Error related to compression operations."""
    pass


class CompressionFailedError(CompressionError):
    """Compression operation failed."""
    pass


class InferenceError(PolyglotError):
    """Error related to inference operations."""
    pass


class GenerationError(InferenceError):
    """Error during text generation."""
    pass


class TokenizationError(PolyglotError):
    """Error related to tokenization."""
    pass


class QuantizationError(PolyglotError):
    """Error related to quantization."""
    pass


class ValidationError(PolyglotError):
    """Validation error."""
    pass


class ConfigurationError(PolyglotError):
    """Configuration error."""
    pass


class HealthCheckError(PolyglotError):
    """Health check error."""
    pass


def handle_polyglot_error(error: Exception, context: dict = None) -> str:
    """
    Handle polyglot error and return user-friendly message.
    
    Args:
        error: Exception to handle
        context: Additional context
        
    Returns:
        User-friendly error message
    """
    if isinstance(error, BackendNotAvailableError):
        return f"Backend not available: {str(error)}"
    
    if isinstance(error, CacheFullError):
        return f"Cache is full: {str(error)}"
    
    if isinstance(error, AttentionShapeError):
        return f"Invalid attention input shape: {str(error)}"
    
    if isinstance(error, ValidationError):
        return f"Validation error: {str(error)}"
    
    if isinstance(error, PolyglotError):
        return f"Polyglot error: {str(error)}"
    
    return f"Error: {str(error)}"













