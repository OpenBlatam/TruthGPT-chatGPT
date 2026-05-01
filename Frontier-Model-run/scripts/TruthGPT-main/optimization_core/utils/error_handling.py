"""
Shared error handling utilities for the entire optimization_core module.

Provides common error handling patterns, custom exceptions, and error recovery.
"""
import logging
import traceback
from typing import Optional, Dict, Any, Callable, Type
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OptimizationCoreError(Exception):
    """Base exception for optimization_core module."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize error.
        
        Args:
            message: Error message
            severity: Error severity
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        """String representation."""
        base = f"[{self.severity.value.upper()}] {self.message}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base += f" ({details_str})"
        return base


class ValidationError(OptimizationCoreError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        """Initialize validation error."""
        details = kwargs.pop('details', {})
        if field:
            details['field'] = field
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            details=details,
            **kwargs
        )


class ConfigurationError(OptimizationCoreError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize configuration error."""
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class ResourceError(OptimizationCoreError):
    """Raised when resource operations fail."""
    
    def __init__(self, message: str, resource: Optional[str] = None, **kwargs):
        """Initialize resource error."""
        details = kwargs.pop('details', {})
        if resource:
            details['resource'] = resource
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class PerformanceError(OptimizationCoreError):
    """Raised when performance issues occur."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize performance error."""
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True,
    log_level: int = logging.ERROR
) -> Optional[OptimizationCoreError]:
    """
    Handle an error with logging and optional conversion.
    
    Args:
        error: Exception to handle
        context: Additional context information
        reraise: Whether to re-raise the error
        log_level: Logging level
    
    Returns:
        Converted error (if not reraise) or None
    """
    context = context or {}
    
    # Convert to OptimizationCoreError if needed
    if isinstance(error, OptimizationCoreError):
        converted_error = error
    else:
        converted_error = OptimizationCoreError(
            message=str(error),
            severity=ErrorSeverity.MEDIUM,
            details=context,
            cause=error
        )
    
    # Log error
    logger.log(
        log_level,
        f"Error: {converted_error.message}",
        extra={
            "severity": converted_error.severity.value,
            "details": converted_error.details,
            "context": context,
            "traceback": traceback.format_exc()
        },
        exc_info=True
    )
    
    if reraise:
        raise converted_error
    
    return converted_error


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_handler: Optional[Callable] = None,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_return: Default return value on error
        error_handler: Optional custom error handler
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            return error_handler(e, *args, **kwargs)
        
        handle_error(e, reraise=False)
        return default_return


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    
    Example:
        @retry_with_backoff(max_attempts=3, initial_delay=1.0)
        def risky_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        import time
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}",
                            exc_info=True
                        )
            
            raise last_exception
        return wrapper
    return decorator


def error_context(operation: str, **context):
    """
    Context manager for error handling with context.
    
    Args:
        operation: Name of operation
        **context: Additional context
    
    Example:
        with error_context("model_loading", model_name="mistral-7b"):
            model = load_model("mistral-7b")
    """
    class ErrorContext:
        def __enter__(self):
            logger.debug(f"Starting {operation}", extra=context)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val:
                handle_error(
                    exc_val,
                    context={"operation": operation, **context},
                    reraise=True
                )
            else:
                logger.debug(f"Completed {operation}", extra=context)
            return False
    
    return ErrorContext()













