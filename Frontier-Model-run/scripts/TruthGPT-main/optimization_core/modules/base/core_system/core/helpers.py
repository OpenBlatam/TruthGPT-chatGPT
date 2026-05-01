"""
Common helper utilities for optimization_core.

Provides decorators, context managers, and utility functions
shared across all modules.
"""
import logging
import time
from typing import Callable, Optional, Dict, Any, TypeVar, List
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


# ════════════════════════════════════════════════════════════════════════════════
# DECORATORS
# ════════════════════════════════════════════════════════════════════════════════

def ensure_initialized(attr_name: str = '_initialized', error_message: Optional[str] = None):
    """
    Decorator to ensure object is initialized before calling method.
    
    Args:
        attr_name: Attribute name to check for initialization
        error_message: Custom error message
    
    Example:
        >>> class MyClass:
        ...     def __init__(self):
        ...         self._initialized = True
        ...     
        ...     @ensure_initialized()
        ...     def do_something(self):
        ...         return "done"
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, attr_name, False):
                class_name = self.__class__.__name__
                message = error_message or f"{class_name} is not initialized. Call initialization first."
                raise RuntimeError(message)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def timing(operation_name: Optional[str] = None, log_level: int = logging.DEBUG):
    """
    Decorator to measure and log function execution time.
    
    Args:
        operation_name: Name of operation (defaults to function name)
        log_level: Logging level
    
    Example:
        >>> @timing("data_processing")
        >>> def process_data():
        ...     time.sleep(1)
    """
    def decorator(func: F) -> F:
        op_name = operation_name or func.__name__
        log = logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                log.log(log_level, f"{op_name} took {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                log.error(f"{op_name} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator


def handle_errors(
    error_type: type = Exception,
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = True
):
    """
    Decorator to handle errors with optional logging and default return.
    
    Args:
        error_type: Type of exception to catch
        default_return: Value to return on error (if not reraise)
        log_error: Whether to log the error
        reraise: Whether to re-raise the error
    
    Example:
        >>> @handle_errors(ValueError, default_return=0)
        >>> def parse_int(s):
        ...     return int(s)
    """
    def decorator(func: F) -> F:
        log = logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                if log_error:
                    log.error(f"{func.__name__} raised {type(e).__name__}: {e}", exc_info=True)
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def retry(
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
        >>> @retry(max_attempts=3, initial_delay=1.0)
        >>> def risky_operation():
        ...     ...
    """
    def decorator(func: F) -> F:
        log = logging.getLogger(func.__module__)
        
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
                        log.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        log.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}",
                            exc_info=True
                        )
            
            raise last_exception
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGERS
# ════════════════════════════════════════════════════════════════════════════════

@contextmanager
def timing_context(operation_name: str, logger_instance: Optional[logging.Logger] = None):
    """
    Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation
        logger_instance: Logger instance (defaults to module logger)
    
    Example:
        >>> with timing_context("data_processing", logger):
        ...     result = process_data()
    """
    log = logger_instance or logger
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        log.debug(f"{operation_name} took {elapsed:.3f}s")


@contextmanager
def error_context(operation: str, **context):
    """
    Context manager for error handling with context.
    
    Args:
        operation: Name of operation
        **context: Additional context information
    
    Example:
        >>> with error_context("model_loading", model_name="mistral-7b"):
        ...     model = load_model("mistral-7b")
    """
    log = logging.getLogger(__name__)
    log.debug(f"Starting {operation}", extra=context)
    try:
        yield
        log.debug(f"Completed {operation}", extra=context)
    except Exception as e:
        log.error(
            f"Failed {operation}: {e}",
            extra={"operation": operation, **context},
            exc_info=True
        )
        raise


# ════════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def batch_items(items: List[T], batch_size: int, truncate: bool = False) -> List[List[T]]:
    """
    Split items into batches of specified size.
    
    Args:
        items: List of items to batch
        batch_size: Maximum batch size (must be positive)
        truncate: If True, only include complete batches (discard incomplete last batch)
    
    Returns:
        List of batches (each batch is a list of items)
    
    Raises:
        ValueError: If batch_size <= 0 or items is not a list
    
    Examples:
        >>> batch_items([1, 2, 3, 4, 5], batch_size=2)
        [[1, 2], [3, 4], [5]]
        >>> batch_items([1, 2, 3, 4, 5], batch_size=2, truncate=True)
        [[1, 2], [3, 4]]
    """
    # Validate inputs
    if not isinstance(items, list):
        raise TypeError(f"items must be a list, got {type(items).__name__}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    # Handle empty list
    if not items:
        return []
    
    # Create batches efficiently using list comprehension
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # Truncate last batch if requested and incomplete
    if truncate and batches and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    
    return batches


def format_duration(seconds: float, precision: int = 3) -> str:
    """
    Format duration in human-readable format with appropriate units.
    
    Args:
        seconds: Duration in seconds (must be non-negative)
        precision: Decimal precision for fractional seconds (default: 3)
    
    Returns:
        Formatted duration string (e.g., "123.456s", "5m 30.5s", "1h 2m 3.4s")
    
    Raises:
        ValueError: If seconds is negative
    
    Examples:
        >>> format_duration(123.456)
        '123.456s'
        >>> format_duration(3661.5)
        '1h 1m 1.5s'
        >>> format_duration(90)
        '1m 30.000s'
    """
    # Validate input
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    if precision < 0:
        raise ValueError(f"precision must be non-negative, got {precision}")
    
    # Format based on duration length
    if seconds < 60:
        # Less than a minute: show seconds only
        return f"{seconds:.{precision}f}s"
    elif seconds < 3600:
        # Less than an hour: show minutes and seconds
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.{precision}f}s"
    else:
        # One hour or more: show hours, minutes, and seconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.{precision}f}s"


def format_size(size_bytes: int) -> str:
    """
    Format size in human-readable format with appropriate units.
    
    Args:
        size_bytes: Size in bytes (must be non-negative)
    
    Returns:
        Formatted size string (e.g., "1.0 KB", "1.5 MB", "2.3 GB")
    
    Raises:
        ValueError: If size_bytes is negative
    
    Examples:
        >>> format_size(1024)
        '1.0 KB'
        >>> format_size(1048576)
        '1.0 MB'
        >>> format_size(1536)
        '1.5 KB'
    """
    # Validate input
    if size_bytes < 0:
        raise ValueError(f"size_bytes must be non-negative, got {size_bytes}")
    
    # Handle zero case
    if size_bytes == 0:
        return "0 B"
    
    # Define size units in order
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(size_bytes)
    
    # Find appropriate unit
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"


def safe_get(dictionary: Dict[str, Any], *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        dictionary: Dictionary to search
        *keys: Keys to traverse
        default: Default value if key not found
    
    Returns:
        Value or default
    
    Example:
        >>> d = {"a": {"b": {"c": 1}}}
        >>> safe_get(d, "a", "b", "c")
        1
        >>> safe_get(d, "a", "x", default=0)
        0
    """
    current = dictionary
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current if current is not None else default


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    
    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


__all__ = [
    # Decorators
    "ensure_initialized",
    "timing",
    "handle_errors",
    "retry",
    # Context managers
    "timing_context",
    "error_context",
    # Utilities
    "batch_items",
    "format_duration",
    "format_size",
    "safe_get",
    "chunk_list",
]



