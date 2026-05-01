"""
Logging utilities for optimization_core.

Provides common logging patterns and utilities.
"""
import logging
import sys
from typing import Optional, Dict, Any, Union
from pathlib import Path


def get_logger(
    name: str,
    level: Optional[int] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Get or create a logger with consistent configuration.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (defaults to INFO)
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )
        
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        if level is None:
            level = logging.INFO
        
        logger.setLevel(level)
    
    return logger


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    root_logger: bool = True
) -> None:
    """
    Set up root logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Optional log file path
        root_logger: Configure root logger (True) or create new (False)
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        handlers.append(file_handler)
    
    formatter = logging.Formatter(format_string)
    
    for handler in handlers:
        handler.setFormatter(formatter)
    
    if root_logger:
        logger = logging.getLogger()
        logger.setLevel(level)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        for handler in handlers:
            logger.addHandler(handler)
    else:
        for handler in handlers:
            handler.setLevel(level)


class LoggerMixin:
    """
    Mixin class to add logging to any class.
    
    Example:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.logger.info("Initialized")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger instance
        level: Logging level
    
    Example:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            return arg1 + arg2
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log(
                level,
                f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger, level: int = logging.INFO):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance
        level: Logging level
    
    Example:
        @log_execution_time(logger)
        def slow_function():
            time.sleep(1)
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.log(
                    level,
                    f"{func.__name__} executed in {elapsed:.3f}s"
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {elapsed:.3f}s: {e}"
                )
                raise
        return wrapper
    return decorator


