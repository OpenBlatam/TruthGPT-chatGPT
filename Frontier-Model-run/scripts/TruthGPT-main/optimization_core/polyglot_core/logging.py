"""
Advanced logging for polyglot_core.

Provides structured logging with performance tracking, backend information,
and integration with metrics.
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json


class PolyglotFormatter(logging.Formatter):
    """
    Custom formatter for polyglot_core logs.
    
    Includes backend information, performance metrics, and structured output.
    """
    
    def __init__(self, include_backend: bool = True, include_metrics: bool = True):
        """
        Initialize formatter.
        
        Args:
            include_backend: Include backend information in logs
            include_metrics: Include performance metrics in logs
        """
        super().__init__()
        self.include_backend = include_backend
        self.include_metrics = include_metrics
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        # Base format
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        level = record.levelname
        name = record.name
        message = record.getMessage()
        
        # Build log line
        parts = [f"[{timestamp}]", f"[{level}]", f"[{name}]"]
        
        # Add backend info if available
        if self.include_backend and hasattr(record, 'backend'):
            parts.append(f"[backend:{record.backend}]")
        
        # Add metrics if available
        if self.include_metrics and hasattr(record, 'duration_ms'):
            parts.append(f"[{record.duration_ms:.2f}ms]")
        
        parts.append(message)
        
        # Add exception info if present
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))
        
        return " ".join(parts)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, 'backend'):
            log_data['backend'] = record.backend
        
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        
        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation
        
        # Add exception if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class PolyglotLogger:
    """
    Enhanced logger for polyglot_core.
    
    Provides structured logging with backend tracking and performance metrics.
    """
    
    def __init__(
        self,
        name: str = "polyglot_core",
        level: int = logging.INFO,
        use_structured: bool = False,
        log_file: Optional[Path] = None
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Log level
            use_structured: Use JSON structured logging
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if use_structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(PolyglotFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            if use_structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(PolyglotFormatter())
            self.logger.addHandler(file_handler)
    
    def log_operation(
        self,
        operation: str,
        backend: Optional[str] = None,
        duration_ms: Optional[float] = None,
        level: int = logging.INFO,
        **kwargs
    ):
        """
        Log an operation with backend and performance info.
        
        Args:
            operation: Operation name
            backend: Backend used
            duration_ms: Duration in milliseconds
            level: Log level
            **kwargs: Additional fields
        """
        extra = {'operation': operation, **kwargs}
        if backend:
            extra['backend'] = backend
        if duration_ms is not None:
            extra['duration_ms'] = duration_ms
        
        self.logger.log(level, f"Operation: {operation}", extra=extra)
    
    def log_backend_selection(
        self,
        feature: str,
        selected_backend: str,
        available_backends: list,
        reason: str = ""
    ):
        """Log backend selection."""
        self.logger.info(
            f"Backend selected for {feature}: {selected_backend}",
            extra={
                'operation': 'backend_selection',
                'feature': feature,
                'backend': selected_backend,
                'available_backends': available_backends,
                'reason': reason
            }
        )
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        backend: Optional[str] = None,
        throughput: Optional[float] = None
    ):
        """Log performance metrics."""
        extra = {
            'operation': operation,
            'duration_ms': duration_ms
        }
        if backend:
            extra['backend'] = backend
        if throughput:
            extra['throughput'] = throughput
        
        self.logger.info(
            f"Performance: {operation} took {duration_ms:.2f}ms",
            extra=extra
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        operation: str,
        backend: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log error with context."""
        extra = {
            'operation': operation,
            'error_type': type(error).__name__
        }
        if backend:
            extra['backend'] = backend
        if context:
            extra.update(context)
        
        self.logger.error(
            f"Error in {operation}: {str(error)}",
            exc_info=True,
            extra=extra
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)


# Global logger
_global_logger: Optional[PolyglotLogger] = None


def get_logger(
    name: str = "polyglot_core",
    level: Optional[int] = None,
    use_structured: bool = False,
    log_file: Optional[Path] = None
) -> PolyglotLogger:
    """
    Get or create global logger.
    
    Args:
        name: Logger name
        level: Log level (default: from config or INFO)
        use_structured: Use JSON structured logging
        log_file: Optional log file path
        
    Returns:
        PolyglotLogger
    """
    global _global_logger
    
    if _global_logger is None:
        if level is None:
            # Try to get from config
            try:
                from .config import get_config
                config = get_config()
                level_map = {
                    'DEBUG': logging.DEBUG,
                    'INFO': logging.INFO,
                    'WARNING': logging.WARNING,
                    'ERROR': logging.ERROR,
                    'CRITICAL': logging.CRITICAL
                }
                level = level_map.get(config.log_level, logging.INFO)
            except Exception:
                level = logging.INFO
        
        _global_logger = PolyglotLogger(
            name=name,
            level=level,
            use_structured=use_structured,
            log_file=log_file
        )
    
    return _global_logger


def setup_logging(
    level: int = logging.INFO,
    use_structured: bool = False,
    log_file: Optional[Path] = None
):
    """
    Setup global logging.
    
    Args:
        level: Log level
        use_structured: Use JSON structured logging
        log_file: Optional log file path
    """
    global _global_logger
    _global_logger = PolyglotLogger(
        name="polyglot_core",
        level=level,
        use_structured=use_structured,
        log_file=log_file
    )













