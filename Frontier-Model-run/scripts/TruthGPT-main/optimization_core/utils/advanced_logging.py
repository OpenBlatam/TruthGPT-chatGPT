"""
Advanced logging utilities for optimization_core.

Provides advanced logging capabilities with multiple handlers and formatters.
"""
import logging
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'levelname',
                          'levelno', 'lineno', 'module', 'msecs', 'message', 'pathname',
                          'process', 'processName', 'relativeCreated', 'thread', 'threadName',
                          'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with multiple handlers."""
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        json_log: bool = False,
        console: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            json_log: Whether to use JSON formatting
            console: Whether to log to console
            max_bytes: Maximum log file size
            backup_count: Number of backup files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            if json_log:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setLevel(level)
            if json_log:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            self.logger.addHandler(file_handler)
    
    def log_with_context(
        self,
        level: int,
        message: str,
        **context
    ):
        """
        Log with additional context.
        
        Args:
            level: Log level
            message: Log message
            **context: Additional context fields
        """
        extra = {f"ctx_{k}": v for k, v in context.items()}
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **context):
        """Log info message with context."""
        self.log_with_context(logging.INFO, message, **context)
    
    def warning(self, message: str, **context):
        """Log warning message with context."""
        self.log_with_context(logging.WARNING, message, **context)
    
    def error(self, message: str, **context):
        """Log error message with context."""
        self.log_with_context(logging.ERROR, message, **context)
    
    def debug(self, message: str, **context):
        """Log debug message with context."""
        self.log_with_context(logging.DEBUG, message, **context)
    
    def critical(self, message: str, **context):
        """Log critical message with context."""
        self.log_with_context(logging.CRITICAL, message, **context)


def setup_logging(
    name: str = "optimization_core",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    json_log: bool = False
) -> StructuredLogger:
    """
    Setup structured logging.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        json_log: Whether to use JSON formatting
    
    Returns:
        Structured logger
    """
    return StructuredLogger(name, level, log_file, json_log)













