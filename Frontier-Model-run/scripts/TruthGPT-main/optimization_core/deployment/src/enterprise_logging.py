"""
Enterprise TruthGPT Logging System
Advanced logging with enterprise features
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import datetime
from contextlib import contextmanager
from enum import Enum
import traceback
import colorlog

class LogLevel(Enum):
    """Log levels for enterprise system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    ENTERPRISE = "ENTERPRISE"
    OPTIMIZATION = "OPTIMIZATION"
    COMPLIANCE = "COMPLIANCE"
    COST = "COST"

class EnterpriseLogger:
    """Enterprise logging system with advanced features."""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup enterprise logger with advanced features."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level.value)
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level.value)
        
        # Color formatter
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)s]%(reset)s %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'blue',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
                'SECURITY': 'magenta',
                'ENTERPRISE': 'purple',
                'OPTIMIZATION': 'green',
                'COMPLIANCE': 'blue',
                'COST': 'cyan'
            }
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with JSON format
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f'{self.name}_{datetime.datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # JSON formatter
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
        )
        file_handler.setFormatter(json_formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
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
    
    def security(self, message: str, **kwargs):
        """Log security message."""
        self.logger.log(25, message, extra=kwargs)
    
    def enterprise(self, message: str, **kwargs):
        """Log enterprise message."""
        self.logger.log(26, message, extra=kwargs)
    
    def optimization(self, message: str, **kwargs):
        """Log optimization message."""
        self.logger.log(27, message, extra=kwargs)
    
    def compliance(self, message: str, **kwargs):
        """Log compliance message."""
        self.logger.log(28, message, extra=kwargs)
    
    def cost(self, message: str, **kwargs):
        """Log cost message."""
        self.logger.log(29, message, extra=kwargs)
    
    @contextmanager
    def log_context(self, operation: str, **kwargs):
        """Context manager for logging operations."""
        try:
            self.info(f"Starting {operation}", **kwargs)
            yield
            self.info(f"Completed {operation}", **kwargs)
        except Exception as e:
            self.error(
                f"Failed {operation}: {str(e)}",
                exception=traceback.format_exc(),
                **kwargs
            )
            raise

# Global logger instance
_enterprise_logger: Optional[EnterpriseLogger] = None

def get_logger(name: str, level: LogLevel = LogLevel.INFO) -> EnterpriseLogger:
    """Get or create enterprise logger."""
    global _enterprise_logger
    if _enterprise_logger is None:
        _enterprise_logger = EnterpriseLogger(name, level)
    return _enterprise_logger

# Example usage
if __name__ == "__main__":
    logger = get_logger("truthgpt-enterprise", LogLevel.INFO)
    
    logger.info("Enterprise TruthGPT logging system initialized")
    logger.security("Security checkpoint passed")
    logger.enterprise("Enterprise features enabled")
    logger.optimization("Optimization system ready")
    logger.compliance("Compliance checks passed")
    logger.cost("Cost optimization enabled")
    
    with logger.log_context("example_operation"):
        logger.debug("Inside operation")
        logger.info("Operation in progress")
    
    logger.error("This is an error message")
    logger.critical("This is a critical message")








