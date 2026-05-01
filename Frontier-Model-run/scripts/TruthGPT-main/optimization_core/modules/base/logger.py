"""
Ultra-fast modular logging system
Following deep learning best practices
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time
import json


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class BaseLogger:
    """Ultra-fast base logger"""
    
    def __init__(self, name: str, config: LoggingConfig):
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with configuration"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.level))
        formatter = logging.Formatter(self.config.format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.file_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.config.file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)


def setup_logging(name: str = "TruthGPT", 
                 config: Optional[LoggingConfig] = None) -> BaseLogger:
    """Setup logging with default configuration"""
    if config is None:
        config = LoggingConfig()
    
    return BaseLogger(name, config)



