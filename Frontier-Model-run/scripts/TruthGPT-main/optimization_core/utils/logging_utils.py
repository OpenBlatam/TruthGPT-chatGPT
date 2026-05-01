"""
Structured logging utilities for deep learning workflows.
Provides proper logging configuration following best practices.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a structured logger with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format string with detailed information
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
        )
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Specialized logger for training workflows with structured metrics.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        tokens_per_sec: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log training step information."""
        msg = (
            f"Step {step} | Epoch {epoch} | Loss: {loss:.4f} | "
            f"LR: {learning_rate:.2e}"
        )
        if tokens_per_sec is not None:
            msg += f" | Tokens/sec: {tokens_per_sec:.1f}"
        if kwargs:
            extra_info = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in kwargs.items())
            msg += f" | {extra_info}"
        self.logger.info(msg)
    
    def log_eval(
        self,
        step: int,
        val_loss: float,
        perplexity: Optional[float] = None,
        improved: bool = False,
        **kwargs
    ) -> None:
        """Log evaluation metrics."""
        msg = (
            f"Eval Step {step} | Val Loss: {val_loss:.4f}"
        )
        if perplexity is not None:
            msg += f" | Perplexity: {perplexity:.2f}"
        msg += f" | Improved: {improved}"
        if kwargs:
            extra_info = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in kwargs.items())
            msg += f" | {extra_info}"
        self.logger.info(msg)
    
    def log_checkpoint(self, step: int, path: str, is_best: bool = False) -> None:
        """Log checkpoint saving."""
        status = "BEST" if is_best else "CHECKPOINT"
        self.logger.info(f"{status} saved at step {step} to {path}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log errors with context."""
        context_msg = f" in {context}" if context else ""
        self.logger.error(f"Error{context_msg}: {error}", exc_info=True)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warnings."""
        if kwargs:
            extra = " | ".join(f"{k}: {v}" for k, v in kwargs.items())
            message = f"{message} | {extra}"
        self.logger.warning(message)
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log informational messages."""
        if kwargs:
            extra = " | ".join(f"{k}: {v}" for k, v in kwargs.items())
            message = f"{message} | {extra}"
        self.logger.info(message)

