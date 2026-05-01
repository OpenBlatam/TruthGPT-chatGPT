"""
Exception classes for optimization_core.

Provides a unified exception hierarchy for consistent error handling.
"""
from enum import Enum
from typing import Optional, Dict, Any
import traceback


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OptimizationCoreError(Exception):
    """
    Base exception for optimization_core module.
    
    All custom exceptions should inherit from this class.
    """
    
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
        self._traceback = traceback.format_exc()
    
    def __str__(self) -> str:
        """String representation."""
        base = f"[{self.severity.value.upper()}] {self.message}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base += f" ({details_str})"
        return base
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
            "traceback": self._traceback,
        }


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
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        """Initialize configuration error."""
        details = kwargs.pop('details', {})
        if config_key:
            details['config_key'] = config_key
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            details=details,
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
    
    def __init__(self, message: str, metric: Optional[str] = None, **kwargs):
        """Initialize performance error."""
        details = kwargs.pop('details', {})
        if metric:
            details['metric'] = metric
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


class ModelError(OptimizationCoreError):
    """Raised when model operations fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        """Initialize model error."""
        details = kwargs.pop('details', {})
        if model_name:
            details['model_name'] = model_name
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            details=details,
            **kwargs
        )


class InferenceError(OptimizationCoreError):
    """Raised when inference operations fail."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize inference error."""
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class DataError(OptimizationCoreError):
    """Raised when data operations fail."""
    
    def __init__(self, message: str, data_source: Optional[str] = None, **kwargs):
        """Initialize data error."""
        details = kwargs.pop('details', {})
        if data_source:
            details['data_source'] = data_source
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            **kwargs
        )


# Re-export from config_base for convenience
from .config_base import ConfigError, ConfigValidationError

__all__ = [
    "ErrorSeverity",
    "OptimizationCoreError",
    "ValidationError",
    "ConfigurationError",
    "ResourceError",
    "PerformanceError",
    "ModelError",
    "InferenceError",
    "DataError",
    "ConfigError",
    "ConfigValidationError",
]













