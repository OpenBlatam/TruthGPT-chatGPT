"""
Inference Engine Exceptions
============================

Custom exceptions for inference engines with detailed error information.
"""

from optimization_core.core.exceptions import (
    InferenceError,
    ModelError,
    ValidationError as CoreValidationError,
    ResourceError,
    ConfigurationError,
)


class InferenceEngineError(InferenceError):
    """
    Base exception for inference engine errors.
    
    Inherits from modules.base.core_system.core.InferenceError for consistency.
    """
    
    def __init__(self, message: str, engine_type: str = None, **kwargs):
        """
        Initialize inference engine error.
        
        Args:
            message: Error message
            engine_type: Type of engine that raised the error
            **kwargs: Additional arguments passed to base class
        """
        details = kwargs.pop('details', {})
        if engine_type:
            details['engine_type'] = engine_type
        super().__init__(message, details=details, **kwargs)
        self.engine_type = engine_type


class EngineInitializationError(InferenceEngineError):
    """Raised when engine initialization fails."""
    pass


class EngineNotInitializedError(InferenceEngineError):
    """Raised when engine is not initialized."""
    pass


class GenerationError(InferenceEngineError):
    """Raised when text generation fails."""
    pass


class ValidationError(CoreValidationError):
    """
    Raised when input validation fails.
    
    Inherits from modules.base.core_system.core.ValidationError for consistency.
    """
    pass


class ModelNotFoundError(ModelError):
    """
    Raised when model file is not found.
    
    Inherits from modules.base.core_system.core.ModelError for consistency.
    """
    pass


class EngineCompilationError(InferenceEngineError):
    """Raised when engine compilation fails."""
    pass


class QuantizationError(InferenceEngineError):
    """Raised when quantization fails."""
    pass


class BatchProcessingError(InferenceEngineError):
    """Raised when batch processing fails."""
    pass


