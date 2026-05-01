"""
Inference Engine Helpers
========================

Helper functions for inference engine operations.
"""

import logging
from typing import List, Optional, Dict, Any, Callable
from functools import wraps

from optimization_core.core.helpers import (
    ensure_initialized as core_ensure_initialized,
    timing_context,
    handle_errors,
    batch_items,
)

from ..exceptions import (
    InferenceEngineError,
    EngineNotInitializedError,
    GenerationError
)

logger = logging.getLogger(__name__)


def ensure_initialized(func: Callable) -> Callable:
    """
    Decorator to ensure engine is initialized before calling method.
    
    Uses core.helpers.ensure_initialized with engine-specific error.
    
    Example:
        >>> @ensure_initialized
        >>> def generate(self, prompts):
        ...     return self._do_generate(prompts)
    """
    @wraps(func)
    @core_ensure_initialized(attr_name='_initialized')
    def wrapper(self, *args, **kwargs):
        # Additional engine-specific check
        if not getattr(self, '_initialized', False):
            engine_type = self.__class__.__name__
            raise EngineNotInitializedError(
                f"Engine is not initialized. Call initialization first.",
                engine_type=engine_type
            )
        return func(self, *args, **kwargs)
    return wrapper


def handle_generation_errors(func: Callable) -> Callable:
    """
    Decorator to handle generation errors with proper exception wrapping.
    
    Example:
        >>> @handle_generation_errors
        >>> def generate(self, prompts):
        ...     return self._raw_generate(prompts)
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except InferenceEngineError:
            # Re-raise inference engine errors as-is
            raise
        except Exception as e:
            engine_type = getattr(self, '__class__', type(self)).__name__
            raise GenerationError(
                f"Generation failed: {str(e)}",
                engine_type=engine_type,
                details={"original_error": type(e).__name__}
            ) from e
    return wrapper


def batch_prompts(
    prompts: List[str],
    max_batch_size: int,
    truncate: bool = False
) -> List[List[str]]:
    """
    Split prompts into batches.
    
    Uses core.helpers.batch_items for consistency.
    
    Args:
        prompts: List of prompts
        max_batch_size: Maximum batch size
        truncate: If True, truncate last batch if needed
    
    Returns:
        List of prompt batches
    
    Raises:
        ValueError: If max_batch_size <= 0
    """
    return batch_items(prompts, max_batch_size, truncate=truncate)


def log_engine_stats(
    engine_name: str,
    operation: str,
    batch_size: int,
    latency_ms: float,
    tokens_generated: Optional[int] = None,
    logger_instance: Optional[logging.Logger] = None
):
    """
    Log engine statistics.
    
    Args:
        engine_name: Name of the engine
        operation: Operation name (e.g., "generate", "batch_generate")
        batch_size: Batch size
        latency_ms: Latency in milliseconds
        tokens_generated: Number of tokens generated (optional)
        logger_instance: Logger instance (optional)
    """
    log = logger_instance or logger
    stats = {
        "engine": engine_name,
        "operation": operation,
        "batch_size": batch_size,
        "latency_ms": latency_ms,
    }
    
    if tokens_generated is not None:
        stats["tokens_generated"] = tokens_generated
        stats["tokens_per_second"] = (tokens_generated / (latency_ms / 1000)) if latency_ms > 0 else 0
    
    log.info(f"Engine stats: {stats}")


def validate_batch_size(batch_size: int, max_batch_size: int, operation: str = "operation"):
    """
    Validate batch size against maximum.
    
    Args:
        batch_size: Requested batch size
        max_batch_size: Maximum allowed batch size
        operation: Operation name for error message
    
    Raises:
        ValueError: If batch_size exceeds max_batch_size
    """
    if batch_size > max_batch_size:
        raise ValueError(
            f"Batch size {batch_size} exceeds maximum {max_batch_size} for {operation}"
        )


def format_error_details(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Format error details for logging.
    
    Args:
        error: Exception that occurred
        context: Additional context information
    
    Returns:
        Dictionary with error details
    """
    details = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if context:
        details.update(context)
    
    return details


