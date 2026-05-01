"""
Error Handler Module
Centralized error handling and recovery strategies
"""

import logging
from typing import Callable, TypeVar, Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorHandler:
    """Handles errors with fallback strategies."""
    
    @staticmethod
    def with_fallback(
        fallback_strategy: Callable[[], T],
        error_message: Optional[str] = None
    ) -> Callable:
        """Decorator to add fallback strategy on error."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if error_message:
                        logger.warning(f"{error_message}: {e}")
                    return fallback_strategy()
            return wrapper
        return decorator
    
    @staticmethod
    def safe_execute(
        func: Callable[..., T],
        fallback: Callable[[], T],
        *args,
        **kwargs
    ) -> T:
        """Safely execute function with fallback."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error in {func.__name__}: {e}. Using fallback.")
            return fallback()


class StrategyErrorHandler:
    """Specialized error handler for optimization strategies."""
    
    @staticmethod
    def apply_with_fallback(
        strategy_func: Callable,
        fallback_strategy: str,
        strategy_name: str,
        model: Any,
        strategy_registry: Any
    ) -> tuple:
        """Apply strategy with fallback to default strategy."""
        try:
            optimized_model = strategy_func(model)
            techniques_applied = [f'ai_{strategy_name}']
            return optimized_model, techniques_applied
        except Exception as e:
            logger.warning(
                f"Failed to apply strategy '{strategy_name}': {e}. "
                f"Falling back to {fallback_strategy}."
            )
            optimized_model = strategy_registry.apply_strategy(model, fallback_strategy)
            techniques_applied = [f'ai_{fallback_strategy}']
            return optimized_model, techniques_applied


