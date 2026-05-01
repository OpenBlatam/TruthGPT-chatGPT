"""
Middleware utilities for optimization_core.

Provides utilities for middleware pattern implementation.
"""
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from pydantic import BaseModel, ConfigDict
from functools import wraps

logger = logging.getLogger(__name__)


class Middleware(BaseModel):
    """Middleware definition."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    func: Callable
    priority: int = 0
    description: Optional[str] = None


class MiddlewareStack:
    """Stack for middleware execution."""
    
    def __init__(self):
        """Initialize middleware stack."""
        self.middlewares: List[Middleware] = []
    
    def add(
        self,
        name: str,
        func: Callable,
        priority: int = 0,
        description: Optional[str] = None
    ):
        """
        Add middleware to stack.
        
        Args:
            name: Middleware name
            func: Middleware function
            priority: Priority (higher = executed first)
            description: Optional description
        """
        middleware = Middleware(
            name=name,
            func=func,
            priority=priority,
            description=description
        )
        
        self.middlewares.append(middleware)
        self.middlewares.sort(key=lambda m: m.priority, reverse=True)
        
        logger.debug(f"Added middleware: {name} (priority: {priority})")
    
    def execute(
        self,
        request: Dict[str, Any],
        handler: Callable
    ) -> Any:
        """
        Execute middleware stack.
        
        Args:
            request: Request data
            handler: Final handler function
        
        Returns:
            Handler result
        """
        def create_middleware_chain(index: int):
            """Create middleware chain."""
            if index >= len(self.middlewares):
                return handler
            
            middleware = self.middlewares[index]
            
            def next_middleware(req: Dict[str, Any]):
                """Next middleware in chain."""
                return create_middleware_chain(index + 1)(req)
            
            def middleware_wrapper(req: Dict[str, Any]):
                """Middleware wrapper."""
                try:
                    return middleware.func(req, next_middleware)
                except Exception as e:
                    logger.error(
                        f"Middleware '{middleware.name}' failed: {e}",
                        exc_info=True
                    )
                    raise
            
            return middleware_wrapper
        
        return create_middleware_chain(0)(request)
    
    def remove(self, name: str):
        """
        Remove middleware from stack.
        
        Args:
            name: Middleware name
        """
        self.middlewares = [m for m in self.middlewares if m.name != name]
        logger.debug(f"Removed middleware: {name}")
    
    def list_middlewares(self) -> List[str]:
        """
        List all middlewares.
        
        Returns:
            List of middleware names
        """
        return [m.name for m in self.middlewares]


def middleware_decorator(
    name: str,
    priority: int = 0
):
    """
    Decorator for creating middleware.
    
    Args:
        name: Middleware name
        priority: Priority
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        """Middleware decorator."""
        @wraps(func)
        def wrapper(request: Dict[str, Any], next_handler: Callable):
            """Middleware wrapper."""
            return func(request, next_handler)
        
        wrapper.middleware_name = name
        wrapper.middleware_priority = priority
        
        return wrapper
    
    return decorator


def create_middleware_stack() -> MiddlewareStack:
    """
    Create a middleware stack.
    
    Returns:
        Middleware stack
    """
    return MiddlewareStack()













