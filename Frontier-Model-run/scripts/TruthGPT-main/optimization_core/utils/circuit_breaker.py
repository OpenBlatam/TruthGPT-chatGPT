"""
Circuit breaker utilities for optimization_core.

Provides utilities for circuit breaker pattern implementation.
"""
import logging
import time
from typing import Callable, Optional, Dict, Any
from enum import Enum
from functools import wraps
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If function fails
        """
        # Check if circuit should transition
        self._check_state_transition()
        
        # Reject if open
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN"
            )
        
        # Try to call function
        try:
            result = func(*args, **kwargs)
            
            # Success
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            else:
                self.failure_count = 0
            
            return result
        
        except self.config.expected_exception as e:
            # Failure
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
            
            raise
    
    def _check_state_transition(self):
        """Check if circuit should transition states."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_state_change >= self.config.timeout:
                self._transition_to_half_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        if self.state != CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}' transitioning to OPEN "
                f"(failures: {self.failure_count})"
            )
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            self.success_count = 0
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0
    
    def reset(self):
        """Reset circuit breaker to CLOSED state."""
        self._transition_to_closed()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state.
        
        Returns:
            State dictionary
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


class CircuitBreakerOpenError(Exception):
    """Error raised when circuit breaker is open."""
    pass


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    Decorator for circuit breaker.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Failure threshold
        success_threshold: Success threshold for half-open
        timeout: Timeout before transitioning to half-open
        expected_exception: Exception type to catch
    
    Returns:
        Decorator function
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        expected_exception=expected_exception
    )
    breaker = CircuitBreaker(name, config)
    
    def decorator(func: Callable):
        """Circuit breaker decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function."""
            return breaker.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator













