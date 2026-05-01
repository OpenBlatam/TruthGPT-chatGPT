"""
Circuit breaker utilities for optimization_core.

Provides reusable circuit breaker implementation for fault tolerance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ════════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER STATE
# ════════════════════════════════════════════════════════════════════════════════

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


# ════════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER CONFIG
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    """Number of failures before opening circuit."""
    
    recovery_timeout: float = 30.0
    """Seconds to wait before attempting recovery (half-open)."""
    
    success_threshold: int = 2
    """Number of successes needed to close circuit from half-open."""
    
    expected_exception: Tuple[type, ...] = (Exception,)
    """Exceptions that count as failures."""
    
    half_open_max_concurrent: int = 1
    """Max concurrent requests in half-open state."""


# ════════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER (SYNC)
# ════════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Circuit breaker for fault tolerance (sync).
    
    Example:
        >>> breaker = CircuitBreaker("api_service")
        >>> try:
        ...     result = breaker.call(risky_function, arg1, arg2)
        ... except CircuitBreakerOpenError:
        ...     # Circuit is open, use fallback
        ...     result = fallback_function()
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration (uses defaults if None)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        # Check state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {self.last_failure_time}"
                )
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return False
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' CLOSED")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during half-open, open again
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' OPEN (failed in half-open)")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN "
                    f"(failures: {self.failure_count})"
                )
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC CIRCUIT BREAKER
# ════════════════════════════════════════════════════════════════════════════════

class AsyncCircuitBreaker:
    """
    Circuit breaker for fault tolerance (async).
    
    Example:
        >>> breaker = AsyncCircuitBreaker("api_service")
        >>> try:
        ...     result = await breaker.call(async_risky_function, arg1)
        ... except CircuitBreakerOpenError:
        ...     result = await fallback_function()
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize async circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration (uses defaults if None)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.lock = asyncio.Lock()
        self.half_open_semaphore = asyncio.Semaphore(
            self.config.half_open_max_concurrent
        )
    
    async def call(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> T:
        """
        Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        async with self.lock:
            # Check state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Last failure: {self.last_failure_time}"
                    )
        
        # Use semaphore in half-open state
        if self.state == CircuitState.HALF_OPEN:
            async with self.half_open_semaphore:
                return await self._execute(func, *args, **kwargs)
        else:
            return await self._execute(func, *args, **kwargs)
    
    async def _execute(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> T:
        """Execute function and handle state transitions."""
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            async with self.lock:
                self._on_success()
            return result
        except self.config.expected_exception as e:
            async with self.lock:
                self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return False
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker '{self.name}' OPEN (failed in half-open)")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN "
                    f"(failures: {self.failure_count})"
                )
    
    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN


# ════════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ════════════════════════════════════════════════════════════════════════════════

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "CircuitState",
    # Config
    "CircuitBreakerConfig",
    # Circuit breakers
    "CircuitBreaker",
    "AsyncCircuitBreaker",
    # Exceptions
    "CircuitBreakerOpenError",
]













