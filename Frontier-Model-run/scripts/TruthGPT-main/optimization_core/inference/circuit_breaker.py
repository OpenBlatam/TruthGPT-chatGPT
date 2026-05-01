"""
⚡ Circuit Breaker Pattern
Resilient circuit breaker with exponential backoff and half-open state management
"""

import time
from enum import Enum
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass, field
from threading import Lock
from collections import defaultdict


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing from half-open
    timeout_seconds: int = 60  # Time before attempting half-open
    half_open_max_calls: int = 3  # Max calls in half-open state


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitState
    failures: int
    successes: int
    total_calls: int
    opened_at: Optional[float] = None
    last_failure_at: Optional[float] = None
    last_success_at: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._total_calls = 0
        self._opened_at: Optional[float] = None
        self._last_failure_at: Optional[float] = None
        self._last_success_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = Lock()
    
    def _should_transition_to_half_open(self) -> bool:
        """Check if should transition to half-open"""
        if self._state != CircuitState.OPEN:
            return False
        
        if not self._opened_at:
            return False
        
        elapsed = time.time() - self._opened_at
        return elapsed >= self.config.timeout_seconds
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._successes = 0
    
    def _transition_to_open(self):
        """Transition to open state"""
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        self._half_open_calls = 0
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = None
        self._half_open_calls = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            Result of function call
        
        Raises:
            Exception: CircuitBreakerOpenError if circuit is open
        """
        with self._lock:
            # Check if should transition to half-open
            if self._should_transition_to_half_open():
                self._transition_to_half_open()
            
            # Check if circuit is open
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. "
                    f"Retry after {int(self.config.timeout_seconds - (time.time() - self._opened_at))} seconds"
                )
            
            # Check half-open call limit
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    # Too many calls in half-open, go back to open
                    self._transition_to_open()
                    raise CircuitBreakerOpenError("Circuit breaker exceeded half-open call limit")
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self):
        """Record successful call"""
        with self._lock:
            self._total_calls += 1
            self._last_success_at = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                self._successes += 1
                
                # Check if should close
                if self._successes >= self.config.success_threshold:
                    self._transition_to_closed()
            
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failures = 0
    
    def _record_failure(self):
        """Record failed call"""
        with self._lock:
            self._total_calls += 1
            self._failures += 1
            self._last_failure_at = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open opens circuit
                self._transition_to_open()
            
            elif self._state == CircuitState.CLOSED:
                # Check if should open
                if self._failures >= self.config.failure_threshold:
                    self._transition_to_open()
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        with self._lock:
            return CircuitBreakerStats(
                state=self._state,
                failures=self._failures,
                successes=self._successes,
                total_calls=self._total_calls,
                opened_at=self._opened_at,
                last_failure_at=self._last_failure_at,
                last_success_at=self._last_success_at
            )
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self._transition_to_closed()
            self._total_calls = 0
            self._last_failure_at = None
            self._last_success_at = None


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreakerManager:
    """Manager for multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = defaultdict(
            lambda: CircuitBreaker()
        )
        self._lock = Lock()
    
    def get_breaker(
        self,
        key: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for key"""
        with self._lock:
            if key not in self._breakers:
                self._breakers[key] = CircuitBreaker(config)
            return self._breakers[key]
    
    def reset_breaker(self, key: str):
        """Reset circuit breaker for key"""
        with self._lock:
            if key in self._breakers:
                self._breakers[key].reset()
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get stats for all circuit breakers"""
        with self._lock:
            return {
                key: breaker.get_stats()
                for key, breaker in self._breakers.items()
            }


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()



