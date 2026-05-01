"""
TruthGPT Polyglot Core - Circuit Breaker
========================================

Resilience pattern implementation for backend failover.

Features:
- Three states: Closed, Open, Half-Open
- Automatic state transitions
- Configurable thresholds
- Per-backend circuit breakers
"""

import time
import threading
from dataclasses import dataclass
from typing import Callable, TypeVar, Optional, Any, Dict
from enum import Enum
from functools import wraps


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout: float = 30.0               # Seconds before trying half-open
    half_open_max_calls: int = 3        # Max calls in half-open state
    failure_rate_threshold: float = 0.5  # Alternative: failure rate based
    minimum_calls: int = 10             # Minimum calls before rate check


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


class CircuitBreaker:
    """
    Circuit breaker for resilient backend calls.
    
    Example:
        >>> cb = CircuitBreaker("rust_backend")
        >>> 
        >>> @cb.protected
        >>> def call_rust():
        ...     return rust_core.process()
        >>> 
        >>> result = cb.call(call_rust)
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._stats = CircuitStats()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        with self._lock:
            return CircuitStats(
                total_calls=self._stats.total_calls,
                successful_calls=self._stats.successful_calls,
                failed_calls=self._stats.failed_calls,
                rejected_calls=self._stats.rejected_calls,
                state_changes=self._stats.state_changes,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time
            )
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False
            
            # Half-open: allow limited requests
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = time.time()
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def record_rejection(self):
        """Record a rejected call (circuit open)."""
        with self._lock:
            self._stats.rejected_calls += 1
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state (must hold lock)."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1
        
        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        
        # Callback outside lock would cause deadlock issues
        if self.on_state_change:
            # Schedule callback for later execution
            threading.Thread(
                target=self.on_state_change,
                args=(old_state, new_state),
                daemon=True
            ).start()
    
    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
    
    def call(self, func: Callable[[], Any], fallback: Optional[Callable[[], Any]] = None) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to call
            fallback: Optional fallback function if circuit is open
            
        Returns:
            Result of func or fallback
            
        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        if not self.allow_request():
            self.record_rejection()
            if fallback:
                return fallback()
            raise CircuitOpenError(f"Circuit {self.name} is open")
        
        try:
            result = func()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise
    
    def protected(self, func: Callable) -> Callable:
        """
        Decorator to protect a function with circuit breaker.
        
        Example:
            >>> @cb.protected
            >>> def my_function():
            ...     return call_backend()
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(lambda: func(*args, **kwargs))
        return wrapper


class CircuitOpenError(Exception):
    """Exception raised when circuit is open."""
    pass


T = TypeVar("T")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Example:
        >>> registry = CircuitBreakerRegistry()
        >>> rust_cb = registry.get_or_create("rust_backend")
        >>> cpp_cb = registry.get_or_create("cpp_backend")
        >>> registry.get_all_stats()
    """
    
    def __init__(self, default_config: Optional[CircuitBreakerConfig] = None):
        self.default_config = default_config or CircuitBreakerConfig()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name,
                    config or self.default_config
                )
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        with self._lock:
            return {
                name: {
                    "state": cb.state.value,
                    "total_calls": cb.stats.total_calls,
                    "successful_calls": cb.stats.successful_calls,
                    "failed_calls": cb.stats.failed_calls,
                    "rejected_calls": cb.stats.rejected_calls,
                    "failure_rate": cb.stats.failure_rate,
                    "state_changes": cb.stats.state_changes
                }
                for name, cb in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()


# Default registry
default_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker from the default registry."""
    return default_registry.get_or_create(name)


# Pre-configured circuit breakers for TruthGPT backends
class BackendCircuitBreakers:
    """Pre-configured circuit breakers for TruthGPT backends."""
    
    def __init__(self):
        self.registry = CircuitBreakerRegistry()
        
        # Rust backend - generally stable
        self.rust = self.registry.get_or_create(
            "rust_core",
            CircuitBreakerConfig(
                failure_threshold=5,
                timeout=30.0,
                success_threshold=3
            )
        )
        
        # C++ backend - might have GPU issues
        self.cpp = self.registry.get_or_create(
            "cpp_core",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout=60.0,
                success_threshold=2
            )
        )
        
        # Go backend - network issues
        self.go = self.registry.get_or_create(
            "go_core",
            CircuitBreakerConfig(
                failure_threshold=5,
                timeout=15.0,
                success_threshold=3
            )
        )
        
        # Julia backend - scientific computing
        self.julia = self.registry.get_or_create(
            "julia_core",
            CircuitBreakerConfig(
                failure_threshold=5,
                timeout=30.0,
                success_threshold=3
            )
        )
        
        # Scala backend - distributed processing
        self.scala = self.registry.get_or_create(
            "scala_core",
            CircuitBreakerConfig(
                failure_threshold=5,
                timeout=45.0,
                success_threshold=3
            )
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all backend circuit breaker stats."""
        return self.registry.get_all_stats()
    
    def reset_all(self):
        """Reset all backend circuit breakers."""
        self.registry.reset_all()





