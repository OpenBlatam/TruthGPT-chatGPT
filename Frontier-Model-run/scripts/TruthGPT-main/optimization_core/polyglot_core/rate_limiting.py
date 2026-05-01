"""
Rate limiting utilities for polyglot_core.

Provides rate limiting and throttling capabilities.
"""

from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import time
import threading


@dataclass
class RateLimit:
    """Rate limit configuration."""
    max_requests: int
    time_window_seconds: float
    name: str = "default"


class RateLimiter:
    """
    Rate limiter for polyglot_core operations.
    
    Implements token bucket algorithm.
    """
    
    def __init__(self, rate_limit: RateLimit):
        """
        Initialize rate limiter.
        
        Args:
            rate_limit: Rate limit configuration
        """
        self.rate_limit = rate_limit
        self._requests = deque()
        self._lock = threading.Lock() if threading else None
    
    def acquire(self, wait: bool = True) -> bool:
        """
        Acquire permission to proceed.
        
        Args:
            wait: Whether to wait if rate limit exceeded
            
        Returns:
            True if permission granted, False otherwise
        """
        now = time.time()
        window_start = now - self.rate_limit.time_window_seconds
        
        if self._lock:
            with self._lock:
                # Remove old requests
                while self._requests and self._requests[0] < window_start:
                    self._requests.popleft()
                
                # Check if limit exceeded
                if len(self._requests) >= self.rate_limit.max_requests:
                    if wait:
                        # Wait until oldest request expires
                        if self._requests:
                            sleep_time = self._requests[0] + self.rate_limit.time_window_seconds - now
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                                return self.acquire(wait=False)
                    return False
                
                # Add current request
                self._requests.append(now)
                return True
        else:
            # Non-threaded version
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()
            
            if len(self._requests) >= self.rate_limit.max_requests:
                if wait and self._requests:
                    sleep_time = self._requests[0] + self.rate_limit.time_window_seconds - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        return self.acquire(wait=False)
                return False
            
            self._requests.append(now)
            return True
    
    def reset(self):
        """Reset rate limiter."""
        if self._lock:
            with self._lock:
                self._requests.clear()
        else:
            self._requests.clear()
    
    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - self.rate_limit.time_window_seconds
        
        if self._lock:
            with self._lock:
                while self._requests and self._requests[0] < window_start:
                    self._requests.popleft()
                return max(0, self.rate_limit.max_requests - len(self._requests))
        else:
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()
            return max(0, self.rate_limit.max_requests - len(self._requests))


class RateLimitDecorator:
    """Decorator for rate limiting functions."""
    
    def __init__(self, rate_limit: RateLimit):
        self.rate_limiter = RateLimiter(rate_limit)
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if not self.rate_limiter.acquire():
                raise RateLimitExceeded(f"Rate limit exceeded for {self.rate_limiter.rate_limit.name}")
            return func(*args, **kwargs)
        return wrapper


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


def rate_limit(max_requests: int, time_window_seconds: float, name: str = "default"):
    """
    Decorator for rate limiting.
    
    Example:
        @rate_limit(max_requests=100, time_window_seconds=60)
        def my_function():
            ...
    """
    limit = RateLimit(max_requests=max_requests, time_window_seconds=time_window_seconds, name=name)
    return RateLimitDecorator(limit)













