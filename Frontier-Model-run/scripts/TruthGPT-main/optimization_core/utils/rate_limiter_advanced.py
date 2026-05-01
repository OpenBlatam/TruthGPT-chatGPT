"""
Advanced rate limiting utilities for optimization_core.

Provides advanced rate limiting with multiple strategies.
"""
import logging
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


from pydantic import BaseModel


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""
    max_requests: int
    window_seconds: float
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_size: Optional[int] = None  # For token bucket


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.requests: deque = deque()
        self.tokens: float = config.max_requests  # For token bucket
        self.last_refill: float = time.time()
    
    def acquire(self) -> bool:
        """
        Try to acquire permission to make a request.
        
        Returns:
            True if allowed, False otherwise
        """
        now = time.time()
        
        if self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._fixed_window_check(now)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_check(now)
        elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_check(now)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def _fixed_window_check(self, now: float) -> bool:
        """Check fixed window rate limit."""
        window_start = now - (now % self.config.window_seconds)
        
        # Remove old requests
        self.requests = deque([
            req_time for req_time in self.requests
            if req_time >= window_start
        ])
        
        if len(self.requests) < self.config.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def _sliding_window_check(self, now: float) -> bool:
        """Check sliding window rate limit."""
        cutoff = now - self.config.window_seconds
        
        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        if len(self.requests) < self.config.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def _token_bucket_check(self, now: float) -> bool:
        """Check token bucket rate limit."""
        # Refill tokens
        elapsed = now - self.last_refill
        tokens_to_add = (elapsed / self.config.window_seconds) * self.config.max_requests
        self.tokens = min(
            self.config.max_requests,
            self.tokens + tokens_to_add
        )
        self.last_refill = now
        
        # Check if we have tokens
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        
        return False
    
    def wait(self):
        """Wait until a request can be made."""
        while not self.acquire():
            if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                if self.requests:
                    next_available = self.requests[0] + self.config.window_seconds
                    sleep_time = max(0, next_available - time.time())
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Statistics dictionary
        """
        now = time.time()
        
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return {
                "strategy": self.config.strategy.value,
                "tokens_available": self.tokens,
                "max_requests": self.config.max_requests,
            }
        else:
            cutoff = now - self.config.window_seconds
            recent_requests = [r for r in self.requests if r >= cutoff]
            
            return {
                "strategy": self.config.strategy.value,
                "requests_in_window": len(recent_requests),
                "max_requests": self.config.max_requests,
                "window_seconds": self.config.window_seconds,
            }


def create_rate_limiter(
    max_requests: int,
    window_seconds: float,
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
) -> AdvancedRateLimiter:
    """
    Create a rate limiter.
    
    Args:
        max_requests: Maximum requests per window
        window_seconds: Window size in seconds
        strategy: Rate limiting strategy
    
    Returns:
        Rate limiter
    """
    config = RateLimitConfig(
        max_requests=max_requests,
        window_seconds=window_seconds,
        strategy=strategy
    )
    return AdvancedRateLimiter(config)













