"""
Rate limiting and throttling utilities for optimization_core.

Provides reusable rate limiting and throttling implementations.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Callable, Dict, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ════════════════════════════════════════════════════════════════════════════════
# RATE LIMITER (SYNC)
# ════════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Rate limiter with sliding window.
    
    Example:
        >>> limiter = RateLimiter(max_requests=100, time_window=60.0)
        >>> if limiter.is_allowed("user123"):
        ...     process_request()
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        time_window: float = 60.0
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = Lock()
    
    def is_allowed(
        self,
        identifier: str = "default"
    ) -> bool:
        """
        Check if request is allowed.
        
        Args:
            identifier: Request identifier (e.g., user ID, IP)
        
        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            cutoff = now - self.time_window
            
            # Remove old requests
            requests = self.requests[identifier]
            while requests and requests[0] < cutoff:
                requests.popleft()
            
            # Check limit
            if len(requests) >= self.max_requests:
                return False
            
            # Add request
            requests.append(now)
            return True
    
    def get_remaining(
        self,
        identifier: str = "default"
    ) -> int:
        """
        Get remaining requests in current window.
        
        Args:
            identifier: Request identifier
        
        Returns:
            Number of remaining requests
        """
        with self.lock:
            now = time.time()
            cutoff = now - self.time_window
            
            requests = self.requests[identifier]
            while requests and requests[0] < cutoff:
                requests.popleft()
            
            return max(0, self.max_requests - len(requests))
    
    def reset(self, identifier: Optional[str] = None) -> None:
        """
        Reset rate limiter for identifier or all.
        
        Args:
            identifier: Identifier to reset (None for all)
        """
        with self.lock:
            if identifier:
                self.requests[identifier].clear()
            else:
                self.requests.clear()


# ════════════════════════════════════════════════════════════════════════════════
# ASYNC RATE LIMITER
# ════════════════════════════════════════════════════════════════════════════════

class AsyncRateLimiter:
    """
    Async rate limiter with sliding window.
    
    Example:
        >>> limiter = AsyncRateLimiter(max_requests=100, time_window=60.0)
        >>> if await limiter.is_allowed("user123"):
        ...     await process_request()
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        time_window: float = 60.0
    ):
        """
        Initialize async rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = asyncio.Lock()
    
    async def is_allowed(
        self,
        identifier: str = "default"
    ) -> bool:
        """
        Check if request is allowed (async).
        
        Args:
            identifier: Request identifier
        
        Returns:
            True if allowed, False if rate limited
        """
        async with self.lock:
            now = time.time()
            cutoff = now - self.time_window
            
            # Remove old requests
            requests = self.requests[identifier]
            while requests and requests[0] < cutoff:
                requests.popleft()
            
            # Check limit
            if len(requests) >= self.max_requests:
                return False
            
            # Add request
            requests.append(now)
            return True
    
    async def get_remaining(
        self,
        identifier: str = "default"
    ) -> int:
        """
        Get remaining requests in current window (async).
        
        Args:
            identifier: Request identifier
        
        Returns:
            Number of remaining requests
        """
        async with self.lock:
            now = time.time()
            cutoff = now - self.time_window
            
            requests = self.requests[identifier]
            while requests and requests[0] < cutoff:
                requests.popleft()
            
            return max(0, self.max_requests - len(requests))
    
    async def reset(self, identifier: Optional[str] = None) -> None:
        """
        Reset rate limiter for identifier or all (async).
        
        Args:
            identifier: Identifier to reset (None for all)
        """
        async with self.lock:
            if identifier:
                self.requests[identifier].clear()
            else:
                self.requests.clear()


# ════════════════════════════════════════════════════════════════════════════════
# THROTTLER (WAITS INSTEAD OF REJECTING)
# ════════════════════════════════════════════════════════════════════════════════

class Throttler:
    """
    Throttler that waits instead of rejecting requests.
    
    Example:
        >>> throttler = Throttler(max_calls=10, period=1.0)
        >>> await throttler.acquire()  # Waits if needed
    """
    
    def __init__(
        self,
        max_calls: int = 10,
        period: float = 1.0
    ):
        """
        Initialize throttler.
        
        Args:
            max_calls: Maximum calls per period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: deque = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire throttle permission (waits if needed).
        
        Example:
            >>> await throttler.acquire()
        """
        async with self.lock:
            now = time.time()
            cutoff = now - self.period
            
            # Remove old calls
            while self.calls and self.calls[0] < cutoff:
                self.calls.popleft()
            
            # Wait if at limit
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    # Remove expired calls after sleep
                    now = time.time()
                    cutoff = now - self.period
                    while self.calls and self.calls[0] < cutoff:
                        self.calls.popleft()
            
            # Record this call
            self.calls.append(time.time())
    
    def reset(self) -> None:
        """Reset throttler."""
        self.calls.clear()


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Rate limiters
    "RateLimiter",
    "AsyncRateLimiter",
    # Throttler
    "Throttler",
]













