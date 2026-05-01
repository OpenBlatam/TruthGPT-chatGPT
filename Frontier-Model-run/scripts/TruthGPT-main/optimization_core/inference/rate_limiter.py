"""
🚦 Advanced Rate Limiter
Sliding window rate limiting with per-endpoint and per-client support
"""

import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple
from threading import Lock
import hashlib


class SlidingWindowRateLimiter:
    """Sliding window rate limiter"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_allowance: int = 10
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.burst_allowance = burst_allowance
        self._windows: Dict[str, deque] = defaultdict(lambda: deque())
        self._hour_windows: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = Lock()
    
    def _clean_old_entries(self, window: deque, window_seconds: int):
        """Remove old entries from window"""
        cutoff = time.time() - window_seconds
        while window and window[0] < cutoff:
            window.popleft()
    
    def check_limit(
        self,
        key: str,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed
        
        Returns:
            (allowed, retry_after_seconds)
        """
        with self._lock:
            rpm_limit = requests_per_minute or self.rpm
            rph_limit = requests_per_hour or self.rph
            
            now = time.time()
            
            # Clean minute window
            minute_window = self._windows[key]
            self._clean_old_entries(minute_window, 60)
            
            # Clean hour window
            hour_window = self._hour_windows[key]
            self._clean_old_entries(hour_window, 3600)
            
            # Check minute limit (with burst allowance)
            minute_count = len(minute_window)
            if minute_count >= rpm_limit + self.burst_allowance:
                # Calculate retry after
                if minute_window:
                    oldest = minute_window[0]
                    retry_after = int(60 - (now - oldest)) + 1
                else:
                    retry_after = 1
                return False, retry_after
            
            # Check hour limit
            hour_count = len(hour_window)
            if hour_count >= rph_limit:
                if hour_window:
                    oldest = hour_window[0]
                    retry_after = int(3600 - (now - oldest)) + 1
                else:
                    retry_after = 3600
                return False, retry_after
            
            # Record request
            minute_window.append(now)
            hour_window.append(now)
            
            return True, None
    
    def reset(self, key: str):
        """Reset limits for a key"""
        with self._lock:
            self._windows.pop(key, None)
            self._hour_windows.pop(key, None)
    
    def get_stats(self, key: str) -> Dict[str, int]:
        """Get current stats for a key"""
        with self._lock:
            minute_window = self._windows[key]
            hour_window = self._hour_windows[key]
            
            self._clean_old_entries(minute_window, 60)
            self._clean_old_entries(hour_window, 3600)
            
            return {
                "requests_last_minute": len(minute_window),
                "requests_last_hour": len(hour_window),
                "rpm_limit": self.rpm,
                "rph_limit": self.rph
            }


class RateLimiterManager:
    """Manager for multiple rate limiters"""
    
    def __init__(self):
        self._limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self._default_limiter = SlidingWindowRateLimiter()
        self._endpoint_limits: Dict[str, Dict[str, int]] = {}
        self._lock = Lock()
    
    def configure_endpoint(
        self,
        endpoint: str,
        requests_per_minute: int,
        requests_per_hour: Optional[int] = None,
        burst_allowance: int = 10
    ):
        """Configure rate limit for specific endpoint"""
        with self._lock:
            self._endpoint_limits[endpoint] = {
                "rpm": requests_per_minute,
                "rph": requests_per_hour or requests_per_minute * 60,
                "burst": burst_allowance
            }
            
            # Create limiter for endpoint
            self._limiters[endpoint] = SlidingWindowRateLimiter(
                requests_per_minute=requests_per_minute,
                requests_per_hour=requests_per_hour or requests_per_minute * 60,
                burst_allowance=burst_allowance
            )
    
    def check_rate_limit(
        self,
        client_key: str,
        endpoint: Optional[str] = None
    ) -> Tuple[bool, Optional[int]]:
        """
        Check rate limit for client and endpoint
        
        Args:
            client_key: Unique client identifier (IP, API key, etc.)
            endpoint: Optional endpoint path for endpoint-specific limits
        
        Returns:
            (allowed, retry_after_seconds)
        """
        with self._lock:
            # Check endpoint-specific limit first
            if endpoint and endpoint in self._limiters:
                limiter = self._limiters[endpoint]
                endpoint_key = f"{endpoint}:{client_key}"
                allowed, retry_after = limiter.check_limit(endpoint_key)
                
                if not allowed:
                    return False, retry_after
            
            # Check global limit
            return self._default_limiter.check_limit(client_key)
    
    def reset_client(self, client_key: str):
        """Reset limits for a client"""
        with self._lock:
            self._default_limiter.reset(client_key)
            for limiter in self._limiters.values():
                limiter.reset(client_key)
    
    def get_client_stats(self, client_key: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limit stats for client"""
        with self._lock:
            stats = {
                "global": self._default_limiter.get_stats(client_key)
            }
            
            if endpoint and endpoint in self._limiters:
                endpoint_key = f"{endpoint}:{client_key}"
                stats["endpoint"] = {
                    endpoint: self._limiters[endpoint].get_stats(endpoint_key)
                }
            
            return stats


# Global rate limiter instance
rate_limiter = RateLimiterManager()



