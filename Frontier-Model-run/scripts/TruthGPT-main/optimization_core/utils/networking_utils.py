"""
Networking utilities for optimization_core.

Provides utilities for network operations and API interactions.
"""
import logging
import time
import requests
from typing import Dict, Any, Optional, List, Callable
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIResponse(BaseModel):
    """API response data structure."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    elapsed_time: float

    @property
    def success(self) -> bool:
        """Check if request was successful."""
        return 200 <= self.status_code < 300


class APIClient:
    """Client for API interactions."""
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for API
            timeout: Request timeout in seconds
            retries: Number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
    
    def request(
        self,
        method: HTTPMethod,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make API request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            headers: Request headers
            params: Query parameters
        
        Returns:
            API response
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.retries + 1):
            try:
                start_time = time.time()
                response = self.session.request(
                    method=method.value,
                    url=url,
                    json=data,
                    headers=headers,
                    params=params,
                    timeout=self.timeout
                )
                elapsed = time.time() - start_time
                
                return APIResponse(
                    status_code=response.status_code,
                    data=response.json() if response.content else None,
                    headers=dict(response.headers),
                    elapsed_time=elapsed
                )
            except requests.exceptions.RequestException as e:
                if attempt < self.retries:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.retries + 1}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Request failed after {self.retries + 1} attempts: {e}")
                    raise
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make GET request."""
        return self.request(HTTPMethod.GET, endpoint, params=params, headers=headers)
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make POST request."""
        return self.request(HTTPMethod.POST, endpoint, data=data, headers=headers)
    
    def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make PUT request."""
        return self.request(HTTPMethod.PUT, endpoint, data=data, headers=headers)
    
    def delete(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make DELETE request."""
        return self.request(HTTPMethod.DELETE, endpoint, headers=headers)


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
    
    def acquire(self) -> bool:
        """
        Acquire permission to make a call.
        
        Returns:
            True if allowed, False otherwise
        """
        now = time.time()
        
        # Remove old calls
        self.calls = [t for t in self.calls if now - t < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False
    
    def wait(self):
        """Wait until a call can be made."""
        while not self.acquire():
            if self.calls:
                sleep_time = self.time_window - (time.time() - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)













