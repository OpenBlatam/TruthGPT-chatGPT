"""
TruthGPT Go Core Python Client

This module provides a Python client for interacting with Go Core services.
Supports HTTP REST API, gRPC, and caching operations.

Usage:
    from truthgpt_go import GoClient
    
    client = GoClient(
        inference_url="http://localhost:8080",
        cache_url="http://localhost:8081",
        grpc_url="localhost:50051"
    )
    
    # HTTP Inference
    result = client.predict("Hello, world!", max_tokens=100)
    
    # Cache operations
    client.cache_put("key1", b"value1")
    value = client.cache_get("key1")
    
    # gRPC streaming
    for token in client.stream_predict("Generate a story"):
        print(token, end="")
"""

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Union
from dataclasses import dataclass, field
from urllib.parse import urljoin

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False


@dataclass
class InferenceRequest:
    """Request for inference."""
    input_text: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class InferenceResponse:
    """Response from inference."""
    output_text: str
    tokens_used: int
    latency_ms: float
    model_id: str


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int
    misses: int
    puts: int
    deletes: int
    bytes_written: int
    bytes_read: int
    hit_rate: float


class GoClientError(Exception):
    """Base exception for Go client errors."""
    pass


class ConnectionError(GoClientError):
    """Connection error."""
    pass


class ResponseError(GoClientError):
    """Response error."""
    pass


class GoClient:
    """
    Python client for TruthGPT Go Core services.
    
    Provides access to:
    - HTTP REST API for inference
    - gRPC for streaming inference
    - Cache operations
    
    Args:
        inference_url: URL of the inference server (e.g., "http://localhost:8080")
        cache_url: URL of the cache service (e.g., "http://localhost:8081")
        grpc_url: gRPC endpoint (e.g., "localhost:50051")
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        inference_url: str = "http://localhost:8080",
        cache_url: str = "http://localhost:8081",
        grpc_url: str = "localhost:50051",
        timeout: float = 30.0
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package is required. Install with: pip install requests")
        
        self.inference_url = inference_url.rstrip("/")
        self.cache_url = cache_url.rstrip("/")
        self.grpc_url = grpc_url
        self.timeout = timeout
        
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "TruthGPT-Go-Client/1.0"
        })
        
        self._grpc_channel = None
        self._inference_stub = None
    
    # ════════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ════════════════════════════════════════════════════════════════════════════
    
    def _build_inference_url(self, path: str) -> str:
        """
        Build inference API URL.
        
        Args:
            path: API path (e.g., "inference", "inference/batch")
        
        Returns:
            Full URL
        """
        return f"{self.inference_url}/api/v1/{path}"
    
    def _build_cache_url(self, path: str = "") -> str:
        """
        Build cache API URL.
        
        Args:
            path: API path (e.g., "cache/key1", "cache/stats")
        
        Returns:
            Full URL
        """
        if path:
            return f"{self.cache_url}/api/v1/{path}"
        return f"{self.cache_url}/api/v1/cache"
    
    def _make_request(
        self,
        method: str,
        url: str,
        service_name: str = "service",
        timeout: Optional[float] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with error handling.
        
        Args:
            method: HTTP method (get, post, put, delete)
            url: Request URL
            service_name: Service name for error messages
            timeout: Request timeout (uses self.timeout if None)
            **kwargs: Additional request arguments
        
        Returns:
            Response object
        
        Raises:
            ConnectionError: If request fails
        """
        timeout = timeout or self.timeout
        
        try:
            response = getattr(self._session, method.lower())(
                url, timeout=timeout, **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to {service_name}: {e}"
            )
    
    def _get_json_response(
        self,
        method: str,
        url: str,
        service_name: str = "service",
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request and return JSON response.
        
        Args:
            method: HTTP method
            url: Request URL
            service_name: Service name for error messages
            timeout: Request timeout
            **kwargs: Additional request arguments
        
        Returns:
            JSON response as dictionary
        
        Raises:
            ConnectionError: If request fails
        """
        response = self._make_request(method, url, service_name, timeout, **kwargs)
        return response.json()
    
    def _parse_inference_response(self, data: Dict[str, Any]) -> InferenceResponse:
        """
        Parse inference response from JSON data.
        
        Args:
            data: JSON response data
        
        Returns:
            InferenceResponse object
        """
        return InferenceResponse(
            output_text=data.get("output", ""),
            tokens_used=data.get("tokens_used", 0),
            latency_ms=data.get("latency_ms", 0),
            model_id=data.get("model_id", "")
        )
    
    def _parse_cache_stats(self, data: Dict[str, Any]) -> CacheStats:
        """
        Parse cache stats from JSON data.
        
        Args:
            data: JSON response data
        
        Returns:
            CacheStats object
        """
        return CacheStats(
            hits=data.get("hits", 0),
            misses=data.get("misses", 0),
            puts=data.get("puts", 0),
            deletes=data.get("deletes", 0),
            bytes_written=data.get("bytes_written", 0),
            bytes_read=data.get("bytes_read", 0),
            hit_rate=data.get("hit_rate", 0.0)
        )
    
    def _check_service_health(
        self,
        url: str,
        service_name: str,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """
        Check health of a service.
        
        Args:
            url: Health check URL
            service_name: Name of the service
            timeout: Request timeout
        
        Returns:
            Health status dictionary
        """
        try:
            response = self._session.get(url, timeout=timeout)
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "code": response.status_code
            }
        except requests.exceptions.RequestException:
            return {"status": "unreachable"}
    
    # ════════════════════════════════════════════════════════════════════════════
    # HTTP INFERENCE
    # ════════════════════════════════════════════════════════════════════════════
    
    def predict(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> InferenceResponse:
        """
        Run inference on input text.
        
        Args:
            input_text: Input text to process
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional options
        
        Returns:
            InferenceResponse with generated text
        """
        url = f"{self.inference_url}/api/v1/inference"
        
        payload = {
            "input": input_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            **kwargs
        }
        
        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            return InferenceResponse(
                output_text=data.get("output", ""),
                tokens_used=data.get("tokens_used", 0),
                latency_ms=data.get("latency_ms", 0),
                model_id=data.get("model_id", "")
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to inference server: {e}")
    
    def batch_predict(
        self,
        inputs: List[str],
        max_tokens: int = 100,
        **kwargs
    ) -> List[InferenceResponse]:
        """
        Run batch inference on multiple inputs.
        
        Args:
            inputs: List of input texts
            max_tokens: Maximum tokens to generate per input
            **kwargs: Additional options
        
        Returns:
            List of InferenceResponse objects
        """
        url = f"{self.inference_url}/api/v1/inference/batch"
        
        payload = [
            {"input": text, "max_tokens": max_tokens, **kwargs}
            for text in inputs
        ]
        
        try:
            response = self._session.post(url, json=payload, timeout=self.timeout * 2)
            response.raise_for_status()
            data = response.json()
            
            return [
                InferenceResponse(
                    output_text=r.get("output", ""),
                    tokens_used=r.get("tokens_used", 0),
                    latency_ms=r.get("latency_ms", 0),
                    model_id=r.get("model_id", "")
                )
                for r in data.get("responses", [])
            ]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to inference server: {e}")
    
    def stream_predict_http(
        self,
        input_text: str,
        max_tokens: int = 100,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream inference results via HTTP Server-Sent Events.
        
        Args:
            input_text: Input text to process
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options
        
        Yields:
            Generated tokens one at a time
        """
        url = f"{self.inference_url}/api/v1/inference/stream"
        
        payload = {
            "input": input_text,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = self._session.post(
                url, 
                json=payload, 
                stream=True, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield data
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to inference server: {e}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # CACHE OPERATIONS
    # ════════════════════════════════════════════════════════════════════════════
    
    def cache_get(self, key: str) -> Optional[bytes]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        url = f"{self.cache_url}/api/v1/cache/{key}"
        
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if data.get("found"):
                return data.get("value", "").encode()
            return None
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to cache service: {e}")
    
    def cache_put(self, key: str, value: Union[bytes, str]) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        
        Returns:
            True if successful
        """
        url = f"{self.cache_url}/api/v1/cache/{key}"
        
        if isinstance(value, str):
            value = value.encode()
        
        try:
            response = self._session.put(url, data=value, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("stored", False)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to cache service: {e}")
    
    def cache_delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted
        """
        url = f"{self.cache_url}/api/v1/cache/{key}"
        
        try:
            response = self._session.delete(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("deleted", False)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to cache service: {e}")
    
    def cache_stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            CacheStats object
        """
        url = f"{self.cache_url}/api/v1/cache/stats"
        
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            return CacheStats(
                hits=data.get("hits", 0),
                misses=data.get("misses", 0),
                puts=data.get("puts", 0),
                deletes=data.get("deletes", 0),
                bytes_written=data.get("bytes_written", 0),
                bytes_read=data.get("bytes_read", 0),
                hit_rate=data.get("hit_rate", 0.0)
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to cache service: {e}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # HEALTH CHECKS
    # ════════════════════════════════════════════════════════════════════════════
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of services.
        
        Returns:
            Dictionary with health status
        """
        result = {}
        
        # Check inference server
        try:
            response = self._session.get(
                f"{self.inference_url}/health",
                timeout=5.0
            )
            result["inference_server"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "code": response.status_code
            }
        except requests.exceptions.RequestException:
            result["inference_server"] = {"status": "unreachable"}
        
        # Check cache service
        try:
            response = self._session.get(
                f"{self.cache_url}/health",
                timeout=5.0
            )
            result["cache_service"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "code": response.status_code
            }
        except requests.exceptions.RequestException:
            result["cache_service"] = {"status": "unreachable"}
        
        return result
    
    # ════════════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGER
    # ════════════════════════════════════════════════════════════════════════════
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the client and release resources."""
        self._session.close()
        if self._grpc_channel:
            self._grpc_channel.close()


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def create_client(
    inference_url: str = "http://localhost:8080",
    **kwargs
) -> GoClient:
    """Create a Go client with default settings."""
    return GoClient(inference_url=inference_url, **kwargs)


if __name__ == "__main__":
    # Simple test
    client = create_client()
    
    print("Health check:", client.health_check())
    
    try:
        result = client.predict("Hello, TruthGPT!")
        print(f"Inference result: {result.output_text}")
    except GoClientError as e:
        print(f"Error: {e}")






