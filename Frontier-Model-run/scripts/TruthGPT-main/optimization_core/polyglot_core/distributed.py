"""
Distributed backend support via Go services.

Provides clients for Go-based gRPC/HTTP services:
- Inference Server
- Cache Service
- Coordinator
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    host: str = "localhost"
    http_port: int = 8080
    grpc_port: int = 50051
    
    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"
    
    @property
    def grpc_addr(self) -> str:
        return f"{self.host}:{self.grpc_port}"


class GoClient:
    """
    Client for Go backend services.
    
    Provides access to:
    - Inference Server (HTTP/gRPC)
    - Cache Service (HTTP/gRPC)
    - Coordinator (HTTP)
    
    Example:
        >>> client = GoClient()
        >>> result = client.predict("Hello, world!", max_tokens=100)
        >>> print(result['text'])
    """
    
    def __init__(
        self,
        inference_endpoint: Optional[ServiceEndpoint] = None,
        cache_endpoint: Optional[ServiceEndpoint] = None,
        use_grpc: bool = False
    ):
        """
        Initialize Go client.
        
        Args:
            inference_endpoint: Inference server endpoint
            cache_endpoint: Cache service endpoint
            use_grpc: Use gRPC instead of HTTP
        """
        self.inference_endpoint = inference_endpoint or ServiceEndpoint(
            host="localhost", http_port=8080, grpc_port=50051
        )
        self.cache_endpoint = cache_endpoint or ServiceEndpoint(
            host="localhost", http_port=8081, grpc_port=50052
        )
        self.use_grpc = use_grpc
        
        self._http_client = None
        self._grpc_channel = None
    
    def _get_http_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.Client(timeout=30.0)
            except ImportError:
                import urllib.request
                self._http_client = "urllib"
        return self._http_client
    
    def predict(
        self,
        text: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using inference server.
        
        Args:
            text: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Dict with 'text', 'tokens', 'latency_ms'
        """
        url = f"{self.inference_endpoint.http_url}/v1/completions"
        
        payload = {
            "prompt": text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        return self._post(url, payload)
    
    def embed(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Dict with 'embeddings' array
        """
        url = f"{self.inference_endpoint.http_url}/v1/embeddings"
        return self._post(url, {"texts": texts})
    
    def cache_get(self, layer: int, position: int, key: str = "") -> Optional[bytes]:
        """Get from cache service."""
        url = f"{self.cache_endpoint.http_url}/cache/{layer}/{position}"
        if key:
            url += f"?key={key}"
        
        try:
            result = self._get(url)
            if result.get("found"):
                import base64
                return base64.b64decode(result["data"])
        except:
            pass
        return None
    
    def cache_put(
        self,
        layer: int,
        position: int,
        data: bytes,
        key: str = ""
    ) -> bool:
        """Put to cache service."""
        import base64
        url = f"{self.cache_endpoint.http_url}/cache/{layer}/{position}"
        
        payload = {
            "data": base64.b64encode(data).decode(),
            "key": key
        }
        
        try:
            self._post(url, payload)
            return True
        except:
            return False
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all services."""
        status = {}
        
        for name, endpoint in [
            ("inference", self.inference_endpoint),
            ("cache", self.cache_endpoint)
        ]:
            try:
                result = self._get(f"{endpoint.http_url}/health")
                status[name] = result.get("status") == "ok"
            except:
                status[name] = False
        
        return status
    
    def _get(self, url: str) -> Dict[str, Any]:
        """HTTP GET request."""
        client = self._get_http_client()
        
        if client == "urllib":
            import urllib.request
            with urllib.request.urlopen(url) as resp:
                return json.loads(resp.read())
        else:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()
    
    def _post(self, url: str, data: Dict) -> Dict[str, Any]:
        """HTTP POST request."""
        client = self._get_http_client()
        
        if client == "urllib":
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        else:
            resp = client.post(url, json=data)
            resp.raise_for_status()
            return resp.json()
    
    def close(self):
        """Close connections."""
        if self._http_client and hasattr(self._http_client, 'close'):
            self._http_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class GoCacheClient:
    """
    Specialized client for Go cache service.
    
    Provides KVCache-compatible interface backed by Go service.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8081
    ):
        self.endpoint = ServiceEndpoint(host=host, http_port=port)
        self._client = GoClient(cache_endpoint=self.endpoint)
    
    def put(
        self,
        layer: int,
        position: int,
        key: Any,
        value: Any,
        tag: str = ""
    ):
        """Store in cache."""
        import numpy as np
        
        if isinstance(key, np.ndarray):
            key = key.tobytes()
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        
        data = key + b"|||" + value
        self._client.cache_put(layer, position, data, tag)
    
    def get(
        self,
        layer: int,
        position: int,
        tag: str = ""
    ) -> Optional[Dict]:
        """Retrieve from cache."""
        data = self._client.cache_get(layer, position, tag)
        if data is None:
            return None
        
        parts = data.split(b"|||")
        if len(parts) != 2:
            return None
        
        import numpy as np
        return {
            'key': np.frombuffer(parts[0], dtype=np.float32),
            'value': np.frombuffer(parts[1], dtype=np.float32)
        }
    
    def remove(self, layer: int, position: int, tag: str = "") -> bool:
        # Not implemented in basic HTTP API
        return False
    
    def contains(self, layer: int, position: int, tag: str = "") -> bool:
        return self.get(layer, position, tag) is not None
    
    def clear(self):
        # Not implemented
        pass
    
    def size(self) -> int:
        return 0  # Not tracked
    
    @property
    def hit_rate(self) -> float:
        return 0.0  # Not tracked locally


# Alias for convenience
DistributedClient = GoClient













