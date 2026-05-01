"""
REST API utilities for polyglot_core.

Provides REST API endpoints and FastAPI integration.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class APIEndpoint:
    """API endpoint definition."""
    path: str
    method: str  # GET, POST, PUT, DELETE
    handler: callable
    description: str = ""
    tags: List[str] = field(default_factory=list)
    response_model: Optional[type] = None


class APIRouter:
    """
    API router for polyglot_core.
    
    Manages REST API endpoints.
    """
    
    def __init__(self, prefix: str = "/api/v1"):
        """
        Initialize API router.
        
        Args:
            prefix: API prefix
        """
        self.prefix = prefix
        self._endpoints: List[APIEndpoint] = []
    
    def register(
        self,
        path: str,
        method: str = "GET",
        description: str = "",
        tags: Optional[List[str]] = None,
        response_model: Optional[type] = None
    ):
        """
        Register endpoint decorator.
        
        Args:
            path: Endpoint path
            method: HTTP method
            description: Endpoint description
            tags: Endpoint tags
            response_model: Response model
        """
        def decorator(handler: callable):
            endpoint = APIEndpoint(
                path=path,
                method=method,
                handler=handler,
                description=description,
                tags=tags or [],
                response_model=response_model
            )
            self._endpoints.append(endpoint)
            return handler
        return decorator
    
    def get_endpoints(self) -> List[APIEndpoint]:
        """Get all registered endpoints."""
        return self._endpoints.copy()
    
    def create_fastapi_app(self):
        """
        Create FastAPI app from registered endpoints.
        
        Returns:
            FastAPI app instance
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
            
            app = FastAPI(
                title="Polyglot Core API",
                description="REST API for polyglot_core",
                version="2.0.0"
            )
            
            for endpoint in self._endpoints:
                full_path = f"{self.prefix}{endpoint.path}"
                
                # Register route
                if endpoint.method == "GET":
                    app.get(full_path, tags=endpoint.tags, summary=endpoint.description)(
                        endpoint.handler
                    )
                elif endpoint.method == "POST":
                    app.post(full_path, tags=endpoint.tags, summary=endpoint.description)(
                        endpoint.handler
                    )
                elif endpoint.method == "PUT":
                    app.put(full_path, tags=endpoint.tags, summary=endpoint.description)(
                        endpoint.handler
                    )
                elif endpoint.method == "DELETE":
                    app.delete(full_path, tags=endpoint.tags, summary=endpoint.description)(
                        endpoint.handler
                    )
            
            return app
        except ImportError:
            raise ImportError("FastAPI is required for API functionality. Install with: pip install fastapi")


# Global API router
_global_api_router = APIRouter()


def get_api_router() -> APIRouter:
    """Get global API router."""
    return _global_api_router


def register_endpoint(path: str, method: str = "GET", **kwargs):
    """Convenience decorator to register endpoint."""
    return _global_api_router.register(path, method, **kwargs)













