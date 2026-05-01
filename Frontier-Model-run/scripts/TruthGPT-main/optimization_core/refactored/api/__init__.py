"""
API Layer
=========

REST API for external integration with:
- FastAPI-based endpoints
- WebSocket support
- Authentication and authorization
- Rate limiting
- API documentation
"""

from .server import APIServer
from .endpoints import optimization_router, monitoring_router, config_router
from .websocket import WebSocketManager
from .auth import AuthenticationManager
from .middleware import RateLimitMiddleware, CORSMiddleware

__all__ = [
    'APIServer',
    'optimization_router',
    'monitoring_router', 
    'config_router',
    'WebSocketManager',
    'AuthenticationManager',
    'RateLimitMiddleware',
    'CORSMiddleware'
]



