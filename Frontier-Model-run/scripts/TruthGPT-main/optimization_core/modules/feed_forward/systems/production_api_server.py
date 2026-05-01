"""
Production API Server for PiMoE System
FastAPI-based production server with:
- RESTful API endpoints
- WebSocket support
- Authentication and authorization
- Rate limiting
- Request validation
- Response caching
- Health checks
- Metrics and monitoring
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import traceback
import os
import signal

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
import redis
from redis.exceptions import RedisError
import jwt
from passlib.context import CryptContext
import hashlib

from .production_pimoe_system import ProductionPiMoESystem, ProductionConfig, ProductionMode

# Pydantic models for API
class PiMoERequest(BaseModel):
    """Request model for PiMoE processing."""
    input_tensor: List[List[List[float]]] = Field(..., description="Input tensor as nested list")
    attention_mask: Optional[List[List[int]]] = Field(None, description="Attention mask")
    return_comprehensive_info: bool = Field(False, description="Return comprehensive information")
    request_id: Optional[str] = Field(None, description="Custom request ID")
    
    @validator('input_tensor')
    def validate_input_tensor(cls, v):
        if not v or not v[0] or not v[0][0]:
            raise ValueError("Input tensor cannot be empty")
        return v

class PiMoEResponse(BaseModel):
    """Response model for PiMoE processing."""
    request_id: str
    output: List[List[List[float]]]
    processing_time: float
    success: bool
    comprehensive_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    uptime: float
    version: str
    system_info: Dict[str, Any]

class MetricsResponse(BaseModel):
    """Metrics response model."""
    metrics: Dict[str, Any]
    timestamp: float

class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    request_id: str

# Authentication
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Prometheus metrics
def get_or_create_metric(collector_type, name, *args, **kwargs):
    """Safely get or create a prometheus metric."""
    # Check if metric already exists in registry
    for collector in REGISTRY._collector_to_names:
        if name in REGISTRY._collector_to_names[collector] or \
           (name + '_total') in REGISTRY._collector_to_names[collector] or \
           (name + '_count') in REGISTRY._collector_to_names[collector] or \
           (name + '_sum') in REGISTRY._collector_to_names[collector] or \
           (name + '_bucket') in REGISTRY._collector_to_names[collector]:
            return collector
    return collector_type(name, *args, **kwargs)

REQUEST_COUNT = get_or_create_metric(Counter, 'pimoe_api_requests', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = get_or_create_metric(Histogram, 'pimoe_api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = get_or_create_metric(Gauge, 'pimoe_api_active_connections', 'Active connections')
MEMORY_USAGE = get_or_create_metric(Gauge, 'pimoe_api_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = get_or_create_metric(Gauge, 'pimoe_api_cpu_usage_percent', 'CPU usage percentage')

class ProductionAPIServer:
    """Production API server for PiMoE system."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.app = FastAPI(
            title="PiMoE Production API",
            description="Production API server for PiMoE token-level routing system",
            version="1.0.0",
            docs_url="/docs" if config.production_mode != ProductionMode.PRODUCTION else None,
            redoc_url="/redoc" if config.production_mode != ProductionMode.PRODUCTION else None
        )
        
        # Initialize PiMoE system
        self.pimoe_system = ProductionPiMoESystem(config)
        
        # Initialize Redis for caching
        self.redis_client = self._init_redis()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Request cache
        self.request_cache = {}
        
        # Authentication
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key")
        self.jwt_algorithm = "HS256"
        
        # Rate limiting
        self.rate_limiter = {}
        
        # Start background tasks
        @self.app.on_event("startup")
        async def startup_event():
            # Start background tasks within the event loop
            try:
                self._start_background_tasks()
            except Exception as e:
                self.pimoe_system.logger.log_error("Failed to start background tasks", e)
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis client for caching."""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()  # Test connection
            return client
        except Exception as e:
            self.pimoe_system.logger.log_warning(f"Redis connection failed: {e}")
            return None
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if self.config.production_mode != ProductionMode.PRODUCTION else ["https://yourdomain.com"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        if self.config.production_mode == ProductionMode.PRODUCTION:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
            )
        
        # Gzip middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for metrics and logging
        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            return response
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            health_status = self.pimoe_system.health_check()
            
            return HealthResponse(
                status=health_status['status'],
                timestamp=time.time(),
                uptime=health_status['uptime'],
                version="1.0.0",
                system_info=health_status
            )
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            # Update system metrics
            self._update_system_metrics()
            
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        @self.app.post("/api/v1/process", response_model=PiMoEResponse)
        async def process_request(
            request: PiMoERequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Process PiMoE request."""
            # Authenticate request
            if not await self._authenticate_request(credentials):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            
            # Rate limiting
            if not await self._check_rate_limit(credentials.credentials):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            # Generate request ID
            request_id = request.request_id or str(uuid.uuid4())
            
            # Check cache
            cache_key = self._generate_cache_key(request)
            cached_response = await self._get_from_cache(cache_key)
            if cached_response:
                return PiMoEResponse(**cached_response)
            
            try:
                # Convert input to tensor
                import torch
                input_tensor = torch.tensor(request.input_tensor, dtype=torch.float32)
                attention_mask = torch.tensor(request.attention_mask) if request.attention_mask else None
                
                # Prepare request data
                request_data = {
                    'request_id': request_id,
                    'input_tensor': input_tensor,
                    'attention_mask': attention_mask,
                    'return_comprehensive_info': request.return_comprehensive_info
                }
                
                # Process request
                response_data = self.pimoe_system.process_request(request_data)
                
                # Convert response
                response = PiMoEResponse(
                    request_id=response_data['request_id'],
                    output=response_data['output'],
                    processing_time=response_data['processing_time'],
                    success=response_data['success'],
                    comprehensive_info=response_data.get('comprehensive_info'),
                    error=response_data.get('error')
                )
                
                # Cache response
                await self._set_cache(cache_key, response.dict())
                
                return response
                
            except Exception as e:
                self.pimoe_system.logger.log_error(f"Request processing failed: {request_id}", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Request processing failed: {str(e)}"
                )
        
        @self.app.get("/api/v1/stats")
        async def get_stats():
            """Get system statistics."""
            stats = self.pimoe_system.get_production_stats()
            return stats
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication."""
            await websocket.accept()
            self.active_connections.append(websocket)
            ACTIVE_CONNECTIONS.set(len(self.active_connections))
            
            try:
                while True:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Process message
                    if message['type'] == 'process':
                        # Process PiMoE request
                        request_data = message['data']
                        response_data = self.pimoe_system.process_request(request_data)
                        
                        # Send response
                        await websocket.send_text(json.dumps({
                            'type': 'response',
                            'data': response_data,
                            'request_id': message['request_id']
                        }))
                    
                    elif message['type'] == 'ping':
                        # Respond to ping
                        await websocket.send_text(json.dumps({
                            'type': 'pong',
                            'timestamp': time.time()
                        }))
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                ACTIVE_CONNECTIONS.set(len(self.active_connections))
            except Exception as e:
                self.pimoe_system.logger.log_error("WebSocket error", e)
                await websocket.close()
    
    async def _authenticate_request(self, credentials: HTTPAuthorizationCredentials) -> bool:
        """Authenticate request using JWT."""
        try:
            token = credentials.credentials
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return True
        except jwt.InvalidTokenError:
            return False
    
    async def _check_rate_limit(self, token: str) -> bool:
        """Check rate limit for token."""
        current_time = time.time()
        window_size = 60  # 1 minute window
        max_requests = 100  # Max requests per window
        
        # Get token hash for rate limiting
        token_hash = hashlib.md5(token.encode()).hexdigest()
        
        # Clean old entries
        if token_hash in self.rate_limiter:
            self.rate_limiter[token_hash] = [
                req_time for req_time in self.rate_limiter[token_hash]
                if current_time - req_time < window_size
            ]
        else:
            self.rate_limiter[token_hash] = []
        
        # Check rate limit
        if len(self.rate_limiter[token_hash]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limiter[token_hash].append(current_time)
        return True
    
    def _generate_cache_key(self, request: PiMoERequest) -> str:
        """Generate cache key for request."""
        # Create hash of request data
        request_data = {
            'input_tensor': request.input_tensor,
            'attention_mask': request.attention_mask,
            'return_comprehensive_info': request.return_comprehensive_info
        }
        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get response from cache."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(f"pimoe:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except RedisError as e:
            self.pimoe_system.logger.log_warning(f"Redis get error: {e}")
        
        return None
    
    async def _set_cache(self, cache_key: str, response_data: Dict[str, Any], ttl: int = 3600):
        """Set response in cache."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                f"pimoe:{cache_key}",
                ttl,
                json.dumps(response_data)
            )
        except RedisError as e:
            self.pimoe_system.logger.log_warning(f"Redis set error: {e}")
    
    def _update_system_metrics(self):
        """Update system metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            MEMORY_USAGE.set(memory_info.rss)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            CPU_USAGE.set(cpu_percent)
            
        except Exception as e:
            self.pimoe_system.logger.log_warning(f"Failed to update system metrics: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks."""
        async def cleanup_task():
            """Cleanup task for rate limiter and cache."""
            while True:
                try:
                    # Cleanup rate limiter
                    current_time = time.time()
                    for token_hash in list(self.rate_limiter.keys()):
                        self.rate_limiter[token_hash] = [
                            req_time for req_time in self.rate_limiter[token_hash]
                            if current_time - req_time < 300  # 5 minutes
                        ]
                        if not self.rate_limiter[token_hash]:
                            del self.rate_limiter[token_hash]
                    
                    await asyncio.sleep(60)  # Run every minute
                except Exception as e:
                    self.pimoe_system.logger.log_error("Cleanup task error", e)
                    await asyncio.sleep(60)
        
        # Start background task
        asyncio.create_task(cleanup_task())
    
    def run(self, host: str = "0.0.0.0", port: int = 8080, **kwargs):
        """Run the API server."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if self.config.production_mode == ProductionMode.PRODUCTION else "debug",
            access_log=True,
            **kwargs
        )

def create_production_api_server(
    hidden_size: int = 512,
    num_experts: int = 8,
    production_mode: ProductionMode = ProductionMode.PRODUCTION,
    **kwargs
) -> ProductionAPIServer:
    """
    Factory function to create production API server.
    """
    config = ProductionConfig(
        hidden_size=hidden_size,
        num_experts=num_experts,
        production_mode=production_mode,
        **kwargs
    )
    
    return ProductionAPIServer(config)

def run_production_api_demo():
    """Run production API server demonstration."""
    print("🚀 Production API Server Demo")
    print("=" * 50)
    
    # Create production API server
    server = create_production_api_server(
        hidden_size=512,
        num_experts=8,
        production_mode=ProductionMode.PRODUCTION,
        max_batch_size=16,
        max_sequence_length=1024,
        enable_monitoring=True,
        enable_metrics=True
    )
    
    print(f"📋 API Server Configuration:")
    print(f"  Hidden Size: {server.config.hidden_size}")
    print(f"  Number of Experts: {server.config.num_experts}")
    print(f"  Production Mode: {server.config.production_mode.value}")
    print(f"  Max Batch Size: {server.config.max_batch_size}")
    print(f"  Max Sequence Length: {server.config.max_sequence_length}")
    
    print(f"\n🌐 API Endpoints:")
    print(f"  Health Check: GET /health")
    print(f"  Process Request: POST /api/v1/process")
    print(f"  System Stats: GET /api/v1/stats")
    print(f"  Metrics: GET /metrics")
    print(f"  WebSocket: WS /ws")
    print(f"  Documentation: GET /docs")
    
    print(f"\n🔐 Authentication:")
    print(f"  Bearer Token required for API endpoints")
    print(f"  JWT Secret: {server.jwt_secret}")
    
    print(f"\n📊 Monitoring:")
    print(f"  Prometheus metrics available at /metrics")
    print(f"  Health checks available at /health")
    print(f"  System statistics available at /api/v1/stats")
    
    print(f"\n🚀 Starting API server...")
    print(f"  Host: 0.0.0.0")
    print(f"  Port: 8080")
    print(f"  Production mode: {server.config.production_mode.value}")
    
    # Start server
    try:
        server.run(host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down API server...")
        server.pimoe_system.shutdown()
        print(f"✅ API server shutdown complete!")

if __name__ == "__main__":
    # Run production API demo
    run_production_api_demo()





