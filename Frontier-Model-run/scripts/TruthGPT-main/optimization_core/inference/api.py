"""
🚀 Production-Ready Inference API
Enterprise-grade FastAPI server with batching, streaming, observability, and resilience.
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx
from fastapi import (
    FastAPI, Request, Response, HTTPException, Header, BackgroundTasks,
    Depends, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from ..configs.loader import load_config
from ..models import build_model


# ============================================================================
# Configuration
# ============================================================================

API_TOKEN = os.environ.get("TRUTHGPT_API_TOKEN", "changeme")
CONFIG_PATH = os.environ.get(
    "TRUTHGPT_CONFIG",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "llm_default.yaml",
    ),
)

# Batching configuration
BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "32"))
BATCH_FLUSH_TIMEOUT_MS = int(os.environ.get("BATCH_FLUSH_TIMEOUT_MS", "20"))

# Rate limiting
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "600"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))

# Circuit breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.environ.get("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_TIMEOUT_SEC = int(os.environ.get("CIRCUIT_BREAKER_TIMEOUT_SEC", "60"))

# Webhooks
WEBHOOK_HMAC_SECRET = os.environ.get("WEBHOOK_HMAC_SECRET", "changeme-secret")
WEBHOOK_TIMESTAMP_WINDOW = int(os.environ.get("WEBHOOK_TIMESTAMP_WINDOW", "300"))

# Observability
ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "true").lower() == "true"
ENABLE_TRACING = os.environ.get("ENABLE_TRACING", "true").lower() == "true"

# CORS
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")


# ============================================================================
# Models
# ============================================================================

class InferRequest(BaseModel):
    """Inference request model"""
    model: str = Field(..., description="Model identifier")
    prompt: str = Field(..., min_length=1, max_length=8192, description="Input prompt")
    params: Dict[str, Any] = Field(default_factory=dict, description="Generation parameters")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key for deduplication")
    
    @validator("params")
    def validate_params(cls, v):
        """Validate and normalize params"""
        max_tokens = v.get("max_new_tokens", 512)
        if max_tokens > 4096:
            raise ValueError("max_new_tokens cannot exceed 4096")
        return v


class InferResponse(BaseModel):
    """Inference response model"""
    id: str = Field(..., description="Request ID")
    model: str = Field(..., description="Model used")
    output: str = Field(..., description="Generated text")
    usage: Dict[str, int] = Field(..., description="Token usage")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    cached: bool = Field(default=False, description="Whether result was cached")


class TokenChunk(BaseModel):
    """Streaming token chunk"""
    text: str = Field(..., description="Token text")
    finish_reason: Optional[str] = Field(None, description="Finish reason if done")


class WebhookPayload(BaseModel):
    """Webhook payload"""
    id: str
    type: str
    payload: Dict[str, Any]
    timestamp: int


# ============================================================================
# State Management
# ============================================================================

class GlobalState:
    """Global application state"""
    model: Optional[Any] = None
    batch_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    rate_limiter: Dict[str, List[float]] = defaultdict(list)
    circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "failures": 0,
        "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
        "opened_at": None
    })
    cache: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Any] = defaultdict(int)


state = GlobalState()


# ============================================================================
# Utility Functions
# ============================================================================

def get_request_id(request: Request) -> str:
    """Extract or generate request ID"""
    rid = request.headers.get("X-Request-ID")
    return rid or str(uuid.uuid4())


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt for caching"""
    return prompt.strip().lower()


def cache_key(model: str, prompt: str, params: Dict[str, Any]) -> str:
    """Generate cache key"""
    normalized_prompt = normalize_prompt(prompt)
    params_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
    key_data = f"{model}:{normalized_prompt}:{params_str}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def check_rate_limit(client_id: str) -> tuple[bool, Optional[int]]:
    """Check rate limit for client"""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    
    # Clean old entries
    state.rate_limiter[client_id] = [
        ts for ts in state.rate_limiter[client_id] if ts > window_start
    ]
    
    # Check limit
    if len(state.rate_limiter[client_id]) >= RATE_LIMIT_RPM:
        retry_after = int(state.rate_limiter[client_id][0] + RATE_LIMIT_WINDOW_SEC - now) + 1
        return False, retry_after
    
    state.rate_limiter[client_id].append(now)
    return True, None


def check_circuit_breaker(model: str) -> bool:
    """Check if circuit breaker allows request"""
    cb = state.circuit_breakers[model]
    
    if cb["state"] == "CLOSED":
        return True
    
    if cb["state"] == "OPEN":
        # Check if timeout has passed
        if cb["opened_at"]:
            elapsed = time.time() - cb["opened_at"]
            if elapsed >= CIRCUIT_BREAKER_TIMEOUT_SEC:
                cb["state"] = "HALF_OPEN"
                cb["failures"] = 0
                return True
        return False
    
    # HALF_OPEN
    return True


def record_circuit_breaker_success(model: str):
    """Record successful request"""
    cb = state.circuit_breakers[model]
    if cb["state"] == "HALF_OPEN":
        cb["state"] = "CLOSED"
        cb["failures"] = 0


def record_circuit_breaker_failure(model: str):
    """Record failed request"""
    cb = state.circuit_breakers[model]
    cb["failures"] += 1
    
    if cb["failures"] >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
        cb["state"] = "OPEN"
        cb["opened_at"] = time.time()


def sign_webhook(secret: str, payload: bytes, timestamp: int) -> str:
    """Sign webhook payload"""
    message = f"{timestamp}.".encode() + payload
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={signature}"


def verify_webhook(secret: str, payload: bytes, header: str) -> bool:
    """Verify webhook signature"""
    try:
        parts = dict(s.split("=") for s in header.split(","))
        timestamp = int(parts.get("t", 0))
        signature = parts.get("v1", "")
        
        # Check timestamp window
        now = int(time.time())
        if abs(now - timestamp) > WEBHOOK_TIMESTAMP_WINDOW:
            return False
        
        # Verify signature
        expected = sign_webhook(secret, payload, timestamp).split("v1=", 1)[1]
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor:
    """Batch processor for efficient inference"""
    
    def __init__(self, max_size: int, flush_timeout_ms: int):
        self.max_size = max_size
        self.flush_timeout_ms = flush_timeout_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running = False
    
    async def start(self):
        """Start batch processor"""
        self.running = True
        asyncio.create_task(self._process_batches())
    
    async def _process_batches(self):
        """Process batches"""
        while self.running:
            batch = []
            timeout_reached = False
            
            # Collect requests until batch is full or timeout
            try:
                first_item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.flush_timeout_ms / 1000.0
                )
                batch.append(first_item)
                
                # Try to fill batch
                while len(batch) < self.max_size:
                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=0.01  # Short timeout for batching
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
            except asyncio.TimeoutError:
                continue
            
            if batch:
                await self._execute_batch(batch)
    
    async def _execute_batch(self, batch: List[Dict[str, Any]]):
        """Execute batch inference"""
        # Group by model and params
        batches_by_model = defaultdict(list)
        for item in batch:
            model_key = item["model"]
            batches_by_model[model_key].append(item)
        
        # Process each model batch
        for model_key, model_batch in batches_by_model.items():
            try:
                # Extract prompts
                prompts = [item["prompt"] for item in model_batch]
                
                # Run inference (simplified - would use actual batching)
                start_time = time.time()
                for item in model_batch:
                    try:
                        result = state.model.infer({
                            "text": item["prompt"],
                            **item.get("params", {})
                        })
                        
                        item["future"].set_result({
                            "result": result,
                            "latency_ms": (time.time() - start_time) * 1000
                        })
                    except Exception as e:
                        item["future"].set_exception(e)
                        
            except Exception as e:
                # Fail all items in batch
                for item in model_batch:
                    if not item["future"].done():
                        item["future"].set_exception(e)
    
    async def enqueue(self, request: InferRequest) -> asyncio.Future:
        """Enqueue request for batching"""
        future = asyncio.Future()
        await self.queue.put({
            "model": request.model,
            "prompt": request.prompt,
            "params": request.params,
            "future": future
        })
        return future


batch_processor = BatchProcessor(BATCH_MAX_SIZE, BATCH_FLUSH_TIMEOUT_MS)


# ============================================================================
# Middleware
# ============================================================================

async def metrics_middleware(request: Request, call_next):
    """Metrics middleware"""
    request_id = get_request_id(request)
    start_time = time.time()
    
    response = await call_next(request)
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Update metrics
    state.metrics["requests_total"] += 1
    state.metrics["request_duration_ms"] += latency_ms
    
    if response.status_code >= 500:
        state.metrics["errors_5xx"] += 1
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{latency_ms:.2f}"
    
    return response


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    # Startup
    print("🚀 Starting Inference API...")
    
    # Load model
    print(f"📦 Loading model from {CONFIG_PATH}...")
    cfg = load_config(CONFIG_PATH, overrides=None)
    state.model = build_model(cfg.model.family, cfg.dict())
    print("✅ Model loaded successfully")
    
    # Start batch processor
    await batch_processor.start()
    print("✅ Batch processor started")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")


app = FastAPI(
    title="Frontier-Model-Run Inference API",
    description="Enterprise-grade inference API with batching, streaming, and observability",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics middleware
if ENABLE_METRICS:
    app.middleware("http")(metrics_middleware)

# ============================================================================
# Routers
# ============================================================================
try:
    from ..agents.api_router import create_agent_router
    from ..agents.client import AgentClient

    _agent_client = AgentClient(use_swarm=False)
    agent_router = create_agent_router(_agent_client)
    app.include_router(agent_router)
    print("Agent API router mounted at /v1/agent")
except ImportError as e:
    _agent_client = None
    print(f"Warning: the Agent API router could not be loaded: {e}")

try:
    from ..agents.messaging.api_routes import create_messaging_router

    if _agent_client is not None:
        messaging_router = create_messaging_router(_agent_client)
        app.include_router(messaging_router)
        print("Messaging webhooks mounted at /webhooks/*")
    else:
        print("Warning: Messaging webhooks skipped (AgentClient not available)")
except ImportError as e:
    print(f"Warning: Messaging webhooks could not be loaded: {e}")

try:
    from ..agents.observability.api_routes import create_observability_router

    observability_router = create_observability_router()
    app.include_router(observability_router)
    print("Observability router mounted at /v1/traces")
except ImportError as e:
    print(f"Warning: Observability router could not be loaded: {e}")

try:
    from ..agents.scheduler import AgentScheduler
    from ..agents.scheduler.api_routes import create_scheduler_router

    if _agent_client is not None:
        _scheduler = AgentScheduler(_agent_client)
        scheduler_router = create_scheduler_router(_scheduler)
        app.include_router(scheduler_router)
        print("Scheduler router mounted at /v1/scheduler")
    else:
        print("Warning: Scheduler skipped (AgentClient not available)")
except ImportError as e:
    print(f"Warning: Scheduler router could not be loaded: {e}")


# ============================================================================
# Authentication
# ============================================================================

async def verify_token(authorization: Optional[str] = Header(None)):
    """Verify API token"""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return token


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "success": True,
        "service": "Frontier-Model-Run Inference API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "infer": "/v1/infer",
            "stream": "/v1/infer/stream",
            "webhooks": "/webhooks/ingest",
            "agent_run": "/v1/agent/run",
            "agent_memory_clear": "/v1/agent/memory/{user_id}"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    model_loaded = state.model is not None
    batch_processor_running = batch_processor.running
    
    health_status = {
        "status": "healthy" if model_loaded and batch_processor_running else "degraded",
        "timestamp": int(time.time()),
        "checks": {
            "model": "loaded" if model_loaded else "not_loaded",
            "batch_processor": "running" if batch_processor_running else "stopped"
        }
    }
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/ready")
async def ready():
    """Readiness check"""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    if not ENABLE_METRICS:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    total_requests = state.metrics.get("requests_total", 0)
    total_duration = state.metrics.get("request_duration_ms", 0)
    avg_duration = total_duration / total_requests if total_requests > 0 else 0
    
    metrics_text = f"""# HELP inference_requests_total Total number of inference requests
# TYPE inference_requests_total counter
inference_requests_total {total_requests}

# HELP inference_request_duration_ms Average request duration in milliseconds
# TYPE inference_request_duration_ms gauge
inference_request_duration_ms {avg_duration:.2f}

# HELP inference_errors_5xx_total Total number of 5xx errors
# TYPE inference_errors_5xx_total counter
inference_errors_5xx_total {state.metrics.get("errors_5xx", 0)}

# HELP inference_cache_hits_total Total number of cache hits
# TYPE inference_cache_hits_total counter
inference_cache_hits_total {state.metrics.get("cache_hits", 0)}
"""
    
    return Response(content=metrics_text, media_type="text/plain")


@app.post("/v1/infer", response_model=InferResponse)
async def infer(
    request: InferRequest,
    req: Request,
    token: str = Depends(verify_token)
):
    """Synchronous inference endpoint"""
    request_id = get_request_id(req)
    client_id = req.client.host if req.client else "unknown"
    
    # Rate limiting
    allowed, retry_after = check_rate_limit(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Circuit breaker
    if not check_circuit_breaker(request.model):
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable (circuit breaker open)"
        )
    
    start_time = time.time()
    
    try:
        # Check cache
        cache_key_str = cache_key(request.model, request.prompt, request.params)
        if cache_key_str in state.cache:
            state.metrics["cache_hits"] += 1
            cached_result = state.cache[cache_key_str]
            return InferResponse(
                id=request_id,
                model=request.model,
                output=cached_result["output"],
                usage=cached_result["usage"],
                latency_ms=(time.time() - start_time) * 1000,
                cached=True
            )
        
        # Run inference (with batching or direct)
        if BATCH_MAX_SIZE > 1:
            future = await batch_processor.enqueue(request)
            batch_result = await future
            result = batch_result["result"]
            latency_ms = batch_result["latency_ms"]
        else:
            result = state.model.infer({
                "text": request.prompt,
                **request.params
            })
            latency_ms = (time.time() - start_time) * 1000
        
        # Cache result
        state.cache[cache_key_str] = {
            "output": result.get("text", ""),
            "usage": result.get("usage", {})
        }
        
        # Record success
        record_circuit_breaker_success(request.model)
        
        return InferResponse(
            id=request_id,
            model=request.model,
            output=result.get("text", ""),
            usage=result.get("usage", {}),
            latency_ms=latency_ms,
            cached=False
        )
        
    except Exception as e:
        record_circuit_breaker_failure(request.model)
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/v1/infer/stream")
async def infer_stream(
    request: InferRequest,
    req: Request,
    token: str = Depends(verify_token)
):
    """Streaming inference endpoint (SSE)"""
    request_id = get_request_id(req)
    client_id = req.client.host if req.client else "unknown"
    
    # Rate limiting
    allowed, retry_after = check_rate_limit(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Circuit breaker
    if not check_circuit_breaker(request.model):
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable"
        )
    
    async def generate_stream() -> AsyncIterator[str]:
        """Generate streaming response"""
        try:
            # Run inference and stream tokens
            result = state.model.infer({
                "text": request.prompt,
                **request.params,
                "stream": True
            })
            
            # Stream tokens (simplified - would use actual streaming)
            text = result.get("text", "")
            for i, char in enumerate(text):
                chunk = TokenChunk(
                    text=char,
                    finish_reason=None if i < len(text) - 1 else "stop"
                )
                yield f"event: token\ndata: {chunk.json()}\n\n"
            
            yield f"event: done\ndata: {{}}\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": request_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


    return {
        "success": True,
        "id": payload.id,
        "received_at": int(time.time())
    }


# ============================================================================
# High-level Swarm & Research Endpoints
# ============================================================================

@app.post("/v1/swarm/ask")
async def swarm_ask(
    prompt: str,
    user_id: str = "api_user",
    token: str = Depends(verify_token)
):
    """High-level swarm interaction endpoint."""
    if _agent_client is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    response = await _agent_client.run(user_id=user_id, prompt=prompt)
    return response


@app.get("/v1/research/papers")
async def list_papers(
    category: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """List available research knowledge base."""
    from ..core.papers import PaperRegistry
    registry = PaperRegistry()
    
    paper_ids = registry.list_available_papers(category=category)
    results = []
    for pid in paper_ids:
        meta = registry.get_paper_metadata(pid)
        results.append({
            "id": pid,
            "category": meta.category,
            "techniques": meta.techniques
        })
    
    return {"total": len(results), "papers": results}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        reload=os.environ.get("ENVIRONMENT", "production") == "development"
    )




