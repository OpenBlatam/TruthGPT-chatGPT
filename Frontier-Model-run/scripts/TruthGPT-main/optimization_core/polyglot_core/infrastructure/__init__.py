"""
Infrastructure modules for polyglot_core.

Rate limiting, circuit breaker, distributed systems, async support,
API, service discovery, and load balancing.

All imports are guarded with try/except to allow graceful degradation
when optional compiled backends (Go, Rust, CUDA) are unavailable.
"""

import logging as _log

_logger = _log.getLogger(__name__)

# ---- Rate Limiting -------------------------------------------------------
try:
    from ..rate_limiting import (
        RateLimit,
        RateLimiter,
        RateLimitExceeded,
        rate_limit,
    )
except ImportError:
    _logger.debug("rate_limiting module not available; skipping")
    RateLimit = RateLimiter = RateLimitExceeded = rate_limit = None  # type: ignore[assignment, misc]

# ---- Circuit Breaker ------------------------------------------------------
try:
    from ..circuit_breaker import (
        CircuitState,
        CircuitBreakerConfig,
        CircuitStats,
        CircuitBreaker,
        CircuitOpenError,
        CircuitBreakerRegistry,
        BackendCircuitBreakers,
        get_circuit_breaker,
    )
except ImportError:
    _logger.debug("circuit_breaker module not available; skipping")
    CircuitState = CircuitBreakerConfig = CircuitStats = CircuitBreaker = None  # type: ignore[assignment, misc]
    CircuitOpenError = CircuitBreakerRegistry = BackendCircuitBreakers = get_circuit_breaker = None  # type: ignore[assignment, misc]

# ---- Distributed (Go / gRPC bindings) -------------------------------------
try:
    from ..distributed import (
        DistributedClient,
        GoClient,
        ServiceEndpoint,
    )
except ImportError:
    _logger.debug("distributed module not available (Go backend missing); skipping")
    DistributedClient = GoClient = ServiceEndpoint = None  # type: ignore[assignment, misc]

# ---- Async Core -----------------------------------------------------------
try:
    from ..async_core import (
        AsyncKVCache,
        AsyncCompressor,
        AsyncInferenceEngine,
        AsyncBatchScheduler,
        WebSocketStreamer,
    )
except ImportError:
    _logger.debug("async_core module not available; skipping")
    AsyncKVCache = AsyncCompressor = AsyncInferenceEngine = None  # type: ignore[assignment, misc]
    AsyncBatchScheduler = WebSocketStreamer = None  # type: ignore[assignment, misc]

# ---- API ------------------------------------------------------------------
try:
    from ..api import (
        APIEndpoint,
        APIRouter,
        get_api_router,
        register_endpoint,
    )
except ImportError:
    _logger.debug("api module not available; skipping")
    APIEndpoint = APIRouter = get_api_router = register_endpoint = None  # type: ignore[assignment, misc]

# ---- Service Discovery ----------------------------------------------------
try:
    from ..service_discovery import (
        ServiceStatus,
        ServiceInfo,
        ServiceRegistry,
        get_service_registry,
        register_service,
    )
except ImportError:
    _logger.debug("service_discovery module not available; skipping")
    ServiceStatus = ServiceInfo = ServiceRegistry = None  # type: ignore[assignment, misc]
    get_service_registry = register_service = None  # type: ignore[assignment, misc]

# ---- Load Balancer ---------------------------------------------------------
try:
    from ..load_balancer import (
        LoadBalanceStrategy,
        BackendInstance,
        LoadBalancer,
        get_load_balancer,
        create_load_balancer,
    )
except ImportError:
    _logger.debug("load_balancer module not available; skipping")
    LoadBalanceStrategy = BackendInstance = LoadBalancer = None  # type: ignore[assignment, misc]
    get_load_balancer = create_load_balancer = None  # type: ignore[assignment, misc]

__all__ = [
    # Rate Limiting
    "RateLimit",
    "RateLimiter",
    "RateLimitExceeded",
    "rate_limit",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitStats",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitBreakerRegistry",
    "BackendCircuitBreakers",
    "get_circuit_breaker",
    # Distributed
    "DistributedClient",
    "GoClient",
    "ServiceEndpoint",
    # Async
    "AsyncKVCache",
    "AsyncCompressor",
    "AsyncInferenceEngine",
    "AsyncBatchScheduler",
    "WebSocketStreamer",
    # API
    "APIEndpoint",
    "APIRouter",
    "get_api_router",
    "register_endpoint",
    # Service Discovery
    "ServiceStatus",
    "ServiceInfo",
    "ServiceRegistry",
    "get_service_registry",
    "register_service",
    # Load Balancer
    "LoadBalanceStrategy",
    "BackendInstance",
    "LoadBalancer",
    "get_load_balancer",
    "create_load_balancer",
]

