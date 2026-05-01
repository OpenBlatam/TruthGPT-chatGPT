"""
Inference Middleware Components

This module contains middleware components: circuit breaker, rate limiter, and cache management.
"""

from __future__ import annotations

import importlib

__all__ = [
    'CircuitBreaker',
    'CircuitState',
    'CircuitBreakerConfig',
    'CircuitBreakerStats',
    'SlidingWindowRateLimiter',
    'RateLimiterManager',
    'CacheManager',
    'InMemoryCache',
    'RedisCache',
]

_LAZY_IMPORTS = {
    'CircuitBreaker': '..circuit_breaker',
    'CircuitState': '..circuit_breaker',
    'CircuitBreakerConfig': '..circuit_breaker',
    'CircuitBreakerStats': '..circuit_breaker',
    'SlidingWindowRateLimiter': '..rate_limiter',
    'RateLimiterManager': '..rate_limiter',
    'CacheManager': '..cache_manager',
    'InMemoryCache': '..cache',
    'RedisCache': '..cache',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for inference middleware components."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(module_path, package=__package__)
        obj = getattr(module, name)
        _import_cache[name] = obj
        return obj
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def list_available_middleware_components() -> list[str]:
    """List all available middleware components."""
    return list(_LAZY_IMPORTS.keys())


def get_middleware_component_info(component_name: str) -> dict[str, any]:
    """Get information about a middleware component."""
    if component_name not in _LAZY_IMPORTS:
        raise ValueError(f"Unknown middleware component: {component_name}")
    
    return {
        'name': component_name,
        'module': _LAZY_IMPORTS[component_name],
        'available': component_name in _import_cache or True,
    }


