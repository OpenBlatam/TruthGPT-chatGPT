"""
Professional inference module with batching, caching, and optimization.

This module provides organized access to inference components:
- core: Core inference components (engine, batch processor, text generator)
- middleware: Middleware components (circuit breaker, rate limiter, cache)
- monitoring: Monitoring components (metrics, observability)
- api: API server components
- utils: Utility functions
"""

from __future__ import annotations

import importlib

# Direct imports for backward compatibility
from .inference_engine import InferenceEngine, BatchProcessor
from .cache_manager import CacheManager
from .text_generator import TextGenerator

# New high-performance engines
try:
    from .vllm_engine import VLLMEngine, AsyncVLLMEngine, create_vllm_engine
    from .tensorrt_llm_engine import TensorRTLLMEngine, create_tensorrt_llm_engine
    from .base_engine import BaseInferenceEngine, GenerationConfig
    from .engine_factory import (
        create_inference_engine,
        EngineType,
        list_available_engines,
    )
    _NEW_ENGINES_AVAILABLE = True
except ImportError:
    _NEW_ENGINES_AVAILABLE = False

# Lazy imports for organized submodules
_LAZY_IMPORTS = {
    'core': '.core',
    'middleware': '.middleware',
    'monitoring': '.monitoring',
    'api': '.api',
    'server': '.server',
    'utils': '.utils',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for inference submodules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name in _import_cache:
        return _import_cache[name]
    
    module_path = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(module_path, package=__package__)
        _import_cache[name] = module
        return module
    except (ImportError, AttributeError) as e:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def list_available_inference_modules() -> list[str]:
    """List all available inference submodules."""
    return list(_LAZY_IMPORTS.keys())


__all__ = [
    # Core components (backward compatible)
    "InferenceEngine",
    "BatchProcessor",
    "CacheManager",
    "TextGenerator",
    # New high-performance engines
    "VLLMEngine",
    "AsyncVLLMEngine",
    "TensorRTLLMEngine",
    "BaseInferenceEngine",
    "GenerationConfig",
    "create_vllm_engine",
    "create_tensorrt_llm_engine",
    "create_inference_engine",
    "EngineType",
    "list_available_engines",
    # Submodules
    "core",
    "middleware",
    "monitoring",
    "api",
    "server",
    "utils",
    "list_available_inference_modules",
]

