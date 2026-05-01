"""
Module Detection Utilities

Detects and tracks availability of polyglot modules.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Try to import all polyglot modules
POLYGLOT_MODULES: Dict[str, bool] = {}


def detect_polyglot_modules():
    """Detect and register all available polyglot modules."""
    global POLYGLOT_MODULES
    
    # Polyglot core
    try:
        try:
            from polyglot_core import (
                Backend, BackendInfo, get_available_backends,
                get_best_backend, is_backend_available,
                UnifiedKVCache, UnifiedCompressor
            )
        except ImportError:
            from optimization_core.polyglot_core import (
                Backend, BackendInfo, get_available_backends,
                get_best_backend, is_backend_available,
                UnifiedKVCache, UnifiedCompressor
            )
        POLYGLOT_MODULES['polyglot_core'] = True
    except ImportError as e:
        logger.warning(f"polyglot_core not available: {e}")
        POLYGLOT_MODULES['polyglot_core'] = False
    
    # Inference
    try:
        from optimization_core.polyglot_core.inference import (
            InferenceEngine, GenerationConfig, InferenceConfig
        )
        POLYGLOT_MODULES['polyglot_inference'] = True
    except ImportError as e:
        logger.warning(f"polyglot_inference not available: {e}")
        POLYGLOT_MODULES['polyglot_inference'] = False
    
    # Engine factory
    try:
        from optimization_core.inference.engine_factory import (
            create_inference_engine, EngineType, list_available_engines
        )
        POLYGLOT_MODULES['inference_engines'] = True
    except ImportError as e:
        logger.warning(f"inference_engines not available: {e}")
        POLYGLOT_MODULES['inference_engines'] = False
    
    # Attention
    try:
        from optimization_core.polyglot_core.attention import Attention
        POLYGLOT_MODULES['attention'] = True
    except ImportError as e:
        logger.warning(f"attention not available: {e}")
        POLYGLOT_MODULES['attention'] = False
    
    # Compression
    try:
        from optimization_core.polyglot_core.compression import Compressor
        POLYGLOT_MODULES['compression'] = True
    except ImportError as e:
        logger.warning(f"compression not available: {e}")
        POLYGLOT_MODULES['compression'] = False
    
    # Cache
    try:
        from optimization_core.polyglot_core.cache import KVCache
        POLYGLOT_MODULES['cache'] = True
    except ImportError as e:
        logger.warning(f"cache not available: {e}")
        POLYGLOT_MODULES['cache'] = False
    
    # Rust backend
    try:
        import truthgpt_rust
        POLYGLOT_MODULES['rust'] = True
    except ImportError:
        POLYGLOT_MODULES['rust'] = False
        logger.warning("Rust backend not available")
    
    # C++ backend
    try:
        import _cpp_core
        POLYGLOT_MODULES['cpp'] = True
    except ImportError:
        POLYGLOT_MODULES['cpp'] = False
        logger.warning("C++ backend not available")
    
    # Go backend
    try:
        from optimization_core.go_core.pkg.client import python_client
        POLYGLOT_MODULES['go'] = True
    except ImportError:
        POLYGLOT_MODULES['go'] = False
        logger.warning("Go backend not available")
    
    return POLYGLOT_MODULES


# Initialize on import
detect_polyglot_modules()













