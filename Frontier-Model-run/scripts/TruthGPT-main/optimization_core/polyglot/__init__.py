"""
Polyglot Integration Module

Provides unified interface to Rust, Go, C++, and Julia cores.
"""
from typing import Optional, Dict, Any

__version__ = "0.2.0"

# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

RUST_AVAILABLE = False
GO_AVAILABLE = False
CPP_AVAILABLE = False
JULIA_AVAILABLE = False

try:
    from truthgpt_rust import PyKVCache, PyFastTokenizer, PyCompressor, PyDataLoader
    RUST_AVAILABLE = True
except ImportError:
    pass

try:
    import _cpp_core as cpp_core
    CPP_AVAILABLE = True
except ImportError:
    pass

try:
    import julia
    julia.install()
    from julia import TruthGPTCore
    JULIA_AVAILABLE = True
except ImportError:
    pass

def get_available_backends() -> Dict[str, bool]:
    """Get status of all backends."""
    return {
        "rust": RUST_AVAILABLE,
        "go": GO_AVAILABLE,
        "cpp": CPP_AVAILABLE,
        "julia": JULIA_AVAILABLE,
    }

def get_backend_info() -> Dict[str, Any]:
    """Get detailed backend information."""
    info = {
        "available": get_available_backends(),
        "recommended": [],
    }
    
    if RUST_AVAILABLE:
        try:
            from truthgpt_rust import get_version, get_system_info
            info["rust"] = {
                "version": get_version(),
                "system": get_system_info(),
            }
            info["recommended"].append("rust")
        except Exception:
            pass
    
    if CPP_AVAILABLE:
        try:
            info["cpp"] = {
                "version": cpp_core.__version__,
                "backends": cpp_core.get_available_backends(),
            }
            info["recommended"].append("cpp")
        except Exception:
            pass
    
    if JULIA_AVAILABLE:
        try:
            info["julia"] = {
                "version": TruthGPTCore.VERSION,
            }
            info["recommended"].append("julia")
        except Exception:
            pass
    
    return info

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

from .kv_cache import KVCache
from .attention import attention
from .compression import Compressor, CompressionAlgorithm, create_compressor
from .tokenizer import Tokenizer, create_tokenizer
from .optimization import HyperparameterOptimizer, create_optimizer
from .data_loader import DataLoader, create_data_loader
from .utils import (
    measure_time,
    get_backend_preference,
    select_best_backend,
    format_performance_stats,
    benchmark_backends,
)

__all__ = [
    "RUST_AVAILABLE",
    "GO_AVAILABLE",
    "CPP_AVAILABLE",
    "JULIA_AVAILABLE",
    "get_available_backends",
    "get_backend_info",
    "KVCache",
    "attention",
    "Compressor",
    "CompressionAlgorithm",
    "create_compressor",
    "Tokenizer",
    "create_tokenizer",
    "HyperparameterOptimizer",
    "create_optimizer",
    "DataLoader",
    "create_data_loader",
    "measure_time",
    "get_backend_preference",
    "select_best_backend",
    "format_performance_stats",
    "benchmark_backends",
]

