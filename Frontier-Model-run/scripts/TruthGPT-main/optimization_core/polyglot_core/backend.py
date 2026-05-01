"""
Backend detection and selection module.

Automatically detects available backends and selects the best one
for each component type.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import importlib
import sys


class Backend(Enum):
    """Available backend implementations."""
    PYTHON = auto()   # Pure Python fallback
    RUST = auto()     # Rust via PyO3
    CPP = auto()      # C++ via PyBind11
    GO = auto()       # Go via gRPC/HTTP


@dataclass
class BackendInfo:
    """Information about a backend."""
    name: str
    backend: Backend
    available: bool
    version: str = ""
    features: List[str] = field(default_factory=list)
    performance_multiplier: float = 1.0
    error: Optional[str] = None
    
    def __str__(self) -> str:
        status = "✓" if self.available else "✗"
        return f"{status} {self.name} ({self.version})"


# Backend detection cache
_backend_cache: Dict[Backend, BackendInfo] = {}


def _detect_rust_backend() -> BackendInfo:
    """Detect Rust backend availability."""
    try:
        from optimization_core import rust_core
        truthgpt_rust = rust_core.truthgpt_rust
        
        features = []
        if hasattr(truthgpt_rust, 'PyKVCache'):
            features.append('kv_cache')
        if hasattr(truthgpt_rust, 'PyCompressor'):
            features.append('compression')
        if hasattr(truthgpt_rust, 'PyTokenizer'):
            features.append('tokenization')
        if hasattr(truthgpt_rust, 'PyBatchDataLoader'):
            features.append('data_loading')
        if hasattr(truthgpt_rust, 'PyAttention'):
            features.append('attention')
        
        version = getattr(truthgpt_rust, '__version__', '1.0.0')
        
        return BackendInfo(
            name="rust_core",
            backend=Backend.RUST,
            available=True,
            version=version,
            features=features,
            performance_multiplier=50.0  # 50x vs Python
        )
    except ImportError as e:
        return BackendInfo(
            name="rust_core",
            backend=Backend.RUST,
            available=False,
            error=str(e)
        )


def _detect_cpp_backend() -> BackendInfo:
    """Detect C++ backend availability."""
    try:
        from optimization_core import _cpp_core
        
        features = []
        if hasattr(_cpp_core, 'attention'):
            features.append('attention')
            features.append('flash_attention')
        if hasattr(_cpp_core, 'memory'):
            features.append('kv_cache')
        
        # Get additional info
        info = {}
        if hasattr(_cpp_core, 'get_system_info'):
            info = _cpp_core.get_system_info()
        
        backends = info.get('backends', [])
        if 'cuda' in backends:
            features.append('cuda')
        if 'cutlass' in backends:
            features.append('cutlass')
        if 'eigen' in backends:
            features.append('eigen')
        if 'tbb' in backends:
            features.append('tbb')
        if 'lz4' in backends:
            features.append('compression')
        
        version = getattr(_cpp_core, '__version__', '1.1.0')
        
        # GPU gives higher multiplier
        perf = 100.0 if 'cuda' in features else 10.0
        
        return BackendInfo(
            name="cpp_core",
            backend=Backend.CPP,
            available=True,
            version=version,
            features=features,
            performance_multiplier=perf
        )
    except ImportError as e:
        return BackendInfo(
            name="cpp_core",
            backend=Backend.CPP,
            available=False,
            error=str(e)
        )


def _detect_go_backend() -> BackendInfo:
    """Detect Go backend availability via gRPC/HTTP."""
    try:
        import grpc
        import requests
        
        # Try to connect to Go services
        features = ['grpc', 'http']
        
        # Check if services are reachable
        try:
            resp = requests.get("http://localhost:8080/health", timeout=1)
            if resp.status_code == 200:
                features.append('inference_server')
        except:
            pass
        
        try:
            resp = requests.get("http://localhost:8081/health", timeout=1)
            if resp.status_code == 200:
                features.append('cache_service')
        except:
            pass
        
        return BackendInfo(
            name="go_core",
            backend=Backend.GO,
            available=len(features) > 2,  # Has at least one service
            version="1.0.0",
            features=features,
            performance_multiplier=20.0  # For distributed workloads
        )
    except ImportError as e:
        return BackendInfo(
            name="go_core",
            backend=Backend.GO,
            available=False,
            error=str(e)
        )


def get_available_backends(force_refresh: bool = False) -> List[BackendInfo]:
    """
    Get list of all available backends.
    
    Args:
        force_refresh: Re-detect backends even if cached
        
    Returns:
        List of BackendInfo for each backend
    """
    global _backend_cache
    
    if not _backend_cache or force_refresh:
        _backend_cache = {
            Backend.PYTHON: BackendInfo(
                name="python",
                backend=Backend.PYTHON,
                available=True,
                version=f"{sys.version_info.major}.{sys.version_info.minor}",
                features=['fallback', 'all'],
                performance_multiplier=1.0
            ),
            Backend.RUST: _detect_rust_backend(),
            Backend.CPP: _detect_cpp_backend(),
            Backend.GO: _detect_go_backend(),
        }
    
    return list(_backend_cache.values())


def is_backend_available(backend: Backend) -> bool:
    """Check if a specific backend is available."""
    get_available_backends()
    return _backend_cache.get(backend, BackendInfo(
        name="unknown", backend=backend, available=False
    )).available


def get_best_backend(feature: str) -> Backend:
    """
    Get the best available backend for a specific feature.
    
    Args:
        feature: Feature name (e.g., 'attention', 'kv_cache', 'compression')
        
    Returns:
        Best available Backend enum value
    """
    backends = get_available_backends()
    
    # Feature to preferred backend mapping
    feature_preferences = {
        'attention': [Backend.CPP, Backend.RUST, Backend.PYTHON],
        'flash_attention': [Backend.CPP, Backend.RUST, Backend.PYTHON],
        'cuda': [Backend.CPP],
        'kv_cache': [Backend.RUST, Backend.CPP, Backend.GO, Backend.PYTHON],
        'compression': [Backend.RUST, Backend.CPP, Backend.PYTHON],
        'tokenization': [Backend.RUST, Backend.PYTHON],
        'data_loading': [Backend.RUST, Backend.PYTHON],
        'inference': [Backend.CPP, Backend.RUST, Backend.GO, Backend.PYTHON],
        'distributed': [Backend.GO, Backend.PYTHON],
        'http': [Backend.GO, Backend.PYTHON],
        'grpc': [Backend.GO, Backend.CPP],
    }
    
    preferences = feature_preferences.get(feature, [Backend.PYTHON])
    
    for backend in preferences:
        info = _backend_cache.get(backend)
        if info and info.available and feature in info.features:
            return backend
    
    # Default to Python
    return Backend.PYTHON


def get_backend_info(backend: Backend) -> Optional[BackendInfo]:
    """Get detailed information about a specific backend."""
    get_available_backends()
    return _backend_cache.get(backend)


def print_backend_status():
    """Print status of all backends to console."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                  TruthGPT Backend Status                      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    for info in get_available_backends():
        status = "✓" if info.available else "✗"
        perf = f"{info.performance_multiplier:.0f}x" if info.available else "N/A"
        
        print(f"║  {status} {info.name:12} v{info.version:8}  Perf: {perf:5}       ║")
        
        if info.features:
            features_str = ", ".join(info.features[:4])
            if len(info.features) > 4:
                features_str += f" +{len(info.features)-4}"
            print(f"║     Features: {features_str:43} ║")
        
        if info.error:
            print(f"║     Error: {info.error[:45]:45} ║")
    
    print("╚══════════════════════════════════════════════════════════════╝\n")













