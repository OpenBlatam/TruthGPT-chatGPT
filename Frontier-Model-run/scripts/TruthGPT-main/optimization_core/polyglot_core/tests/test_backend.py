"""
Tests for backend detection and selection.
"""

import pytest
from optimization_core.polyglot_core.backend import (
    Backend,
    BackendInfo,
    get_available_backends,
    get_best_backend,
    is_backend_available,
    get_backend_info,
)


def test_backend_enum():
    """Test Backend enum."""
    assert Backend.PYTHON in Backend
    assert Backend.RUST in Backend
    assert Backend.CPP in Backend
    assert Backend.GO in Backend


def test_get_available_backends():
    """Test getting available backends."""
    backends = get_available_backends()
    
    assert len(backends) >= 1  # At least Python
    assert all(isinstance(b, BackendInfo) for b in backends)
    
    # Python should always be available
    python_backend = next((b for b in backends if b.backend == Backend.PYTHON), None)
    assert python_backend is not None
    assert python_backend.available is True


def test_is_backend_available():
    """Test checking backend availability."""
    # Python should always be available
    assert is_backend_available(Backend.PYTHON) is True
    
    # Others may or may not be available
    rust_available = is_backend_available(Backend.RUST)
    assert isinstance(rust_available, bool)


def test_get_best_backend():
    """Test getting best backend for features."""
    # Should always return a backend
    best = get_best_backend('kv_cache')
    assert best in Backend
    
    best = get_best_backend('attention')
    assert best in Backend
    
    best = get_best_backend('compression')
    assert best in Backend
    
    # Unknown feature should default to Python
    best = get_best_backend('unknown_feature')
    assert best == Backend.PYTHON


def test_get_backend_info():
    """Test getting backend info."""
    info = get_backend_info(Backend.PYTHON)
    assert info is not None
    assert isinstance(info, BackendInfo)
    assert info.available is True


def test_backend_info_str():
    """Test BackendInfo string representation."""
    info = BackendInfo(
        name="test",
        backend=Backend.PYTHON,
        available=True,
        version="1.0.0"
    )
    assert "test" in str(info)
    assert "✓" in str(info) or "✗" in str(info)













