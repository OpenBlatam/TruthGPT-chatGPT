"""
Enterprise Utilities Module

This module contains enterprise-grade utilities for authentication, caching,
monitoring, and cloud integration.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, List

__all__ = [
    'EnterpriseAuth',
    'EnterpriseCache',
    'EnterpriseMonitor',
    'EnterpriseMetrics',
    'EnterpriseCloudIntegration',
    'EnterpriseTruthGPTAdapter',
]

_LAZY_IMPORTS = {
    'EnterpriseAuth': 'optimization_core.modules.enterprise.auth',
    'EnterpriseCache': 'optimization_core.modules.enterprise.cache',
    'EnterpriseMonitor': 'optimization_core.modules.enterprise.monitor',
    'EnterpriseMetrics': 'optimization_core.modules.enterprise.metrics',
    'EnterpriseCloudIntegration': 'optimization_core.modules.enterprise.cloud_integration',
    'EnterpriseTruthGPTAdapter': 'optimization_core.adapters.enterprise_truthgpt_adapter',
}

# Thread-safe cache for loaded modules
_import_cache: Dict[str, Any] = {}
_cache_lock = threading.RLock()


def __getattr__(name: str):
    """Lazy import system for enterprise utility modules."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    with _cache_lock:
        if name in _import_cache:
            return _import_cache[name]
        
        if name not in _LAZY_IMPORTS:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        
        module_path = _LAZY_IMPORTS[name]
        try:
            # Use absolute imports
            module = importlib.import_module(module_path)
            obj = getattr(module, name)
            _import_cache[name] = obj
            return obj
        except (ImportError, AttributeError) as e:
            raise AttributeError(
                f"module '{__name__}' has no attribute '{name}'. "
                f"Failed to import from '{module_path}': {e}"
            ) from e



