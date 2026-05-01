"""
Legacy redirection hub for optimization_core.core.

This module provides backward compatibility for code that still imports from 
optimization_core.core. It lazily redirects requests to the new location:
optimization_core.modules.base.core_system.core.
"""
from __future__ import annotations
import importlib
import sys
import types
from typing import Any

# The new location of the core package
NEW_CORE_BASE = "optimization_core.modules.base.core_system.core"

# Map of common submodules for faster resolution
_SUBMODULE_MAP = {
    "metrics_base": f"{NEW_CORE_BASE}.metrics_base",
    "performance_utils": f"{NEW_CORE_BASE}.performance_utils",
    "serialization": f"{NEW_CORE_BASE}.serialization",
    "config_utils": f"{NEW_CORE_BASE}.config_utils",
    "paper_base": f"{NEW_CORE_BASE}.paper_base",
    "service_registry": f"{NEW_CORE_BASE}.service_registry",
    "architecture": f"{NEW_CORE_BASE}.architecture",
    "config": f"{NEW_CORE_BASE}.config",
    "validators": f"{NEW_CORE_BASE}.validators",
    "exceptions": f"{NEW_CORE_BASE}.exceptions",
    "interfaces": f"{NEW_CORE_BASE}.interfaces",
    "helpers": f"{NEW_CORE_BASE}.helpers",
    "cache_utils": f"{NEW_CORE_BASE}.cache_utils",
    "validation": f"{NEW_CORE_BASE}.validation",
    "monitoring": f"{NEW_CORE_BASE}.monitoring",
    "unified_optimizer": f"{NEW_CORE_BASE}.unified_optimizer",
    "base_truthgpt_optimizer": f"{NEW_CORE_BASE}.base_truthgpt_optimizer",
}

def __getattr__(name: str) -> Any:
    """Lazily import from the new core location."""
    if name.startswith('_'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    # Try the explicit map first
    if name in _SUBMODULE_MAP:
        target = _SUBMODULE_MAP[name]
    else:
        # Fallback to direct mapping
        target = f"{NEW_CORE_BASE}.{name}"
    
    try:
        # If it's a module we're looking for, return it
        return importlib.import_module(target)
    except ImportError:
        # If it's an attribute inside the main __init__ of the new core
        new_core_mod = importlib.import_module(NEW_CORE_BASE)
        if hasattr(new_core_mod, name):
            return getattr(new_core_mod, name)
        
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Also tried redirecting to '{target}' and '{NEW_CORE_BASE}'."
        )

# Note: For 'from optimization_core.core.xxx import ...' cases, the package-level
# __getattr__ above handles redirection. Direct 'import optimization_core.core.xxx'
# is handled by the wildcard re-export files (e.g. core/config.py).
