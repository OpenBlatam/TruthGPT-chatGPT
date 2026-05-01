"""
Cache utilities for optimization_core.

This module re-exports cache utilities from modules.base.core_system.core.cache_utils for backward compatibility.
New code should import directly from modules.base.core_system.core.cache_utils.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..modules.base.core_system.core.cache_utils import (
        CacheStats,
        MemoryCache,
        DiskCache,
        cached,
    )
else:
    # Re-export from modules.base.core_system.core.cache_utils for backward compatibility
    from ..modules.base.core_system.core.cache_utils import (
        CacheStats,
        MemoryCache,
        DiskCache,
        cached,
    )

__all__ = [
    'CacheStats',
    'MemoryCache',
    'DiskCache',
    'cached',
]





