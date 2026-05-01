"""
Profiler shim for modules.base.
"""
from .core_system.core.performance_utils import (
    PerformanceProfiler as Profiler,
    profile_function,
)

class ProfilerConfig:
    pass

def profile_class(cls):
    return cls

__all__ = ['Profiler', 'ProfilerConfig', 'profile_function', 'profile_class']
