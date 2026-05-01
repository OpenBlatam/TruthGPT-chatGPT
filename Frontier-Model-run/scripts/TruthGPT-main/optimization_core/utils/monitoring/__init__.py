"""
Monitoring Utilities Module

This module contains utilities for monitoring training, visualizing results,
and tracking experiments.
"""

from __future__ import annotations

import importlib

__all__ = [
    'MonitorTraining',
    'RealTimePerformanceMonitor',
    'TruthGPTMonitoring',
    'VisualizeTraining',
    'CompareRuns',
    'ExperimentTracker',
]

_LAZY_IMPORTS = {
    'MonitorTraining': '..monitor_training',
    'RealTimePerformanceMonitor': '..real_time_performance_monitor',
    'TruthGPTMonitoring': '..truthgpt_monitoring',
    'VisualizeTraining': '..visualize_training',
    'CompareRuns': '..compare_runs',
    'ExperimentTracker': '..experiment_tracker',
}

_import_cache = {}


def __getattr__(name: str):
    """Lazy import system for monitoring utility modules."""
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

