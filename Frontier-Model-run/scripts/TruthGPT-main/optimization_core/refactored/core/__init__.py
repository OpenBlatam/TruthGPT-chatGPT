"""
Core Architecture Components
===========================

This module contains the core architecture components for the refactored framework:
- Architecture: Main framework orchestrator
- Factory: Optimizer creation and management
- Container: Dependency injection system
- Config: Unified configuration management
- Monitoring: Metrics collection and analysis
- Caching: Intelligent caching system
"""

from .architecture import OptimizationFramework
from .factory import OptimizerFactory
from .container import DependencyContainer
from .config import UnifiedConfig
from .monitoring import MetricsCollector
from .caching import CacheManager

__all__ = [
    'OptimizationFramework',
    'OptimizerFactory',
    'DependencyContainer', 
    'UnifiedConfig',
    'MetricsCollector',
    'CacheManager'
]



