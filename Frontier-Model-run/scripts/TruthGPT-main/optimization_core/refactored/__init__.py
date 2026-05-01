"""
Refactored TruthGPT Optimization Framework
==========================================

A completely refactored, production-ready optimization framework with:
- Unified architecture with proper abstractions
- Factory pattern for optimizer creation
- Plugin system for extensibility
- Dependency injection container
- Async processing capabilities
- Comprehensive monitoring and metrics
- Intelligent caching layer
- REST API for external integration

Author: TruthGPT Team
Version: 3.0.0
License: MIT
"""

from .core.architecture import OptimizationFramework
from .core.factory import OptimizerFactory
from .core.container import DependencyContainer
from .core.config import UnifiedConfig
from .core.monitoring import MetricsCollector
from .core.caching import CacheManager
from .api.server import APIServer

__version__ = "3.0.0"
__author__ = "TruthGPT Team"
__license__ = "MIT"

# Main framework instance
framework = OptimizationFramework()

# Quick access to main components
factory = OptimizerFactory()
container = DependencyContainer()
config = UnifiedConfig()
metrics = MetricsCollector()
cache = CacheManager()
api_server = APIServer()

__all__ = [
    'OptimizationFramework',
    'OptimizerFactory', 
    'DependencyContainer',
    'UnifiedConfig',
    'MetricsCollector',
    'CacheManager',
    'APIServer',
    'framework',
    'factory',
    'container',
    'config',
    'metrics',
    'cache',
    'api_server'
]



