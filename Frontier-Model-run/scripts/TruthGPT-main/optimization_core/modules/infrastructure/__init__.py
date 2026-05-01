"""
ML Core Infrastructure Package
==============================

Core infrastructure systems including adaptive optimization strategies,
auto-performance tuning, AI utilities, plugin architecture, and event bus.
"""

from .adaptive_optimization_strategies import (
    AdaptiveOptimizationStrategies,
    AdaptationTrigger,
    AdaptationAction,
    AdaptationRule,
    AdaptationContext,
    AdaptationDecision,
    create_adaptive_optimization_strategies
)
from .auto_performance_optimizer import (
    AutoPerformanceOptimizer,
    OptimizationConfig,
    OptimizationTarget,
    OptimizationStrategy,
    PerformanceMetrics,
    OptimizationResult
)
from .ai_utils import (
    AIUtils,
    AIOptimizationLevel,
    create_ai_utils,
    optimize_with_ai_utils
)
from .plugin_system import (
    PluginInfo,
    PluginRegistry,
    BasePlugin,
    register_plugin,
    get_plugin,
    list_plugins
)
from .event_system import (
    EventType,
    Event,
    EventEmitter,
    EventBus,
    get_event_bus,
    get_emitter
)

__all__ = [
    'AdaptiveOptimizationStrategies',
    'AdaptationTrigger',
    'AdaptationAction',
    'AdaptationRule',
    'AdaptationContext',
    'AdaptationDecision',
    'create_adaptive_optimization_strategies',
    'AutoPerformanceOptimizer',
    'OptimizationConfig',
    'OptimizationTarget',
    'OptimizationStrategy',
    'PerformanceMetrics',
    'OptimizationResult',
    'AIUtils',
    'AIOptimizationLevel',
    'create_ai_utils',
    'optimize_with_ai_utils',
    'PluginInfo',
    'PluginRegistry',
    'BasePlugin',
    'register_plugin',
    'get_plugin',
    'list_plugins',
    'EventType',
    'Event',
    'EventEmitter',
    'EventBus',
    'get_event_bus',
    'get_emitter'
]
