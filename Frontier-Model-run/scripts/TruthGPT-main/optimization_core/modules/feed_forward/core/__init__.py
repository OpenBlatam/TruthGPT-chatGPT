"""
PiMoE Core Components
=====================
Core implementations of the Progressive Mixture of Experts system.
"""

from __future__ import annotations
from optimization_core.utils.dependency_manager import resolve_lazy_import

_LAZY_IMPORTS = {
    # Interfaces
    'ExpertType': '.interfaces',
    'ExpertStatus': '.interfaces',
    'ExpertResult': '.interfaces',
    'ExpertConfig': '.interfaces',
    'RoutingDecision': '.interfaces',
    'ProductionMode': '.interfaces',
    'LogLevel': '.interfaces',
    'OptimizationLevel': '.interfaces',
    'LoggerProtocol': '.interfaces',
    'MonitorProtocol': '.interfaces',
    'PiMoEProcessorProtocol': '.interfaces',
    'SystemConfig': '.interfaces',
    'ProductionConfig': '.interfaces',
    'RequestData': '.interfaces',
    'ResponseData': '.interfaces',
    
    # Unified PiMoE
    'UnifiedPiMoESystem': '.pimoe',
    'PerformanceTracker': '.pimoe',
    'create_unified_pimoe': '.pimoe',
    
    # Refactored Base
    'BaseConfig': '.refactored_pimoe_base',
    'BaseService': '.refactored_pimoe_base',
    'DIContainer': '.refactored_pimoe_base',
    'EventBus': '.refactored_pimoe_base',
    'Event': '.refactored_pimoe_base',
    'ResourceManager': '.refactored_pimoe_base',
    'MetricsCollector': '.refactored_pimoe_base',
    'HealthChecker': '.refactored_pimoe_base',
    'BasePiMoESystem': '.refactored_pimoe_base',
    'ServiceFactory': '.refactored_pimoe_base',
    
    # Feed-Forward (Modular)
    'FeedForward': '.feed_forward',
    'GatedFeedForward': '.feed_forward',
    'SwiGLU': '.feed_forward',
    'ReGLU': '.feed_forward',
    'GeGLU': '.feed_forward',
    'ModularFeedForward': '.feed_forward',
    'AdaptiveFeedForward': '.feed_forward',
    'FeedForwardBase': '.feed_forward',
    'create_feed_forward': '.feed_forward',
    'create_swiglu': '.feed_forward',
    'create_gated_ffn': '.feed_forward',
}

def __getattr__(name: str):
    """Lazy import system for PiMoE core components."""
    return resolve_lazy_import(name, __package__ or 'core', _LAZY_IMPORTS)

__all__ = list(_LAZY_IMPORTS.keys())

