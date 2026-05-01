"""
Feed-Forward Module for TruthGPT Optimization Core
===================================================

Unified architecture providing:
- layers: Low-level FFN implementations (SwiGLU, GeGLU, ReGLU, etc.)
- experts: Modular expert definitions (Base, Specialized)
- routing: Token-level routing (BaseRouter, TokenLevelRouter)
- blocks: High-level composable blocks (MoEBlock)
- systems: Production PiMoE systems
- next_generation_ai: Experimental NAS, Quantum AI
"""

from __future__ import annotations
import logging
from optimization_core.utils.dependency_manager import resolve_lazy_import

_logger = logging.getLogger(__name__)

_LAZY_IMPORTS = {
    # ── NEW UNIFIED ARCHITECTURE ─────────────────────────────────────
    # Layers (GLU variants)
    'FeedForwardBase': '.layers.ffn',
    'FeedForward': '.layers.ffn',
    'GatedFeedForward': '.layers.ffn',
    'SwiGLU': '.swiglu',
    'SwigluMLP': '.swiglu',
    'ReGLU': '.layers.ffn',
    'GeGLU': '.layers.ffn',
    'create_feed_forward': '.layers.ffn',

    # Experts
    'BaseExpert': '.experts.base',
    'ExpertType': '.experts.base',
    'SpecializedExpert': '.experts.specialized',

    # Routing
    'BaseRouter': '.routing.base',
    'RoutingResult': '.routing.base',
    'TokenLevelRouter': '.routing.token_router',

    # Blocks
    'MoEBlock': '.blocks.moe_block',

    # ── LEGACY / BACKWARD COMPAT (still functional) ─────────────────
    # Core (legacy unified engine)
    'ModularFeedForward': '.core.feed_forward',
    'AdaptiveFeedForward': '.core.feed_forward',
    'create_swiglu': '.core.feed_forward',
    'create_gated_ffn': '.core.feed_forward',

    # Enhanced PiMoE integration
    'EnhancedPiMoEIntegration': '.core.enhanced_pimoe_integration',
    'AdaptivePiMoE': '.core.enhanced_pimoe_integration',
    'OptimizationMetrics': '.core.enhanced_pimoe_integration',
    'create_enhanced_pimoe_integration': '.core.enhanced_pimoe_integration',

    # Base/Core refactored protocols
    'SystemConfig': '.core.refactored_pimoe_base',
    'LoggerProtocol': '.core.refactored_pimoe_base',
    'MonitorProtocol': '.core.refactored_pimoe_base',
    'ErrorHandlerProtocol': '.core.refactored_pimoe_base',
    'RequestQueueProtocol': '.core.refactored_pimoe_base',
    'PiMoEProcessorProtocol': '.core.refactored_pimoe_base',
    'BaseService': '.core.refactored_pimoe_base',
    'BaseConfig': '.core.refactored_pimoe_base',
    'ServiceFactory': '.core.refactored_pimoe_base',
    'DIContainer': '.core.refactored_pimoe_base',
    'EventBus': '.core.refactored_pimoe_base',
    'ResourceManager': '.core.refactored_pimoe_base',
    'MetricsCollector': '.core.refactored_pimoe_base',
    'HealthChecker': '.core.refactored_pimoe_base',
    'BasePiMoESystem': '.core.refactored_pimoe_base',

    # Legacy routing
    'PiMoESystem': '.routing.pimoe_router',
    'PiMoEExpert': '.routing.pimoe_router',
    'RoutingDecision': '.routing.pimoe_router',
    'create_pimoe_system': '.routing.pimoe_router',

    'AdvancedPiMoESystem': '.routing.advanced_pimoe_routing',
    'RoutingStrategy': '.routing.advanced_pimoe_routing',
    'AdvancedRoutingConfig': '.routing.advanced_pimoe_routing',
    'AttentionBasedRouter': '.routing.advanced_pimoe_routing',
    'HierarchicalRouter': '.routing.advanced_pimoe_routing',
    'DynamicExpertScaler': '.routing.advanced_pimoe_routing',
    'CrossExpertCommunicator': '.routing.advanced_pimoe_routing',
    'NeuralArchitectureSearchRouter': '.routing.advanced_pimoe_routing',

    # Systems - Point to package to leverage its __init__.py exports
    'ProductionPiMoESystem': '.systems',
    'ProductionConfig': '.systems',
    'ProductionMode': '.systems',
    'ProductionLogger': '.systems',
    'ProductionMonitor': '.systems',
    'ProductionErrorHandler': '.systems',
    'ProductionRequestQueue': '.systems',
    'create_production_pimoe_system': '.systems',

    'ProductionAPIServer': '.systems',
    'PiMoERequest': '.systems',
    'PiMoEResponse': '.systems',
    'HealthResponse': '.systems',
    'MetricsResponse': '.systems',
    'WebSocketMessage': '.systems',
    'create_production_api_server': '.systems',
    'run_production_api_demo': '.systems',

    'ProductionDeployment': '.systems',
    'DeploymentEnvironment': '.systems',
    'ScalingStrategy': '.systems',
    'DockerConfig': '.systems',
    'KubernetesConfig': '.systems',
    'MonitoringConfig': '.systems',
    'LoadBalancerConfig': '.systems',
    'create_production_deployment': '.systems',

    'RefactoredProductionPiMoESystem': '.systems',
    'create_refactored_production_system': '.systems',

    # Optimization
    'AdvancedArchitectureOptimizer': '.optimization',
    'AdvancedArchitectureConfig': '.optimization',
}

def __getattr__(name: str):
    """Lazy import system for feed_forward submodules."""
    return resolve_lazy_import(name, __package__ or 'feed_forward', _LAZY_IMPORTS)

__all__ = list(_LAZY_IMPORTS.keys())

__version__ = "2.0.0"

