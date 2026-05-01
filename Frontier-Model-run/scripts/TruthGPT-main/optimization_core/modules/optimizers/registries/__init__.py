"""
Registries subpackage for TruthGPT Optimizers.
"""

from .advanced_optimization_registry import (
    AdvancedOptimizationConfig as AdvancedOptimizationConfigV1,
    get_advanced_optimization_config as get_config_v1,
    apply_advanced_optimizations as apply_v1,
    get_advanced_optimization_report as get_report_v1
)

from .advanced_optimization_registry_v2 import (
    AdvancedOptimizationConfig as AdvancedOptimizationConfigV2,
    get_advanced_optimization_config as get_config_v2,
    apply_advanced_optimizations as apply_v2,
    get_advanced_optimization_report as get_report_v2
)

__all__ = [
    'AdvancedOptimizationConfigV1',
    'get_config_v1',
    'apply_v1',
    'get_report_v1',
    'AdvancedOptimizationConfigV2',
    'get_config_v2',
    'apply_v2',
    'get_report_v2',
]

