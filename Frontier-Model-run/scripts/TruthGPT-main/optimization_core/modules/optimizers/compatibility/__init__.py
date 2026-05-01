"""
Compatibility Module
====================

This module contains compatibility layers and utilities.
"""

from .legacy_shims import (
    AdvancedTruthGPTOptimizer,
    ExpertTruthGPTOptimizer,
    UltimateTruthGPTOptimizer,
    SupremeTruthGPTOptimizer,
    EnterpriseTruthGPTOptimizer,
    UltraFastTruthGPTOptimizer,
    UltraSpeedTruthGPTOptimizer,
    HyperSpeedTruthGPTOptimizer,
    LightningSpeedTruthGPTOptimizer,
    InfiniteTruthGPTOptimizer,
    TranscendentTruthGPTOptimizer,
    TransformerOptimizer,
)

from .generic_shims import (
    UltraSpeedOptimizer,
    SuperSpeedOptimizer,
    LightningSpeedOptimizer,
    UltraFastOptimizer,
    MasterOptimizer,
    UltimateOptimizer,
    ExtremeOptimizationEngine,
    UltimateOptimizerCompat,
)

from .shims.enhanced_optimization_core import EnhancedOptimizationCore
from .shims.hybrid_optimization_core import HybridOptimizationCore

__all__ = [
    # Legacy shims
    'AdvancedTruthGPTOptimizer',
    'ExpertTruthGPTOptimizer',
    'UltimateTruthGPTOptimizer',
    'SupremeTruthGPTOptimizer',
    'EnterpriseTruthGPTOptimizer',
    'UltraFastTruthGPTOptimizer',
    'UltraSpeedTruthGPTOptimizer',
    'HyperSpeedTruthGPTOptimizer',
    'LightningSpeedTruthGPTOptimizer',
    'InfiniteTruthGPTOptimizer',
    'TranscendentTruthGPTOptimizer',
    'TransformerOptimizer',
    # Generic shims
    'UltraSpeedOptimizer',
    'SuperSpeedOptimizer',
    'LightningSpeedOptimizer',
    'UltraFastOptimizer',
    'MasterOptimizer',
    'UltimateOptimizer',
    'ExtremeOptimizationEngine',
    'UltimateOptimizerCompat',
    # Existing shims in subfolder
    'EnhancedOptimizationCore',
    'HybridOptimizationCore',
]

