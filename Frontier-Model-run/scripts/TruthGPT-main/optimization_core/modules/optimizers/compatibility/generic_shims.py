"""
Compatibility Shims for Generic Optimizers
===========================================
Provides backward compatibility for old generic optimizer classes.
"""

import warnings
from typing import Dict, Any, Optional

try:
    from ..core.generic_optimizer import (
        UnifiedGenericOptimizer,
        GenericOptimizationLevel,
        GenericOptimizationResult,
    )
except (ImportError, ValueError):
    from optimization_core.modules.optimizers.core.generic_optimizer import (
        UnifiedGenericOptimizer,
        GenericOptimizationLevel,
        GenericOptimizationResult,
    )


def _create_generic_compat_optimizer(level: GenericOptimizationLevel, optimizer_type: str):
    """Create a compatibility optimizer wrapper."""
    class CompatGenericOptimizer(UnifiedGenericOptimizer):
        """Compatibility wrapper for old generic optimizer classes."""
        
        def __init__(self, config: Optional[Dict[str, Any]] = None):
            warnings.warn(
                f"This optimizer class is deprecated. Use UnifiedGenericOptimizer with level={level.value}, type={optimizer_type} instead.",
                DeprecationWarning,
                stacklevel=3
            )
            super().__init__(config=config, level=level, optimizer_type=optimizer_type)
    
    return CompatGenericOptimizer


# Speed optimizers
UltraSpeedOptimizer = _create_generic_compat_optimizer(GenericOptimizationLevel.ULTRA_SPEED, "speed")
SuperSpeedOptimizer = _create_generic_compat_optimizer(GenericOptimizationLevel.SUPER_SPEED, "speed")
LightningSpeedOptimizer = _create_generic_compat_optimizer(GenericOptimizationLevel.LIGHTNING_SPEED, "speed")
UltraFastOptimizer = _create_generic_compat_optimizer(GenericOptimizationLevel.ULTRA_FAST, "speed")

# Master optimizers
MasterOptimizer = _create_generic_compat_optimizer(GenericOptimizationLevel.MASTER, "master")
UltimateOptimizer = _create_generic_compat_optimizer(GenericOptimizationLevel.ULTIMATE, "master")

# Extreme optimizers  
ExtremeOptimizationEngine = _create_generic_compat_optimizer(GenericOptimizationLevel.EXTREME, "extreme")

# Additional master-level optimizers
UltimateOptimizerCompat = _create_generic_compat_optimizer(GenericOptimizationLevel.ULTIMATE, "master")

# Note: ultimate_optimizer.py is more complex (integration system) and may need special handling
# For now, we provide basic compatibility

__all__ = [
    'UltraSpeedOptimizer',
    'SuperSpeedOptimizer',
    'LightningSpeedOptimizer',
    'UltraFastOptimizer',
    'MasterOptimizer',
    'UltimateOptimizer',
    'ExtremeOptimizationEngine',
    'UltimateOptimizerCompat',
]

