"""
Causal Inference Sub-Package
============================

Modular causal inference system.
"""
from .enums import CausalMethod, CausalEffectType
from .config import CausalConfig
from .discovery import CausalDiscovery
from .estimation import CausalEffectEstimator
from .analysis import SensitivityAnalyzer, RobustnessChecker
from .system import CausalInferenceSystem

# Legacy alias
CausalInferenceEngine = CausalInferenceSystem

__all__ = [
    'CausalMethod',
    'CausalEffectType',
    'CausalConfig',
    'CausalDiscovery',
    'CausalEffectEstimator',
    'SensitivityAnalyzer',
    'RobustnessChecker',
    'CausalInferenceSystem',
    'CausalInferenceEngine'
]

