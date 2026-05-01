"""
Constants for TruthGPT Optimization Core
Defines all optimization constants and configurations

This module now imports from the organized constants/ submodule structure
for better maintainability while maintaining 100% backward compatibility.

For new code, consider importing directly from the organized modules:
- from optimization_core.constants.enums import OptimizationLevel
- from optimization_core.constants.performance import SPEED_IMPROVEMENTS
- from optimization_core.constants.configurations import DEFAULT_CONFIGS
- from optimization_core.constants.messages import ERROR_MESSAGES
- from optimization_core.constants.version import VERSION_INFO
"""

# Import all constants from organized modules for backward compatibility
# This maintains 100% backward compatibility with existing code
from .constants import *

# Re-export everything for backward compatibility
__all__ = [
    # Enums
    'OptimizationFramework',
    'OptimizationLevel',
    'OptimizationType',
    'OptimizationTechnique',
    'OptimizationMetric',
    'OptimizationResult',
    # Performance
    'SPEED_IMPROVEMENTS',
    'MEMORY_REDUCTIONS',
    'ENERGY_EFFICIENCIES',
    'ACCURACY_PRESERVATIONS',
    'FRAMEWORK_BENEFITS',
    'TECHNIQUE_BENEFITS',
    'PERFORMANCE_THRESHOLDS',
    # Configurations
    'DEFAULT_CONFIGS',
    'OPTIMIZATION_PROFILES',
    'HARDWARE_CONFIGS',
    'SOFTWARE_CONFIGS',
    'MODEL_CONFIGS',
    'DATASET_CONFIGS',
    'TRAINING_CONFIGS',
    'EVALUATION_CONFIGS',
    'DEPLOYMENT_CONFIGS',
    'MONITORING_CONFIGS',
    'LOGGING_CONFIGS',
    'SECURITY_CONFIGS',
    'COMPLIANCE_CONFIGS',
    'QUALITY_CONFIGS',
    'DOCUMENTATION_CONFIGS',
    # Messages
    'ERROR_MESSAGES',
    'SUCCESS_MESSAGES',
    'WARNING_MESSAGES',
    'INFO_MESSAGES',
    'DEBUG_MESSAGES',
    # Version
    'VERSION_INFO',
]

