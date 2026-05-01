"""
Constants Module for TruthGPT Optimization Core

This module provides organized access to all optimization constants.
All constants are re-exported here for backward compatibility.
"""

# Import all constants from organized submodules
from .enums import (
    OptimizationFramework,
    OptimizationLevel,
    OptimizationType,
    OptimizationTechnique,
    OptimizationMetric,
    OptimizationResult,
)

from .performance import (
    SPEED_IMPROVEMENTS,
    MEMORY_REDUCTIONS,
    ENERGY_EFFICIENCIES,
    ACCURACY_PRESERVATIONS,
    FRAMEWORK_BENEFITS,
    TECHNIQUE_BENEFITS,
    PERFORMANCE_THRESHOLDS,
)

from .configurations import (
    DEFAULT_CONFIGS,
    OPTIMIZATION_PROFILES,
    HARDWARE_CONFIGS,
    SOFTWARE_CONFIGS,
    MODEL_CONFIGS,
    DATASET_CONFIGS,
    TRAINING_CONFIGS,
    EVALUATION_CONFIGS,
    DEPLOYMENT_CONFIGS,
    MONITORING_CONFIGS,
    LOGGING_CONFIGS,
    SECURITY_CONFIGS,
    COMPLIANCE_CONFIGS,
    QUALITY_CONFIGS,
    DOCUMENTATION_CONFIGS,
)

from .messages import (
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    WARNING_MESSAGES,
    INFO_MESSAGES,
    DEBUG_MESSAGES,
)

from .version import VERSION_INFO

# Export all constants for backward compatibility
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


