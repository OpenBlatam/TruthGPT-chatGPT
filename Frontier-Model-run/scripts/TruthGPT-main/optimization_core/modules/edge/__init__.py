"""
Edge Computing Module
======================

Edge inference and deployment adapters for optimization_core.
"""

from .edge_inference_adapter import EdgeInferenceAdapter, EdgeGenerateResult, EdgeLoadResult

__all__ = [
    'EdgeInferenceAdapter',
    'EdgeGenerateResult',
    'EdgeLoadResult',
]

