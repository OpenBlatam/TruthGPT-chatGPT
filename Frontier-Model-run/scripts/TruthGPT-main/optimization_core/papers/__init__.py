"""
Research Papers Implementations for TruthGPT Optimization Core.
These modules contain exact implementations of state-of-the-art research papers.
"""

from .fp16_stability import FP16Stability
from .elastic_reasoning import ElasticReasoning
from .chain_of_draft import ChainOfDraft

__all__ = [
    "FP16Stability",
    "ElasticReasoning",
    "ChainOfDraft"
]

