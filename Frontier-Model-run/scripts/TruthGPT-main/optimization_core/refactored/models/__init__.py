"""
Deep Learning Models Module
===========================

Production-ready model implementations following deep learning best practices:
- PyTorch-based model architectures
- Proper weight initialization and normalization
- Mixed precision training support
- GPU utilization optimization
- Modular and extensible design
"""

from .base import BaseModel, ModelConfig
from .transformer import TransformerOptimizer, TransformerConfig
from .diffusion import DiffusionOptimizer, DiffusionConfig
from .hybrid import HybridOptimizer, HybridConfig
from .quantum import QuantumOptimizer, QuantumConfig

__all__ = [
    'BaseModel',
    'ModelConfig',
    'TransformerOptimizer',
    'TransformerConfig',
    'DiffusionOptimizer', 
    'DiffusionConfig',
    'HybridOptimizer',
    'HybridConfig',
    'QuantumOptimizer',
    'QuantumConfig'
]



