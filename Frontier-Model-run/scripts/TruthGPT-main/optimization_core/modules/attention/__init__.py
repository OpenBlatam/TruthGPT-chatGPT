"""
Attention module for TruthGPT Optimization Core
Contains multi-head attention and specialized attention implementations.
"""

from __future__ import annotations
from optimization_core.utils.dependency_manager import resolve_lazy_import

_LAZY_IMPORTS = {
    # Multi-Head Attention
    'MultiHeadAttention': '.multi_head_attention',
    'ScaledDotProductAttention': '.multi_head_attention',
    'create_multi_head_attention': '.multi_head_attention',
    
    # Flash Attention
    'FlashAttention': '.flash_attention',
    'FlashAttentionV2': '.flash_attention',
    'create_flash_attention': '.flash_attention',
    
    # Sparse Attention
    'SparseAttention': '.sparse_attention',
    'LocalAttention': '.sparse_attention',
    'StridedAttention': '.sparse_attention',
    'create_sparse_attention': '.sparse_attention',
    
    # Cross Attention
    'CrossAttention': '.cross_attention',
    'create_cross_attention': '.cross_attention',
}

def __getattr__(name: str):
    """Lazy import system for attention components."""
    return resolve_lazy_import(name, __package__ or 'attention', _LAZY_IMPORTS)

__all__ = list(_LAZY_IMPORTS.keys())

