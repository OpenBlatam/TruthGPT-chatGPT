"""
Ultra-fast modular model components
Following deep learning best practices
"""

from .transformer import TransformerModel, TransformerConfig, TransformerBlock
from .attention import MultiHeadAttention, FlashAttention, AttentionConfig
from .feedforward import FeedForward, SwiGLU, GatedMLP, FeedForwardConfig
from .embeddings import TokenEmbedding, PositionalEmbedding, EmbeddingConfig
from .normalization import LayerNorm, RMSNorm, NormalizationConfig
from .optimizer import ModelOptimizer, OptimizerConfig
from .compiler import ModelCompiler, CompilationConfig

__all__ = [
    # Transformer components
    'TransformerModel', 'TransformerConfig', 'TransformerBlock',
    
    # Attention
    'MultiHeadAttention', 'FlashAttention', 'AttentionConfig',
    
    # Feedforward
    'FeedForward', 'SwiGLU', 'GatedMLP', 'FeedForwardConfig',
    
    # Embeddings
    'TokenEmbedding', 'PositionalEmbedding', 'EmbeddingConfig',
    
    # Normalization
    'LayerNorm', 'RMSNorm', 'NormalizationConfig',
    
    # Optimization
    'ModelOptimizer', 'OptimizerConfig',
    
    # Compilation
    'ModelCompiler', 'CompilationConfig'
]
