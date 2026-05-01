"""
Transformer Block module for TruthGPT Optimization Core
Contains transformer block implementations with various optimizations
"""

from .transformer_block import (
    TransformerBlock,
    PreNormTransformerBlock,
    PostNormTransformerBlock,
    create_transformer_block
)

from .attention_block import (
    AttentionBlock,
    create_attention_block
)

from .feed_forward_block import (
    FeedForwardBlock,
    create_feed_forward_block
)

__all__ = [
    # Transformer Blocks
    'TransformerBlock',
    'PreNormTransformerBlock',
    'PostNormTransformerBlock',
    'create_transformer_block',
    
    # Attention Blocks
    'AttentionBlock',
    'create_attention_block',
    
    # Feed-Forward Blocks
    'FeedForwardBlock',
    'create_feed_forward_block'
]





