"""
Efficient Transformer Decoder with Optimized K/V Caching
Implements the suggested improvements for TruthGPT inference optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math
import logging
from dataclasses import dataclass
import time
from contextlib import contextmanager

from ..attention.efficient_kv_cache import (
    EfficientMultiHeadAttention, 
    KVCacheConfig, 
    PrefillDecodeOptimizer,
    MemoryEfficientAttention
)

logger = logging.getLogger(__name__)

@dataclass
class DecoderConfig:
    """Configuration for efficient decoder."""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    use_kv_cache: bool = True
    cache_config: Optional[KVCacheConfig] = None
    use_flash_attention: bool = True
    use_memory_efficient_attention: bool = True
    max_sequence_length: int = 2048

class EfficientTransformerDecoder(nn.Module):
    """
    Efficient Transformer Decoder with optimized K/V caching.
    
    This implementation provides:
    - Efficient K/V cache reuse for sequential generation
    - Separate prefill and decode phases
    - Memory-optimized attention computation
    - Automatic cache management
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.d_ff = config.d_ff
        self.dropout = config.dropout
        self.activation = config.activation
        self.max_sequence_length = config.max_sequence_length
        
        # Initialize components
        self._build_layers()
        self._setup_optimizations()
        
        # Performance tracking
        self.prefill_time = 0.0
        self.decode_times = []
        self.cache_hit_rates = []
    
    def _build_layers(self):
        """Build the decoder layers."""
        # Embedding layers
        self.embedding = nn.Embedding(self.config.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_sequence_length, self.d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            EfficientDecoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=self.activation,
                use_kv_cache=self.config.use_kv_cache,
                cache_config=self.config.cache_config,
                layer_idx=i
            )
            for i in range(self.n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.config.vocab_size)
    
    def _setup_optimizations(self):
        """Setup optimization components."""
        # Memory efficient attention
        if self.config.use_memory_efficient_attention:
            self.memory_optimizer = MemoryEfficientAttention(
                use_checkpointing=True,
                use_mixed_precision=True
            )
        else:
            self.memory_optimizer = None
        
        # Prefill/decode optimizer
        self.prefill_decode_optimizer = PrefillDecodeOptimizer(
            self, 
            self.config.cache_config
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        cache_position: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with efficient K/V caching.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            use_cache: Whether to use K/V cache
            cache_position: Position in sequence for caching
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing outputs and cache information
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # Get embeddings
        x = self.embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Process through decoder layers
        hidden_states = x
        cache_info = {}
        
        for i, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )
            
            hidden_states = layer_output['hidden_states']
            
            # Collect cache information
            if 'cache_info' in layer_output:
                cache_info[f'layer_{i}'] = layer_output['cache_info']
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Compute logits
        logits = self.output_projection(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'cache_info': cache_info
        }
    
    def prefill_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process the prefill phase (initial prompt processing).
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        start_time = time.time()
        
        # Use memory-efficient forward if enabled
        if self.memory_optimizer:
            with self.memory_optimizer.memory_efficient_forward(self):
                outputs = self.prefill_decode_optimizer.prefill_phase(
                    input_ids, attention_mask, **kwargs
                )
        else:
            outputs = self.prefill_decode_optimizer.prefill_phase(
                input_ids, attention_mask, **kwargs
            )
        
        self.prefill_time = time.time() - start_time
        logger.info(f"Prefill phase completed in {self.prefill_time:.4f}s")
        
        return outputs
    
    def decode_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process the decode phase (token-by-token generation).
        
        Args:
            input_ids: New token IDs (typically single token)
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        start_time = time.time()
        
        # Use memory-efficient forward if enabled
        if self.memory_optimizer:
            with self.memory_optimizer.memory_efficient_forward(self):
                outputs = self.prefill_decode_optimizer.decode_phase(
                    input_ids, attention_mask, **kwargs
                )
        else:
            outputs = self.prefill_decode_optimizer.decode_phase(
                input_ids, attention_mask, **kwargs
            )
        
        decode_time = time.time() - start_time
        self.decode_times.append(decode_time)
        
        # Track cache performance
        cache_stats = self.get_cache_stats()
        if cache_stats:
            self.cache_hit_rates.append(cache_stats.get('hit_rate', 0.0))
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using efficient K/V caching.
        
        Args:
            input_ids: Initial input tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            **kwargs: Additional arguments
            
        Returns:
            Generated token IDs
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Process prefill phase
        outputs = self.prefill_phase(input_ids, **kwargs)
        generated_ids = input_ids.clone()
        
        # Generate tokens one by one
        for _ in range(max_length - input_ids.size(1)):
            # Get logits for the last token
            logits = outputs['logits'][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Process decode phase for next token
            outputs = self.decode_phase(next_token, **kwargs)
        
        return generated_ids
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from all layers."""
        stats = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_cache_stats'):
                layer_stats = layer.get_cache_stats()
                if layer_stats:
                    stats[f'layer_{i}'] = layer_stats
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all K/V caches."""
        for layer in self.layers:
            if hasattr(layer, 'clear_cache'):
                layer.clear_cache()
        
        if hasattr(self.prefill_decode_optimizer, 'reset'):
            self.prefill_decode_optimizer.reset()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_decode_time = sum(self.decode_times) / len(self.decode_times) if self.decode_times else 0.0
        avg_cache_hit_rate = sum(self.cache_hit_rates) / len(self.cache_hit_rates) if self.cache_hit_rates else 0.0
        
        return {
            'prefill_time': self.prefill_time,
            'avg_decode_time': avg_decode_time,
            'total_decode_tokens': len(self.decode_times),
            'avg_cache_hit_rate': avg_cache_hit_rate,
            'cache_stats': self.get_cache_stats()
        }

class EfficientDecoderLayer(nn.Module):
    """
    Single decoder layer with efficient K/V caching.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_kv_cache: bool = True,
        cache_config: Optional[KVCacheConfig] = None,
        layer_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.layer_idx = layer_idx
        
        # Self-attention with K/V cache
        self.self_attention = EfficientMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_kv_cache=use_kv_cache,
            cache_config=cache_config
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU()
        }
        return activations.get(activation, nn.GELU())
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        cache_position: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the decoder layer.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            use_cache: Whether to use K/V cache
            cache_position: Position in sequence for caching
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing outputs and cache information
        """
        # Self-attention with residual connection
        attn_output, cache_info = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            mask=attention_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        
        hidden_states = self.norm1(hidden_states + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + ffn_output)
        
        return {
            'hidden_states': hidden_states,
            'cache_info': cache_info
        }
    
    def clear_cache(self) -> None:
        """Clear K/V cache for this layer."""
        if hasattr(self.self_attention, 'clear_cache'):
            self.self_attention.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this layer."""
        if hasattr(self.self_attention, 'get_cache_stats'):
            return self.self_attention.get_cache_stats()
        return {}

# Factory functions
def create_efficient_decoder(config: DecoderConfig) -> EfficientTransformerDecoder:
    """Create an efficient transformer decoder."""
    return EfficientTransformerDecoder(config)

def create_decoder_config(
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1,
    activation: str = "gelu",
    use_kv_cache: bool = True,
    cache_config: Optional[KVCacheConfig] = None,
    **kwargs
) -> DecoderConfig:
    """Create a decoder configuration."""
    return DecoderConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        use_kv_cache=use_kv_cache,
        cache_config=cache_config,
        **kwargs
    )





