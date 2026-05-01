"""
Ultra-Modular Efficient Decoder for Prefill and Decode Phases
Separates prefill (process prompt) and decode (generate token by token) phases
Optimizes for K/V cache reuse and minimal memory overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

# Import the modular K/V cache
from ..attention.ultra_modular_kv_cache import (
    KVCacheModule,
    KVCacheConfig,
    create_kv_cache,
    create_kv_cache_config,
    CacheStrategy,
    MemoryLayout
)

logger = logging.getLogger(__name__)

class DecodePhase(Enum):
    """Decode phase types."""
    PREFILL = "prefill"          # Process entire prompt
    DECODE = "decode"            # Generate token by token
    BATCH_DECODE = "batch_decode" # Batch token generation

class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    CONSERVATIVE = "conservative"   # Minimal memory usage
    BALANCED = "balanced"           # Balance speed and memory
    AGGRESSIVE = "aggressive"       # Speed over memory
    AUTO = "auto"                   # Automatic based on resources

@dataclass
class DecoderConfig:
    """Configuration for efficient decoder."""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    max_sequence_length: int = 4096
    use_cache: bool = True
    cache_config: Optional[KVCacheConfig] = None
    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False
    batch_size: int = 1
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UltraModularDecoder(nn.Module):
    """
    Ultra-Modular Efficient Decoder with Prefill and Decode phases.
    
    Key features:
    - Separate prefill (process prompt) and decode (generate token by token) phases
    - Reuses K/V cache for each new token instead of recalculating from scratch
    - Minimizes memory overhead and latency between tokens
    - Modular design for easy integration
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Setup K/V cache
        self._setup_kv_cache()
        
        # Decoder layers
        self._setup_decoder_layers()
        
        # Performance tracking
        self.performance_stats = {
            'prefill_time': [],
            'decode_time': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tokens': 0
        }
        
        logger.info(f"Ultra-Modular Decoder initialized with {config.n_layers} layers")
    
    def _setup_kv_cache(self):
        """Setup K/V cache system."""
        if self.config.cache_config is None:
            cache_config = create_kv_cache_config(
                max_cache_size=self.config.max_sequence_length,
                cache_strategy=CacheStrategy.ADAPTIVE,
                memory_layout=MemoryLayout.DENSE,
                use_compression=True,
                compression_ratio=0.3,
                use_quantization=True,
                quantization_bits=8
            )
        else:
            cache_config = self.config.cache_config
        
        self.kv_cache = create_kv_cache(cache_config)
        self.kv_cache.to(self.device)
    
    def _setup_decoder_layers(self):
        """Setup decoder layers."""
        self.layers = nn.ModuleList([
            self._create_decoder_layer()
            for _ in range(self.config.n_layers)
        ])
    
    def _create_decoder_layer(self) -> nn.Module:
        """Create a single decoder layer."""
        return DecoderLayer(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            use_flash_attention=self.config.use_flash_attention
        )
    
    def prefill_phase(self, input_ids: torch.Tensor, 
                     attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Prefill phase: Process entire prompt and populate K/V cache.
        
        Args:
            input_ids: Input token ids
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with outputs and cache state
        """
        start_time = time.time()
        
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings
        x = self._create_embeddings(input_ids)
        
        # Process through decoder layers and populate cache
        cache_state = {}
        
        for layer_id, layer in enumerate(self.layers):
            # Self-attention with cache building
            x, cache_entry = self._self_attention_with_cache(
                x, layer_id, cache_state
            )
            
            # Feed-forward
            x = layer.feed_forward(x)
            
            # Store cache entry
            cache_state[layer_id] = cache_entry
        
        # Compute final output
        output = self._compute_output(x)
        
        prefill_time = time.time() - start_time
        self.performance_stats['prefill_time'].append(prefill_time)
        
        logger.info(f"Prefill phase completed in {prefill_time:.4f}s for {seq_len} tokens")
        
        return {
            'output': output,
            'cache_state': cache_state,
            'prefill_time': prefill_time,
            'phase': DecodePhase.PREFILL
        }
    
    def decode_phase(self, last_token_ids: torch.Tensor, 
                    cache_state: Dict[int, Any]) -> Dict[str, Any]:
        """
        Decode phase: Generate next token using cached K/V.
        
        Args:
            last_token_ids: Last generated token ids
            cache_state: Cached K/V from previous tokens
            
        Returns:
            Dictionary with outputs and updated cache state
        """
        start_time = time.time()
        
        batch_size = last_token_ids.shape[0]
        
        # Create embeddings for last token
        x = self._create_embeddings(last_token_ids)
        
        # Process through decoder layers with cache reuse
        new_cache_state = {}
        
        for layer_id, layer in enumerate(self.layers):
            # Get existing cache entry
            old_cache = cache_state.get(layer_id)
            
            # Self-attention with cache reuse
            x, new_cache_entry = self._self_attention_with_cache_reuse(
                x, layer_id, old_cache, new_cache_state
            )
            
            # Feed-forward
            x = layer.feed_forward(x)
            
            # Store new cache entry
            new_cache_state[layer_id] = new_cache_entry
        
        # Compute final output
        output = self._compute_output(x)
        
        decode_time = time.time() - start_time
        self.performance_stats['decode_time'].append(decode_time)
        self.performance_stats['total_tokens'] += batch_size
        
        logger.debug(f"Decode phase completed in {decode_time:.4f}s")
        
        return {
            'output': output,
            'cache_state': new_cache_state,
            'decode_time': decode_time,
            'phase': DecodePhase.DECODE
        }
    
    def _self_attention_with_cache(self, x: torch.Tensor, layer_id: int,
                                   cache_state: Dict[int, Any]) -> Tuple[torch.Tensor, Any]:
        """Compute self-attention and build cache."""
        # Generate Q, K, V
        q, k, v = self._generate_qkv(x, layer_id)
        
        # Compute attention
        attn_output = self._compute_attention(q, k, v)
        
        # Store in cache
        position = cache_state.get('position', 0)
        cache_entry = {
            'key': k,
            'value': v,
            'position': position
        }
        
        self.kv_cache.set_cache_entry(layer_id, position, k, v)
        
        # Update position
        cache_state['position'] = position + 1
        
        return attn_output, cache_entry
    
    def _self_attention_with_cache_reuse(self, x: torch.Tensor, layer_id: int,
                                        old_cache: Any,
                                        cache_state: Dict[int, Any]) -> Tuple[torch.Tensor, Any]:
        """Compute self-attention with K/V cache reuse."""
        # Generate Q, K, V for new token
        q, k, v = self._generate_qkv(x, layer_id)
        
        # Reuse old cache if available
        if old_cache is not None:
            # Get cached K, V
            cached_k = old_cache['key']
            cached_v = old_cache['value']
            
            # Concatenate new K/V to cached K/V
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            
            self.performance_stats['cache_hits'] += 1
        else:
            self.performance_stats['cache_misses'] += 1
        
        # Compute attention with concatenated K/V
        attn_output = self._compute_attention(q, k, v)
        
        # Store new cache entry
        position = cache_state.get('position', 0)
        cache_entry = {
            'key': k,
            'value': v,
            'position': position
        }
        
        self.kv_cache.update_cache_for_token(layer_id, position, k, v)
        
        # Update position
        cache_state['position'] = position + 1
        
        return attn_output, cache_entry
    
    def _generate_qkv(self, x: torch.Tensor, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate Q, K, V from input."""
        layer = self.layers[layer_id]
        q = layer.self_attn.q_proj(x)
        k = layer.self_attn.k_proj(x)
        v = layer.self_attn.v_proj(x)
        
        return q, k, v
    
    def _compute_attention(self, q: torch.Tensor, k: torch.Tensor,
                          v: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        scale = 1.0 / (q.size(-1) ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)
        
        return output
    
    def _create_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create embeddings from input ids."""
        # Simplified embedding
        embedding = nn.Embedding(
            num_embeddings=50257,
            embedding_dim=self.config.d_model
        ).to(self.device)
        
        return embedding(input_ids)
    
    def _compute_output(self, x: torch.Tensor) -> torch.Tensor:
        """Compute final output from decoder state."""
        # Simplified output layer
        output_layer = nn.Linear(
            self.config.d_model,
            50257  # vocabulary size
        ).to(self.device)
        
        return output_layer(x)
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text by alternating prefill and decode phases.
        
        Args:
            input_ids: Initial input ids
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token ids
        """
        generated_ids = input_ids.clone()
        
        # Prefill phase
        prefill_result = self.prefill_phase(input_ids)
        cache_state = prefill_result['cache_state']
        
        # Decode phase - generate tokens one by one
        for _ in range(max_length - input_ids.shape[1]):
            # Get last token
            last_token_ids = generated_ids[:, -1:]
            
            # Decode next token
            decode_result = self.decode_phase(last_token_ids, cache_state)
            
            # Get next token
            next_token_logits = decode_result['output'][:, -1, :]
            next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)
            
            # Append to generated ids
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Update cache state
            cache_state = decode_result['cache_state']
        
        return generated_ids
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_calls = len(self.performance_stats['prefill_time']) + len(self.performance_stats['decode_time'])
        cache_hits = self.performance_stats['cache_hits']
        cache_misses = self.performance_stats['cache_misses']
        
        avg_prefill_time = sum(self.performance_stats['prefill_time']) / len(self.performance_stats['prefill_time']) if self.performance_stats['prefill_time'] else 0.0
        avg_decode_time = sum(self.performance_stats['decode_time']) / len(self.performance_stats['decode_time']) if self.performance_stats['decode_time'] else 0.0
        
        cache_hit_rate = (cache_hits / (cache_hits + cache_misses) * 100) if (cache_hits + cache_misses) > 0 else 0.0
        
        return {
            'performance_stats': self.performance_stats.copy(),
            'cache_stats': self.kv_cache.get_cache_stats(),
            'avg_prefill_time': avg_prefill_time,
            'avg_decode_time': avg_decode_time,
            'total_tokens': self.performance_stats['total_tokens'],
            'cache_hit_rate': cache_hit_rate,
            'total_calls': total_calls
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.kv_cache.clear_cache()
        
        # Reset performance stats
        self.performance_stats = {
            'prefill_time': [],
            'decode_time': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'total_tokens': 0
        }

class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, use_flash_attention: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder layer."""
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

# Factory functions
def create_ultra_modular_decoder(config: DecoderConfig = None) -> UltraModularDecoder:
    """Create an ultra-modular decoder."""
    if config is None:
        config = DecoderConfig()
    return UltraModularDecoder(config)

def create_decoder_config(**kwargs) -> DecoderConfig:
    """Create a decoder configuration."""
    return DecoderConfig(**kwargs)



