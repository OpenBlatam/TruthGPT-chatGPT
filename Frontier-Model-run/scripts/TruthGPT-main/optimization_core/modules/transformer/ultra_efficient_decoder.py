"""
Ultra-Efficient Transformer Decoder with Advanced K/V Caching
Optimized prefill and decode phases with minimal memory overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
import math
import logging
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from enum import Enum

from ..attention.ultra_efficient_kv_cache import (
    UltraEfficientAttention,
    UltraKVCacheConfig,
    create_ultra_efficient_attention,
    create_ultra_cache_config
)

logger = logging.getLogger(__name__)

class DecodePhase(Enum):
    """Decode phase types."""
    PREFILL = "prefill"      # Process entire prompt
    DECODE = "decode"        # Generate token by token
    HYBRID = "hybrid"        # Mixed prefill/decode

class MemoryStrategy(Enum):
    """Memory management strategies."""
    AGGRESSIVE = "aggressive"    # Maximum memory optimization
    BALANCED = "balanced"        # Balanced memory/speed
    SPEED = "speed"              # Maximum speed

@dataclass
class UltraDecoderConfig:
    """Ultra-efficient decoder configuration."""
    
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    vocab_size: int = 50000
    max_sequence_length: int = 4096
    
    # Attention configuration
    dropout: float = 0.1
    activation: str = "gelu"
    use_sparse_attention: bool = True
    sparse_attention_ratio: float = 0.1
    
    # Cache configuration
    cache_config: Optional[UltraKVCacheConfig] = None
    use_kv_cache: bool = True
    cache_warming_enabled: bool = True
    prefetch_enabled: bool = True
    
    # Memory optimization
    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    use_gradient_checkpointing: bool = True
    use_activation_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # Performance optimization
    use_parallel_processing: bool = True
    num_workers: int = 4
    use_cuda_streams: bool = True
    use_async_processing: bool = True
    
    # Advanced features
    use_quantization: bool = True
    quantization_bits: int = 8
    use_compression: bool = True
    compression_ratio: float = 0.3
    
    # Monitoring
    enable_profiling: bool = True
    enable_metrics: bool = True
    log_frequency: int = 100

class UltraEfficientDecoder(nn.Module):
    """
    Ultra-efficient transformer decoder with advanced optimizations.
    
    Features:
    - Optimized prefill and decode phases
    - Advanced K/V caching
    - Memory-efficient attention
    - Parallel processing
    - Quantization and compression
    """
    
    def __init__(self, config: UltraDecoderConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.d_ff = config.d_ff
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length
        
        # Initialize components
        self._build_layers()
        self._setup_optimizations()
        self._setup_performance_tracking()
        
        # Phase management
        self.current_phase = DecodePhase.PREFILL
        self.current_position = 0
        self.sequence_length = 0
        
        # Performance tracking
        self.prefill_times = []
        self.decode_times = []
        self.cache_hit_rates = []
        self.memory_usage = []
        
    def _build_layers(self):
        """Build decoder layers with optimizations."""
        # Embedding layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_sequence_length, self.d_model)
        
        # Create cache config if not provided
        if self.config.cache_config is None:
            self.config.cache_config = create_ultra_cache_config(
                max_cache_size=self.max_sequence_length,
                use_compression=self.config.use_compression,
                compression_ratio=self.config.compression_ratio
            )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            UltraDecoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.config.dropout,
                activation=self.config.activation,
                cache_config=self.config.cache_config,
                use_sparse_attention=self.config.use_sparse_attention,
                sparse_attention_ratio=self.config.sparse_attention_ratio,
                layer_idx=i
            )
            for i in range(self.n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Quantization
        if self.config.use_quantization:
            self._apply_quantization()
    
    def _setup_optimizations(self):
        """Setup performance optimizations."""
        # Gradient checkpointing
        if self.config.use_gradient_checkpointing:
            for layer in self.layers:
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()
        
        # Mixed precision
        if self.config.use_mixed_precision:
            self.half()
        
        # CUDA streams for parallel processing
        if torch.cuda.is_available() and self.config.use_cuda_streams:
            self.cuda_streams = [torch.cuda.Stream() for _ in range(self.config.num_workers)]
        else:
            self.cuda_streams = None
        
        # Thread pool for async processing
        if self.config.use_async_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_workers)
        else:
            self.thread_pool = None
    
    def _setup_performance_tracking(self):
        """Setup performance tracking."""
        self.performance_metrics = {
            'total_prefill_time': 0.0,
            'total_decode_time': 0.0,
            'total_tokens_generated': 0,
            'cache_hit_rate': 0.0,
            'memory_usage': 0.0,
            'throughput': 0.0
        }
    
    def _apply_quantization(self):
        """Apply quantization to the model."""
        if self.config.quantization_bits == 8:
            # 8-bit quantization
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = module.weight.data.round().clamp(-128, 127)
        elif self.config.quantization_bits == 4:
            # 4-bit quantization
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = module.weight.data.round().clamp(-8, 7)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        cache_position: Optional[int] = None,
        phase: Optional[DecodePhase] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Ultra-efficient forward pass with phase optimization.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            use_cache: Whether to use K/V cache
            cache_position: Position in sequence for caching
            phase: Decode phase (prefill/decode)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing outputs and cache information
        """
        batch_size, seq_len = input_ids.size()
        
        # Determine phase
        if phase is None:
            phase = self._determine_phase(seq_len)
        
        self.current_phase = phase
        self.sequence_length = seq_len
        
        # Create position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # Get embeddings
        x = self.embedding(input_ids)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        # Apply dropout
        x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # Process through decoder layers with phase optimization
        hidden_states = x
        cache_info = {}
        
        for i, layer in enumerate(self.layers):
            if phase == DecodePhase.PREFILL:
                # Prefill phase - process entire sequence
                layer_output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    phase=phase,
                    **kwargs
                )
            else:
                # Decode phase - process with caching
                layer_output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    phase=phase,
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
            'cache_info': cache_info,
            'phase': phase.value
        }
    
    def _determine_phase(self, seq_len: int) -> DecodePhase:
        """Determine the appropriate decode phase."""
        if seq_len > 1:
            return DecodePhase.PREFILL
        else:
            return DecodePhase.DECODE
    
    def prefill_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized prefill phase processing.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        start_time = time.time()
        
        # Use memory-efficient forward
        with self._memory_efficient_context():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                phase=DecodePhase.PREFILL,
                **kwargs
            )
        
        prefill_time = time.time() - start_time
        self.prefill_times.append(prefill_time)
        self.performance_metrics['total_prefill_time'] += prefill_time
        
        # Cache warming
        if self.config.cache_warming_enabled:
            self._warm_cache(input_ids, attention_mask)
        
        logger.info(f"Prefill phase completed in {prefill_time:.4f}s")
        return outputs
    
    def decode_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Optimized decode phase processing.
        
        Args:
            input_ids: New token IDs (typically single token)
            attention_mask: Attention mask
            cache_position: Position in sequence for caching
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        start_time = time.time()
        
        # Use memory-efficient forward
        with self._memory_efficient_context():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                cache_position=cache_position,
                phase=DecodePhase.DECODE,
                **kwargs
            )
        
        decode_time = time.time() - start_time
        self.decode_times.append(decode_time)
        self.performance_metrics['total_decode_time'] += decode_time
        self.performance_metrics['total_tokens_generated'] += 1
        
        # Track cache performance
        cache_stats = self.get_cache_stats()
        if cache_stats:
            hit_rate = cache_stats.get('hit_rate', 0.0)
            self.cache_hit_rates.append(hit_rate)
            self.performance_metrics['cache_hit_rate'] = sum(self.cache_hit_rates) / len(self.cache_hit_rates)
        
        # Update position
        self.current_position += 1
        
        return outputs
    
    def _memory_efficient_context(self):
        """Memory-efficient context manager."""
        if self.config.memory_strategy == MemoryStrategy.AGGRESSIVE:
            return self._aggressive_memory_context()
        elif self.config.memory_strategy == MemoryStrategy.BALANCED:
            return self._balanced_memory_context()
        else:
            return self._speed_memory_context()
    
    @contextmanager
    def _aggressive_memory_context(self):
        """Aggressive memory optimization context."""
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            for layer in self.layers:
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()
        
        # Enable activation checkpointing
        if self.config.use_activation_checkpointing:
            for layer in self.layers:
                if hasattr(layer, 'activation_checkpointing_enable'):
                    layer.activation_checkpointing_enable()
        
        try:
            yield
        finally:
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @contextmanager
    def _balanced_memory_context(self):
        """Balanced memory optimization context."""
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            for layer in self.layers:
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()
        
        try:
            yield
        finally:
            # Light cleanup
            gc.collect()
    
    @contextmanager
    def _speed_memory_context(self):
        """Speed-optimized context."""
        try:
            yield
        finally:
            # Minimal cleanup
            pass
    
    def _warm_cache(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Warm up the cache for better performance."""
        if not self.config.cache_warming_enabled:
            return
        
        # Process a few tokens to warm up the cache
        warmup_tokens = min(10, input_ids.size(1))
        warmup_ids = input_ids[:, :warmup_tokens]
        
        with torch.no_grad():
            self.forward(
                input_ids=warmup_ids,
                attention_mask=attention_mask[:, :warmup_tokens] if attention_mask is not None else None,
                use_cache=True,
                phase=DecodePhase.PREFILL
            )
    
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
        Generate text using ultra-efficient K/V caching.
        
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
        for i in range(max_length - input_ids.size(1)):
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
            outputs = self.decode_phase(
                next_token, 
                cache_position=input_ids.size(1) + i,
                **kwargs
            )
        
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
        
        # Reset position tracking
        self.current_position = 0
        self.sequence_length = 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        # Calculate throughput
        total_time = self.performance_metrics['total_prefill_time'] + self.performance_metrics['total_decode_time']
        total_tokens = self.performance_metrics['total_tokens_generated']
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        
        # Calculate average times
        avg_prefill_time = sum(self.prefill_times) / len(self.prefill_times) if self.prefill_times else 0.0
        avg_decode_time = sum(self.decode_times) / len(self.decode_times) if self.decode_times else 0.0
        
        # Calculate cache hit rate
        cache_hit_rate = sum(self.cache_hit_rates) / len(self.cache_hit_rates) if self.cache_hit_rates else 0.0
        
        return {
            'performance_metrics': self.performance_metrics,
            'avg_prefill_time': avg_prefill_time,
            'avg_decode_time': avg_decode_time,
            'cache_hit_rate': cache_hit_rate,
            'throughput': throughput,
            'cache_stats': self.get_cache_stats(),
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if torch.cuda.is_available():
            return {
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024 * 1024),  # MB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024 * 1024),    # MB
                'gpu_memory_cached': torch.cuda.memory_cached() / (1024 * 1024)         # MB
            }
        else:
            return {
                'cpu_memory_usage': 0.0
            }

class UltraDecoderLayer(nn.Module):
    """
    Ultra-efficient decoder layer with advanced optimizations.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        cache_config: Optional[UltraKVCacheConfig] = None,
        use_sparse_attention: bool = True,
        sparse_attention_ratio: float = 0.1,
        layer_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.layer_idx = layer_idx
        
        # Self-attention with ultra-efficient caching
        self.self_attention = create_ultra_efficient_attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            cache_config=cache_config,
            use_sparse_attention=use_sparse_attention
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
        phase: Optional[DecodePhase] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the decoder layer.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            use_cache: Whether to use K/V cache
            cache_position: Position in sequence for caching
            phase: Decode phase
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
def create_ultra_efficient_decoder(config: UltraDecoderConfig) -> UltraEfficientDecoder:
    """Create an ultra-efficient transformer decoder."""
    return UltraEfficientDecoder(config)

def create_ultra_decoder_config(**kwargs) -> UltraDecoderConfig:
    """Create an ultra-efficient decoder configuration."""
    return UltraDecoderConfig(**kwargs)





