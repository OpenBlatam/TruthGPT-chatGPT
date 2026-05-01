"""
Builder pattern for polyglot_core.

Provides fluent builders for complex configurations.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .cache import KVCache, KVCacheConfig, EvictionStrategy
from .attention import Attention, AttentionConfig, AttentionPattern
from .compression import Compressor, CompressionConfig, CompressionAlgorithm
from .inference import InferenceEngine, InferenceConfig, GenerationConfig
from .backend import Backend


class CacheBuilder:
    """Builder for KV Cache configuration."""
    
    def __init__(self):
        self._max_size: Optional[int] = None
        self._eviction_strategy: Optional[EvictionStrategy] = None
        self._backend: Optional[Backend] = None
        self._enable_compression: bool = False
        self._enable_quantization: bool = False
        self._async_mode: bool = False
        self._metadata: Dict[str, Any] = {}
    
    def with_max_size(self, max_size: int) -> 'CacheBuilder':
        """Set maximum cache size."""
        self._max_size = max_size
        return self
    
    def with_eviction_strategy(self, strategy: EvictionStrategy) -> 'CacheBuilder':
        """Set eviction strategy."""
        self._eviction_strategy = strategy
        return self
    
    def with_backend(self, backend: Backend) -> 'CacheBuilder':
        """Set backend."""
        self._backend = backend
        return self
    
    def with_compression(self, enabled: bool = True) -> 'CacheBuilder':
        """Enable/disable compression."""
        self._enable_compression = enabled
        return self
    
    def with_quantization(self, enabled: bool = True) -> 'CacheBuilder':
        """Enable/disable quantization."""
        self._enable_quantization = enabled
        return self
    
    def with_async_mode(self, enabled: bool = True) -> 'CacheBuilder':
        """Enable/disable async mode."""
        self._async_mode = enabled
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'CacheBuilder':
        """Add metadata."""
        self._metadata[key] = value
        return self
    
    def build(self) -> KVCache:
        """Build KV Cache instance."""
        config = KVCacheConfig(
            max_size=self._max_size or 8192,
            eviction_strategy=self._eviction_strategy or EvictionStrategy.LRU,
            enable_compression=self._enable_compression,
            enable_quantization=self._enable_quantization,
            async_mode=self._async_mode,
            **self._metadata
        )
        
        return KVCache(config=config, backend=self._backend)


class AttentionBuilder:
    """Builder for Attention configuration."""
    
    def __init__(self):
        self._d_model: Optional[int] = None
        self._n_heads: Optional[int] = None
        self._d_kv: Optional[int] = None
        self._backend: Optional[Backend] = None
        self._use_flash: bool = False
        self._use_sparse: bool = False
        self._dropout: float = 0.0
        self._metadata: Dict[str, Any] = {}
    
    def with_dimensions(self, d_model: int, n_heads: int, d_kv: Optional[int] = None) -> 'AttentionBuilder':
        """Set dimensions."""
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_kv = d_kv
        return self
    
    def with_backend(self, backend: Backend) -> 'AttentionBuilder':
        """Set backend."""
        self._backend = backend
        return self
    
    def with_flash_attention(self, enabled: bool = True) -> 'AttentionBuilder':
        """Enable/disable flash attention."""
        self._use_flash = enabled
        return self
    
    def with_sparse_attention(self, enabled: bool = True) -> 'AttentionBuilder':
        """Enable/disable sparse attention."""
        self._use_sparse = enabled
        return self
    
    def with_dropout(self, dropout: float) -> 'AttentionBuilder':
        """Set dropout rate."""
        self._dropout = dropout
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'AttentionBuilder':
        """Add metadata."""
        self._metadata[key] = value
        return self
    
    def build(self) -> Attention:
        """Build Attention instance."""
        if not self._d_model or not self._n_heads:
            raise ValueError("d_model and n_heads are required")
        
        config = AttentionConfig(
            d_model=self._d_model,
            n_heads=self._n_heads,
            d_kv=self._d_kv,
            use_flash_attention=self._use_flash,
            use_sparse_attention=self._use_sparse,
            dropout=self._dropout,
            **self._metadata
        )
        
        return Attention(config=config, backend=self._backend)


class InferenceBuilder:
    """Builder for Inference Engine configuration."""
    
    def __init__(self):
        self._backend: Optional[Backend] = None
        self._max_batch_size: Optional[int] = None
        self._max_sequence_length: Optional[int] = None
        self._use_cache: bool = True
        self._use_kv_cache: bool = True
        self._temperature: float = 1.0
        self._top_p: float = 1.0
        self._top_k: Optional[int] = None
        self._metadata: Dict[str, Any] = {}
    
    def with_backend(self, backend: Backend) -> 'InferenceBuilder':
        """Set backend."""
        self._backend = backend
        return self
    
    def with_batch_size(self, max_batch_size: int) -> 'InferenceBuilder':
        """Set maximum batch size."""
        self._max_batch_size = max_batch_size
        return self
    
    def with_sequence_length(self, max_sequence_length: int) -> 'InferenceBuilder':
        """Set maximum sequence length."""
        self._max_sequence_length = max_sequence_length
        return self
    
    def with_caching(self, use_cache: bool = True, use_kv_cache: bool = True) -> 'InferenceBuilder':
        """Enable/disable caching."""
        self._use_cache = use_cache
        self._use_kv_cache = use_kv_cache
        return self
    
    def with_sampling(self, temperature: float = 1.0, top_p: float = 1.0, top_k: Optional[int] = None) -> 'InferenceBuilder':
        """Set sampling parameters."""
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'InferenceBuilder':
        """Add metadata."""
        self._metadata[key] = value
        return self
    
    def build(self) -> InferenceEngine:
        """Build Inference Engine instance."""
        config = InferenceConfig(
            max_batch_size=self._max_batch_size,
            max_sequence_length=self._max_sequence_length,
            use_cache=self._use_cache,
            use_kv_cache=self._use_kv_cache,
            **self._metadata
        )
        
        generation_config = GenerationConfig(
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k
        )
        
        return InferenceEngine(config=config, generation_config=generation_config, backend=self._backend)


# Convenience functions
def cache_builder() -> CacheBuilder:
    """Create cache builder."""
    return CacheBuilder()


def attention_builder() -> AttentionBuilder:
    """Create attention builder."""
    return AttentionBuilder()


def inference_builder() -> InferenceBuilder:
    """Create inference builder."""
    return InferenceBuilder()













