"""
Ultra-Modular K/V Cache System for Efficient Decoding
Modular cache that reuses K/V cache for each new token instead of recalculating from scratch
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategies for K/V cache management."""
    FIFO = "fifo"                      # First In First Out
    LRU = "lru"                        # Least Recently Used
    LFU = "lfu"                        # Least Frequently Used
    ADAPTIVE = "adaptive"              # Adaptive based on usage
    MULTI_LEVEL = "multi_level"        # Multi-level caching
    COMPRESSED = "compressed"          # Compressed storage

class MemoryLayout(Enum):
    """Memory layout strategies for K/V cache."""
    DENSE = "dense"                    # Dense storage
    SPARSE = "sparse"                   # Sparse storage
    BLOCKED = "blocked"                 # Block-wise storage
    PACKED = "packed"                   # Packed storage

@dataclass
class KVCacheConfig:
    """Configuration for K/V cache."""
    max_cache_size: int = 8192
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    memory_layout: MemoryLayout = MemoryLayout.DENSE
    use_compression: bool = True
    compression_ratio: float = 0.3
    use_quantization: bool = True
    quantization_bits: int = 8
    prefetch_next: bool = True
    cache_reuse: bool = True
    memory_efficient: bool = True

@dataclass
class CacheEntry:
    """Entry in the K/V cache."""
    key: torch.Tensor
    value: torch.Tensor
    layer_id: int
    position: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    compressed: bool = False

class KVCacheModule(nn.Module):
    """
    Ultra-Modular K/V Cache Module for efficient token-by-token decoding.
    
    Key features:
    - Reuses K/V cache for each new token instead of recalculating from scratch
    - Minimizes memory overhead and latency between tokens
    - Modular design for easy integration
    - Automatic cache management
    """
    
    def __init__(self, config: KVCacheConfig):
        super().__init__()
        self.config = config
        
        # Cache storage
        self.cache: Dict[int, OrderedDict[int, CacheEntry]] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0
        }
        
        # Compression and quantization
        self._setup_compression()
        
        logger.info(f"K/V Cache Module initialized with strategy: {config.cache_strategy}")
    
    def _setup_compression(self):
        """Setup compression mechanisms."""
        if self.config.use_compression:
            self.compression_module = CompressionModule(
                compression_ratio=self.config.compression_ratio
            )
        
        if self.config.use_quantization:
            self.quantization_module = QuantizationModule(
                bits=self.config.quantization_bits
            )
    
    def get_cache_entry(self, layer_id: int, position: int) -> Optional[CacheEntry]:
        """Get cache entry for specific layer and position."""
        if layer_id not in self.cache:
            return None
        
        if position not in self.cache[layer_id]:
            return None
        
        entry = self.cache[layer_id][position]
        entry.access_count += 1
        entry.last_accessed = time.time()
        
        self.cache_stats['hits'] += 1
        
        return entry
    
    def set_cache_entry(self, layer_id: int, position: int, key: torch.Tensor, 
                       value: torch.Tensor):
        """Set cache entry for specific layer and position."""
        if layer_id not in self.cache:
            self.cache[layer_id] = OrderedDict()
        
        # Check if we need to evict
        if len(self.cache[layer_id]) >= self.config.max_cache_size:
            self._evict_entry(layer_id)
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            layer_id=layer_id,
            position=position
        )
        
        # Apply compression if enabled
        if self.config.use_compression:
            entry = self._compress_entry(entry)
        
        # Apply quantization if enabled
        if self.config.use_quantization:
            entry = self._quantize_entry(entry)
        
        self.cache[layer_id][position] = entry
    
    def _evict_entry(self, layer_id: int):
        """Evict an entry using the configured strategy."""
        if layer_id not in self.cache or not self.cache[layer_id]:
            return
        
        if self.config.cache_strategy == CacheStrategy.LRU:
            # Remove least recently used
            self.cache[layer_id].popitem(last=False)
        
        elif self.config.cache_strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_entry = min(
                self.cache[layer_id].items(),
                key=lambda x: x[1].access_count
            )
            self.cache[layer_id].pop(min_entry[0])
        
        elif self.config.cache_strategy == CacheStrategy.FIFO:
            # Remove first entry
            self.cache[layer_id].popitem(last=False)
        
        self.cache_stats['evictions'] += 1
    
    def _compress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Compress cache entry."""
        if self.config.use_compression and hasattr(self, 'compression_module'):
            entry.key = self.compression_module.compress(entry.key)
            entry.value = self.compression_module.compress(entry.value)
            entry.compressed = True
            self.cache_stats['compressions'] += 1
        
        return entry
    
    def _quantize_entry(self, entry: CacheEntry) -> CacheEntry:
        """Quantize cache entry."""
        if self.config.use_quantization and hasattr(self, 'quantization_module'):
            entry.key = self.quantization_module.quantize(entry.key)
            entry.value = self.quantization_module.quantize(entry.value)
        
        return entry
    
    def update_cache_for_token(self, layer_id: int, position: int, 
                              key: torch.Tensor, value: torch.Tensor):
        """
        Update cache for a new token.
        Reuses existing cache and only updates necessary entries.
        """
        # Check if we have cached data
        existing_entry = self.get_cache_entry(layer_id, position)
        
        if existing_entry is None:
            # Cache miss - store new entry
            self.cache_stats['misses'] += 1
            self.set_cache_entry(layer_id, position, key, value)
        else:
            # Cache hit - reuse existing entry
            # In decode phase, we only append new K/V to existing cache
            self._append_to_cache(layer_id, position, key, value)
    
    def _append_to_cache(self, layer_id: int, position: int, 
                         key: torch.Tensor, value: torch.Tensor):
        """Append new K/V to existing cache without full recalculation."""
        if layer_id not in self.cache or position not in self.cache[layer_id]:
            return
        
        entry = self.cache[layer_id][position]
        
        # Append new K/V to existing cache
        if not entry.compressed:
            entry.key = torch.cat([entry.key, key], dim=2)
            entry.value = torch.cat([entry.value, value], dim=2)
        else:
            # Decompress, append, recompress
            entry = self._expand_entry(entry)
            entry.key = torch.cat([entry.key, key], dim=2)
            entry.value = torch.cat([entry.value, value], dim=2)
            entry = self._compress_entry(entry)
    
    def _expand_entry(self, entry: CacheEntry) -> CacheEntry:
        """Expand compressed entry."""
        if self.config.use_compression and hasattr(self, 'compression_module'):
            entry.key = self.compression_module.decompress(entry.key)
            entry.value = self.compression_module.decompress(entry.value)
            entry.compressed = False
        
        return entry
    
    def clear_cache(self, layer_id: Optional[int] = None):
        """Clear cache for specific layer or all layers."""
        if layer_id is None:
            self.cache.clear()
        elif layer_id in self.cache:
            self.cache[layer_id].clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'cache_stats': self.cache_stats.copy(),
            'hit_rate': hit_rate,
            'total_entries': sum(len(cache) for cache in self.cache.values()),
            'memory_usage': self._estimate_memory_usage(),
            'cache_strategy': self.config.cache_strategy.value,
            'memory_layout': self.config.memory_layout.value
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        total_params = sum(
            entry.key.numel() + entry.value.numel()
            for cache in self.cache.values()
            for entry in cache.values()
        )
        
        # Assume FP16 (2 bytes per parameter)
        memory_mb = (total_params * 2) / (1024 * 1024)
        return memory_mb
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                layer_id: int, position: int, use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with K/V cache reuse.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            layer_id: Layer identifier
            position: Position in sequence
            use_cache: Whether to use cache
            
        Returns:
            Tuple of (output, updated cache entry)
        """
        if use_cache and self.config.cache_reuse:
            # Try to reuse existing cache
            existing_entry = self.get_cache_entry(layer_id, position)
            
            if existing_entry is not None:
                # Cache hit - reuse existing K/V
                key = existing_entry.key
                value = existing_entry.value
            else:
                # Cache miss - store new entry
                self.set_cache_entry(layer_id, position, key, value)
        
        # Standard attention computation
        output = self._compute_attention(query, key, value)
        
        return output, (key, value)
    
    def _compute_attention(self, query: torch.Tensor, key: torch.Tensor,
                           value: torch.Tensor) -> torch.Tensor:
        """Compute standard scaled dot-product attention."""
        # This is a simplified implementation
        # Actual implementation would use efficient attention mechanisms
        scale = 1.0 / (query.size(-1) ** 0.5)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, value)
        
        return output

class CompressionModule(nn.Module):
    """Module for compressing K/V cache entries."""
    
    def __init__(self, compression_ratio: float = 0.3):
        super().__init__()
        self.compression_ratio = compression_ratio
    
    def compress(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress tensor using specified ratio."""
        # Simplified compression - actual implementation would use proper algorithms
        return tensor
    
    def decompress(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decompress tensor."""
        return tensor

class QuantizationModule(nn.Module):
    """Module for quantizing K/V cache entries."""
    
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified bit width."""
        # Simplified quantization
        scale = 2 ** (self.bits - 1) - 1
        tensor = torch.clamp(tensor * scale, -scale, scale - 1)
        tensor = tensor / scale
        
        return tensor
    
    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor."""
        scale = 2 ** (self.bits - 1) - 1
        return tensor * scale

# Factory functions
def create_kv_cache(config: KVCacheConfig = None) -> KVCacheModule:
    """Create a K/V cache module."""
    if config is None:
        config = KVCacheConfig()
    return KVCacheModule(config)

def create_kv_cache_config(**kwargs) -> KVCacheConfig:
    """Create a K/V cache configuration."""
    return KVCacheConfig(**kwargs)



