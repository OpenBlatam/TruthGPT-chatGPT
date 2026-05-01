"""
Ultra-Efficient K/V Cache Architecture for TruthGPT
Advanced implementation with optimized prefill/decode phases
Minimizes memory overhead and latency between tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
import math
import logging
from .base import BaseOptimizationModel, CudaResourceManager
from abc import ABC, abstractmethod
import gc
import time
from contextlib import contextmanager
import threading
from collections import deque
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In, First Out
    ADAPTIVE = "adaptive" # Adaptive based on access patterns
    COMPRESSED = "compressed" # Compressed storage

class MemoryLayout(Enum):
    """Memory layout strategies."""
    CONTIGUOUS = "contiguous"     # Contiguous memory layout
    CHUNKED = "chunked"          # Chunked memory layout
    SPARSE = "sparse"            # Sparse memory layout
    HIERARCHICAL = "hierarchical" # Hierarchical memory layout

class UltraKVCacheConfig(BaseOptimizationModel):
    """Ultra-efficient K/V cache configuration."""
    
    # Cache size and management
    max_cache_size: int = 8192
    cache_chunk_size: int = 512
    max_sequence_length: int = 4096
    
    # Memory optimization
    cache_dtype: torch.dtype = torch.float16
    use_compression: bool = True
    compression_ratio: float = 0.3
    use_memory_mapping: bool = True
    memory_layout: MemoryLayout = MemoryLayout.HIERARCHICAL
    
    # Cache strategy
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    eviction_threshold: float = 0.8
    warming_enabled: bool = True
    prefetch_enabled: bool = True
    
    # Performance optimization
    use_async_loading: bool = True
    use_parallel_processing: bool = True
    num_workers: int = 4
    use_cuda_streams: bool = True
    
    # Advanced features
    use_quantization: bool = True
    quantization_bits: int = 8
    use_sparse_attention: bool = True
    sparse_attention_ratio: float = 0.1
    
    # Monitoring
    enable_profiling: bool = True
    enable_metrics: bool = True
    log_frequency: int = 100

class UltraKVCache:
    """
    Ultra-efficient Key-Value cache with advanced optimizations.
    
    Features:
    - Hierarchical memory layout
    - Adaptive eviction strategies
    - Compression and quantization
    - Async loading and prefetching
    - Parallel processing
    - Memory mapping
    """
    
    def __init__(self, config: UltraKVCacheConfig):
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_patterns: Dict[str, List[float]] = {}
        self.frequency_counts: Dict[str, int] = {}
        self.access_times: Dict[str, float] = {}
        
        # Memory management
        self.memory_pool = {}
        self.chunk_allocator = ChunkAllocator(config.cache_chunk_size)
        self.compression_engine = CompressionEngine(config.compression_ratio)
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.compression_savings = 0.0
        
        # Async processing
        self.async_queue = deque()
        self.prefetch_queue = deque()
        self.worker_threads = []
        self._setup_async_workers()
        
        # CUDA streams for parallel processing
        self.cuda_streams = CudaResourceManager.get_streams(
            config.num_workers, 
            enabled=config.use_cuda_streams
        )
    
    def _setup_async_workers(self):
        """Setup async worker threads."""
        if self.config.use_async_loading:
            for i in range(self.config.num_workers):
                thread = threading.Thread(target=self._async_worker, daemon=True)
                thread.start()
                self.worker_threads.append(thread)
    
    def _async_worker(self):
        """Async worker for background processing."""
        while True:
            try:
                if self.async_queue:
                    task = self.async_queue.popleft()
                    self._process_async_task(task)
                elif self.prefetch_queue:
                    task = self.prefetch_queue.popleft()
                    self._process_prefetch_task(task)
                else:
                    time.sleep(0.001)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Async worker error: {e}")
    
    def _process_async_task(self, task: Dict[str, Any]):
        """Process async task."""
        task_type = task.get('type')
        if task_type == 'compress':
            self._compress_cache_entry(task['key'], task['data'])
        elif task_type == 'decompress':
            self._decompress_cache_entry(task['key'])
        elif task_type == 'evict':
            self._evict_cache_entry(task['key'])
    
    def _process_prefetch_task(self, task: Dict[str, Any]):
        """Process prefetch task."""
        # Implement prefetching logic
        pass
    
    def get(self, layer_idx: int, position: int, key: str = None) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached K/V states with ultra-efficient retrieval."""
        cache_key = f"{layer_idx}_{position}_{key}" if key else f"{layer_idx}_{position}"
        
        # Update access patterns
        current_time = time.time()
        if cache_key in self.access_patterns:
            self.access_patterns[cache_key].append(current_time)
        else:
            self.access_patterns[cache_key] = [current_time]
        
        self.frequency_counts[cache_key] = self.frequency_counts.get(cache_key, 0) + 1
        self.access_times[cache_key] = current_time
        
        if cache_key in self.cache:
            self.hit_count += 1
            
            # Check if data is compressed
            if self.cache[cache_key].get('compressed', False):
                # Async decompression
                if self.config.use_async_loading:
                    self.async_queue.append({
                        'type': 'decompress',
                        'key': cache_key
                    })
                else:
                    self._decompress_cache_entry(cache_key)
            
            return self.cache[cache_key].get('data')
        else:
            self.miss_count += 1
            return None
    
    def put(self, layer_idx: int, position: int, kv_states: Dict[str, torch.Tensor], 
            key: str = None, priority: float = 1.0) -> None:
        """Store K/V states with ultra-efficient storage."""
        cache_key = f"{layer_idx}_{position}_{key}" if key else f"{layer_idx}_{position}"
        
        # Check if we need to evict
        if self._should_evict():
            self._evict_entries()
        
        # Prepare data for storage
        cache_entry = {
            'data': kv_states,
            'layer_idx': layer_idx,
            'position': position,
            'priority': priority,
            'timestamp': time.time(),
            'compressed': False,
            'access_count': 0
        }
        
        # Apply compression if enabled
        if self.config.use_compression:
            if self.config.use_async_loading:
                self.async_queue.append({
                    'type': 'compress',
                    'key': cache_key,
                    'data': cache_entry
                })
            else:
                self._compress_cache_entry(cache_key, cache_entry)
        
        # Store in cache
        self.cache[cache_key] = cache_entry
        
        # Update memory pool
        self._update_memory_pool(cache_key, cache_entry)
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        current_size = len(self.cache)
        return current_size >= self.config.max_cache_size * self.config.eviction_threshold
    
    def _evict_entries(self):
        """Evict entries based on strategy."""
        if self.config.cache_strategy == CacheStrategy.LRU:
            self._evict_lru()
        elif self.config.cache_strategy == CacheStrategy.LFU:
            self._evict_lfu()
        elif self.config.cache_strategy == CacheStrategy.FIFO:
            self._evict_fifo()
        elif self.config.cache_strategy == CacheStrategy.ADAPTIVE:
            self._evict_adaptive()
        elif self.config.cache_strategy == CacheStrategy.COMPRESSED:
            self._evict_compressed()
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        if not self.access_times:
            return
        
        # Sort by access time
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        entries_to_evict = len(self.cache) - int(self.config.max_cache_size * 0.8)
        
        for i in range(min(entries_to_evict, len(sorted_entries))):
            key = sorted_entries[i][0]
            self._evict_cache_entry(key)
    
    def _evict_lfu(self):
        """Evict least frequently used entries."""
        if not self.frequency_counts:
            return
        
        # Sort by frequency
        sorted_entries = sorted(self.frequency_counts.items(), key=lambda x: x[1])
        entries_to_evict = len(self.cache) - int(self.config.max_cache_size * 0.8)
        
        for i in range(min(entries_to_evict, len(sorted_entries))):
            key = sorted_entries[i][0]
            self._evict_cache_entry(key)
    
    def _evict_fifo(self):
        """Evict first in, first out."""
        if not self.cache:
            return
        
        # Get oldest entries
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1]['timestamp'])
        entries_to_evict = len(self.cache) - int(self.config.max_cache_size * 0.8)
        
        for i in range(min(entries_to_evict, len(sorted_entries))):
            key = sorted_entries[i][0]
            self._evict_cache_entry(key)
    
    def _evict_adaptive(self):
        """Adaptive eviction based on access patterns."""
        if not self.access_patterns:
            return
        
        # Calculate scores based on recency, frequency, and priority
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            access_times = self.access_patterns.get(key, [])
            frequency = self.frequency_counts.get(key, 0)
            recency = current_time - self.access_times.get(key, 0)
            priority = entry.get('priority', 1.0)
            
            # Adaptive scoring
            recency_score = 1.0 / (1.0 + recency)
            frequency_score = math.log(1.0 + frequency)
            priority_score = priority
            
            scores[key] = recency_score * frequency_score * priority_score
        
        # Evict lowest scoring entries
        sorted_entries = sorted(scores.items(), key=lambda x: x[1])
        entries_to_evict = len(self.cache) - int(self.config.max_cache_size * 0.8)
        
        for i in range(min(entries_to_evict, len(sorted_entries))):
            key = sorted_entries[i][0]
            self._evict_cache_entry(key)
    
    def _evict_compressed(self):
        """Evict compressed entries first."""
        compressed_entries = [k for k, v in self.cache.items() if v.get('compressed', False)]
        
        if compressed_entries:
            # Evict compressed entries first
            for key in compressed_entries[:len(compressed_entries)//2]:
                self._evict_cache_entry(key)
        else:
            # Fallback to LRU
            self._evict_lru()
    
    def _evict_cache_entry(self, key: str):
        """Evict a specific cache entry."""
        if key in self.cache:
            del self.cache[key]
            self.eviction_count += 1
            
            # Clean up tracking data
            if key in self.access_patterns:
                del self.access_patterns[key]
            if key in self.frequency_counts:
                del self.frequency_counts[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def _compress_cache_entry(self, key: str, data: Dict[str, Any]):
        """Compress cache entry."""
        if key in self.cache:
            compressed_data = self.compression_engine.compress(data['data'])
            self.cache[key]['data'] = compressed_data
            self.cache[key]['compressed'] = True
            self.compression_savings += self.compression_engine.get_savings_ratio()
    
    def _decompress_cache_entry(self, key: str):
        """Decompress cache entry."""
        if key in self.cache and self.cache[key].get('compressed', False):
            decompressed_data = self.compression_engine.decompress(self.cache[key]['data'])
            self.cache[key]['data'] = decompressed_data
            self.cache[key]['compressed'] = False
    
    def _update_memory_pool(self, key: str, entry: Dict[str, Any]):
        """Update memory pool with new entry."""
        # Implement memory pool management
        pass
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.access_patterns.clear()
        self.frequency_counts.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.compression_savings = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'eviction_count': self.eviction_count,
            'compression_savings': self.compression_savings,
            'current_size': len(self.cache),
            'max_size': self.config.max_cache_size,
            'memory_usage': self._get_memory_usage(),
            'access_patterns': len(self.access_patterns),
            'frequency_distribution': self._get_frequency_distribution()
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_memory = 0
        compressed_memory = 0
        
        for entry in self.cache.values():
            if entry.get('compressed', False):
                compressed_memory += self._estimate_tensor_size(entry['data'])
            else:
                total_memory += self._estimate_tensor_size(entry['data'])
        
        return {
            'total_memory_mb': total_memory / (1024 * 1024),
            'compressed_memory_mb': compressed_memory / (1024 * 1024),
            'compression_ratio': compressed_memory / (total_memory + compressed_memory) if (total_memory + compressed_memory) > 0 else 0
        }
    
    def _estimate_tensor_size(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> float:
        """Estimate tensor size in bytes."""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, dict):
            return sum(self._estimate_tensor_size(v) for v in data.values())
        return 0
    
    def _get_frequency_distribution(self) -> Dict[str, int]:
        """Get frequency distribution of cache accesses."""
        if not self.frequency_counts:
            return {}
        
        frequencies = list(self.frequency_counts.values())
        return {
            'min_frequency': min(frequencies),
            'max_frequency': max(frequencies),
            'avg_frequency': sum(frequencies) / len(frequencies),
            'median_frequency': sorted(frequencies)[len(frequencies)//2]
        }

class ChunkAllocator:
    """Memory chunk allocator for efficient memory management."""
    
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.available_chunks = deque()
        self.allocated_chunks = set()
    
    def allocate(self, size: int) -> int:
        """Allocate memory chunk."""
        if self.available_chunks:
            return self.available_chunks.popleft()
        else:
            # Create new chunk
            chunk_id = len(self.allocated_chunks)
            self.allocated_chunks.add(chunk_id)
            return chunk_id
    
    def deallocate(self, chunk_id: int):
        """Deallocate memory chunk."""
        if chunk_id in self.allocated_chunks:
            self.allocated_chunks.remove(chunk_id)
            self.available_chunks.append(chunk_id)

class CompressionEngine:
    """Advanced compression engine for cache data."""
    
    def __init__(self, target_ratio: float):
        self.target_ratio = target_ratio
        self.compression_stats = {
            'total_compressed': 0,
            'total_original': 0,
            'compression_ratios': []
        }
    
    def compress(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress tensor data."""
        compressed_data = {}
        
        for key, tensor in data.items():
            if tensor.dtype == torch.float16:
                # Already compressed
                compressed_data[key] = tensor
            else:
                # Apply compression
                compressed_tensor = self._compress_tensor(tensor)
                compressed_data[key] = compressed_tensor
        
        return compressed_data
    
    def decompress(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decompress tensor data."""
        decompressed_data = {}
        
        for key, tensor in data.items():
            decompressed_tensor = self._decompress_tensor(tensor)
            decompressed_data[key] = decompressed_tensor
        
        return decompressed_data
    
    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress individual tensor."""
        # Apply quantization-based compression
        if tensor.dtype == torch.float32:
            # Convert to float16
            return tensor.half()
        elif tensor.dtype == torch.float16:
            # Apply 8-bit quantization
            return self._quantize_tensor(tensor, 8)
        else:
            return tensor
    
    def _decompress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decompress individual tensor."""
        # Reverse compression
        if tensor.dtype == torch.float16:
            return tensor.float()
        else:
            return tensor
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize tensor to specified number of bits."""
        # Simple quantization implementation
        scale = 2 ** (bits - 1) - 1
        quantized = torch.round(tensor * scale) / scale
        return quantized
    
    def get_savings_ratio(self) -> float:
        """Get compression savings ratio."""
        if self.compression_stats['total_original'] > 0:
            return 1.0 - (self.compression_stats['total_compressed'] / self.compression_stats['total_original'])
        return 0.0

class UltraEfficientAttention(nn.Module):
    """
    Ultra-efficient attention with advanced K/V caching.
    
    Features:
    - Hierarchical cache management
    - Adaptive compression
    - Parallel processing
    - Memory optimization
    - Sparse attention support
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        cache_config: Optional[UltraKVCacheConfig] = None,
        use_sparse_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.use_sparse_attention = use_sparse_attention
        
        # Validate dimensions
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        # Linear transformations
        self.query_linear = nn.Linear(d_model, d_model, bias=bias)
        self.key_linear = nn.Linear(d_model, d_model, bias=bias)
        self.value_linear = nn.Linear(d_model, d_model, bias=bias)
        self.output_linear = nn.Linear(d_model, d_model, bias=bias)
        
        # Ultra-efficient K/V cache
        self.cache_config = cache_config or UltraKVCacheConfig()
        self.kv_cache = UltraKVCache(self.cache_config)
        
        # Sparse attention
        if use_sparse_attention:
            self.sparse_attention = SparseAttention(
                d_model, n_heads, 
                sparsity_ratio=self.cache_config.sparse_attention_ratio
            )
        
        # Performance optimization
        self.scale = math.sqrt(self.head_dim)
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
        # CUDA streams for parallel processing
        self.cuda_streams = CudaResourceManager.get_streams(
            self.cache_config.num_workers,
            enabled=self.cache_config.use_cuda_streams
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        cache_position: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Ultra-efficient forward pass with advanced caching.
        
        Args:
            query: Query tensor
            key: Key tensor (optional)
            value: Value tensor (optional)
            mask: Attention mask
            use_cache: Whether to use K/V cache
            cache_position: Position in sequence for caching
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, cached_kv_states)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Use query as key/value if not provided
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Apply linear transformations
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Try to use cached K/V states
        cached_kv = None
        if use_cache and cache_position is not None:
            cached_kv = self.kv_cache.get(0, cache_position)
        
        # Compute attention with optimization
        if self.use_sparse_attention and self.sparse_attention:
            output = self._sparse_attention_forward(query, key, value, mask, cached_kv)
        elif self.use_flash_attention and not cached_kv:
            output = self._flash_attention_forward(query, key, value, mask)
        else:
            output = self._standard_attention_forward(query, key, value, mask, cached_kv)
        
        # Update cache if enabled
        if use_cache and cache_position is not None:
            kv_states = {
                'key': key.detach(),
                'value': value.detach()
            }
            self.kv_cache.put(0, cache_position, kv_states)
            cached_kv = kv_states
        
        # Reshape back to original format
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Apply output linear transformation
        output = self.output_linear(output)
        
        return output, cached_kv
    
    def _sparse_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cached_kv: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Sparse attention forward pass."""
        return self.sparse_attention(query, key, value, mask, cached_kv)
    
    def _flash_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Flash attention forward pass."""
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
        )
    
    def _standard_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cached_kv: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Standard attention forward pass with caching."""
        
        # Use cached K/V if available
        if cached_kv is not None:
            key = torch.cat([cached_kv['key'], key], dim=2)
            value = torch.cat([cached_kv['value'], value], dim=2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output
    
    def clear_cache(self):
        """Clear the K/V cache."""
        self.kv_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.kv_cache.get_stats()

class SparseAttention(nn.Module):
    """Sparse attention mechanism for memory efficiency."""
    
    def __init__(self, d_model: int, n_heads: int, sparsity_ratio: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.sparsity_ratio = sparsity_ratio
        self.head_dim = d_model // n_heads
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cached_kv: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Sparse attention forward pass."""
        
        # Use cached K/V if available
        if cached_kv is not None:
            key = torch.cat([cached_kv['key'], key], dim=2)
            value = torch.cat([cached_kv['value'], value], dim=2)
        
        batch_size, n_heads, seq_len, head_dim = query.shape
        
        # Create sparse attention pattern
        sparse_pattern = self._create_sparse_pattern(seq_len, self.sparsity_ratio)
        
        # Apply sparse attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply sparse mask
        if sparse_pattern is not None:
            scores = scores.masked_fill(~sparse_pattern, -1e9)
        
        # Apply regular mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output
    
    def _create_sparse_pattern(self, seq_len: int, sparsity_ratio: float) -> Optional[torch.Tensor]:
        """Create sparse attention pattern."""
        if sparsity_ratio >= 1.0:
            return None
        
        # Create random sparse pattern
        num_connections = int(seq_len * seq_len * sparsity_ratio)
        pattern = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # Add random connections
        indices = torch.randperm(seq_len * seq_len)[:num_connections]
        rows = indices // seq_len
        cols = indices % seq_len
        pattern[rows, cols] = True
        
        # Ensure diagonal is always connected
        pattern.fill_diagonal_(True)
        
        return pattern.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

# Factory functions
def create_ultra_efficient_attention(
    d_model: int,
    n_heads: int,
    dropout: float = 0.1,
    cache_config: Optional[UltraKVCacheConfig] = None,
    use_sparse_attention: bool = True
) -> UltraEfficientAttention:
    """Create ultra-efficient attention module."""
    return UltraEfficientAttention(
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        cache_config=cache_config,
        use_sparse_attention=use_sparse_attention
    )

def create_ultra_cache_config(**kwargs) -> UltraKVCacheConfig:
    """Create ultra-efficient cache configuration."""
    return UltraKVCacheConfig(**kwargs)


import math
from typing import Optional, Tuple

import torch


class KVCacheBase:
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        raise NotImplementedError

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    @property
    def length(self) -> int:
        raise NotImplementedError


class PagedKVCache(KVCacheBase):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_tokens: int,
        block_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_tokens = max_tokens
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        self.num_blocks = math.ceil(max_tokens / block_size)
        self.k_blocks = torch.empty(
            (self.num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device
        )
        self.v_blocks = torch.empty(
            (self.num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device
        )
        self._cursor = 0

    @torch.no_grad()
    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        # key/value: [B, H, T, D] or [H, T, D]; we only support B=1 streaming here
        if key.dim() == 4:
            key = key.squeeze(0)
            value = value.squeeze(0)
        assert key.shape == value.shape, "K/V must have same shape"
        _, t, _ = key.shape  # [H, T, D]
        if self._cursor + t > self.max_tokens:
            raise RuntimeError("KV cache capacity exceeded")

        while t > 0:
            block_index = self._cursor // self.block_size
            block_offset = self._cursor % self.block_size
            writable = min(t, self.block_size - block_offset)
            t_start = 0
            t_end = writable
            self.k_blocks[block_index, :, block_offset : block_offset + writable].copy_(key[:, t_start:t_end])
            self.v_blocks[block_index, :, block_offset : block_offset + writable].copy_(value[:, t_start:t_end])
            key = key[:, writable:]
            value = value[:, writable:]
            t -= writable
            self._cursor += writable

    @torch.no_grad()
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cursor == 0:
            shape = (1, self.num_heads, 0, self.head_dim)
            empty = torch.empty(shape, dtype=self.dtype, device=self.device)
            return empty, empty
        full_blocks = self._cursor // self.block_size
        tail = self._cursor % self.block_size
        if tail == 0:
            k = self.k_blocks[:full_blocks]
            v = self.v_blocks[:full_blocks]
            k = k.reshape(full_blocks * self.block_size, self.num_heads, self.head_dim).transpose(0, 1)
            v = v.reshape(full_blocks * self.block_size, self.num_heads, self.head_dim).transpose(0, 1)
        else:
            k_main = self.k_blocks[:full_blocks]
            v_main = self.v_blocks[:full_blocks]
            k_tail = self.k_blocks[full_blocks, :, :tail]
            v_tail = self.v_blocks[full_blocks, :, :tail]
            k_main = k_main.reshape(full_blocks * self.block_size, self.num_heads, self.head_dim)
            v_main = v_main.reshape(full_blocks * self.block_size, self.num_heads, self.head_dim)
            k = torch.cat([k_main, k_tail], dim=0).transpose(0, 1)
            v = torch.cat([v_main, v_tail], dim=0).transpose(0, 1)
        # return shapes: [H, T, D] -> expand to [B=1, H, T, D]
        return k.unsqueeze(0), v.unsqueeze(0)

    @torch.no_grad()
    def reset(self) -> None:
        self._cursor = 0

    @property
    def length(self) -> int:
        return self._cursor


@torch.no_grad()
def sdpa_with_cache(
    query: torch.Tensor,
    key_cache: PagedKVCache,
    value_cache: PagedKVCache,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # query: [B, H, Tq, D]; kv from cache
    k, v = key_cache.get_kv()
    q = query
    # Use PyTorch SDPA which supports variable key/value sequence length
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=False
    )
    return out





