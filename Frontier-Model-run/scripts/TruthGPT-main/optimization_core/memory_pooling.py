"""
Advanced Memory Pooling and Caching Optimizations for TruthGPT
Implements sophisticated memory management techniques for enhanced performance
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import threading
import time
from collections import OrderedDict
import gc

class TensorPool:
    """Advanced tensor memory pool for efficient memory reuse."""
    
    def __init__(self, max_pool_size: int = 1000, cleanup_threshold: float = 0.8):
        self.pools = {}
        self.max_pool_size = max_pool_size
        self.cleanup_threshold = cleanup_threshold
        self.access_times = {}
        self.lock = threading.Lock()
        self.total_tensors = 0
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                   device: torch.device = None) -> torch.Tensor:
        """Get a tensor from the pool or create a new one."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        key = (shape, dtype, device)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].pop()
                tensor.zero_()
                self.access_times[id(tensor)] = time.time()
                return tensor
            else:
                tensor = torch.zeros(shape, dtype=dtype, device=device)
                self.access_times[id(tensor)] = time.time()
                self.total_tensors += 1
                
                if self.total_tensors > self.max_pool_size * self.cleanup_threshold:
                    self._cleanup_old_tensors()
                
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the pool for reuse."""
        if not isinstance(tensor, torch.Tensor):
            return
        
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        device = tensor.device
        key = (shape, dtype, device)
        
        with self.lock:
            if key not in self.pools:
                self.pools[key] = []
            
            if len(self.pools[key]) < self.max_pool_size // 10:
                self.pools[key].append(tensor.detach())
                self.access_times[id(tensor)] = time.time()
    
    def _cleanup_old_tensors(self):
        """Clean up old tensors to free memory."""
        current_time = time.time()
        cleanup_age = 300
        
        tensors_to_remove = []
        for tensor_id, access_time in self.access_times.items():
            if current_time - access_time > cleanup_age:
                tensors_to_remove.append(tensor_id)
        
        for tensor_id in tensors_to_remove:
            del self.access_times[tensor_id]
        
        for key in list(self.pools.keys()):
            self.pools[key] = [t for t in self.pools[key] if id(t) in self.access_times]
            if not self.pools[key]:
                del self.pools[key]
        
        self.total_tensors = sum(len(pool) for pool in self.pools.values())
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            total_pooled = sum(len(pool) for pool in self.pools.values())
            pool_shapes = list(self.pools.keys())
            
            return {
                'total_pooled_tensors': total_pooled,
                'total_tensor_shapes': len(pool_shapes),
                'max_pool_size': self.max_pool_size,
                'pool_utilization': total_pooled / self.max_pool_size if self.max_pool_size > 0 else 0,
                'active_shapes': pool_shapes[:10]
            }

class ActivationCache:
    """LRU cache for activation tensors to avoid recomputation."""
    
    def __init__(self, max_size: int = 100, max_memory_mb: float = 1000):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _get_tensor_memory_mb(self, tensor: torch.Tensor) -> float:
        """Calculate tensor memory usage in MB."""
        return tensor.numel() * tensor.element_size() / (1024 * 1024)
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached activation tensor."""
        with self.lock:
            if key in self.cache:
                tensor = self.cache.pop(key)
                self.cache[key] = tensor
                self.hits += 1
                return tensor.clone()
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, tensor: torch.Tensor):
        """Cache activation tensor."""
        tensor_memory = self._get_tensor_memory_mb(tensor)
        
        with self.lock:
            while (len(self.cache) >= self.max_size or 
                   self.memory_usage + tensor_memory > self.max_memory_mb):
                if not self.cache:
                    break
                oldest_key, oldest_tensor = self.cache.popitem(last=False)
                self.memory_usage -= self._get_tensor_memory_mb(oldest_tensor)
            
            self.cache[key] = tensor.clone().detach()
            self.memory_usage += tensor_memory
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.memory_usage = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.memory_usage,
                'max_memory_mb': self.max_memory_mb,
                'hit_rate': hit_rate,
                'total_hits': self.hits,
                'total_misses': self.misses
            }

class GradientCache:
    """Cache for gradient computations to avoid recomputation."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
    
    def get_gradient_key(self, inputs: torch.Tensor, weights: torch.Tensor) -> str:
        """Generate a key for gradient caching."""
        input_hash = hash(tuple(inputs.flatten()[:100].tolist()))
        weight_hash = hash(tuple(weights.flatten()[:100].tolist()))
        return f"{input_hash}_{weight_hash}_{inputs.shape}_{weights.shape}"
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached gradient."""
        with self.lock:
            if key in self.cache:
                gradient = self.cache.pop(key)
                self.cache[key] = gradient
                return gradient.clone()
            return None
    
    def put(self, key: str, gradient: torch.Tensor):
        """Cache gradient."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = gradient.clone().detach()

class MemoryPoolingOptimizer:
    """Memory pooling and caching optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tensor_pool = TensorPool(
            max_pool_size=config.get('tensor_pool_size', 1000),
            cleanup_threshold=config.get('cleanup_threshold', 0.8)
        )
        self.activation_cache = ActivationCache(
            max_size=config.get('activation_cache_size', 100),
            max_memory_mb=config.get('activation_cache_memory_mb', 1000)
        )
        self.gradient_cache = GradientCache(
            max_size=config.get('gradient_cache_size', 50)
        )
        self.enable_tensor_pooling = config.get('enable_tensor_pooling', True)
        self.enable_activation_caching = config.get('enable_activation_caching', True)
        self.enable_gradient_caching = config.get('enable_gradient_caching', False)
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply memory pooling optimizations to model."""
        if self.enable_tensor_pooling:
            model = self._add_tensor_pooling_hooks(model)
        
        if self.enable_activation_caching:
            model = self._add_activation_caching_hooks(model)
        
        return model
    
    def _add_tensor_pooling_hooks(self, model: nn.Module) -> nn.Module:
        """Add tensor pooling hooks to model layers."""
        def pooling_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                pass
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(pooling_hook)
        
        return model
    
    def _add_activation_caching_hooks(self, model: nn.Module) -> nn.Module:
        """Add activation caching hooks to model layers."""
        def caching_hook(module, input, output):
            if isinstance(output, torch.Tensor) and not module.training:
                input_tensor = input[0] if isinstance(input, tuple) else input
                cache_key = f"{module.__class__.__name__}_{hash(tuple(input_tensor.flatten()[:50].tolist()))}"
                
                self.activation_cache.put(cache_key, output)
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                module.register_forward_hook(caching_hook)
        
        return model
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization statistics."""
        stats = {
            'tensor_pool': self.tensor_pool.get_stats(),
            'activation_cache': self.activation_cache.get_stats(),
            'gradient_cache': {
                'cache_size': len(self.gradient_cache.cache),
                'max_size': self.gradient_cache.max_size
            }
        }
        
        if torch.cuda.is_available():
            stats['cuda_memory'] = {
                'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024)
            }
        
        return stats
    
    def clear_caches(self):
        """Clear all caches and force garbage collection."""
        self.activation_cache.clear()
        self.gradient_cache.cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def create_memory_pooling_optimizer(config: Dict[str, Any]) -> MemoryPoolingOptimizer:
    """Create memory pooling optimizer from configuration."""
    return MemoryPoolingOptimizer(config)

_global_tensor_pool = None
_global_activation_cache = None

def get_global_tensor_pool() -> TensorPool:
    """Get global tensor pool instance."""
    global _global_tensor_pool
    if _global_tensor_pool is None:
        _global_tensor_pool = TensorPool()
    return _global_tensor_pool

def get_global_activation_cache() -> ActivationCache:
    """Get global activation cache instance."""
    global _global_activation_cache
    if _global_activation_cache is None:
        _global_activation_cache = ActivationCache()
    return _global_activation_cache
