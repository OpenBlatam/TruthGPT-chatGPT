"""
Python wrapper for optimization_core C++ extensions.

This module provides a high-level Python interface to the C++ backend,
with fallbacks to pure Python implementations when the C++ module is unavailable.

Usage:
    from optimization_core.cpp_core.python import cpp_wrapper as cpp

    # Attention
    attn = cpp.FlashAttention(num_heads=12, head_dim=64)
    output = attn.forward(q, k, v)

    # KV Cache
    cache = cpp.KVCache(max_size=10000)
    cache.put(layer=0, position=0, key=k_data, value=v_data)
    k, v = cache.get(layer=0, position=0)

    # Inference
    engine = cpp.InferenceEngine(seed=42)
    tokens = engine.generate(prompt_ids, model.forward, max_tokens=100)
"""

import numpy as np
from typing import Optional, Tuple, List, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

# Try to import C++ module
try:
    from optimization_core._cpp_core import (
        FlashAttention as _CppFlashAttention,
        KVCache as _CppKVCache,
        InferenceEngine as _CppInferenceEngine,
        compress_lz4,
        decompress_lz4,
    )
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False
    print("[optimization_core] C++ backend not available, using Python fallback")


# ============================================================================
# Enums and Configurations
# ============================================================================

class EvictionStrategy(Enum):
    """KV cache eviction strategies."""
    LRU = "lru"      # Least Recently Used
    LFU = "lfu"      # Least Frequently Used
    FIFO = "fifo"    # First In First Out
    ADAPTIVE = "adaptive"


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    num_heads: int = 8
    head_dim: int = 64
    use_flash: bool = True
    use_causal_mask: bool = True
    dropout: float = 0.0
    block_size: int = 64  # For Flash Attention

    @property
    def d_model(self) -> int:
        return self.num_heads * self.head_dim


@dataclass
class CacheConfig:
    """Configuration for KV cache."""
    max_size: int = 10000
    eviction_strategy: EvictionStrategy = EvictionStrategy.LRU
    enable_compression: bool = True
    compression_threshold: int = 512  # Compress entries larger than this


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    num_beams: int = 1

    @classmethod
    def greedy(cls) -> "GenerationConfig":
        return cls(do_sample=False, temperature=1.0)

    @classmethod
    def sampling(cls, temperature: float = 0.7, top_p: float = 0.9) -> "GenerationConfig":
        return cls(do_sample=True, temperature=temperature, top_p=top_p)


@dataclass
class GenerationResult:
    """Result from text generation."""
    token_ids: List[int]
    tokens_generated: int
    generation_time_ms: float

    @property
    def tokens_per_second(self) -> float:
        if self.generation_time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.generation_time_ms / 1000.0)


# ============================================================================
# Flash Attention
# ============================================================================

class FlashAttention:
    """
    Memory-efficient Flash Attention implementation.
    
    Uses C++ backend if available, falls back to optimized NumPy.
    
    Args:
        config: AttentionConfig or individual parameters
        num_heads: Number of attention heads
        head_dim: Dimension per head
    """

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        **kwargs
    ):
        if config is not None:
            self.config = config
        else:
            self.config = AttentionConfig(num_heads=num_heads, head_dim=head_dim, **kwargs)

        self._use_cpp = HAS_CPP_BACKEND and self.config.use_flash

        if self._use_cpp:
            self._cpp_attn = _CppFlashAttention(
                self.config.num_heads,
                self.config.head_dim,
                self.config.block_size,
                self.config.use_causal_mask
            )

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            query: [batch, seq, dim] or [batch, heads, seq, head_dim]
            key: Same shape as query
            value: Same shape as query
            attention_mask: Optional [batch, seq, seq] mask
            
        Returns:
            Output tensor with same shape as query
        """
        if self._use_cpp:
            return self._cpp_forward(query, key, value, attention_mask)
        else:
            return self._numpy_forward(query, key, value, attention_mask)

    def _cpp_forward(self, q, k, v, mask):
        """C++ backend forward pass."""
        batch = q.shape[0]
        seq_len = q.shape[1] if q.ndim == 3 else q.shape[2]

        # Ensure contiguous C-order arrays
        q = np.ascontiguousarray(q.flatten(), dtype=np.float32)
        k = np.ascontiguousarray(k.flatten(), dtype=np.float32)
        v = np.ascontiguousarray(v.flatten(), dtype=np.float32)

        output = self._cpp_attn.forward(q, k, v, batch, seq_len, mask)
        return output.reshape(batch, seq_len, self.config.d_model)

    def _numpy_forward(self, q, k, v, mask):
        """NumPy fallback forward pass."""
        batch, seq_len = q.shape[:2]
        dim = q.shape[-1]
        
        # Reshape for multi-head attention
        q = q.reshape(batch, seq_len, self.config.num_heads, self.config.head_dim)
        k = k.reshape(batch, seq_len, self.config.num_heads, self.config.head_dim)
        v = v.reshape(batch, seq_len, self.config.num_heads, self.config.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.config.head_dim)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Apply causal mask
        if self.config.use_causal_mask:
            causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32) * -1e9, k=1)
            scores = scores + causal_mask

        # Apply attention mask
        if mask is not None:
            scores = scores + mask

        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-9)

        # Apply attention
        output = np.matmul(weights, v)

        # Transpose back and reshape
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)
        return output

    def __repr__(self):
        backend = "C++" if self._use_cpp else "NumPy"
        return f"FlashAttention(heads={self.config.num_heads}, dim={self.config.head_dim}, backend={backend})"


# ============================================================================
# KV Cache
# ============================================================================

class KVCache:
    """
    High-performance KV cache with optional compression.
    
    Args:
        config: CacheConfig or individual parameters
        max_size: Maximum number of entries
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        max_size: int = 10000,
        **kwargs
    ):
        if config is not None:
            self.config = config
        else:
            self.config = CacheConfig(max_size=max_size, **kwargs)

        self._use_cpp = HAS_CPP_BACKEND

        if self._use_cpp:
            self._cpp_cache = _CppKVCache(
                self.config.max_size,
                self.config.eviction_strategy.value,
                self.config.enable_compression
            )
        else:
            self._cache: Dict[Tuple[int, int, str], bytes] = {}
            self._access_order: List[Tuple[int, int, str]] = []
            self._hit_count = 0
            self._miss_count = 0

    def put(
        self,
        layer: int,
        position: int,
        data: np.ndarray,
        tag: str = ""
    ) -> None:
        """Store key-value data in cache."""
        if self._use_cpp:
            self._cpp_cache.put(layer, position, data.tobytes(), tag)
        else:
            key = (layer, position, tag)
            self._cache[key] = self._maybe_compress(data.tobytes())
            
            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            # Evict if necessary
            while len(self._cache) > self.config.max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

    def get(
        self,
        layer: int,
        position: int,
        tag: str = "",
        dtype: np.dtype = np.float32
    ) -> Optional[np.ndarray]:
        """Retrieve key-value data from cache."""
        if self._use_cpp:
            data = self._cpp_cache.get(layer, position, tag)
            if data is not None:
                return np.frombuffer(data, dtype=dtype)
            return None
        else:
            key = (layer, position, tag)
            if key in self._cache:
                self._hit_count += 1
                # Update access order
                self._access_order.remove(key)
                self._access_order.append(key)
                data = self._maybe_decompress(self._cache[key])
                return np.frombuffer(data, dtype=dtype)
            else:
                self._miss_count += 1
                return None

    def clear(self) -> None:
        """Clear all cache entries."""
        if self._use_cpp:
            self._cpp_cache.clear()
        else:
            self._cache.clear()
            self._access_order.clear()

    @property
    def size(self) -> int:
        """Current number of entries."""
        if self._use_cpp:
            return self._cpp_cache.size()
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        if self._use_cpp:
            return self._cpp_cache.hit_rate()
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def _maybe_compress(self, data: bytes) -> bytes:
        """Compress data if enabled and above threshold."""
        if (self.config.enable_compression and 
            len(data) >= self.config.compression_threshold and
            HAS_CPP_BACKEND):
            return compress_lz4(data)
        return data

    def _maybe_decompress(self, data: bytes) -> bytes:
        """Decompress data if it was compressed."""
        # In practice, we'd store a flag indicating compression
        if HAS_CPP_BACKEND:
            try:
                return decompress_lz4(data)
            except:
                return data
        return data

    def __repr__(self):
        backend = "C++" if self._use_cpp else "Python"
        return f"KVCache(size={self.size}/{self.config.max_size}, backend={backend})"


# ============================================================================
# Inference Engine
# ============================================================================

class InferenceEngine:
    """
    Text generation engine with various sampling strategies.
    
    Args:
        seed: Random seed for reproducibility
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._use_cpp = HAS_CPP_BACKEND

        if self._use_cpp:
            self._cpp_engine = _CppInferenceEngine(seed)
        else:
            self._rng = np.random.default_rng(seed)

    def generate(
        self,
        input_ids: List[int],
        forward_fn: Callable[[List[int]], np.ndarray],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate tokens given input IDs.
        
        Args:
            input_ids: Input token IDs
            forward_fn: Function that takes tokens and returns logits [vocab_size]
            config: Generation configuration
            
        Returns:
            GenerationResult with generated tokens and statistics
        """
        if config is None:
            config = GenerationConfig(**kwargs)

        start_time = time.perf_counter()
        
        tokens = list(input_ids)
        generated_count = 0

        for _ in range(config.max_new_tokens):
            # Get logits from model
            logits = forward_fn(tokens)
            
            if isinstance(logits, np.ndarray):
                logits = logits.flatten()

            # Sample next token
            next_token = self._sample(logits, tokens, config)
            tokens.append(int(next_token))
            generated_count += 1

            # Check for EOS
            if config.eos_token_id is not None and next_token == config.eos_token_id:
                break

        generation_time = (time.perf_counter() - start_time) * 1000

        return GenerationResult(
            token_ids=tokens,
            tokens_generated=generated_count,
            generation_time_ms=generation_time
        )

    def _sample(
        self,
        logits: np.ndarray,
        prev_tokens: List[int],
        config: GenerationConfig
    ) -> int:
        """Sample next token from logits."""
        logits = logits.astype(np.float64)

        # Apply repetition penalty
        if config.repetition_penalty != 1.0:
            for token in set(prev_tokens):
                if 0 <= token < len(logits):
                    if logits[token] > 0:
                        logits[token] /= config.repetition_penalty
                    else:
                        logits[token] *= config.repetition_penalty

        # Greedy decoding
        if not config.do_sample:
            return int(np.argmax(logits))

        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature

        # Convert to probabilities
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        # Apply top-k
        if config.top_k > 0:
            indices = np.argsort(probs)[-config.top_k:]
            mask = np.zeros_like(probs)
            mask[indices] = 1
            probs = probs * mask
            probs = probs / probs.sum()

        # Apply top-p (nucleus sampling)
        if config.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumsum, config.top_p) + 1
            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = 1
            probs = probs * mask
            probs = probs / probs.sum()

        # Sample
        return int(self._rng.choice(len(probs), p=probs))

    def __repr__(self):
        backend = "C++" if self._use_cpp else "NumPy"
        return f"InferenceEngine(seed={self.seed}, backend={backend})"


# ============================================================================
# Convenience Functions
# ============================================================================

def check_cpp_backend() -> Dict[str, bool]:
    """Check availability of C++ backend components."""
    return {
        "available": HAS_CPP_BACKEND,
        "flash_attention": HAS_CPP_BACKEND,
        "kv_cache": HAS_CPP_BACKEND,
        "inference_engine": HAS_CPP_BACKEND,
        "compression": HAS_CPP_BACKEND,
    }


def benchmark_backend(size: int = 10000) -> Dict[str, float]:
    """
    Benchmark C++ vs Python backend performance.
    
    Returns dict with operation times in milliseconds.
    """
    import time
    
    results = {}
    
    # Create test data
    data = np.random.randn(4, 256, 512).astype(np.float32)
    q, k, v = data, data, data
    
    # Test attention
    attn = FlashAttention(num_heads=8, head_dim=64)
    
    start = time.perf_counter()
    for _ in range(10):
        _ = attn.forward(q, k, v)
    results["attention_ms"] = (time.perf_counter() - start) * 100
    
    # Test cache
    cache = KVCache(max_size=size)
    test_data = np.random.randn(512).astype(np.float32)
    
    start = time.perf_counter()
    for i in range(size):
        cache.put(0, i, test_data)
    results["cache_put_ms"] = (time.perf_counter() - start) * 1000
    
    start = time.perf_counter()
    for i in range(size):
        _ = cache.get(0, i)
    results["cache_get_ms"] = (time.perf_counter() - start) * 1000
    
    results["backend"] = "C++" if HAS_CPP_BACKEND else "Python"
    
    return results


# ============================================================================
# Module Info
# ============================================================================

__all__ = [
    "FlashAttention",
    "KVCache",
    "InferenceEngine",
    "AttentionConfig",
    "CacheConfig",
    "GenerationConfig",
    "GenerationResult",
    "EvictionStrategy",
    "check_cpp_backend",
    "benchmark_backend",
    "HAS_CPP_BACKEND",
]

if __name__ == "__main__":
    print("optimization_core C++ Wrapper")
    print("=" * 40)
    print(f"C++ Backend: {'Available' if HAS_CPP_BACKEND else 'Not Available'}")
    print()
    
    status = check_cpp_backend()
    for key, val in status.items():
        print(f"  {key}: {'✓' if val else '✗'}")
    
    print()
    print("Benchmark:")
    bench = benchmark_backend(1000)
    for key, val in bench.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")













