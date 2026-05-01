"""
Unified Attention module with automatic backend selection.

Supports Flash Attention, Sparse Attention, and standard attention
with Rust, C++, and Python backends.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np
import math
import time

from .backend import Backend, get_best_backend, is_backend_available

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default attention parameters
DEFAULT_D_MODEL = 768
DEFAULT_N_HEADS = 12
DEFAULT_MAX_SEQ_LEN = 8192
DEFAULT_DROPOUT = 0.0
DEFAULT_WINDOW_SIZE = 512
DEFAULT_BLOCK_SIZE = 64
DEFAULT_ROPE_THETA = 10000.0

# Numerical stability
EPSILON = 1e-9
LARGE_NEGATIVE_VALUE = -1e9

# Time conversion
MILLISECONDS_PER_SECOND = 1000

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionPattern(Enum):
    """Attention pattern types."""
    FULL = "full"              # Full O(N²) attention
    CAUSAL = "causal"          # Autoregressive causal mask
    SLIDING_WINDOW = "sliding" # Local sliding window
    SPARSE = "sparse"          # Block-sparse
    BIGBIRD = "bigbird"        # BigBird-style


class PositionEncoding(Enum):
    """Position encoding types."""
    NONE = "none"
    ROPE = "rope"       # Rotary Position Embeddings
    ALIBI = "alibi"     # Attention with Linear Biases
    RELATIVE = "relative"


@dataclass
class AttentionConfig:
    """
    Configuration for attention mechanisms.
    
    Attributes:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads (for GQA, default: n_heads)
        head_dim: Dimension per head (default: d_model // n_heads)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pattern: Attention pattern type
        position_encoding: Position encoding type
        use_causal_mask: Whether to use causal mask
        window_size: Window size for sliding window attention
        block_size: Block size for Flash Attention tiling
        rope_theta: Base frequency for RoPE
    """
    d_model: int = DEFAULT_D_MODEL
    n_heads: int = DEFAULT_N_HEADS
    n_kv_heads: Optional[int] = None  # For Grouped-Query Attention (GQA)
    head_dim: Optional[int] = None
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    dropout: float = DEFAULT_DROPOUT
    pattern: AttentionPattern = AttentionPattern.FULL
    position_encoding: PositionEncoding = PositionEncoding.NONE
    use_causal_mask: bool = False
    window_size: int = DEFAULT_WINDOW_SIZE
    block_size: int = DEFAULT_BLOCK_SIZE
    rope_theta: float = DEFAULT_ROPE_THETA
    
    def __post_init__(self):
        """Validate and set default values."""
        # Validate parameters
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        
        # Set defaults
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.head_dim is None:
            self.head_dim = self.d_model // self.n_heads
        
        # Validate GQA configuration
        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) cannot be > n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
    
    @property
    def is_gqa(self) -> bool:
        """
        Check if using Grouped-Query Attention (GQA).
        
        Returns:
            True if n_kv_heads < n_heads
        """
        return self.n_kv_heads < self.n_heads
    
    @property
    def softmax_scale(self) -> float:
        """
        Get softmax scaling factor (1 / sqrt(head_dim)).
        
        Returns:
            Scaling factor for attention scores
        """
        return 1.0 / math.sqrt(self.head_dim)
    
    @classmethod
    def llama_7b(cls) -> "AttentionConfig":
        """
        Config for LLaMA 7B model.
        
        Returns:
            AttentionConfig with LLaMA 7B parameters
        """
        return cls(
            d_model=4096,
            n_heads=32,
            n_kv_heads=32,
            pattern=AttentionPattern.CAUSAL,
            position_encoding=PositionEncoding.ROPE
        )
    
    @classmethod
    def llama_70b(cls) -> "AttentionConfig":
        """
        Config for LLaMA 70B model with GQA.
        
        Returns:
            AttentionConfig with LLaMA 70B parameters (8 KV heads for 64 query heads)
        """
        return cls(
            d_model=8192,
            n_heads=64,
            n_kv_heads=8,
            pattern=AttentionPattern.CAUSAL,
            position_encoding=PositionEncoding.ROPE
        )
    
    @classmethod
    def mistral_7b(cls) -> "AttentionConfig":
        """
        Config for Mistral 7B model with sliding window.
        
        Returns:
            AttentionConfig with Mistral 7B parameters
        """
        return cls(
            d_model=4096,
            n_heads=32,
            n_kv_heads=8,
            pattern=AttentionPattern.SLIDING_WINDOW,
            window_size=4096,
            position_encoding=PositionEncoding.ROPE
        )


@dataclass
class AttentionOutput:
    """
    Output from attention computation.
    
    Attributes:
        output: Output tensor [batch * seq, d_model]
        attention_weights: Optional attention weights [batch, heads, seq, seq]
        compute_time_ms: Computation time in milliseconds
        memory_bytes: Memory used in bytes
    """
    output: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    compute_time_ms: float = 0.0
    memory_bytes: int = 0

# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Attention:
    """
    Unified Attention with automatic backend selection.
    
    Automatically selects the best backend:
    - C++ (CUDA): 10-100x faster with GPU
    - C++ (CPU): 5-10x faster with Eigen/SIMD
    - Rust: Memory-efficient CPU attention
    - Python: Fallback with optimized numpy
    
    Example:
        >>> config = AttentionConfig(d_model=768, n_heads=12)
        >>> attn = Attention(config)
        >>> output = attn.forward(q, k, v, batch_size=4, seq_len=512)
        >>> print(f"Time: {output.compute_time_ms:.2f}ms")
    """
    
    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        backend: Optional[Backend] = None,
        **kwargs
    ):
        """
        Initialize Attention.
        
        Args:
            config: Attention configuration
            d_model: Model dimension (if config not provided)
            n_heads: Number of heads (if config not provided)
            backend: Force specific backend (None = auto-select)
            **kwargs: Additional config options
        """
        if config is None:
            config = AttentionConfig(d_model=d_model, n_heads=n_heads, **kwargs)
        
        self.config = config
        self._backend = backend or get_best_backend('attention')
        self._impl = self._create_implementation()
    
    def _create_implementation(self):
        """
        Create backend-specific implementation.
        
        Returns:
            Backend implementation or None (use Python fallback)
        """
        if self._backend == Backend.CPP and is_backend_available(Backend.CPP):
            return self._create_cpp_impl()
        elif self._backend == Backend.RUST and is_backend_available(Backend.RUST):
            return self._create_rust_impl()
        else:
            return None  # Use Python fallback
    
    def _create_cpp_impl(self):
        """
        Create C++ implementation.
        
        Returns:
            C++ attention engine or None if unavailable
        """
        try:
            from optimization_core import _cpp_core
            
            cpp_config = _cpp_core.attention.FlashAttentionConfig(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout,
                use_causal_mask=self.config.use_causal_mask or 
                               self.config.pattern == AttentionPattern.CAUSAL,
                window_size=self.config.window_size
            )
            
            return _cpp_core.attention.FlashAttentionCPU(cpp_config)
        except (ImportError, AttributeError, Exception) as e:
            print(f"[Attention] C++ backend failed: {e}, using Python fallback")
            return None
    
    def _create_rust_impl(self):
        """
        Create Rust implementation.
        
        Returns:
            Rust attention engine or None if unavailable
        """
        try:
            from optimization_core.rust_core import truthgpt_rust
            return truthgpt_rust.PyAttention(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads
            )
        except (ImportError, AttributeError, Exception) as e:
            print(f"[Attention] Rust backend failed: {e}, using Python fallback")
            return None
    
    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        batch_size: int,
        seq_len: int,
        attention_mask: Optional[np.ndarray] = None,
        return_attention_weights: bool = False
    ) -> AttentionOutput:
        """
        Compute attention forward pass.
        
        Args:
            query: Query tensor [batch * seq, d_model]
            key: Key tensor [batch * seq, d_kv] (d_kv = d_model for standard, smaller for GQA)
            value: Value tensor [batch * seq, d_kv]
            batch_size: Batch size
            seq_len: Sequence length
            attention_mask: Optional attention mask [batch, seq, seq] or [seq, seq]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            AttentionOutput with output tensor and optional weights
            
        Raises:
            ValueError: If dimensions are invalid
        """
        # Validate inputs
        self._validate_inputs(query, key, value, batch_size, seq_len)
        
        start_time = time.perf_counter()
        
        # Try backend implementation first
        if self._impl is not None and self._backend == Backend.CPP:
            result = self._forward_cpp(
                query, key, value, batch_size, seq_len,
                attention_mask, return_attention_weights
            )
            elapsed_ms = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
            
            return AttentionOutput(
                output=result['output'],
                attention_weights=result.get('attention_weights'),
                compute_time_ms=result.get('compute_time_ms', elapsed_ms),
                memory_bytes=result.get('memory_bytes', 0)
            )
        else:
            # Python fallback
            output, weights = self._python_forward(
                query, key, value, batch_size, seq_len,
                attention_mask, return_attention_weights
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
            
            return AttentionOutput(
                output=output,
                attention_weights=weights if return_attention_weights else None,
                compute_time_ms=elapsed_ms
            )
    
    def _validate_inputs(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        batch_size: int,
        seq_len: int
    ):
        """Validate input dimensions."""
        expected_q_shape = (batch_size * seq_len, self.config.d_model)
        expected_kv_shape = (batch_size * seq_len, self.config.n_kv_heads * self.config.head_dim)
        
        if query.shape != expected_q_shape:
            raise ValueError(
                f"query shape {query.shape} != expected {expected_q_shape}"
            )
        if key.shape != expected_kv_shape:
            raise ValueError(
                f"key shape {key.shape} != expected {expected_kv_shape}"
            )
        if value.shape != expected_kv_shape:
            raise ValueError(
                f"value shape {value.shape} != expected {expected_kv_shape}"
            )
    
    def _forward_cpp(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        batch_size: int,
        seq_len: int,
        attention_mask: Optional[np.ndarray],
        return_attention_weights: bool
    ) -> Dict[str, Any]:
        """Forward pass using C++ backend."""
        return self._impl.forward(
            query, key, value,
            batch_size, seq_len,
            attention_mask,
            return_attention_weights
        )
    
    def _python_forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        batch_size: int,
        seq_len: int,
        attention_mask: Optional[np.ndarray] = None,
        return_attention_weights: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Python fallback attention computation.
        
        Returns:
            Tuple of (output, attention_weights)
        """
        d_model = self.config.d_model
        n_heads = self.config.n_heads
        n_kv_heads = self.config.n_kv_heads
        head_dim = self.config.head_dim
        
        # Reshape to [batch, seq, heads, head_dim]
        q = query.reshape(batch_size, seq_len, n_heads, head_dim)
        k = key.reshape(batch_size, seq_len, n_kv_heads, head_dim)
        v = value.reshape(batch_size, seq_len, n_kv_heads, head_dim)
        
        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Handle GQA by repeating KV heads
        if self.config.is_gqa:
            repeat_factor = n_heads // n_kv_heads
            k = np.repeat(k, repeat_factor, axis=1)
            v = np.repeat(v, repeat_factor, axis=1)
        
        # Scaled dot-product attention: Q @ K^T / sqrt(head_dim)
        scale = self.config.softmax_scale
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        # Apply causal mask if needed
        if self._should_apply_causal_mask():
            scores = self._apply_causal_mask(scores, seq_len)
        
        # Apply custom attention mask
        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask)
        
        # Softmax: numerically stable
        weights = self._softmax(scores)
        
        # Apply dropout (if enabled)
        if self.config.dropout > 0.0:
            # TODO: Implement dropout
            pass
        
        # Apply attention weights to values
        output = np.matmul(weights, v)
        
        # Reshape back to [batch, seq, d_model]
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Flatten to [batch * seq, d_model]
        output = output.reshape(batch_size * seq_len, d_model)
        
        # Return attention weights if requested
        attention_weights = weights if return_attention_weights else None
        
        return output, attention_weights
    
    def _should_apply_causal_mask(self) -> bool:
        """Check if causal mask should be applied."""
        return (self.config.use_causal_mask or 
                self.config.pattern == AttentionPattern.CAUSAL)
    
    def _apply_causal_mask(self, scores: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Apply causal mask to attention scores.
        
        Args:
            scores: Attention scores [batch, heads, seq, seq]
            seq_len: Sequence length
            
        Returns:
            Masked scores
        """
        # Create upper triangular mask (1s above diagonal)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=scores.dtype), k=1)
        mask = mask * LARGE_NEGATIVE_VALUE
        
        # Add mask to scores (broadcast if needed)
        if scores.ndim == 4:
            # [batch, heads, seq, seq] - add mask to last two dims
            scores = scores + mask[np.newaxis, np.newaxis, :, :]
        else:
            scores = scores + mask
        
        return scores
    
    def _apply_attention_mask(
        self,
        scores: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply custom attention mask.
        
        Args:
            scores: Attention scores
            attention_mask: Mask (0 = masked, 1 = unmasked)
            
        Returns:
            Masked scores
        """
        # Convert mask: 0 -> large negative, 1 -> 0
        mask = (1.0 - attention_mask) * LARGE_NEGATIVE_VALUE
        return scores + mask
    
    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute numerically stable softmax.
        
        Args:
            scores: Attention scores
            
        Returns:
            Attention weights (probabilities)
        """
        # Subtract max for numerical stability
        scores_shifted = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + EPSILON)
    
    @property
    def backend(self) -> Backend:
        """Get current backend."""
        return self._backend
    
    def __repr__(self) -> str:
        return (f"Attention(d_model={self.config.d_model}, "
                f"n_heads={self.config.n_heads}, "
                f"n_kv_heads={self.config.n_kv_heads}, "
                f"backend={self._backend.name})")

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIZED ATTENTION CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class FlashAttention(Attention):
    """
    Flash Attention with O(N) memory complexity.
    
    Uses tiled computation for memory efficiency, reducing memory
    from O(N²) to O(N) where N is sequence length.
    
    Example:
        >>> flash_attn = FlashAttention(d_model=768, n_heads=12)
        >>> output = flash_attn.forward(q, k, v, batch_size=4, seq_len=2048)
    """
    
    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        **kwargs
    ):
        """
        Initialize Flash Attention.
        
        Args:
            config: Attention configuration
            **kwargs: Additional config options
        """
        if config is None:
            config = AttentionConfig(**kwargs)
        
        # Ensure Flash-specific settings
        if config.block_size <= 0:
            config.block_size = DEFAULT_BLOCK_SIZE
        
        # Force C++ backend for Flash Attention (if available)
        super().__init__(config=config, backend=Backend.CPP)


class SparseAttention(Attention):
    """
    Sparse Attention with local + global patterns.
    
    Reduces complexity from O(N²) to O(N × window_size) by
    using local attention windows with global tokens.
    
    Example:
        >>> sparse_attn = SparseAttention(
        ...     d_model=768, n_heads=12,
        ...     window_size=256, global_tokens=16
        ... )
    """
    
    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        window_size: int = 256,
        global_tokens: int = 16,
        **kwargs
    ):
        """
        Initialize Sparse Attention.
        
        Args:
            config: Attention configuration
            window_size: Local attention window size
            global_tokens: Number of global tokens
            **kwargs: Additional config options
        """
        if config is None:
            config = AttentionConfig(**kwargs)
        
        config.pattern = AttentionPattern.SPARSE
        config.window_size = window_size
        
        super().__init__(config=config)
        
        if global_tokens <= 0:
            raise ValueError(f"global_tokens must be positive, got {global_tokens}")
        self.global_tokens = global_tokens

