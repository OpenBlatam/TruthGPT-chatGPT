"""
Core modules for polyglot_core.

Core functionality: backend detection, cache, attention, compression, inference, tokenization, quantization.
"""

# Import from parent directory for backward compatibility
import sys
from pathlib import Path

# Add parent to path if needed
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Re-export core modules
from ..backend import (
    Backend,
    BackendInfo,
    get_available_backends,
    get_best_backend,
    is_backend_available,
    get_backend_info,
    print_backend_status,
)

from ..cache import (
    KVCache,
    KVCacheConfig,
    EvictionStrategy,
    CacheStats,
)

from ..attention import (
    Attention,
    AttentionConfig,
    AttentionPattern,
    PositionEncoding,
    FlashAttention,
    SparseAttention,
)

from ..compression import (
    Compressor,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionStats,
    compress,
    decompress,
)

from ..inference import (
    InferenceEngine,
    InferenceConfig,
    GenerationConfig,
    GenerationResult,
    TokenSampler,
)

from ..tokenization import (
    Tokenizer,
    TokenizerConfig,
)

from ..quantization import (
    Quantizer,
    QuantizationConfig,
    QuantizationType,
    QuantizationStats,
    quantize_weights,
    dequantize_weights,
)

__all__ = [
    # Backend
    "Backend",
    "BackendInfo",
    "get_available_backends",
    "get_best_backend",
    "is_backend_available",
    "get_backend_info",
    "print_backend_status",
    # Cache
    "KVCache",
    "KVCacheConfig",
    "EvictionStrategy",
    "CacheStats",
    # Attention
    "Attention",
    "AttentionConfig",
    "AttentionPattern",
    "PositionEncoding",
    "FlashAttention",
    "SparseAttention",
    # Compression
    "Compressor",
    "CompressionConfig",
    "CompressionAlgorithm",
    "CompressionStats",
    "compress",
    "decompress",
    # Inference
    "InferenceEngine",
    "InferenceConfig",
    "GenerationConfig",
    "GenerationResult",
    "TokenSampler",
    # Tokenization
    "Tokenizer",
    "TokenizerConfig",
    # Quantization
    "Quantizer",
    "QuantizationConfig",
    "QuantizationType",
    "QuantizationStats",
    "quantize_weights",
    "dequantize_weights",
]













