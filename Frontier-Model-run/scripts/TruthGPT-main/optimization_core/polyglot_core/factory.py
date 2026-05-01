"""
Factory pattern for polyglot_core.

Provides factory methods for creating components with automatic backend selection.
"""

from typing import Optional, Dict, Any, Type, TypeVar, Generic, List
from dataclasses import dataclass
from enum import Enum

from .backend import Backend, get_best_backend, is_backend_available
from .cache import KVCache, KVCacheConfig, EvictionStrategy
from .attention import Attention, AttentionConfig, AttentionPattern
from .compression import Compressor, CompressionConfig, CompressionAlgorithm
from .inference import InferenceEngine, InferenceConfig
from .tokenization import Tokenizer, TokenizerConfig
from .quantization import Quantizer, QuantizationConfig, QuantizationType


T = TypeVar('T')


class ComponentType(str, Enum):
    """Component types."""
    CACHE = "cache"
    ATTENTION = "attention"
    COMPRESSOR = "compressor"
    INFERENCE = "inference"
    TOKENIZER = "tokenizer"
    QUANTIZER = "quantizer"


@dataclass
class FactoryConfig:
    """Factory configuration."""
    preferred_backend: Optional[Backend] = None
    auto_select_backend: bool = True
    fallback_to_python: bool = True
    strict_mode: bool = False


class ComponentFactory:
    """
    Factory for creating polyglot_core components.
    
    Automatically selects the best backend for each component.
    """
    
    def __init__(self, config: Optional[FactoryConfig] = None):
        """
        Initialize factory.
        
        Args:
            config: Factory configuration
        """
        self.config = config or FactoryConfig()
        self._cache: Dict[str, Any] = {}
    
    def create_cache(
        self,
        max_size: int = 8192,
        eviction_strategy: EvictionStrategy = EvictionStrategy.LRU,
        backend: Optional[Backend] = None,
        **kwargs
    ) -> KVCache:
        """
        Create KV Cache instance.
        
        Args:
            max_size: Maximum cache size
            eviction_strategy: Eviction strategy
            backend: Optional backend override
            **kwargs: Additional cache config
            
        Returns:
            KVCache instance
        """
        backend = self._select_backend(backend, [Backend.RUST, Backend.CPP, Backend.PYTHON])
        
        config = KVCacheConfig(
            max_size=max_size,
            eviction_strategy=eviction_strategy,
            **kwargs
        )
        
        return KVCache(config=config, backend=backend)
    
    def create_attention(
        self,
        d_model: int,
        n_heads: int,
        backend: Optional[Backend] = None,
        **kwargs
    ) -> Attention:
        """
        Create Attention instance.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            backend: Optional backend override
            **kwargs: Additional attention config
            
        Returns:
            Attention instance
        """
        backend = self._select_backend(backend, [Backend.CPP, Backend.RUST, Backend.PYTHON])
        
        config = AttentionConfig(
            d_model=d_model,
            n_heads=n_heads,
            **kwargs
        )
        
        return Attention(config=config, backend=backend)
    
    def create_compressor(
        self,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4,
        backend: Optional[Backend] = None,
        **kwargs
    ) -> Compressor:
        """
        Create Compressor instance.
        
        Args:
            algorithm: Compression algorithm
            backend: Optional backend override
            **kwargs: Additional compression config
            
        Returns:
            Compressor instance
        """
        backend = self._select_backend(backend, [Backend.RUST, Backend.CPP, Backend.PYTHON])
        
        config = CompressionConfig(
            algorithm=algorithm,
            **kwargs
        )
        
        return Compressor(config=config, backend=backend)
    
    def create_inference_engine(
        self,
        backend: Optional[Backend] = None,
        **kwargs
    ) -> InferenceEngine:
        """
        Create Inference Engine instance.
        
        Args:
            backend: Optional backend override
            **kwargs: Additional inference config
            
        Returns:
            InferenceEngine instance
        """
        backend = self._select_backend(backend, [Backend.CPP, Backend.RUST, Backend.PYTHON])
        
        config = InferenceConfig(**kwargs)
        
        return InferenceEngine(config=config, backend=backend)
    
    def create_tokenizer(
        self,
        model_name: str,
        backend: Optional[Backend] = None,
        **kwargs
    ) -> Tokenizer:
        """
        Create Tokenizer instance.
        
        Args:
            model_name: Tokenizer model name
            backend: Optional backend override
            **kwargs: Additional tokenizer config
            
        Returns:
            Tokenizer instance
        """
        backend = self._select_backend(backend, [Backend.RUST, Backend.PYTHON])
        
        config = TokenizerConfig(
            model_name=model_name,
            **kwargs
        )
        
        return Tokenizer(config=config, backend=backend)
    
    def create_quantizer(
        self,
        quantization_type: QuantizationType = QuantizationType.INT8,
        backend: Optional[Backend] = None,
        **kwargs
    ) -> Quantizer:
        """
        Create Quantizer instance.
        
        Args:
            quantization_type: Quantization type
            backend: Optional backend override
            **kwargs: Additional quantizer config
            
        Returns:
            Quantizer instance
        """
        backend = self._select_backend(backend, [Backend.CPP, Backend.RUST, Backend.PYTHON])
        
        config = QuantizationConfig(
            quantization_type=quantization_type,
            **kwargs
        )
        
        return Quantizer(config=config, backend=backend)
    
    def _select_backend(
        self,
        override: Optional[Backend],
        preferred_order: List[Backend]
    ) -> Backend:
        """
        Select backend based on configuration.
        
        Args:
            override: Optional backend override
            preferred_order: Preferred backend order
            
        Returns:
            Selected backend
        """
        if override:
            if self.config.strict_mode and not is_backend_available(override):
                raise ValueError(f"Backend {override} not available in strict mode")
            return override
        
        if self.config.preferred_backend:
            if is_backend_available(self.config.preferred_backend):
                return self.config.preferred_backend
        
        if self.config.auto_select_backend:
            for backend in preferred_order:
                if is_backend_available(backend):
                    return backend
        
        if self.config.fallback_to_python:
            return Backend.PYTHON
        
        raise RuntimeError("No available backend found")


# Global factory
_global_factory = ComponentFactory()


def get_factory(config: Optional[FactoryConfig] = None) -> ComponentFactory:
    """
    Get component factory.
    
    Args:
        config: Optional factory configuration
        
    Returns:
        ComponentFactory instance
    """
    if config:
        return ComponentFactory(config)
    return _global_factory


def create_component(
    component_type: ComponentType,
    **kwargs
) -> Any:
    """
    Convenience function to create components.
    
    Args:
        component_type: Component type
        **kwargs: Component-specific arguments
        
    Returns:
        Component instance
    """
    factory = get_factory()
    
    if component_type == ComponentType.CACHE:
        return factory.create_cache(**kwargs)
    elif component_type == ComponentType.ATTENTION:
        return factory.create_attention(**kwargs)
    elif component_type == ComponentType.COMPRESSOR:
        return factory.create_compressor(**kwargs)
    elif component_type == ComponentType.INFERENCE:
        return factory.create_inference_engine(**kwargs)
    elif component_type == ComponentType.TOKENIZER:
        return factory.create_tokenizer(**kwargs)
    elif component_type == ComponentType.QUANTIZER:
        return factory.create_quantizer(**kwargs)
    else:
        raise ValueError(f"Unknown component type: {component_type}")


