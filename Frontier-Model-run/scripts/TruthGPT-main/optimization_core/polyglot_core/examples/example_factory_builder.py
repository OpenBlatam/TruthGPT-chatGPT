"""
Example: Factory and Builder Patterns

Demonstrates factory and builder patterns for creating components.
"""

from optimization_core.polyglot_core import (
    get_factory,
    FactoryConfig,
    ComponentType,
    create_component,
    cache_builder,
    attention_builder,
    inference_builder,
    Backend,
    EvictionStrategy,
    CompressionAlgorithm,
)


def example_factory():
    """Example: Using Factory pattern."""
    print("=" * 60)
    print("Example: Factory Pattern")
    print("=" * 60)
    
    # Create factory with configuration
    factory_config = FactoryConfig(
        preferred_backend=Backend.RUST,
        auto_select_backend=True,
        fallback_to_python=True
    )
    
    factory = get_factory(factory_config)
    
    # Create components using factory
    cache = factory.create_cache(
        max_size=16384,
        eviction_strategy=EvictionStrategy.LRU
    )
    
    attention = factory.create_attention(
        d_model=768,
        n_heads=12
    )
    
    compressor = factory.create_compressor(
        algorithm=CompressionAlgorithm.LZ4
    )
    
    print(f"✅ Created components using factory:")
    print(f"   Cache: {type(cache).__name__}")
    print(f"   Attention: {type(attention).__name__}")
    print(f"   Compressor: {type(compressor).__name__}")
    
    # Using convenience function
    cache2 = create_component(
        ComponentType.CACHE,
        max_size=8192,
        eviction_strategy=EvictionStrategy.LFU
    )
    
    print(f"   Cache2 (convenience): {type(cache2).__name__}")
    print()


def example_builder():
    """Example: Using Builder pattern."""
    print("=" * 60)
    print("Example: Builder Pattern")
    print("=" * 60)
    
    # Cache Builder
    cache = (cache_builder()
        .with_max_size(32768)
        .with_eviction_strategy(EvictionStrategy.ADAPTIVE)
        .with_backend(Backend.RUST)
        .with_compression(True)
        .with_quantization(True)
        .with_metadata("tag", "production")
        .build())
    
    print(f"✅ Built cache: {type(cache).__name__}")
    print(f"   Max size: {cache.config.max_size}")
    print(f"   Eviction: {cache.config.eviction_strategy}")
    print(f"   Compression: {cache.config.enable_compression}")
    
    # Attention Builder
    attention = (attention_builder()
        .with_dimensions(d_model=1024, n_heads=16, d_kv=64)
        .with_backend(Backend.CPP)
        .with_flash_attention(True)
        .with_dropout(0.1)
        .build())
    
    print(f"\n✅ Built attention: {type(attention).__name__}")
    print(f"   d_model: {attention.config.d_model}")
    print(f"   n_heads: {attention.config.n_heads}")
    print(f"   Flash: {attention.config.use_flash_attention}")
    
    # Inference Builder
    inference = (inference_builder()
        .with_backend(Backend.CPP)
        .with_batch_size(32)
        .with_sequence_length(2048)
        .with_caching(use_cache=True, use_kv_cache=True)
        .with_sampling(temperature=0.7, top_p=0.9, top_k=50)
        .build())
    
    print(f"\n✅ Built inference: {type(inference).__name__}")
    print(f"   Batch size: {inference.config.max_batch_size}")
    print(f"   Sequence length: {inference.config.max_sequence_length}")
    print(f"   Temperature: {inference.generation_config.temperature}")
    print()


def example_combined():
    """Example: Combining Factory and Builder."""
    print("=" * 60)
    print("Example: Factory + Builder")
    print("=" * 60)
    
    # Use factory for simple cases
    factory = get_factory()
    simple_cache = factory.create_cache(max_size=8192)
    
    # Use builder for complex cases
    complex_cache = (cache_builder()
        .with_max_size(65536)
        .with_eviction_strategy(EvictionStrategy.ADAPTIVE)
        .with_compression(True)
        .with_quantization(True)
        .with_async_mode(True)
        .build())
    
    print(f"✅ Simple cache (factory): {type(simple_cache).__name__}")
    print(f"✅ Complex cache (builder): {type(complex_cache).__name__}")
    print(f"   Features: compression={complex_cache.config.enable_compression}, "
          f"quantization={complex_cache.config.enable_quantization}, "
          f"async={complex_cache.config.async_mode}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Polyglot Core - Factory & Builder Examples")
    print("=" * 60 + "\n")
    
    example_factory()
    example_builder()
    example_combined()
    
    print("=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)













