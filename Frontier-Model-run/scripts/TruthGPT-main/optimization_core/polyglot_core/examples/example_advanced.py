"""
Example: Advanced features.

Demonstrates logging, validation, health checks, and optimization.
"""

import numpy as np
from optimization_core.polyglot_core import (
    KVCache,
    Attention,
    AttentionConfig,
    get_logger,
    setup_logging,
    validate_tensor,
    validate_attention_inputs,
    check_health,
    print_health_status,
    get_optimizer,
    optimize_backend,
    Backend,
)


def main():
    print("=" * 80)
    print("Polyglot Core - Advanced Features Example")
    print("=" * 80)
    print()
    
    # 1. Setup logging
    print("1. Setting up logging...")
    print("-" * 80)
    setup_logging(use_structured=False)
    logger = get_logger()
    
    logger.info("Polyglot Core initialized")
    logger.log_operation("cache_creation", backend="rust")
    
    # 2. Validation
    print("\n2. Input validation...")
    print("-" * 80)
    
    # Validate tensor
    try:
        tensor = np.random.randn(10, 20).astype(np.float32)
        validated = validate_tensor(
            tensor,
            name="input",
            dtype=np.float32,
            shape=(10, 20)
        )
        print(f"✓ Tensor validated: shape {validated.shape}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Validate attention inputs
    try:
        q = np.random.randn(2 * 8, 256).astype(np.float32)
        k = np.random.randn(2 * 8, 256).astype(np.float32)
        v = np.random.randn(2 * 8, 256).astype(np.float32)
        
        q, k, v = validate_attention_inputs(q, k, v, batch_size=2, seq_len=8, d_model=256)
        print(f"✓ Attention inputs validated")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # 3. Health checks
    print("\n3. Health checks...")
    print("-" * 80)
    
    health = check_health()
    print(f"Overall status: {health.status.value}")
    print(f"Healthy components: {health.summary.get('healthy', 0)}")
    print(f"Unhealthy components: {health.summary.get('unhealthy', 0)}")
    
    # Print detailed status
    print_health_status()
    
    # 4. Optimization
    print("\n4. Automatic optimization...")
    print("-" * 80)
    
    def create_cache(backend):
        return KVCache(max_size=10000, backend=backend)
    
    def cache_operation(backend):
        cache = create_cache(backend)
        k = np.random.randn(64).astype(np.float32)
        v = np.random.randn(64).astype(np.float32)
        cache.put(layer=0, position=0, key=k, value=v)
        return cache.get(layer=0, position=0)
    
    try:
        result = optimize_backend(
            "kv_cache",
            cache_operation,
            backends=["python", "rust", "cpp"],
            iterations=5
        )
        
        print(f"Best backend: {result.best_backend}")
        print(f"Performance gain: {result.performance_gain:.1f}%")
        if result.recommendations:
            for rec in result.recommendations:
                print(f"  • {rec}")
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # 5. Logging with context
    print("\n5. Contextual logging...")
    print("-" * 80)
    
    cache = KVCache(max_size=1000)
    k = np.random.randn(32).astype(np.float32)
    v = np.random.randn(32).astype(np.float32)
    
    import time
    start = time.perf_counter()
    cache.put(layer=0, position=0, key=k, value=v)
    duration_ms = (time.perf_counter() - start) * 1000
    
    logger.log_performance(
        "cache_put",
        duration_ms,
        backend=cache.backend.name
    )
    
    logger.log_operation(
        "cache_get",
        backend=cache.backend.name,
        duration_ms=0.5
    )
    
    print("\n" + "=" * 80)
    print("Advanced Features Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()













