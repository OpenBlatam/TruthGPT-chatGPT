"""
Example: Using decorators and context managers.

Demonstrates decorators, events, and context managers.
"""

import numpy as np
from optimization_core.polyglot_core import (
    KVCache,
    Attention,
    AttentionConfig,
    profile_operation,
    validate_inputs,
    handle_errors,
    retry_on_failure,
    measure_performance,
    operation_context,
    backend_context,
    performance_context,
    on_event,
    EventType,
    validate_tensor,
)


def main():
    print("=" * 80)
    print("Polyglot Core - Decorators & Context Managers Example")
    print("=" * 80)
    print()
    
    # 1. Decorators
    print("1. Using Decorators:")
    print("-" * 80)
    
    @profile_operation("cache_operation", backend="rust")
    @measure_performance("cache_get")
    def get_from_cache(cache, layer, position):
        return cache.get(layer, position)
    
    @validate_inputs(
        tensor=lambda x: validate_tensor(x, name="input", dtype=np.float32)
    )
    def process_tensor(tensor):
        return tensor * 2
    
    @handle_errors(fallback=lambda: None, log_errors=True)
    @retry_on_failure(max_retries=3, delay=0.1)
    def risky_operation():
        # Simulate operation that might fail
        import random
        if random.random() < 0.3:
            raise ValueError("Random failure")
        return "Success"
    
    # Use decorated functions
    cache = KVCache(max_size=1000)
    k = np.random.randn(32).astype(np.float32)
    v = np.random.randn(32).astype(np.float32)
    cache.put(layer=0, position=0, key=k, value=v)
    
    result = get_from_cache(cache, 0, 0)
    print(f"✓ Cache get result: {result is not None}")
    
    tensor = np.random.randn(10, 20).astype(np.float32)
    processed = process_tensor(tensor)
    print(f"✓ Tensor processed: shape {processed.shape}")
    
    for i in range(5):
        result = risky_operation()
        print(f"  Attempt {i+1}: {result}")
    
    # 2. Context Managers
    print("\n2. Using Context Managers:")
    print("-" * 80)
    
    with operation_context("cache_operations", backend="rust"):
        cache.put(layer=0, position=1, key=k, value=v)
        result = cache.get(layer=0, position=1)
        print(f"✓ Operation context: {result is not None}")
    
    with backend_context("rust"):
        cache = KVCache(max_size=1000)
        print(f"✓ Backend context: {cache.backend.name}")
    
    with performance_context(threshold_ms=10.0):
        # Simulate work
        import time
        time.sleep(0.001)
        print("✓ Performance context: operation completed")
    
    # 3. Events
    print("\n3. Using Events:")
    print("-" * 80)
    
    def on_operation_completed(event):
        print(f"  Event: {event.event_type.value} - {event.data.get('operation', 'unknown')}")
    
    def on_backend_selected(event):
        print(f"  Event: {event.event_type.value} - {event.data.get('backend', 'unknown')}")
    
    on_event(EventType.OPERATION_COMPLETED, on_operation_completed)
    on_event(EventType.BACKEND_SELECTED, on_backend_selected)
    
    # Operations will emit events
    with operation_context("test_operation"):
        cache.put(layer=0, position=2, key=k, value=v)
    
    with backend_context("rust"):
        pass
    
    print("\n" + "=" * 80)
    print("Decorators & Context Managers Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()













