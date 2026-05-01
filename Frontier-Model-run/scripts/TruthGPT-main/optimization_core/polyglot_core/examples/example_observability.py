"""
Example: Observability and Rate Limiting.

Demonstrates tracing, monitoring, and rate limiting.
"""

import numpy as np
from optimization_core.polyglot_core import (
    KVCache,
    Attention,
    AttentionConfig,
    get_observability,
    trace,
    rate_limit,
    RateLimit,
    RateLimiter,
)


def main():
    print("=" * 80)
    print("Polyglot Core - Observability & Rate Limiting Example")
    print("=" * 80)
    print()
    
    # 1. Observability / Tracing
    print("1. Observability / Tracing:")
    print("-" * 80)
    
    observability = get_observability()
    
    # Trace operations
    with observability.trace("cache_operations", {"backend": "rust"}) as span:
        cache = KVCache(max_size=1000)
        k = np.random.randn(32).astype(np.float32)
        v = np.random.randn(32).astype(np.float32)
        
        span.add_attribute("cache_size", cache.max_size)
        span.add_event("cache_put")
        cache.put(layer=0, position=0, key=k, value=v)
        
        span.add_event("cache_get")
        result = cache.get(layer=0, position=0)
        
        span.add_attribute("result_found", result is not None)
    
    # Export traces
    traces = observability.tracer.export_traces()
    print(f"Total spans: {len(traces['spans'])}")
    for span_data in traces['spans']:
        print(f"  - {span_data['name']}: {span_data['duration_ms']:.2f}ms ({span_data['status']})")
    
    # 2. Rate Limiting
    print("\n2. Rate Limiting:")
    print("-" * 80)
    
    # Create rate limiter
    rate_limit_config = RateLimit(max_requests=5, time_window_seconds=1.0, name="cache_ops")
    limiter = RateLimiter(rate_limit_config)
    
    # Test rate limiting
    print("Testing rate limiting (5 requests per second):")
    for i in range(7):
        if limiter.acquire(wait=False):
            print(f"  Request {i+1}: ✓ Allowed")
        else:
            print(f"  Request {i+1}: ✗ Rate limited")
            remaining = limiter.get_remaining()
            print(f"    Remaining: {remaining}")
    
    # 3. Rate Limit Decorator
    print("\n3. Rate Limit Decorator:")
    print("-" * 80)
    
    @rate_limit(max_requests=3, time_window_seconds=1.0, name="test_function")
    def rate_limited_function(x):
        return x * 2
    
    print("Calling rate-limited function:")
    for i in range(5):
        try:
            result = rate_limited_function(i)
            print(f"  Call {i+1}: ✓ Result = {result}")
        except Exception as e:
            print(f"  Call {i+1}: ✗ {e}")
    
    # 4. Combined Observability and Rate Limiting
    print("\n4. Combined Observability & Rate Limiting:")
    print("-" * 80)
    
    @rate_limit(max_requests=10, time_window_seconds=1.0)
    def traced_and_rate_limited_operation(x):
        with trace("operation", {"input": x}):
            # Simulate work
            import time
            time.sleep(0.01)
            return x * 2
    
    print("Combined operation:")
    for i in range(5):
        try:
            result = traced_and_rate_limited_operation(i)
            print(f"  Operation {i+1}: ✓ Result = {result}")
        except Exception as e:
            print(f"  Operation {i+1}: ✗ {e}")
    
    # Export final traces
    final_traces = observability.tracer.export_traces()
    print(f"\nTotal traces collected: {len(final_traces['spans'])}")
    
    print("\n" + "=" * 80)
    print("Observability & Rate Limiting Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()













