"""
Example: Advanced Infrastructure Features

Demonstrates API, Service Discovery, and Load Balancing.
"""

import asyncio
from optimization_core.polyglot_core import (
    get_api_router,
    register_endpoint,
    get_service_registry,
    register_service,
    create_load_balancer,
    LoadBalanceStrategy,
    KVCache,
    Backend,
)


def example_api():
    """Example: REST API setup."""
    print("=" * 60)
    print("Example: REST API")
    print("=" * 60)
    
    router = get_api_router()
    
    @router.register("/cache/get", method="GET", description="Get from cache")
    def get_cache(layer: int, position: int):
        """Get cache entry."""
        return {"layer": layer, "position": position, "value": "cached"}
    
    @router.register("/cache/set", method="POST", description="Set cache entry")
    def set_cache(layer: int, position: int, value: str):
        """Set cache entry."""
        return {"success": True, "layer": layer, "position": position}
    
    @router.register("/health", method="GET", description="Health check")
    def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    # Create FastAPI app (if FastAPI is installed)
    try:
        app = router.create_fastapi_app()
        print(f"✅ FastAPI app created with {len(router.get_endpoints())} endpoints")
        print("\nEndpoints:")
        for endpoint in router.get_endpoints():
            print(f"  {endpoint.method} {endpoint.path} - {endpoint.description}")
    except ImportError:
        print("⚠️  FastAPI not installed. Install with: pip install fastapi")
    
    print()


def example_service_discovery():
    """Example: Service Discovery."""
    print("=" * 60)
    print("Example: Service Discovery")
    print("=" * 60)
    
    registry = get_service_registry()
    
    # Register services
    cache_service_id = register_service(
        "cache_service",
        "localhost",
        8080,
        version="1.0.0",
        backend="rust"
    )
    
    attention_service_id = register_service(
        "attention_service",
        "localhost",
        8081,
        version="1.0.0",
        backend="cpp"
    )
    
    inference_service_id = register_service(
        "inference_service",
        "localhost",
        8082,
        version="1.0.0",
        backend="python"
    )
    
    print(f"✅ Registered {len(registry.list_services())} services")
    
    # Discover services
    cache_services = registry.discover("cache_service")
    print(f"\n📡 Discovered {len(cache_services)} cache services")
    
    # Update heartbeat
    registry.heartbeat(cache_service_id)
    print(f"💓 Updated heartbeat for cache_service")
    
    # List all services
    print("\nAll services:")
    for service in registry.list_services():
        print(f"  - {service.name} ({service.address}:{service.port}) - {service.status.value}")
    
    print()


def example_load_balancer():
    """Example: Load Balancing."""
    print("=" * 60)
    print("Example: Load Balancing")
    print("=" * 60)
    
    # Create multiple cache instances
    cache_rust = KVCache(max_size=8192, backend=Backend.RUST)
    cache_cpp = KVCache(max_size=8192, backend=Backend.CPP)
    cache_python = KVCache(max_size=8192, backend=Backend.PYTHON)
    
    # Create load balancer
    balancer = create_load_balancer(LoadBalanceStrategy.LEAST_LATENCY)
    
    # Add instances with different weights
    balancer.add_instance("rust1", cache_rust, weight=2.0)
    balancer.add_instance("cpp1", cache_cpp, weight=1.5)
    balancer.add_instance("python1", cache_python, weight=1.0)
    
    print(f"✅ Created load balancer with {len(balancer._instances)} instances")
    print(f"   Strategy: {balancer.strategy.value}")
    
    # Execute operations through load balancer
    def cache_operation(cache, key: str, value: str):
        """Cache operation."""
        cache.set(key, value)
        return cache.get(key)
    
    # Test load balancing
    print("\n🔄 Testing load balancing:")
    for i in range(5):
        instance = balancer.select_instance()
        if instance:
            result = balancer.execute(cache_operation, f"key_{i}", f"value_{i}")
            print(f"  Request {i+1}: {instance.id} -> {result}")
    
    # Get statistics
    stats = balancer.get_stats()
    print(f"\n📊 Load Balancer Stats:")
    print(f"   Total instances: {stats['total_instances']}")
    print(f"   Healthy instances: {stats['healthy_instances']}")
    print(f"   Strategy: {stats['strategy']}")
    
    print()


def example_integrated():
    """Example: Integrated Infrastructure."""
    print("=" * 60)
    print("Example: Integrated Infrastructure")
    print("=" * 60)
    
    # Service Discovery + Load Balancer
    registry = get_service_registry()
    
    # Register services
    register_service("cache", "localhost", 8080, backend="rust")
    register_service("cache", "localhost", 8081, backend="cpp")
    
    # Discover and create load balancer
    services = registry.discover("cache")
    balancer = create_load_balancer(LoadBalanceStrategy.ROUND_ROBIN)
    
    for service in services:
        cache = KVCache(max_size=8192)
        balancer.add_instance(
            f"{service.name}_{service.port}",
            cache,
            weight=1.0
        )
    
    print(f"✅ Integrated setup:")
    print(f"   Services discovered: {len(services)}")
    print(f"   Load balancer instances: {len(balancer._instances)}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Polyglot Core - Infrastructure Examples")
    print("=" * 60 + "\n")
    
    example_api()
    example_service_discovery()
    example_load_balancer()
    example_integrated()
    
    print("=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)













