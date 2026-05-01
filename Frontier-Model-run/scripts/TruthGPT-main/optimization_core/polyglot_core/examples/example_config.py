"""
Example: Configuration management.

Demonstrates how to use the configuration system.
"""

from optimization_core.polyglot_core import (
    PolyglotConfig,
    ConfigManager,
    Environment,
    get_config,
    load_config,
    save_config,
    KVCache,
    Attention,
)


def main():
    print("=" * 80)
    print("Polyglot Core - Configuration Example")
    print("=" * 80)
    print()
    
    # 1. Default configuration
    print("1. Default Configuration:")
    print("-" * 80)
    config = PolyglotConfig.default()
    print(f"Environment: {config.environment.value}")
    print(f"Debug: {config.debug}")
    print(f"Cache max size: {config.cache['default_max_size']}")
    print(f"Preferred KV cache backend: {config.preferred_backends['kv_cache']}")
    
    # 2. Production configuration
    print("\n2. Production Configuration:")
    print("-" * 80)
    prod_config = PolyglotConfig.production()
    print(f"Environment: {prod_config.environment.value}")
    print(f"Debug: {prod_config.debug}")
    print(f"Log level: {prod_config.log_level}")
    print(f"Max memory: {prod_config.resources['max_memory_gb']} GB")
    
    # 3. Development configuration
    print("\n3. Development Configuration:")
    print("-" * 80)
    dev_config = PolyglotConfig.development()
    print(f"Environment: {dev_config.environment.value}")
    print(f"Debug: {dev_config.debug}")
    print(f"Profiling enabled: {dev_config.performance['enable_profiling']}")
    print(f"Benchmarking enabled: {dev_config.performance['enable_benchmarking']}")
    
    # 4. Save and load configuration
    print("\n4. Save/Load Configuration:")
    print("-" * 80)
    config_file = "polyglot_config.yaml"
    save_config(prod_config, config_file)
    print(f"Saved configuration to: {config_file}")
    
    loaded_config = load_config(config_file)
    print(f"Loaded configuration: {loaded_config.environment.value}")
    
    # 5. Use configuration
    print("\n5. Using Configuration:")
    print("-" * 80)
    config = get_config()
    
    # Create cache with config settings
    cache = KVCache(max_size=config.cache['default_max_size'])
    print(f"Cache created with max_size: {cache.max_size}")
    
    # Create attention with config settings
    attention = Attention(
        AttentionConfig(
            d_model=config.attention['default_d_model'],
            n_heads=config.attention['default_n_heads']
        )
    )
    print(f"Attention created with d_model: {config.attention['default_d_model']}")
    
    # 6. Configuration manager
    print("\n6. Configuration Manager:")
    print("-" * 80)
    manager = ConfigManager()
    current = manager.get_config()
    print(f"Current config environment: {current.environment.value}")
    
    # Update config
    manager.update_config(debug=True)
    updated = manager.get_config()
    print(f"Updated debug: {updated.debug}")
    
    print("\n" + "=" * 80)
    print("Configuration Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()













