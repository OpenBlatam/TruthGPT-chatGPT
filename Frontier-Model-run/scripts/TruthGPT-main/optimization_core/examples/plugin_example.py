"""
Example demonstrating plugin system for extensibility.
"""
from modules.base.core_system.core.plugin_system import Plugin, PluginManager
from modules.base.core_system.core.service_registry import ServiceRegistry
from modules.base.core_system.core.dynamic_factory import DynamicFactory


class CustomOptimizerPlugin(Plugin):
    """Example plugin that adds a custom optimizer."""
    
    @property
    def name(self) -> str:
        return "custom_optimizer"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def initialize(self, registry: ServiceRegistry) -> None:
        """Initialize plugin and register components."""
        import torch.optim
        
        # Register custom optimizer factory
        def create_custom_optimizer(parameters, lr=0.001, **kwargs):
            # Example: Custom optimizer wrapper
            return torch.optim.Adam(parameters, lr=lr, **kwargs)
        
        registry.register("custom_optimizer", create_custom_optimizer, singleton=False)
        print(f"Plugin '{self.name}' initialized: Custom optimizer registered")


class CustomDatasetPlugin(Plugin):
    """Example plugin that adds a custom dataset loader."""
    
    @property
    def name(self) -> str:
        return "custom_dataset"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def initialize(self, registry: ServiceRegistry) -> None:
        """Register custom dataset loader."""
        def load_custom_dataset(path: str):
            # Example custom dataset loading
            with open(path, 'r') as f:
                return f.read().splitlines()
        
        registry.register("custom_dataset_loader", load_custom_dataset, singleton=False)
        print(f"Plugin '{self.name}' initialized: Custom dataset loader registered")
    
    def get_dependencies(self) -> list[str]:
        """This plugin has no dependencies."""
        return []


def plugin_example():
    """Demonstrate plugin system."""
    print("=" * 60)
    print("Plugin System Example")
    print("=" * 60)
    
    # Create plugin manager
    manager = PluginManager()
    
    # Register plugins
    print("\n1. Registering plugins...")
    manager.register_plugin(CustomOptimizerPlugin())
    manager.register_plugin(CustomDatasetPlugin())
    
    # List plugins
    print(f"\n2. Registered plugins: {manager.list_plugins()}")
    print(f"   Active plugins: {manager.list_active_plugins()}")
    
    # Use plugin services
    print("\n3. Using plugin services...")
    registry = manager._registry
    
    # Get custom optimizer
    optimizer_factory = registry.get("custom_optimizer")
    print(f"   Custom optimizer factory: {optimizer_factory}")
    
    # Get custom dataset loader
    dataset_loader = registry.get("custom_dataset_loader")
    print(f"   Custom dataset loader: {dataset_loader}")
    
    # Deactivate plugin
    print("\n4. Deactivating plugin...")
    manager.deactivate_plugin("custom_optimizer")
    print(f"   Active plugins: {manager.list_active_plugins()}")


if __name__ == "__main__":
    plugin_example()



