"""
Unified Factory System
======================
Centralized access to all factory functions in optimization_core.
"""

# Import registry
from .registry import Registry

# Import all factory modules
from .attention import (
    ATTENTION_BACKENDS,
    sdpa_attention,
    build_sdpa,
    build_flash,
    build_triton,
)

from .optimizer import (
    OPTIMIZERS,
    build_adamw,
    build_lion,
    build_adafactor,
)

from .datasets import (
    DATASETS,
    build_hf,
    build_jsonl,
    build_webdataset,
)

from .callbacks import (
    CALLBACKS,
    build_print,
    build_wandb,
    build_tensorboard,
)

from .collate import (
    COLLATE as COLLATORS,
    build_lm_collate,
    build_cv_collate,
)

from .kv_cache import (
    KV_CACHE as KV_CACHE_FACTORIES,
    build_none as build_kv_cache_none,
    build_paged as build_kv_cache_paged,
)

from .memory import (
    MEMORY_MANAGERS as MEMORY_FACTORIES,
    build_adaptive as build_memory_adaptive,
    build_static as build_memory_static,
)

from .metrics import (
    METRICS,
    metric_loss,
    metric_ppl,
)

# Create helper functions
def get_attention_backend(name: str, *args, **kwargs):
    """Get an attention backend by name."""
    return ATTENTION_BACKENDS.build(name, *args, **kwargs)

def get_optimizer(name: str, *args, **kwargs):
    """Get an optimizer by name."""
    return OPTIMIZERS.build(name, *args, **kwargs)

def get_dataset(name: str, *args, **kwargs):
    """Get a dataset by name."""
    return DATASETS.build(name, *args, **kwargs)

def get_callback(name: str, *args, **kwargs):
    """Get a callback by name."""
    return CALLBACKS.build(name, *args, **kwargs)

def get_collator(name: str, *args, **kwargs):
    """Get a collator by name."""
    return COLLATORS.build(name, *args, **kwargs)

def get_kv_cache(name: str, *args, **kwargs):
    """Get a KV cache by name."""
    return KV_CACHE_FACTORIES.build(name, *args, **kwargs)

def get_memory(name: str, *args, **kwargs):
    """Get a memory manager by name."""
    return MEMORY_FACTORIES.build(name, *args, **kwargs)

def get_metric(name: str, *args, **kwargs):
    """Get a metric by name."""
    return METRICS.build(name, *args, **kwargs)

# Alias for backward compatibility
build_dataset = get_dataset
build_callback = get_callback
build_collator = get_collator
build_kv_cache = get_kv_cache
build_memory = get_memory
build_metric = get_metric


# Unified factory function
def create_factory(factory_type: str = "optimizer", name: str = None, *args, **kwargs):
    """
    Unified factory function to create any factory item.
    
    Args:
        factory_type: Type of factory to use. Options:
            - "optimizer" - OPTIMIZERS registry
            - "attention" - ATTENTION_BACKENDS registry
            - "dataset" - DATASETS registry
            - "callback" - CALLBACKS registry
            - "collator" - COLLATORS registry
            - "kv_cache" - KV_CACHE_FACTORIES registry
            - "memory" - MEMORY_FACTORIES registry
            - "metric" - METRICS registry
        name: Name of the item to create
        *args, **kwargs: Arguments to pass to the factory function
    
    Returns:
        The created item
    """
    factory_type = factory_type.lower()
    
    factory_map = {
        "optimizer": OPTIMIZERS,
        "attention": ATTENTION_BACKENDS,
        "dataset": DATASETS,
        "callback": CALLBACKS,
        "collator": COLLATORS,
        "kv_cache": KV_CACHE_FACTORIES,
        "memory": MEMORY_FACTORIES,
        "metric": METRICS,
    }
    
    if factory_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown factory type: '{factory_type}'. "
            f"Available types: {available}"
        )
    
    registry = factory_map[factory_type]
    
    if name is None:
        raise ValueError(f"Name is required for factory type '{factory_type}'")
    
    return registry.build(name, *args, **kwargs)


# Registry of all available factories
FACTORY_REGISTRY = {
    "optimizer": {
        "registry": OPTIMIZERS,
        "module": "factories.optimizer",
        "description": "Optimizer factory for creating optimizers",
        "helper": get_optimizer,
    },
    "attention": {
        "registry": ATTENTION_BACKENDS,
        "module": "factories.attention",
        "description": "Attention backend factory",
        "helper": get_attention_backend,
    },
    "dataset": {
        "registry": DATASETS,
        "module": "factories.datasets",
        "description": "Dataset factory",
        "helper": get_dataset,
    },
    "callback": {
        "registry": CALLBACKS,
        "module": "factories.callbacks",
        "description": "Callback factory",
        "helper": get_callback,
    },
    "collator": {
        "registry": COLLATORS,
        "module": "factories.collate",
        "description": "Collator factory",
        "helper": get_collator,
    },
    "kv_cache": {
        "registry": KV_CACHE_FACTORIES,
        "module": "factories.kv_cache",
        "description": "KV cache factory",
        "helper": get_kv_cache,
    },
    "memory": {
        "registry": MEMORY_FACTORIES,
        "module": "factories.memory",
        "description": "Memory factory",
        "helper": get_memory,
    },
    "metric": {
        "registry": METRICS,
        "module": "factories.metrics",
        "description": "Metric factory",
        "helper": get_metric,
    },
}


def list_available_factories() -> list:
    """List all available factory types."""
    return list(FACTORY_REGISTRY.keys())


def list_factory_items(factory_type: str) -> list:
    """
    List all available items in a specific factory.
    
    Args:
        factory_type: Type of factory
    
    Returns:
        List of available item names
    """
    if factory_type not in FACTORY_REGISTRY:
        raise ValueError(f"Unknown factory type: {factory_type}")
    
    registry = FACTORY_REGISTRY[factory_type]["registry"]
    return list(registry._items.keys())


def get_factory_info(factory_type: str) -> dict:
    """
    Get information about a specific factory.
    
    Args:
        factory_type: Type of factory
    
    Returns:
        Dictionary with factory information
    """
    if factory_type not in FACTORY_REGISTRY:
        raise ValueError(f"Unknown factory type: {factory_type}")
    
    factory_entry = FACTORY_REGISTRY[factory_type]
    registry = factory_entry["registry"]
    
    return {
        "type": factory_type,
        "module": factory_entry["module"],
        "description": factory_entry["description"],
        "available_items": list(registry._items.keys()),
        "item_count": len(registry._items),
    }


__all__ = [
    # Registry
    "Registry",
    # Attention factories
    "ATTENTION_BACKENDS",
    "sdpa_attention",
    "build_sdpa",
    "build_flash",
    "build_triton",
    "get_attention_backend",
    # Optimizer factories
    "OPTIMIZERS",
    "build_adamw",
    "build_lion",
    "build_adafactor",
    "get_optimizer",
    # Dataset factories
    "DATASETS",
    "build_hf",
    "build_jsonl",
    "build_webdataset",
    "build_dataset",
    "get_dataset",
    # Callback factories
    "CALLBACKS",
    "build_print",
    "build_wandb",
    "build_tensorboard",
    "build_callback",
    "get_callback",
    # Collator factories
    "COLLATORS",
    "build_lm_collate",
    "build_cv_collate",
    "build_collator",
    "get_collator",
    # KV cache factories
    "KV_CACHE_FACTORIES",
    "build_kv_cache_none",
    "build_kv_cache_paged",
    "build_kv_cache",
    "get_kv_cache",
    # Memory factories
    "MEMORY_FACTORIES",
    "build_memory_adaptive",
    "build_memory_static",
    "build_memory",
    "get_memory",
    # Metric factories
    "METRICS",
    "metric_loss",
    "metric_ppl",
    "build_metric",
    "get_metric",
    # Unified factory
    "create_factory",
    # Registry
    "FACTORY_REGISTRY",
    "list_available_factories",
    "list_factory_items",
    "get_factory_info",
]


