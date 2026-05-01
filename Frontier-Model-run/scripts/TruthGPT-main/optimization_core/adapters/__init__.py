"""
Unified Adapters System — Pydantic-First Architecture.
======================================================
Centralized access to all adapter classes in optimization_core.
"""

from typing import Dict, List, Optional, Type, Any

from pydantic import BaseModel, Field

from .base import BaseDynamicAdapter, ObjectStore, ObjectEntry, StoreStats, AdapterRunResult

# Import core adapters
from .optimizer_adapter import (
    OptimizerAdapter,
    PyTorchOptimizerAdapter,
)

from .data_adapter import (
    DataAdapter,
    HuggingFaceDataAdapter,
    JSONLDataAdapter,
)

from .model_adapter import (
    ModelAdapter,
    HuggingFaceModelAdapter,
)

from .training_adapter import (
    TrainingAdapter,
)

# Import edge adapters
try:
    from ..modules.edge.edge_inference_adapter import (
        EdgeInferenceAdapter,
    )
except ImportError:
    EdgeInferenceAdapter = None

# Import TruthGPT adapters
from .truthgpt_adapters import (
    TruthGPTAdapter,
)

from .enterprise_truthgpt_adapter import (
    EnterpriseTruthGPTAdapter,
)


# ---------------------------------------------------------------------------
# Pydantic Models for Registry
# ---------------------------------------------------------------------------

class AdapterRegistryEntry(BaseModel):
    """Typed metadata for a registered adapter class."""
    adapter_class_name: Optional[str] = None
    module: str
    description: str
    available: bool = True


class AdapterInfo(BaseModel):
    """Typed response for adapter introspection queries."""
    adapter_type: str
    subtype: Optional[str] = None
    class_name: Optional[str] = None
    module: Optional[str] = None
    description: Optional[str] = None
    subtypes: Optional[Dict[str, "AdapterRegistryEntry"]] = None


# ---------------------------------------------------------------------------
# Unified adapter factory
# ---------------------------------------------------------------------------

def create_adapter(
    adapter_type: str = "optimizer",
    adapter_subtype: str = "pytorch",
    config: Optional[Dict[str, Any]] = None,
):
    """
    Unified factory function to create adapters.

    Args:
        adapter_type: Type of adapter to create (optimizer, data, model, edge, truthgpt, enterprise).
        adapter_subtype: Subtype of adapter (pytorch, huggingface, jsonl, etc.).
        config: Optional configuration dictionary.

    Returns:
        The requested adapter instance.
    """
    if config is None:
        config = {}

    adapter_type = adapter_type.lower()
    adapter_subtype = adapter_subtype.lower()

    factory_map = {
        "optimizer": {
            "pytorch": lambda cfg: PyTorchOptimizerAdapter(**cfg),
        },
        "data": {
            "huggingface": lambda cfg: HuggingFaceDataAdapter(**cfg),
            "jsonl": lambda cfg: JSONLDataAdapter(**cfg),
        },
        "model": {
            "huggingface": lambda cfg: HuggingFaceModelAdapter(**cfg),
        },
        "edge": {
            "inference": lambda cfg: EdgeInferenceAdapter(**cfg) if EdgeInferenceAdapter else None,
        },
        "truthgpt": {
            "default": lambda cfg: TruthGPTAdapter(adapter_config=cfg),
        },
        "enterprise": {
            "default": lambda cfg: EnterpriseTruthGPTAdapter(adapter_config=cfg),
        },
        "training": {
            "default": lambda cfg: TrainingAdapter(**cfg),
        },
    }

    if adapter_type not in factory_map:
        available = ", ".join(factory_map.keys())
        raise ValueError(
            f"Unknown adapter type: '{adapter_type}'. "
            f"Available types: {available}"
        )

    subtype_map = factory_map[adapter_type]

    if adapter_subtype not in subtype_map:
        available = ", ".join(subtype_map.keys())
        raise ValueError(
            f"Unknown adapter subtype '{adapter_subtype}' for type '{adapter_type}'. "
            f"Available subtypes: {available}"
        )

    factory = subtype_map[adapter_subtype]
    adapter = factory(config)

    if adapter is None:
        raise ImportError(
            f"Adapter type '{adapter_type}' with subtype '{adapter_subtype}' "
            f"is not available (module not found)"
        )

    return adapter


# ---------------------------------------------------------------------------
# Typed Adapter Registry
# ---------------------------------------------------------------------------

def _entry(cls: Optional[Any], module: str, description: str) -> AdapterRegistryEntry:
    """Helper to create registry entries, robust against Mock objects."""
    name = None
    if cls:
        # Safely extract name even if cls is a MagicMock
        try:
            # Check if it's a mock by looking for mock-specific attrs
            if hasattr(cls, "_mock_return_value") or "Mock" in str(type(cls)):
                name = str(cls).split('.')[-1].strip('>')
                if not name or "Mock" in name:
                     name = "MockAdapter"
            else:
                name = getattr(cls, "__name__", str(cls))
        except Exception:
            name = "UnknownAdapter"
            
    if not isinstance(name, str):
        name = str(name)

    return AdapterRegistryEntry(
        adapter_class_name=name,
        module=module,
        description=description,
        available=cls is not None,
    )


ADAPTER_REGISTRY: Dict[str, Dict[str, AdapterRegistryEntry]] = {
    "optimizer": {
        "pytorch": _entry(PyTorchOptimizerAdapter, "adapters.optimizer_adapter", "PyTorch optimizer adapter"),
    },
    "data": {
        "huggingface": _entry(HuggingFaceDataAdapter, "adapters.data_adapter", "HuggingFace data adapter"),
        "jsonl": _entry(JSONLDataAdapter, "adapters.data_adapter", "JSONL data adapter"),
    },
    "model": {
        "huggingface": _entry(HuggingFaceModelAdapter, "adapters.model_adapter", "HuggingFace model adapter"),
    },
    "edge": {
        "inference": _entry(EdgeInferenceAdapter, "modules.edge.edge_inference_adapter", "Edge inference adapter"),
    },
    "truthgpt": {
        "default": _entry(TruthGPTAdapter, "adapters.truthgpt_adapters", "TruthGPT adapter"),
    },
    "enterprise": {
        "default": _entry(EnterpriseTruthGPTAdapter, "adapters.enterprise_truthgpt_adapter", "Enterprise TruthGPT adapter"),
    },
    "training": {
        "default": _entry(TrainingAdapter, "adapters.training_adapter", "Training orchestration adapter"),
    },
}


def list_available_adapter_types() -> List[str]:
    """List all available adapter types."""
    return list(ADAPTER_REGISTRY.keys())


def list_available_adapter_subtypes(adapter_type: str) -> List[str]:
    """List all available subtypes for an adapter type."""
    if adapter_type not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return list(ADAPTER_REGISTRY[adapter_type].keys())


def get_adapter_info(adapter_type: str, adapter_subtype: Optional[str] = None) -> AdapterInfo:
    """
    Get typed information about a specific adapter.

    Returns:
        AdapterInfo model with adapter details.
    """
    if adapter_type not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    if adapter_subtype is None:
        return AdapterInfo(
            adapter_type=adapter_type,
            subtypes=ADAPTER_REGISTRY[adapter_type],
        )

    if adapter_subtype not in ADAPTER_REGISTRY[adapter_type]:
        raise ValueError(f"Unknown adapter subtype: {adapter_subtype}")

    entry = ADAPTER_REGISTRY[adapter_type][adapter_subtype]

    if not entry.available:
        raise ImportError(
            f"Adapter type '{adapter_type}' with subtype '{adapter_subtype}' "
            f"is not available (module not found)"
        )

    return AdapterInfo(
        adapter_type=adapter_type,
        subtype=adapter_subtype,
        class_name=entry.adapter_class_name,
        module=entry.module,
        description=entry.description,
    )


__all__ = [
    # Base
    "BaseDynamicAdapter",
    "ObjectStore",
    "ObjectEntry",
    "StoreStats",
    "AdapterRunResult",
    # Pydantic models
    "AdapterRegistryEntry",
    "AdapterInfo",
    # Optimizer adapters
    "OptimizerAdapter",
    "PyTorchOptimizerAdapter",
    # Data adapters
    "DataAdapter",
    "HuggingFaceDataAdapter",
    "JSONLDataAdapter",
    # Model adapters
    "ModelAdapter",
    "HuggingFaceModelAdapter",
    # Edge adapters
    "EdgeInferenceAdapter",
    # TruthGPT adapters
    "TruthGPTAdapter",
    "EnterpriseTruthGPTAdapter",
    # Unified factory
    "create_adapter",
    # Registry
    "ADAPTER_REGISTRY",
    "list_available_adapter_types",
    "list_available_adapter_subtypes",
    "get_adapter_info",
]

# --- Dynamic MCP Adapters Registration ---
try:
    from ..agents.registry import registry

    # Register all core adapters to make them available as tools to agents
    registry.register("pytorch_optimizer", PyTorchOptimizerAdapter)
    registry.register("huggingface_data", HuggingFaceDataAdapter)
    registry.register("jsonl_data", JSONLDataAdapter)
    registry.register("huggingface_model", HuggingFaceModelAdapter)
    registry.register("training_orchestrator", TrainingAdapter)
    registry.register("truthgpt_adapter", TruthGPTAdapter)
    registry.register("enterprise_adapter", EnterpriseTruthGPTAdapter)
    
    if EdgeInferenceAdapter:
        registry.register("edge_inference", EdgeInferenceAdapter)

except Exception as e:
    logger.warning("Failed to register adapters in ComponentRegistry: %s", e)

