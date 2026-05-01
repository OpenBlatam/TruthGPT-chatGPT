"""
Unified Models System
=====================
Centralized access to all model-related classes in optimization_core.
"""

from __future__ import annotations
from optimization_core.utils.dependency_manager import resolve_lazy_import

_LAZY_IMPORTS = {
    # Managers & Builders
    'ModelManager': '.model_manager',
    'ModelBuilder': '.model_builder',
    
    # HuggingFace
    'HFTransformersModel': '.hf_transformers',
    'create_hf_transformers_model': '.hf_transformers',
    'HFDiffusersModel': '.hf_diffusers',
    'create_hf_diffusers_model': '.hf_diffusers',
    
    # Diffusion
    'DiffusionManager': '.diffusion_manager',
    
    # Utils
    'AttentionUtils': '.attention_utils',
}

def __getattr__(name: str):
    """Lazy import system for model components."""
    return resolve_lazy_import(name, __package__ or 'models', _LAZY_IMPORTS)

def create_model(
    model_type: str = "manager",
    config: dict = None
):
    """
    Unified factory function to create models or model managers.
    
    Args:
        model_type: Type of model to create. Options:
            - "manager" - ModelManager
            - "builder" - ModelBuilder
            - "diffusion" - DiffusionManager
            - "hf_transformers" - HFTransformersModel
            - "hf_diffusers" - HFDiffusersModel
        config: Optional configuration dictionary
    
    Returns:
        The requested model instance
    """
    if config is None:
        config = {}
    
    model_type = model_type.lower()
    
    # Import factories lazily
    if model_type == "manager":
        from .model_manager import ModelManager
        return ModelManager(**config)
    elif model_type == "builder":
        from .model_builder import ModelBuilder
        return ModelBuilder()  # Builder usually doesn't take config in init
    elif model_type == "diffusion":
        from .diffusion_manager import DiffusionManager
        return DiffusionManager(config)
    elif model_type == "hf_transformers":
        from .hf_transformers import create_hf_transformers_model
        return create_hf_transformers_model(config)
    elif model_type == "hf_diffusers":
        # Handle potential missing dependency gracefully
        try:
            from .hf_diffusers import create_hf_diffusers_model
            return create_hf_diffusers_model(config)
        except ImportError:
            raise ImportError("hf_diffusers module not available")
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

# Registry of all available models
MODEL_REGISTRY = {
    "manager": {
        "class": "ModelManager",
        "module": "models.model_manager",
        "description": "Model manager for loading and saving models",
    },
    "builder": {
        "class": "ModelBuilder",
        "module": "models.model_builder",
        "description": "Model builder for constructing models",
    },
    "diffusion": {
        "class": "DiffusionManager",
        "module": "models.diffusion_manager",
        "description": "Diffusion model manager",
    },
    "hf_transformers": {
        "class": "HFTransformersModel",
        "module": "models.hf_transformers",
        "description": "HuggingFace Transformers model",
    },
    "hf_diffusers": {
        "class": "HFDiffusersModel",
        "module": "models.hf_diffusers",
        "description": "HuggingFace Diffusers model",
    },
}

def list_available_models() -> list:
    """List all available model types."""
    return list(MODEL_REGISTRY.keys())

def get_model_info(model_type: str) -> dict:
    """Get information about a specific model."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    entry = MODEL_REGISTRY[model_type]
    return {
        "type": model_type,
        "class": entry["class"],
        "module": entry["module"],
        "description": entry["description"],
    }

__all__ = list(_LAZY_IMPORTS.keys()) + [
    "create_model",
    "MODEL_REGISTRY",
    "list_available_models",
    "get_model_info"
]

