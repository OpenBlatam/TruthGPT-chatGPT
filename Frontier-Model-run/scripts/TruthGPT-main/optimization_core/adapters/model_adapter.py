"""
Model adapters — Pydantic-First Architecture.

The ``process()`` method performs *real* model I/O and stores/retrieves
torch.nn.Module instances via the global ObjectStore, returning typed
Pydantic results with lightweight ``model_id`` strings.
"""

import logging
from typing import Dict, Any, List, Optional

import torch
from pydantic import BaseModel, Field, computed_field

from .base import BaseDynamicAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------

class ModelInfoResult(BaseModel):
    """Typed model statistics."""
    num_parameters: int = 0
    trainable_parameters: int = 0
    model_type: Optional[str] = None
    vocab_size: Optional[int] = None

    @computed_field  # type: ignore[misc]
    @property
    def trainable_pct(self) -> float:
        if self.num_parameters == 0:
            return 0.0
        return round(self.trainable_parameters / self.num_parameters * 100, 2)


class ModelLoadResult(BaseModel):
    """Typed result from a model load action."""
    status: str = "success"
    model_id: str
    info: ModelInfoResult


class ModelSaveResult(BaseModel):
    """Typed result from a model save action."""
    status: str = "success"
    message: str


class ModelListResult(BaseModel):
    """Typed result from a model list action."""
    status: str = "success"
    models: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Adapter
# ---------------------------------------------------------------------------

class ModelAdapter(BaseDynamicAdapter):
    """Base adapter for model operations."""

    name: str = "model_adapter"
    description: str = (
        "Adapter to load, save, and inspect PyTorch models. Input JSON: "
        "{'action': 'load'|'save'|'info'|'list', 'path': 'str', 'model_id': 'str'}"
    )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        action = input_data.get("action")
        kwargs = input_data.get("kwargs", {})

        if action == "load":
            path = input_data.get("path", "")
            model = self.load_model(path, **kwargs)
            model_id = self.store.put(
                model,
                kind="model",
                meta={"path": path, "num_params": sum(p.numel() for p in model.parameters())},
            )
            info = self.get_model_info(model)
            return ModelLoadResult(model_id=model_id, info=info).model_dump()

        elif action == "save":
            model_id = input_data.get("model_id", "")
            path = input_data.get("path", "")
            model = self.store.get(model_id)
            self.save_model(model, path, **kwargs)
            return ModelSaveResult(message=f"Model saved to {path}").model_dump()

        elif action == "info":
            model_id = input_data.get("model_id", "")
            model = self.store.get(model_id)
            info = self.get_model_info(model)
            return {"status": "success", "model_id": model_id, **info.model_dump()}

        elif action == "list":
            ids = self.store.list_ids(kind="model")
            return ModelListResult(models=ids).model_dump()

        else:
            raise ValueError(f"Unknown model action: '{action}'. Use 'load', 'save', 'info', or 'list'.")

    def load_model(self, model_path: str, **kwargs) -> torch.nn.Module:
        """Load a model.  Override in subclasses."""
        raise NotImplementedError("Subclass must implement load_model()")

    def save_model(self, model: torch.nn.Module, path: str, **kwargs) -> None:
        """Save a model.  Override in subclasses."""
        raise NotImplementedError("Subclass must implement save_model()")

    def get_model_info(self, model: torch.nn.Module) -> ModelInfoResult:
        """Return typed model statistics."""
        return ModelInfoResult(
            num_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )


class HuggingFaceModelAdapter(ModelAdapter):
    """Adapter for HuggingFace Transformers models."""

    name: str = "hf_model_adapter"
    description: str = (
        "Load/save HuggingFace Transformers models. Input JSON: "
        "{'action': 'load', 'path': 'meta-llama/Llama-2-7b', 'kwargs': {'device_map': 'auto'}}"
    )

    def load_model(self, model_path: str, **kwargs) -> torch.nn.Module:
        """Load HuggingFace model."""
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=kwargs.get("torch_dtype"),
            device_map=kwargs.get("device_map"),
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

    def save_model(self, model: torch.nn.Module, path: str, **kwargs) -> None:
        """Save HuggingFace model."""
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(
            path,
            safe_serialization=kwargs.get("safe_serialization", True),
        )

    def get_model_info(self, model: torch.nn.Module) -> ModelInfoResult:
        """Get detailed HuggingFace model information."""
        base_model = model.module if hasattr(model, "module") else model
        info = super().get_model_info(model)

        if hasattr(base_model, "config"):
            info.model_type = getattr(base_model.config, "model_type", "unknown")
            info.vocab_size = getattr(base_model.config, "vocab_size", 0)

        return info

