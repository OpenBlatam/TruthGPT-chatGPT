"""
Optimizer adapters — Pydantic-First Architecture.

The ``process()`` method creates *real* PyTorch optimizers from a model
stored in the ObjectStore.  It returns typed Pydantic results with an
``optimizer_id`` that downstream tools can consume via JSON.
"""

import logging
from typing import Dict, Any, Iterator, List, Optional

import torch
from pydantic import BaseModel, Field

from .base import BaseDynamicAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------

class OptimizerCreateResult(BaseModel):
    """Typed result from an optimizer create action."""
    status: str = "success"
    optimizer_id: str
    optimizer_type: str
    model_id: str


class OptimizerStateResult(BaseModel):
    """Typed result from an optimizer state query."""
    status: str = "success"
    optimizer_id: str
    type_name: str
    param_groups: int
    lr: Optional[float] = None


class OptimizerListResult(BaseModel):
    """Typed result from an optimizer list action."""
    status: str = "success"
    optimizers: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Adapter
# ---------------------------------------------------------------------------

class OptimizerAdapter(BaseDynamicAdapter):
    """Base adapter for optimizer operations."""

    name: str = "optimizer_adapter"
    description: str = (
        "Adapter to create and manage PyTorch optimizers. Input JSON: "
        "{'action': 'create'|'get_state'|'list', 'model_id': 'str', "
        "'optimizer_type': 'adamw', 'kwargs': {'lr': 1e-4}}"
    )

    OPTIMIZER_MAP: Dict[str, type] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically process optimizer operations based on input."""
        try:
            if action == "create":
                model_id = input_data.get("model_id", "")
                if not model_id:
                    raise ValueError("model_id is required for action='create'")
                    
                optimizer_type = input_data.get("optimizer_type", "adamw").lower()
                
                if not self.store:
                    raise RuntimeError("Adapter ObjectStore is not initialized.")
                    
                model = self.store.get(model_id)
                if model is None:
                    raise ValueError(f"Model {model_id} not found in ObjectStore")
                
                if not hasattr(model, "parameters"):
                    raise ValueError(f"Object {model_id} is not a valid model (missing .parameters())")

                # Enhancement: Suggest fused kernels if on CUDA
                if torch.cuda.is_available() and "fused" not in kwargs:
                    try:
                        kwargs["fused"] = True
                    except Exception:
                        pass # Some optimizers don't support fused

                optimizer = self.create_optimizer(
                    model.parameters(),
                    optimizer_type=optimizer_type,
                    **kwargs,
                )
                optimizer_id = self.store.put(
                    optimizer,
                    kind="optimizer",
                    meta={"model_id": model_id, "type": optimizer_type, **kwargs},
                )
                return OptimizerCreateResult(
                    optimizer_id=optimizer_id,
                    optimizer_type=optimizer_type,
                    model_id=model_id,
                ).model_dump()

            elif action == "get_state":
                optimizer_id = input_data.get("optimizer_id", "")
                if not optimizer_id:
                    raise ValueError("optimizer_id is required for action='get_state'")
                    
                if not self.store:
                    raise RuntimeError("Adapter ObjectStore is not initialized.")
                    
                optimizer = self.store.get(optimizer_id)
                if optimizer is None:
                    raise ValueError(f"Optimizer {optimizer_id} not found in ObjectStore")
                    
                state = self.get_optimizer_state(optimizer, optimizer_id)
                return state.model_dump()

            elif action == "list":
                if not self.store:
                    raise RuntimeError("Adapter ObjectStore is not initialized.")
                ids = self.store.list_ids(kind="optimizer")
                return OptimizerListResult(optimizers=ids).model_dump()

            else:
                raise ValueError(f"Unknown optimizer action: '{action}'. Use 'create', 'get_state', or 'list'.")
        except Exception as e:
            logger.error(f"Optimizer adapter error: {e}")
            return {"status": "error", "message": str(e)}

    def create_optimizer(
        self,
        parameters: Iterator[torch.nn.Parameter],
        optimizer_type: str = "adamw",
        **kwargs,
    ) -> torch.optim.Optimizer:
        """Create a PyTorch optimizer from the registry."""
        opt_cls = self.OPTIMIZER_MAP.get(optimizer_type)
        if opt_cls is None:
            available = ", ".join(self.OPTIMIZER_MAP.keys())
            raise ValueError(f"Unknown optimizer type: '{optimizer_type}'. Available: {available}")
        return opt_cls(parameters, **kwargs)

    def get_optimizer_state(self, optimizer: torch.optim.Optimizer, optimizer_id: str = "") -> OptimizerStateResult:
        """Get typed optimizer state summary."""
        return OptimizerStateResult(
            optimizer_id=optimizer_id,
            type_name=type(optimizer).__name__,
            param_groups=len(optimizer.param_groups),
            lr=optimizer.param_groups[0].get("lr") if optimizer.param_groups else None,
        )


class PyTorchOptimizerAdapter(OptimizerAdapter):
    """
    PyTorch-specific optimizer adapter.

    Inherits all logic from ``OptimizerAdapter``.  Exists as a named
    subclass for registry compatibility and future PyTorch-specific
    extensions (e.g., fused kernels, CUDA graphs).
    """

    name: str = "pytorch_optimizer_adapter"
    description: str = (
        "Create and manage PyTorch optimizers from a model_id. Input JSON: "
        "{'action': 'create', 'model_id': 'model_xxx', 'optimizer_type': 'adamw', "
        "'kwargs': {'lr': 1e-4, 'weight_decay': 0.01}}"
    )

