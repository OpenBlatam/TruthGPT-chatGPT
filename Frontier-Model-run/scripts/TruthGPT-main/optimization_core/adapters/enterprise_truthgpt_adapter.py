"""
Enterprise TruthGPT Adapter — Pydantic-First Architecture
"""

import logging
from typing import Dict, Any

from .base import BaseDynamicAdapter
from ..modules.enterprise import (
    AdapterConfig,
    EnterpriseTruthGPTModel,
    EnterpriseModelInfo,
)
from pydantic import BaseModel, Field


class EnterpriseAdapterCreateResult(BaseModel):
    """Typed result from enterprise create action."""
    status: str = "success"
    model_id: str
    info: EnterpriseModelInfo


class EnterpriseAdapterForwardResult(BaseModel):
    """Typed result from enterprise forward action."""
    status: str = "success"
    output_shape: list[int]


class EnterpriseTruthGPTAdapter(BaseDynamicAdapter):
    """Enterprise TruthGPT dynamic adapter for agent tools."""
    
    name: str = "enterprise_truthgpt_adapter"
    description: str = (
        "Enterprise TruthGPT operations. Input JSON: "
        "{'action': 'create'|'info'|'optimize', 'kwargs': {}}"
    )
    
    adapter_config: AdapterConfig = Field(default_factory=AdapterConfig)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically process enterprise model operations."""
        if not self.store:
            raise RuntimeError("Enterprise Adapter ObjectStore is not initialized.")
            
        action = input_data.get("action")
        kwargs = input_data.get("kwargs", {})
        
        try:
            if action == "create":
                config = self.adapter_config
                if "config" in kwargs:
                    config = AdapterConfig.model_validate(kwargs["config"])
                    
                model = EnterpriseTruthGPTModel(config)
                
                model_id = self.store.put(
                    model,
                    kind="enterprise_model",
                    meta={"type": "TruthGPT", "vocab_size": config.vocab_size}
                )
                
                info = model.get_model_info()
                
                return EnterpriseAdapterCreateResult(
                    model_id=model_id,
                    info=info
                ).model_dump()
            
            elif action == "info":
                model_id = input_data.get("model_id", "")
                if not model_id:
                    raise ValueError("model_id is required for action='info'")
                model = self.store.get(model_id)
                if not model:
                    raise ValueError(f"Model {model_id} not found.")
                info = model.get_model_info()
                return {"status": "success", "model_id": model_id, **info.model_dump()}
                
            elif action == "optimize":
                model_id = input_data.get("model_id", "")
                if not model_id:
                    raise ValueError("model_id is required for action='optimize'")
                model = self.store.get(model_id)
                if not model:
                    raise ValueError(f"Model {model_id} not found.")
                
                try:
                    model.optimize_for_inference()
                    return {"status": "success", "message": f"Model {model_id} optimized for inference."}
                except Exception as e:
                    return {"status": "error", "message": f"Optimization failed: {str(e)}"}
            
            elif action == "analyze":
                model_id = input_data.get("model_id", "")
                model = self.store.get(model_id)
                if not model:
                    raise ValueError(f"Model {model_id} not found.")
                
                # Check if model has advanced stats (like Holographic/Fractal)
                stats = {}
                if hasattr(model, "get_holographic_quantum_computing_stats"):
                    stats = model.get_holographic_quantum_computing_stats()
                elif hasattr(model, "get_stats"):
                    stats = model.get_stats()
                    
                return {"status": "success", "model_id": model_id, "analysis": stats}
                
            else:
                raise ValueError(f"Unknown truthgpt enterprise action: '{action}'. Use 'create', 'info', 'optimize', or 'analyze'.")
        except Exception as e:
            self.logger.error(f"Enterprise adapter error: {e}")
            return {"status": "error", "message": str(e)}

