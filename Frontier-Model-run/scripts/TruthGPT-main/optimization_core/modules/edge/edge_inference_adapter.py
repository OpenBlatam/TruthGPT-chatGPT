import logging
import torch
import time
from typing import Dict, Any, Optional, List

from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimization_core.adapters.base import BaseDynamicAdapter
from optimization_core.modules.memory.advanced_memory_manager import create_advanced_memory_manager
from optimization_core.modules.attention.attn_autotune import choose_best_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------

class EdgeInferenceConfig(BaseModel):
    """Configuration for Edge Inference."""
    model_name: str = "gpt2"
    device: str = "auto"
    use_amp: bool = True
    max_new_tokens: int = 64


class GenerationConfig(BaseModel):
    """Structured generation parameters."""
    max_new_tokens: Optional[int] = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1


class EdgeGenerateResult(BaseModel):
    """Typed result from an edge generate action."""
    status: str = "success"
    text: str
    edge_id: str
    elapsed_ms: float


class EdgeLoadResult(BaseModel):
    """Typed result from an edge load action."""
    status: str = "success"
    edge_id: str
    model_name: str
    dtype: str
    backend: str


class EdgeStateResult(BaseModel):
    """Typed result from an edge state query."""
    status: str = "success"
    edge_id: str
    model_name: str
    backend: str
    device: str
    dtype: str


class EdgeListResult(BaseModel):
    """Typed result from an edge list action."""
    status: str = "success"
    edges: List[str]


# ---------------------------------------------------------------------------
# Core Adapter
# ---------------------------------------------------------------------------

class EdgeInferenceAdapter(BaseDynamicAdapter):
    """
    Adapter for Edge Inference operations.
    """

    name: str = "edge_inference_adapter"
    description: str = (
        "Adapter for edge inference. Input JSON: "
        "{'action': 'load'|'generate'|'unload', 'model_name': 'gpt2', 'prompt': 'hello', 'kwargs': {}}"
    )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        action = input_data.get("action")
        kwargs = input_data.get("kwargs", {})

        if action == "load":
            config = EdgeInferenceConfig(**input_data) if not kwargs else EdgeInferenceConfig(**{**input_data, **kwargs})
            model_name = config.model_name
            
            # Load and initialize model
            mm = create_advanced_memory_manager()
            dtype = mm.select_dtype_adaptive()
            
            if config.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(config.device)
                
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model.to(device)
            model.eval()
            
            h = getattr(model.config, "n_head", 12)
            d = getattr(model.config, "n_embd", 768) // max(1, h)
            backend = choose_best_backend(h=h, t=128, d=d, dtype=dtype)
            
            # Storing the state bundle in ObjectStore
            edge_bundle = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "dtype": dtype,
                "config": config,
                "backend": backend,
                "use_sdpa": backend == "sdpa"
            }
            
            edge_id = self.store.put(
                edge_bundle,
                kind="edge_inference",
                meta={"model_name": model_name, "backend": backend}
            )
            
            return EdgeLoadResult(
                edge_id=edge_id,
                model_name=model_name,
                dtype=str(dtype),
                backend=backend,
            ).model_dump()

        elif action == "generate":
            edge_id = input_data.get("edge_id", "")
            prompt = input_data.get("prompt", "")
            
            bundle = self.store.get(edge_id)
            if not bundle:
                raise ValueError(f"Edge handle {edge_id} not found.")

            model = bundle["model"]
            tokenizer = bundle["tokenizer"]
            device = bundle["device"]
            dtype = bundle["dtype"]
            base_config = bundle["config"]
            
            gen_params = GenerationConfig(**kwargs)
            max_new_tokens = gen_params.max_new_tokens or base_config.max_new_tokens

            logger.info("Edge inference generating with edge_id: %s", edge_id)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            autocast_dtype = dtype if device.type == "cuda" else None
            
            start_time = time.monotonic()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=autocast_dtype):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=gen_params.temperature,
                        top_p=gen_params.top_p,
                        top_k=gen_params.top_k,
                        repetition_penalty=gen_params.repetition_penalty,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
            
            elapsed = (time.monotonic() - start_time) * 1000
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logger.info("Edge generation completed in %.2fms", elapsed)
            
            return EdgeGenerateResult(
                text=text,
                edge_id=edge_id,
                elapsed_ms=round(elapsed, 2)
            ).model_dump()

        elif action == "get_state":
            edge_id = input_data.get("edge_id", "")
            bundle = self.store.get(edge_id)
            if not bundle:
                raise ValueError(f"Edge handle {edge_id} not found.")
            
            return EdgeStateResult(
                edge_id=edge_id,
                model_name=bundle["config"].model_name,
                backend=bundle["backend"],
                device=str(bundle["device"]),
                dtype=str(bundle["dtype"])
            ).model_dump()

        elif action == "list":
            ids = self.store.list_ids(kind="edge_inference")
            return EdgeListResult(edges=ids).model_dump()

        elif action == "unload":
            edge_id = input_data.get("edge_id", "")
            logger.info("Unloading edge model: %s", edge_id)
            
            if self.store.delete(edge_id):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"status": "success", "message": f"Model {edge_id} unloaded", "edge_id": edge_id}
            else:
                return {"status": "error", "message": f"Model {edge_id} not found", "edge_id": edge_id}

        else:
            raise ValueError(f"Unknown edge inference action: '{action}'. Use 'load', 'generate', 'get_state', 'list', or 'unload'.")

