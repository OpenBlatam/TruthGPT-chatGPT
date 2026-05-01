"""
HuggingFace Transformers Integration
====================================

Optimization wrappers for HF Transformers models.
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

# Try to import register_model, handle failure if not available yet (in older structure)
try:
    from . import register_model
except ImportError:
    # Minimal stub if registry import loops
    def register_model(name):
        def decorator(cls):
            return cls
        return decorator

@register_model("hf-transformers")
class HFTransformersModel:
    """
    Wrapper for HuggingFace Transformers models with optimization support.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[nn.Module] = None
        
        # Auto-load if config provides model name
        if self.config.get("model", {}).get("name_or_path"):
            self.load(self.config)

    def _resolve_dtype(self, mixed_precision: str) -> Optional[torch.dtype]:
        if mixed_precision == "bf16":
            return torch.bfloat16
        if mixed_precision == "fp16":
            return torch.float16
        return None

    def load(self, cfg: Dict[str, Any]) -> None:
        """
        Load model from configuration.
        
        Args:
            cfg: Configuration dictionary
        """
        try:
            name = cfg["model"]["name_or_path"]
            mp = cfg.get("training", {}).get("mixed_precision", "no")
            dtype = self._resolve_dtype(mp)
            
            logger.info(f"Loading HF model: {name} (dtype={dtype})")

            self.tokenizer = AutoTokenizer.from_pretrained(name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
            
            # Smart device placement
            self.to_device()
            self.model.eval()
            
            # Optimization: Disable cache for training (if applicable) but enable for inference by default
            # Actually for inference we want cache. The previous code set it to True.
            if hasattr(self.model, "config"):
                self.model.config.use_cache = True
                
            logger.info("HF model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")
            raise

    def to_device(self) -> None:
        """Move model to best available device."""
        if not self.model:
            return
            
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        self.model.to(device)

    @torch.inference_mode()
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Perform inference.
        
        Args:
            inputs: Dictionary with 'text' and generation params.
            
        Returns:
            Dictionary with 'text' output.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        try:
            device = next(self.model.parameters()).device
            text_input = inputs.get("text", "")
            
            toks = self.tokenizer(text_input, return_tensors="pt").to(device)
            
            gen_kwargs = {
                "max_new_tokens": int(inputs.get("max_new_tokens", 64)),
                "do_sample": inputs.get("do_sample", True),
                "temperature": float(inputs.get("temperature", 0.8)),
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if "top_p" in inputs:
                gen_kwargs["top_p"] = float(inputs["top_p"])
            if "top_k" in inputs:
                gen_kwargs["top_k"] = int(inputs["top_k"])

            output_ids = self.model.generate(**toks, **gen_kwargs)
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            return {"text": generated_text}
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

def create_hf_transformers_model(config: Dict[str, Any]) -> HFTransformersModel:
    """Factory function."""
    return HFTransformersModel(config)

