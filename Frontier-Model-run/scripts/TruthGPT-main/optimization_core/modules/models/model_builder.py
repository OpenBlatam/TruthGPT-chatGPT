"""
Model Builder Module
====================

Builder pattern for constructing models with various configurations.
"""
from typing import Dict, Any, Optional, Union
import torch
import logging

from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class ModelBuilder:
    """
    Builder pattern for constructing models with various configurations.
    """
    
    def __init__(self) -> None:
        self._manager = ModelManager()
        self._model_name: Optional[str] = None
        self._torch_dtype: Optional[torch.dtype] = None
        self._device_map: Optional[str] = None
        self._gradient_checkpointing: bool = True
        self._lora_config: Optional[Dict[str, Any]] = None
        self._multi_gpu: bool = False
        self._torch_compile: bool = False
        self._compile_mode: str = "default"
        self._device_settings: Dict[str, Any] = {}
    
    def with_model_name(self, name: str) -> "ModelBuilder":
        """Set model name or path."""
        self._model_name = name
        return self
    
    def with_dtype(self, dtype: torch.dtype) -> "ModelBuilder":
        """Set model dtype."""
        self._torch_dtype = dtype
        return self
    
    def with_device_map(self, device_map: str) -> "ModelBuilder":
        """Set device mapping strategy."""
        self._device_map = device_map
        return self
    
    def with_gradient_checkpointing(self, enabled: bool = True) -> "ModelBuilder":
        """Enable/disable gradient checkpointing."""
        self._gradient_checkpointing = enabled
        return self
    
    def with_lora(
        self,
        enabled: bool = True,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05
    ) -> "ModelBuilder":
        """Configure LoRA."""
        self._lora_config = {
            "enabled": enabled,
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
        }
        return self
    
    def with_multi_gpu(self, enabled: bool = True) -> "ModelBuilder":
        """Enable multi-GPU training."""
        self._multi_gpu = enabled
        return self
    
    def with_torch_compile(self, enabled: bool = True, mode: str = "default") -> "ModelBuilder":
        """Enable torch.compile."""
        self._torch_compile = enabled
        self._compile_mode = mode
        return self
    
    def with_device_settings(self, allow_tf32: bool = True, matmul_precision: str = "high") -> "ModelBuilder":
        """Configure device settings."""
        self._device_settings = {
            "allow_tf32": allow_tf32,
            "matmul_precision": matmul_precision,
        }
        return self
    
    def build(self) -> torch.nn.Module:
        """Build the configured model."""
        if not self._model_name:
            raise ValueError("model_name must be set")
        
        # Configure device settings first
        if self._device_settings:
            self._manager.configure_device_settings(**self._device_settings)
        
        # Load model
        model = self._manager.load_model(
            model_name=self._model_name,
            torch_dtype=self._torch_dtype,
            device_map=self._device_map,
            gradient_checkpointing=self._gradient_checkpointing,
            lora_config=self._lora_config,
        )
        
        # Apply optimizations
        if self._multi_gpu:
            model = self._manager.enable_multi_gpu(model)
        
        if self._torch_compile:
            model = self._manager.enable_torch_compile(model, self._compile_mode)
        
        return model

