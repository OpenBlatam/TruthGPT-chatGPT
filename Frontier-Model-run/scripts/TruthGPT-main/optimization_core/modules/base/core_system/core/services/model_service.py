"""
Model service for managing model lifecycle.
"""
import logging
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

import torch

from .base_service import BaseService
from ..event_system import EventType
from ...models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ModelLoadConfig(BaseModel):
    """Configuration for loading models into memory."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    torch_dtype: Optional[Any] = Field(default=None, description="PyTorch data type for model weights")
    device_map: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Device map for multi-GPU distribution")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing to save VRAM")
    lora: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for LoRA adapters")

class ModelSaveConfig(BaseModel):
    """Configuration for saving models securely to disk."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    tokenizer: Optional[Any] = Field(default=None, description="Tokenizer object associated with the model")
    safe_serialization: bool = Field(default=True, description="Enforce safetensors serialization over pickle")

class ModelService(BaseService):
    """
    Service for model management operations.
    Supports strict Pydantic validation for IO configurations.
    """
    
    def __init__(self, **kwargs):
        """Initialize model service."""
        super().__init__(name="ModelService", **kwargs)
        self.model_manager: Optional[ModelManager] = None
    
    def _do_initialize(self) -> None:
        """Initialize model manager."""
        self.model_manager = ModelManager()
    
    def load_model(
        self,
        model_name: str,
        config: Optional[Union[Dict[str, Any], ModelLoadConfig]] = None
    ) -> torch.nn.Module:
        """
        Load a model.
        
        Args:
            model_name: Model name or path
            config: Optional model configuration
        
        Returns:
            Loaded model
        """
        if config is None:
            config = ModelLoadConfig()
        elif isinstance(config, dict):
            config = ModelLoadConfig(**config)
        
        try:
            model = self.model_manager.load_model(
                model_name=model_name,
                torch_dtype=config.torch_dtype,
                device_map=config.device_map,
                gradient_checkpointing=config.gradient_checkpointing,
                lora_config=config.lora,
            )
            
            self.emit(EventType.MODEL_LOADED, {
                "model_name": model_name,
                "config": config.model_dump(),
            })
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            self.emit(EventType.ERROR_OCCURRED, {
                "error": str(e),
                "operation": "load_model",
            })
            raise
    
    def save_model(
        self,
        model: torch.nn.Module,
        path: str,
        config: Optional[Union[Dict[str, Any], ModelSaveConfig]] = None
    ) -> None:
        """
        Save a model.
        
        Args:
            model: Model to save
            path: Save path
            config: Optional save configuration
        """
        if config is None:
            config = ModelSaveConfig()
        elif isinstance(config, dict):
            config = ModelSaveConfig(**config)
        
        try:
            self.model_manager.save_model(
                model=model,
                path=path,
                tokenizer=config.tokenizer,
                safe_serialization=config.safe_serialization,
            )
            
            self.emit(EventType.MODEL_SAVED, {
                "path": path,
                "config": config.model_dump(),
            })
            
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {e}", exc_info=True)
            self.emit(EventType.ERROR_OCCURRED, {
                "error": str(e),
                "operation": "save_model",
            })
            raise
    
    def optimize_model(
        self,
        model: torch.nn.Module,
        optimizations: list[str]
    ) -> torch.nn.Module:
        """
        Optimize a model.
        
        Args:
            model: Model to optimize
            optimizations: List of optimization names
        
        Returns:
            Optimized model
        """
        from ...optimization.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        optimized = optimizer.optimize_model(model, optimizations)
        
        return optimized



