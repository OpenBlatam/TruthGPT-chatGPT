"""
Inference Engine Factory.

Provides a unified factory for creating inference engines with automatic
selection based on availability and requirements.
"""
import logging
from typing import Optional, Union, Dict, Any
from pathlib import Path

from optimization_core.core.factory_base import CallableFactory, FactoryError

from .base_engine import BaseInferenceEngine
from .vllm_engine import VLLMEngine, AsyncVLLMEngine, VLLM_AVAILABLE
from .tensorrt_llm_engine import TensorRTLLMEngine, TENSORRT_LLM_AVAILABLE

logger = logging.getLogger(__name__)


class EngineType:
    """Enum-like class for engine types."""
    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt_llm"
    ASYNC_VLLM = "async_vllm"
    AUTO = "auto"


class InferenceEngineFactory(CallableFactory):
    """
    Factory for creating inference engines.
    
    Uses the base factory pattern for consistency.
    """
    
    def __init__(self):
        super().__init__(default_type=EngineType.AUTO)
        self._register_engines()
        self._prefer_gpu = True
    
    def _register_engines(self):
        """Register all engine types."""
        self.register_creator(EngineType.VLLM, self._create_vllm)
        self.register_creator(EngineType.ASYNC_VLLM, self._create_async_vllm)
        self.register_creator(EngineType.TENSORRT_LLM, self._create_tensorrt)
    
    def _check_availability(self, component_type: str) -> bool:
        """Check if engine type is available."""
        if component_type == EngineType.VLLM:
            return VLLM_AVAILABLE
        elif component_type == EngineType.ASYNC_VLLM:
            return VLLM_AVAILABLE
        elif component_type == EngineType.TENSORRT_LLM:
            return TENSORRT_LLM_AVAILABLE
        return False
    
    def _create_vllm(self, model: Union[str, Path], **kwargs) -> VLLMEngine:
        """Create vLLM engine."""
        if not VLLM_AVAILABLE:
            raise FactoryError(
                "vLLM is not available. Install with: pip install vllm>=0.2.0"
            )
        return VLLMEngine(model=str(model), **kwargs)
    
    def _create_async_vllm(self, model: Union[str, Path], **kwargs) -> AsyncVLLMEngine:
        """Create async vLLM engine."""
        if not VLLM_AVAILABLE:
            raise FactoryError(
                "vLLM is not available. Install with: pip install vllm>=0.2.0"
            )
        return AsyncVLLMEngine(model=str(model), **kwargs)
    
    def _create_tensorrt(self, model: Union[str, Path], **kwargs) -> TensorRTLLMEngine:
        """Create TensorRT-LLM engine."""
        if not TENSORRT_LLM_AVAILABLE:
            raise FactoryError(
                "TensorRT-LLM is not available. Install with: "
                "pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com"
            )
        return TensorRTLLMEngine(model_path=str(model), **kwargs)
    
    def select_best(self, prefer_gpu: Optional[bool] = None) -> str:
        """
        Select best available engine.
        
        Args:
            prefer_gpu: Prefer GPU-accelerated engines (uses instance default if None)
        
        Returns:
            Selected engine type
        """
        prefer = prefer_gpu if prefer_gpu is not None else self._prefer_gpu
        
        if prefer and TENSORRT_LLM_AVAILABLE:
            return EngineType.TENSORRT_LLM
        
        if VLLM_AVAILABLE:
            return EngineType.VLLM
        
        raise FactoryError(
            "No inference engines available. Install at least one of: "
            "vLLM (pip install vllm) or TensorRT-LLM"
        )


_factory_instance = InferenceEngineFactory()


def create_inference_engine(
    model: Union[str, Path],
    engine_type: str = EngineType.AUTO,
    prefer_gpu: bool = True,
    **kwargs
) -> BaseInferenceEngine:
    """
    Factory function to create inference engine.
    
    Automatically selects the best available engine based on:
    - Engine availability
    - GPU availability
    - User preferences
    
    Args:
        model: Model name or path
        engine_type: Engine type (vllm, tensorrt_llm, async_vllm, auto)
        prefer_gpu: Prefer GPU-accelerated engines
        **kwargs: Engine-specific arguments
    
    Returns:
        Inference engine instance
    
    Raises:
        FactoryError: If engine type is invalid or unavailable
    """
    _factory_instance._prefer_gpu = prefer_gpu
    
    if engine_type == EngineType.AUTO:
        engine_type = _factory_instance.select_best(prefer_gpu=prefer_gpu)
        logger.info(f"Auto-selected engine: {engine_type}")
    
    return _factory_instance.create(engine_type, model=model, **kwargs)


def list_available_engines() -> Dict[str, bool]:
    """
    List all available inference engines.
    
    Returns:
        Dictionary mapping engine types to availability
    """
    return _factory_instance.list_available()


