"""
Base inference engine abstract class.

Provides a common interface and shared functionality for all inference engines.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from ..agents.models import InferenceResult

logger = logging.getLogger(__name__)


class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    
    Provides common interface and shared functionality for all inference engines.
    """
    
    def __init__(
        self,
        model: Union[str, Path],
        **kwargs
    ):
        """
        Initialize base inference engine.
        
        Args:
            model: Model name or path
            **kwargs: Additional engine-specific arguments
        """
        self.model_path = Path(model) if isinstance(model, (str, Path)) else model
        self._initialized = False
    
    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> Union[InferenceResult, List[InferenceResult]]:
        """
        Generate text from prompts and return structured results.
        """
        pass
    
    @abstractmethod
    def _initialize_engine(self, **kwargs) -> Any:
        """
        Initialize the underlying engine.
        
        Args:
            **kwargs: Engine-specific initialization arguments
        
        Returns:
            Initialized engine object
        """
        pass
    
    def __call__(
        self,
        prompts: Union[str, List[str]],
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Convenience method for generation.
        
        Args:
            prompts: Single prompt or list of prompts
            **kwargs: Generation parameters
        
        Returns:
            Generated text(s)
        """
        return self.generate(prompts, **kwargs)
    
    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized
    
    def _set_initialized(self, value: bool = True):
        """Set initialization status."""
        self._initialized = value
    
    def __repr__(self) -> str:
        """String representation of engine."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_path}, "
            f"initialized={self._initialized})"
        )


class GenerationConfig:
    """
    Configuration class for text generation parameters.
    
    Provides a structured way to pass generation parameters.
    """
    
    def __init__(
        self,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        stop: Optional[Union[str, List[str]]] = None,
        repetition_penalty: float = 1.0,
        **kwargs
    ):
        """
        Initialize generation config.
        
        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            repetition_penalty: Repetition penalty
            **kwargs: Additional parameters
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop
        self.repetition_penalty = repetition_penalty
        self.extra_params = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop,
            "repetition_penalty": self.repetition_penalty,
            **self.extra_params
        }
    
    def __repr__(self) -> str:
        """String representation."""
        params = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"GenerationConfig({params})"













