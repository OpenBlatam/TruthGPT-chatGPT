"""
Core interfaces for optimization_core.

Defines abstract interfaces for all components.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop_sequences: Optional[List[str]] = None


class IComponent(ABC):
    """Base interface for all components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get component name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Get component version."""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the component.
        
        Args:
            **kwargs: Initialization parameters
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.
        
        Returns:
            Status dictionary
        """
        pass


class IInferenceEngine(IComponent):
    """Interface for inference engines."""
    
    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts.
        
        Args:
            prompts: Input prompt(s)
            config: Generation configuration
            **kwargs: Additional parameters
        
        Returns:
            Generated text(s)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        """
        pass


class IDataProcessor(IComponent):
    """Interface for data processors."""
    
    @abstractmethod
    def process(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Process data.
        
        Args:
            data: Data to process
            **kwargs: Processing parameters
        
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate data.
        
        Args:
            data: Data to validate
        
        Returns:
            True if valid
        """
        pass

