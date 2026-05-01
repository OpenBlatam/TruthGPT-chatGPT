"""
Base classes for optimization_core.

Provides base implementations for common components.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from .interfaces import IComponent, IInferenceEngine, IDataProcessor, GenerationConfig

logger = logging.getLogger(__name__)


class BaseComponent(IComponent):
    """Base implementation for all components."""
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        **kwargs
    ):
        """
        Initialize base component.
        
        Args:
            name: Component name
            version: Component version
            **kwargs: Additional parameters
        """
        self._name = name
        self._version = version
        self._initialized = False
        self._metadata: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """Get component name."""
        return self._name
    
    @property
    def version(self) -> str:
        """Get component version."""
        return self._version
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the component.
        
        Args:
            **kwargs: Initialization parameters
        
        Returns:
            True if successful
        """
        try:
            self._initialize_impl(**kwargs)
            self._initialized = True
            logger.info(f"Component '{self.name}' initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize component '{self.name}': {e}", exc_info=True)
            return False
    
    def _initialize_impl(self, **kwargs):
        """Implementation-specific initialization."""
        pass
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self._cleanup_impl()
            self._initialized = False
            logger.info(f"Component '{self.name}' cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up component '{self.name}': {e}", exc_info=True)
    
    def _cleanup_impl(self):
        """Implementation-specific cleanup."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.
        
        Returns:
            Status dictionary
        """
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "metadata": self._metadata,
        }
    
    def set_metadata(self, key: str, value: Any):
        """
        Set metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata.
        
        Args:
            key: Metadata key
            default: Default value
        
        Returns:
            Metadata value
        """
        return self._metadata.get(key, default)


class BaseInferenceEngine(BaseComponent, IInferenceEngine):
    """Base implementation for inference engines."""
    
    def __init__(
        self,
        model_name: str,
        name: Optional[str] = None,
        version: str = "1.0.0",
        **kwargs
    ):
        """
        Initialize base inference engine.
        
        Args:
            model_name: Model name
            name: Component name (defaults to model name)
            version: Component version
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name or f"inference_engine_{model_name}",
            version=version,
            **kwargs
        )
        self.model_name = model_name
    
    def generate(
        self,
        prompts,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ):
        """
        Generate text from prompts.
        
        Args:
            prompts: Input prompt(s)
            config: Generation configuration
            **kwargs: Additional parameters
        
        Returns:
            Generated text(s)
        """
        if not self._initialized:
            raise RuntimeError(f"Component '{self.name}' not initialized")
        
        if config is None:
            config = GenerationConfig()
        
        return self._generate_impl(prompts, config, **kwargs)
    
    @abstractmethod
    def _generate_impl(
        self,
        prompts,
        config: GenerationConfig,
        **kwargs
    ):
        """Implementation-specific generation."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "component_name": self.name,
            "version": self.version,
        }


class BaseDataProcessor(BaseComponent, IDataProcessor):
    """Base implementation for data processors."""
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        **kwargs
    ):
        """
        Initialize base data processor.
        
        Args:
            name: Component name
            version: Component version
            **kwargs: Additional parameters
        """
        super().__init__(name=name, version=version, **kwargs)
    
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
        if not self._initialized:
            raise RuntimeError(f"Component '{self.name}' not initialized")
        
        if not self.validate(data):
            raise ValueError(f"Invalid data for processor '{self.name}'")
        
        return self._process_impl(data, **kwargs)
    
    @abstractmethod
    def _process_impl(self, data: Any, **kwargs) -> Any:
        """Implementation-specific processing."""
        pass
    
    def validate(self, data: Any) -> bool:
        """
        Validate data.
        
        Args:
            data: Data to validate
        
        Returns:
            True if valid
        """
        return self._validate_impl(data)
    
    def _validate_impl(self, data: Any) -> bool:
        """
        Implementation-specific validation.
        
        Args:
            data: Data to validate
        
        Returns:
            True if valid
        """
        return data is not None













