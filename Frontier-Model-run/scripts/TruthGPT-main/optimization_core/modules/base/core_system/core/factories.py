"""
Factory classes for optimization_core.

Provides factory pattern implementations.
"""
import logging
from typing import Dict, Type, Optional, Any
from .interfaces import IComponent, IInferenceEngine, IDataProcessor

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Factory for creating components."""
    
    def __init__(self):
        """Initialize component factory."""
        self._registry: Dict[str, Type[IComponent]] = {}
    
    def register(
        self,
        name: str,
        component_class: Type[IComponent]
    ):
        """
        Register a component class.
        
        Args:
            name: Component name
            component_class: Component class
        """
        self._registry[name] = component_class
        logger.debug(f"Registered component: {name}")
    
    def create(
        self,
        name: str,
        **kwargs
    ) -> Optional[IComponent]:
        """
        Create a component instance.
        
        Args:
            name: Component name
            **kwargs: Component parameters
        
        Returns:
            Component instance or None
        """
        if name not in self._registry:
            logger.error(f"Component '{name}' not registered")
            return None
        
        try:
            component_class = self._registry[name]
            component = component_class(**kwargs)
            if component.initialize(**kwargs):
                return component
            return None
        except Exception as e:
            logger.error(f"Failed to create component '{name}': {e}", exc_info=True)
            return None
    
    def list_components(self) -> list:
        """
        List all registered components.
        
        Returns:
            List of component names
        """
        return list(self._registry.keys())


class InferenceEngineFactory(ComponentFactory):
    """Factory for inference engines."""
    
    def create_engine(
        self,
        engine_type: str,
        model: str,
        **kwargs
    ) -> Optional[IInferenceEngine]:
        """
        Create an inference engine.
        
        Args:
            engine_type: Engine type
            model: Model name
            **kwargs: Engine parameters
        
        Returns:
            Inference engine instance or None
        """
        return self.create(engine_type, model_name=model, **kwargs)


class DataProcessorFactory(ComponentFactory):
    """Factory for data processors."""
    
    def create_processor(
        self,
        processor_type: str,
        **kwargs
    ) -> Optional[IDataProcessor]:
        """
        Create a data processor.
        
        Args:
            processor_type: Processor type
            **kwargs: Processor parameters
        
        Returns:
            Data processor instance or None
        """
        return self.create(processor_type, **kwargs)


# Global factories
_global_component_factory = ComponentFactory()
_global_inference_factory = InferenceEngineFactory()
_global_processor_factory = DataProcessorFactory()


def get_component_factory() -> ComponentFactory:
    """Get global component factory."""
    return _global_component_factory


def get_inference_factory() -> InferenceEngineFactory:
    """Get global inference factory."""
    return _global_inference_factory


def get_processor_factory() -> DataProcessorFactory:
    """Get global processor factory."""
    return _global_processor_factory













