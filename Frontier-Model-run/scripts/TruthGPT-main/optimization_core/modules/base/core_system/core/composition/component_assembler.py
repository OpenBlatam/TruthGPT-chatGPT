"""
Component assembler for composing complex systems from simple components.
"""
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..service_registry import ServiceRegistry, ServiceContainer

logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """Configuration for a component."""
    name: str
    component_type: str
    implementation: str
    config: Dict[str, Any]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ComponentAssembler:
    """
    Assembles components into complete systems.
    Implements composition pattern for maximum modularity.
    """
    
    def __init__(self, container: Optional[ServiceContainer] = None):
        """
        Initialize component assembler.
        
        Args:
            container: Optional service container
        """
        self.container = container or ServiceContainer()
        self.components: Dict[str, ComponentConfig] = {}
    
    def register_component(self, config: ComponentConfig) -> None:
        """
        Register a component configuration.
        
        Args:
            config: Component configuration
        """
        self.components[config.name] = config
        logger.debug(f"Component registered: {config.name}")
    
    def assemble(self, component_names: List[str]) -> Dict[str, Any]:
        """
        Assemble components into a complete system.
        
        Args:
            component_names: List of component names to assemble
        
        Returns:
            Dictionary of assembled components
        """
        assembled = {}
        processed = set()
        
        def resolve_dependencies(name: str):
            """Recursively resolve dependencies."""
            if name in processed:
                return
            
            if name not in self.components:
                raise ValueError(f"Component '{name}' not found")
            
            config = self.components[name]
            
            # Resolve dependencies first
            for dep in config.dependencies:
                resolve_dependencies(dep)
            
            # Create component
            component = self._create_component(config)
            assembled[name] = component
            processed.add(name)
        
        # Resolve all components
        for name in component_names:
            resolve_dependencies(name)
        
        logger.info(f"Assembled {len(assembled)} components")
        return assembled
    
    def _create_component(self, config: ComponentConfig) -> Any:
        """
        Create a component instance.
        
        Args:
            config: Component configuration
        
        Returns:
            Component instance
        """
        # Map component types to factories
        factory_map = {
            "model": self._create_model,
            "optimizer": self._create_optimizer,
            "scheduler": self._create_scheduler,
            "data_loader": self._create_data_loader,
            "evaluator": self._create_evaluator,
        }
        
        factory = factory_map.get(config.component_type)
        if not factory:
            raise ValueError(f"Unknown component type: {config.component_type}")
        
        return factory(config)
    
    def _create_model(self, config: ComponentConfig) -> Any:
        """Create model component."""
        from ..adapters.model_adapter import HuggingFaceModelAdapter
        
        adapter = HuggingFaceModelAdapter()
        return adapter.load_model(config.implementation, **config.config)
    
    def _create_optimizer(self, config: ComponentConfig) -> Any:
        """Create optimizer component."""
        from ..adapters.optimizer_adapter import PyTorchOptimizerAdapter
        
        # Get model from container
        model = self.container.get("model")
        
        adapter = PyTorchOptimizerAdapter()
        return adapter.create_optimizer(
            model.parameters(),
            optimizer_type=config.implementation,
            **config.config
        )
    
    def _create_scheduler(self, config: ComponentConfig) -> Any:
        """Create scheduler component."""
        from transformers import get_cosine_schedule_with_warmup
        
        optimizer = self.container.get("optimizer")
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.config.get("num_warmup_steps", 0),
            num_training_steps=config.config.get("num_training_steps", 1000),
        )
    
    def _create_data_loader(self, config: ComponentConfig) -> Any:
        """Create data loader component."""
        from ..adapters.data_adapter import HuggingFaceDataAdapter
        from ...data import DataLoaderFactory
        
        adapter = HuggingFaceDataAdapter()
        train_texts, val_texts = adapter.load_data(
            config.implementation,
            **config.config
        )
        
        tokenizer = self.container.get("tokenizer")
        return DataLoaderFactory.create_train_loader(
            texts=train_texts,
            tokenizer=tokenizer,
            **config.config.get("loader_config", {})
        )
    
    def _create_evaluator(self, config: ComponentConfig) -> Any:
        """Create evaluator component."""
        from ...training.evaluator import Evaluator
        
        return Evaluator(**config.config)



