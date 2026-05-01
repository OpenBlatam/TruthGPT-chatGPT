"""
Paper Factory - Factory for creating paper-based components
===========================================================

Integrates papers with the framework's factory system.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Any, Type

from ..factory_base import BaseFactory, FactoryError
from .paper_registry import PaperRegistry, get_paper_registry
from .paper_adapter import PaperAdapter, ModelEnhancer
from .paper_metadata import PaperModule

logger = logging.getLogger(__name__)

__all__ = ['PaperFactory', 'create_paper_component']


class PaperFactory(BaseFactory):
    """
    Factory for creating paper-based components.
    
    Integrates research papers with the framework's factory system,
    allowing papers to be used as components.
    """
    
    def __init__(self, registry: Optional[PaperRegistry] = None):
        """
        Initialize paper factory.
        
        Args:
            registry: Paper registry instance
        """
        super().__init__()
        self.registry = registry or get_paper_registry()
        self.adapter = PaperAdapter(self.registry)
        self._component_cache: Dict[str, Any] = {}
    
    def create(
        self,
        name: str,
        **kwargs
    ) -> Optional[Any]:
        """
        Create a paper-based component.
        
        Args:
            name: Paper ID or component name
            **kwargs: Component configuration
        
        Returns:
            Component instance or None
        """
        try:
            # Try to load as paper ID first
            paper_module = self.registry.load_paper(name)
            if paper_module and paper_module.is_available():
                # Create paper component
                config = paper_module.create_config(**kwargs)
                component = paper_module.create_module(config)
                logger.info(f"✅ Created paper component: {name}")
                return component
            
            # Try factory registry
            return super().create(name, **kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create paper component {name}: {e}")
            return None
    
    def register_paper(self, paper_id: str):
        """
        Register a paper for use in factory.
        
        Args:
            paper_id: Paper identifier
        """
        if self.adapter.load_paper(paper_id):
            logger.info(f"✅ Registered paper: {paper_id}")
        else:
            logger.warning(f"⚠️  Failed to register paper: {paper_id}")
    
    def list_available_components(self) -> list:
        """
        List all available paper components.
        
        Returns:
            List of component names (paper IDs)
        """
        papers = self.registry.list_papers()
        paper_ids = [p.paper_id for p in papers]
        
        # Add factory-registered components
        factory_components = super().list_components()
        
        return paper_ids + factory_components
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a component.
        
        Args:
            name: Component name or paper ID
        
        Returns:
            Component information dictionary
        """
        # Try as paper ID
        paper_module = self.registry.load_paper(name)
        if paper_module:
            return {
                'type': 'paper',
                'paper_id': name,
                'metadata': paper_module.metadata.to_dict(),
                'available': paper_module.is_available()
            }
        
        # Try factory
        if name in self._registry:
            return {
                'type': 'factory',
                'name': name,
                'registered': True
            }
        
        return None


def create_paper_component(
    paper_id: str,
    **config
) -> Optional[Any]:
    """
    Convenience function to create a paper component.
    
    Args:
        paper_id: Paper identifier
        **config: Component configuration
    
    Returns:
        Component instance or None
    
    Example:
        >>> component = create_paper_component("2503.00735v3", hidden_dim=512)
    """
    factory = PaperFactory()
    return factory.create(paper_id, **config)





