"""
Paper Adapter - Adapter for integrating papers with small models
=================================================================

Allows small models to leverage research paper techniques, making them
more capable when integrated into the framework.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .paper_registry import PaperRegistry, get_paper_registry
from .paper_metadata import PaperModule
from .paper_component import PaperComponent, PaperComponentMetrics
from .paper_validator import (
    PaperValidator,
    validate_paper_id,
    validate_paper_config
)

logger = logging.getLogger(__name__)

__all__ = ['PaperAdapter', 'ModelEnhancer', 'EnhancementConfig']


@dataclass
class EnhancementConfig:
    """Configuration for model enhancement."""
    paper_ids: List[str]
    apply_sequentially: bool = True
    merge_outputs: bool = False
    preserve_original: bool = True


class PaperAdapter:
    """
    Adapter for applying paper techniques to models.
    
    This adapter allows small models to leverage research paper techniques,
    effectively making them more capable when integrated into the framework.
    
    Follows architecture specifications:
    - Provides metrics and observability
    - Implements lazy loading for performance
    - Supports caching for efficiency
    - Handles errors gracefully
    """
    
    def __init__(
        self,
        registry: Optional[PaperRegistry] = None,
        enable_cache: bool = True,
        max_cache_size: int = 50
    ):
        """
        Initialize paper adapter.
        
        Args:
            registry: Paper registry instance (creates default if None)
            enable_cache: Enable component caching
            max_cache_size: Maximum cache size
        """
        self.registry = registry or get_paper_registry()
        self._loaded_papers: Dict[str, PaperModule] = {}
        self._components: Dict[str, PaperComponent] = {}
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self._metrics: Dict[str, Any] = {
            'total_applications': 0,
            'successful_applications': 0,
            'failed_applications': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def load_paper(self, paper_id: str, create_component: bool = True) -> bool:
        """
        Load a paper for use (lazy loading).
        
        Args:
            paper_id: Paper identifier
            create_component: Create PaperComponent instance
        
        Returns:
            True if loaded successfully
        """
        # Check cache first
        if paper_id in self._loaded_papers:
            self._metrics['cache_hits'] += 1
            return True
        
        self._metrics['cache_misses'] += 1
        
        # Load from registry
        paper_module = self.registry.load_paper(paper_id)
        if paper_module and paper_module.is_available():
            self._loaded_papers[paper_id] = paper_module
            
            # Create component if requested
            if create_component and self.enable_cache:
                component = PaperComponent(paper_module)
                component.initialize()
                self._components[paper_id] = component
                self._evict_cache_if_needed()
            
            logger.info(f"✅ Loaded paper: {paper_id}")
            return True
        
        logger.warning(f"⚠️  Failed to load paper: {paper_id}")
        return False
    
    def _evict_cache_if_needed(self):
        """Evict components from cache if needed (LRU)."""
        if len(self._components) > self.max_cache_size:
            # Remove oldest (first) component
            oldest_id = next(iter(self._components))
            component = self._components.pop(oldest_id)
            component.cleanup()
            logger.debug(f"Evicted component from cache: {oldest_id}")
    
    def get_component(self, paper_id: str) -> Optional[PaperComponent]:
        """
        Get paper component (lazy creation).
        
        Args:
            paper_id: Paper identifier
        
        Returns:
            PaperComponent or None
        """
        if paper_id in self._components:
            return self._components[paper_id]
        
        if paper_id in self._loaded_papers:
            component = PaperComponent(self._loaded_papers[paper_id])
            component.initialize()
            self._components[paper_id] = component
            self._evict_cache_if_needed()
            return component
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get adapter metrics.
        
        Returns:
            Metrics dictionary
        """
        component_metrics = {
            paper_id: comp.get_metrics()
            for paper_id, comp in self._components.items()
        }
        
        return {
            **self._metrics,
            'loaded_papers': len(self._loaded_papers),
            'cached_components': len(self._components),
            'components': component_metrics,
        }
    
    def apply_paper(
        self,
        model: nn.Module,
        paper_id: str,
        config: Optional[Dict[str, Any]] = None,
        use_component: bool = True,
        validate: bool = True
    ) -> nn.Module:
        """
        Apply a paper technique to a model.
        
        Args:
            model: Model to enhance
            paper_id: Paper identifier
            config: Optional configuration for the paper
            use_component: Use PaperComponent for caching
            validate: Enable validation (recommended)
        
        Returns:
            Enhanced model
        
        Raises:
            RuntimeError: If PyTorch is not available
            ValueError: If validation fails or paper is not available
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install PyTorch to use paper adapters.")
        
        # Validate inputs
        if validate:
            # Validate model
            is_valid, error = PaperValidator.validate_model_for_enhancement(model)
            if not is_valid:
                raise ValueError(f"Invalid model: {error}")
            
            # Validate paper_id
            is_valid, error = validate_paper_id(paper_id)
            if not is_valid:
                raise ValueError(f"Invalid paper ID: {error}")
            
            # Validate config if provided
            if config:
                is_valid, error = validate_paper_config(config)
                if not is_valid:
                    raise ValueError(f"Invalid config: {error}")
        
        self._metrics['total_applications'] += 1
        start_time = time.time()
        
        try:
            # Use component if available and requested
            if use_component and self.enable_cache:
                component = self.get_component(paper_id)
                if component:
                    paper_config = component.create_config(**(config or {}))
                    paper_module_instance = component.create_module(paper_config)
                    enhanced_model = self._wrap_model(model, paper_module_instance, paper_id)
                    self._metrics['successful_applications'] += 1
                    logger.info(f"✅ Applied paper {paper_id} to model (via component)")
                    return enhanced_model
            
            # Fallback to direct loading
            if paper_id not in self._loaded_papers:
                if not self.load_paper(paper_id, create_component=False):
                    available = self.list_available_papers()
                    raise ValueError(
                        f"Paper {paper_id} not available. "
                        f"Available papers: {available[:10] if available else 'None'}"
                    )
            
            paper_module = self._loaded_papers[paper_id]
            
            if not paper_module.is_available():
                raise ValueError(f"Paper {paper_id} is loaded but not available (error: {paper_module.error})")
            
            # Create paper config
            paper_config = paper_module.create_config(**(config or {}))
            
            # Create paper module instance
            paper_component = paper_module.create_module(paper_config)
            
            # Wrap model with paper component
            enhanced_model = self._wrap_model(model, paper_component, paper_id)
            
            self._metrics['successful_applications'] += 1
            apply_time = time.time() - start_time
            logger.info(f"✅ Applied paper {paper_id} to model in {apply_time:.3f}s")
            return enhanced_model
            
        except Exception as e:
            self._metrics['failed_applications'] += 1
            logger.error(f"Error applying paper {paper_id}: {e}", exc_info=True)
            raise ValueError(f"Failed to apply paper {paper_id}: {e}") from e
    
    def _wrap_model(
        self,
        model: nn.Module,
        paper_component: nn.Module,
        paper_id: str
    ) -> nn.Module:
        """
        Wrap model with paper component.
        
        Args:
            model: Original model
            paper_component: Paper component to apply
            paper_id: Paper identifier
        
        Returns:
            Wrapped model
        """
        class EnhancedModel(nn.Module):
            """Model enhanced with paper technique."""
            
            def __init__(self, base_model: nn.Module, paper_component: nn.Module, paper_id: str):
                super().__init__()
                self.base_model = base_model
                self.paper_component = paper_component
                self.paper_id = paper_id
            
            def forward(self, *args, **kwargs):
                # Apply base model
                base_output = self.base_model(*args, **kwargs)
                
                # Apply paper component if it's a tensor
                if isinstance(base_output, torch.Tensor):
                    paper_output, _ = self.paper_component(base_output, **kwargs)
                    return paper_output
                elif isinstance(base_output, (tuple, list)):
                    # Handle tuple/list outputs
                    if len(base_output) > 0 and isinstance(base_output[0], torch.Tensor):
                        paper_output, _ = self.paper_component(base_output[0], **kwargs)
                        return (paper_output,) + base_output[1:]
                return base_output
        
        return EnhancedModel(model, paper_component, paper_id)
    
    def list_available_papers(self, category: Optional[str] = None) -> List[str]:
        """
        List available papers.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of paper IDs
        """
        papers = self.registry.list_papers(category=category)
        return [p.paper_id for p in papers]
    
    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a paper.
        
        Args:
            paper_id: Paper identifier
        
        Returns:
            Paper information dictionary or None
        """
        paper_module = self.registry.load_paper(paper_id)
        if paper_module:
            return paper_module.metadata.to_dict()
        return None


class ModelEnhancer:
    """
    High-level interface for enhancing small models with research papers.
    
    This is the main entry point for making small models more capable
    by integrating research paper techniques.
    """
    
    def __init__(self, registry: Optional[PaperRegistry] = None):
        """
        Initialize model enhancer.
        
        Args:
            registry: Paper registry instance
        """
        self.adapter = PaperAdapter(registry)
    
    def enhance_model(
        self,
        model: nn.Module,
        enhancement_config: EnhancementConfig,
        validate: bool = True
    ) -> nn.Module:
        """
        Enhance a model with multiple papers.
        
        Args:
            model: Model to enhance
            enhancement_config: Enhancement configuration
            validate: Enable validation (recommended)
        
        Returns:
            Enhanced model
        
        Raises:
            RuntimeError: If PyTorch is not available
            ValueError: If validation fails
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Validate enhancement config
        if validate:
            is_valid, error = PaperValidator.validate_enhancement_config(
                enhancement_config.paper_ids
            )
            if not is_valid:
                raise ValueError(f"Invalid enhancement config: {error}")
        
        enhanced_model = model
        applied_papers = []
        failed_papers = []
        
        for paper_id in enhancement_config.paper_ids:
            try:
                enhanced_model = self.adapter.apply_paper(
                    enhanced_model,
                    paper_id,
                    validate=validate
                )
                applied_papers.append(paper_id)
                logger.info(f"✅ Enhanced model with paper: {paper_id}")
            except Exception as e:
                failed_papers.append((paper_id, str(e)))
                logger.error(f"Failed to apply paper {paper_id}: {e}")
                if not enhancement_config.preserve_original:
                    raise
        
        # Log summary
        logger.info(
            f"Enhancement complete: {len(applied_papers)}/{len(enhancement_config.paper_ids)} "
            f"papers applied successfully"
        )
        if failed_papers:
            logger.warning(f"Failed papers: {[p[0] for p in failed_papers]}")
        
        return enhanced_model
    
    def suggest_papers(
        self,
        model_size: str = "small",
        category: Optional[str] = None,
        max_memory_impact: str = "medium"
    ) -> List[str]:
        """
        Suggest papers for a model based on criteria.
        
        Args:
            model_size: Model size (small/medium/large)
            category: Paper category
            max_memory_impact: Maximum memory impact
        
        Returns:
            List of suggested paper IDs
        """
        papers = self.adapter.registry.search_papers(
            category=category,
            max_memory_impact=max_memory_impact
        )
        
        # Filter by model size compatibility
        if model_size == "small":
            papers = [p for p in papers if p.memory_impact in ["low", "medium"]]
        elif model_size == "medium":
            papers = [p for p in papers if p.memory_impact in ["low", "medium", "high"]]
        
        # Sort by speedup and accuracy improvement
        papers.sort(
            key=lambda p: (
                p.speedup or 0,
                p.accuracy_improvement or 0
            ),
            reverse=True
        )
        
        return [p.paper_id for p in papers[:10]]  # Top 10
    
    def create_enhancement_plan(
        self,
        model_size: str = "small",
        goals: List[str] = None
    ) -> EnhancementConfig:
        """
        Create an enhancement plan for a model.
        
        Args:
            model_size: Model size
            goals: List of goals (e.g., ["speed", "accuracy", "memory"])
        
        Returns:
            Enhancement configuration
        """
        if goals is None:
            goals = ["speed", "accuracy"]
        
        suggested_papers = []
        
        for goal in goals:
            if goal == "speed":
                papers = self.adapter.registry.search_papers(
                    min_speedup=1.5,
                    max_memory_impact="medium"
                )
            elif goal == "accuracy":
                papers = self.adapter.registry.search_papers(
                    min_accuracy=5.0,
                    max_memory_impact="medium"
                )
            elif goal == "memory":
                papers = self.adapter.registry.search_papers(
                    max_memory_impact="low"
                )
            else:
                continue
            
            # Add top papers for this goal
            papers.sort(key=lambda p: p.speedup or 0, reverse=True)
            for paper in papers[:3]:
                if paper.paper_id not in suggested_papers:
                    suggested_papers.append(paper.paper_id)
        
        return EnhancementConfig(paper_ids=suggested_papers)


