"""
Paper Component - IComponent implementation for papers
=====================================================

Implements IComponent interface for paper-based components, following
the architecture specifications.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..interfaces import IComponent
from ..base_classes import BaseComponent
from ..exceptions import OptimizationCoreError, ResourceError
from .paper_metadata import PaperModule

logger = logging.getLogger(__name__)

__all__ = ['PaperComponent', 'PaperComponentMetrics']


@dataclass
class PaperComponentMetrics:
    """Metrics for paper component."""
    paper_id: str
    load_count: int = 0
    apply_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_load_time: float = 0.0
    total_apply_time: float = 0.0
    last_load_time: Optional[float] = None
    last_apply_time: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'paper_id': self.paper_id,
            'load_count': self.load_count,
            'apply_count': self.apply_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'avg_load_time': (
                self.total_load_time / self.load_count
                if self.load_count > 0 else 0.0
            ),
            'avg_apply_time': (
                self.total_apply_time / self.apply_count
                if self.apply_count > 0 else 0.0
            ),
            'success_rate': (
                self.success_count / self.apply_count
                if self.apply_count > 0 else 0.0
            ),
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            'last_load_time': self.last_load_time,
            'last_apply_time': self.last_apply_time,
        }


class PaperComponent(BaseComponent):
    """
    Paper component implementing IComponent interface.
    
    Follows architecture specifications exactly:
    - Implements IComponent interface
    - Provides metrics and observability
    - Handles initialization and cleanup
    - Provides status information per spec
    """
    
    def __init__(
        self,
        paper_module: PaperModule,
        **kwargs
    ):
        """
        Initialize paper component.
        
        Args:
            paper_module: Loaded paper module
            **kwargs: Additional parameters
        """
        super().__init__(
            name=f"paper_{paper_module.metadata.paper_id}",
            version=str(paper_module.metadata.year) if paper_module.metadata.year else "1.0.0",
            **kwargs
        )
        self.paper_module = paper_module
        self.metrics = PaperComponentMetrics(paper_id=paper_module.metadata.paper_id)
        self._config_cache: Dict[str, Any] = {}
        self._module_cache: Dict[str, Any] = {}
        self._start_time: float = time.time()
        self._last_error: Optional[str] = None
        self._ready: bool = False
    
    def _initialize_impl(self, **kwargs):
        """Implementation-specific initialization."""
        # Check if paper module is available
        if not self.paper_module.is_available():
            self._last_error = f"Paper module not available: {self.paper_module.error}"
            self._ready = False
            raise ResourceError(
                self._last_error,
                resource=f"paper_{self.paper_module.metadata.paper_id}"
            )
        
        self._ready = True
        self._last_error = None
        
        # Pre-warm cache if requested
        if kwargs.get('prewarm_cache', False):
            try:
                self._prewarm_cache()
            except Exception as e:
                self._last_error = str(e)
                logger.warning(f"Failed to pre-warm cache: {e}")
    
    def _cleanup_impl(self) -> None:
        """
        Implementation-specific cleanup.
        
        Per IComponent spec: cleanup() should be idempotent (safe to call multiple times).
        """
        self._config_cache.clear()
        self._module_cache.clear()
        self._ready = False
        # Note: cleanup() returns None per IComponent spec
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.
        
        Returns:
            Status dictionary per IComponent specification:
            {
                "name": str,
                "version": str,
                "initialized": bool,
                "ready": bool,
                "health": str,  # "healthy", "degraded", "unhealthy"
                "metrics": Dict[str, Any],
                "last_error": Optional[str],
                "uptime_seconds": float
            }
        """
        # Calculate health status
        health = self._calculate_health()
        
        # Calculate uptime
        uptime_seconds = time.time() - self._start_time
        
        # Base status per spec
        status = {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "ready": self._ready and self.paper_module.is_available(),
            "health": health,
            "metrics": self.metrics.to_dict(),
            "last_error": self._last_error,
            "uptime_seconds": uptime_seconds,
        }
        
        # Add paper-specific metadata (not in spec but useful)
        status["paper_metadata"] = {
            "paper_id": self.paper_module.metadata.paper_id,
            "paper_name": self.paper_module.metadata.paper_name,
            "category": self.paper_module.metadata.category,
            "available": self.paper_module.is_available(),
        }
        
        return status
    
    def _calculate_health(self) -> str:
        """
        Calculate component health status.
        
        Returns:
            "healthy", "degraded", or "unhealthy"
        """
        if not self._initialized:
            return "unhealthy"
        
        if not self.paper_module.is_available():
            return "unhealthy"
        
        if self._last_error:
            return "degraded"
        
        # Check error rate
        if self.metrics.apply_count > 0:
            error_rate = self.metrics.error_count / self.metrics.apply_count
            if error_rate > 0.5:  # More than 50% errors
                return "unhealthy"
            elif error_rate > 0.1:  # More than 10% errors
                return "degraded"
        
        return "healthy"
    
    def create_config(self, **kwargs) -> Any:
        """
        Create paper config with caching.
        
        Args:
            **kwargs: Config parameters
        
        Returns:
            Config instance
        """
        # Create cache key
        cache_key = str(sorted(kwargs.items()))
        
        # Check cache
        if cache_key in self._config_cache:
            self.metrics.cache_hits += 1
            return self._config_cache[cache_key]
        
        self.metrics.cache_misses += 1
        
        # Create config
        start_time = time.time()
        try:
            config = self.paper_module.create_config(**kwargs)
            self._config_cache[cache_key] = config
            self.metrics.load_count += 1
            self.metrics.total_load_time += time.time() - start_time
            self.metrics.last_load_time = time.time()
            self._last_error = None
            return config
        except Exception as e:
            self.metrics.error_count += 1
            self._last_error = f"Error creating config: {e}"
            logger.error(self._last_error)
            raise
    
    def create_module(self, config: Any) -> Any:
        """
        Create paper module with caching.
        
        Args:
            config: Paper config
        
        Returns:
            Module instance
        """
        # Create cache key from config
        cache_key = str(id(config))
        
        # Check cache
        if cache_key in self._module_cache:
            self.metrics.cache_hits += 1
            return self._module_cache[cache_key]
        
        self.metrics.cache_misses += 1
        
        # Create module
        start_time = time.time()
        try:
            module = self.paper_module.create_module(config)
            self._module_cache[cache_key] = module
            self.metrics.apply_count += 1
            self.metrics.success_count += 1
            self.metrics.total_apply_time += time.time() - start_time
            self.metrics.last_apply_time = time.time()
            self._last_error = None
            return module
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.apply_count += 1
            self._last_error = f"Error creating module: {e}"
            logger.error(self._last_error)
            raise
    
    def _prewarm_cache(self):
        """Pre-warm cache with default config."""
        try:
            default_config = self.create_config()
            self.create_module(default_config)
            logger.info(f"Cache pre-warmed for {self.name}")
        except Exception as e:
            logger.warning(f"Failed to pre-warm cache: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component metrics.
        
        Returns:
            Metrics dictionary
        """
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset metrics."""
        self.metrics = PaperComponentMetrics(paper_id=self.paper_module.metadata.paper_id)


