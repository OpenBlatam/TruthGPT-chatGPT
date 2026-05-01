"""
Paper Registry - Registry for research papers
=============================================

Provides discovery, loading, and management of research papers integrated into the framework.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Try to import from papers source
_papers_source = Path(__file__).resolve().parents[5] / "truthgpt_collected" / "integration_code" / "papers"

# Initialize availability flags
_PAPERS_AVAILABLE = False
SourcePaperMetadata = None
SourcePaperModule = None
PaperRegistryRefactored = None

# Try importing from papers source directory
if _papers_source.exists():
    try:
        # Add papers directory to path
        if str(_papers_source) not in sys.path:
            sys.path.insert(0, str(_papers_source))
        
        # Add papers/core directory to path
        papers_core_path = _papers_source / "core"
        if papers_core_path.exists() and str(papers_core_path) not in sys.path:
            sys.path.insert(0, str(papers_core_path))
        
        # Try to import
        try:
            from metadata_extractor import MetadataExtractor, PaperMetadata as SourcePaperMetadata
            from paper_registry_refactored import PaperRegistryRefactored, PaperModule as SourcePaperModule
            _PAPERS_AVAILABLE = True
            logger.info("✅ Papers source modules loaded successfully")
        except ImportError as e:
            logger.warning(f"Could not import papers source modules: {e}")
            # Try alternative import paths
            try:
                from modules.base.core_system.core.metadata_extractor import MetadataExtractor, PaperMetadata as SourcePaperMetadata
                from modules.base.core_system.core.paper_registry_refactored import PaperRegistryRefactored, PaperModule as SourcePaperModule
                _PAPERS_AVAILABLE = True
                logger.info("✅ Papers source modules loaded from alternative path")
            except ImportError:
                pass
    except Exception as e:
        logger.warning(f"Error setting up papers source: {e}")
else:
    logger.warning(f"Papers source directory not found: {_papers_source}")

from .paper_metadata import PaperMetadata, PaperModule

__all__ = ['PaperRegistry', 'get_paper_registry']


class PaperRegistry:
    """
    Registry for research papers integrated into TruthGPT core.
    
    Wraps the existing paper registry from truthgpt_collected and provides
    a unified interface for the core framework.
    """
    
    def __init__(
        self,
        papers_base_dir: Optional[Path] = None,
        enable_disk_cache: bool = True,
        preload_popular: bool = True,
        max_cache_size: int = 100
    ):
        """
        Initialize paper registry.
        
        Args:
            papers_base_dir: Base directory for papers (defaults to truthgpt_collected/integration_code/papers)
            enable_disk_cache: Enable disk caching
            preload_popular: Preload popular papers
            max_cache_size: Maximum cache size
        """
        if not _PAPERS_AVAILABLE:
            logger.warning("Papers source not available. Registry will be empty.")
            self._registry = None
            self._papers_base_dir = papers_base_dir or _papers_source
            return
        
        # Use default papers directory if not specified
        if papers_base_dir is None:
            papers_base_dir = _papers_source
        
        # Initialize underlying registry
        try:
            self._registry = PaperRegistryRefactored(
                papers_base_dir=papers_base_dir,
                enable_disk_cache=enable_disk_cache,
                preload_popular=preload_popular,
                max_cache_size=max_cache_size
            )
            paper_count = len(self._registry.registry) if hasattr(self._registry, 'registry') else 0
            logger.info(f"✅ Paper registry initialized with {paper_count} papers")
        except Exception as e:
            logger.error(f"Failed to initialize paper registry: {e}", exc_info=True)
            self._registry = None
            logger.warning("Paper registry will operate in limited mode")
    
    def discover_papers(self) -> int:
        """
        Discover papers in the papers directory.
        
        Returns:
            Number of papers discovered
        """
        if not self._registry:
            return 0
        
        try:
            # Trigger discovery if needed
            count = len(self._registry.registry)
            logger.info(f"📚 Discovered {count} papers")
            return count
        except Exception as e:
            logger.error(f"Error discovering papers: {e}")
            return 0
    
    def load_paper(self, paper_id: str, force_reload: bool = False) -> Optional[PaperModule]:
        """
        Load a paper by ID.
        
        Args:
            paper_id: Paper identifier
            force_reload: Force reload even if cached
        
        Returns:
            PaperModule if successful, None otherwise
        """
        if not self._registry:
            logger.warning(f"Registry not available, cannot load paper: {paper_id}")
            return None
        
        try:
            source_module = self._registry.load_paper(paper_id, force_reload)
            if not source_module or not source_module.loaded:
                return None
            
            # Convert to our PaperModule format
            return PaperModule(
                metadata=self._convert_metadata(source_module.metadata),
                config_class=source_module.config_class,
                module_class=source_module.module_class,
                module=source_module.module,
                loaded=source_module.loaded,
                load_time=source_module.load_time,
                error=source_module.error,
                cache_key=source_module.cache_key
            )
        except Exception as e:
            logger.error(f"Error loading paper {paper_id}: {e}")
            return None
    
    def search_papers(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_speedup: Optional[float] = None,
        min_accuracy: Optional[float] = None,
        max_memory_impact: Optional[str] = None
    ) -> List[PaperMetadata]:
        """
        Search papers with filters.
        
        Args:
            query: Text query
            category: Paper category
            min_speedup: Minimum speedup
            min_accuracy: Minimum accuracy improvement
            max_memory_impact: Maximum memory impact (low/medium/high)
        
        Returns:
            List of matching PaperMetadata
        """
        if not self._registry:
            return []
        
        try:
            results = self._registry.search_papers(
                query=query,
                category=category,
                min_speedup=min_speedup,
                min_accuracy=min_accuracy,
                max_memory_impact=max_memory_impact
            )
            return [self._convert_metadata(meta) for meta in results]
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    def list_papers(self, category: Optional[str] = None) -> List[PaperMetadata]:
        """
        List all papers.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of PaperMetadata
        """
        if not self._registry:
            return []
        
        try:
            papers = self._registry.list_papers(category=category)
            return [self._convert_metadata(meta) for meta in papers]
        except Exception as e:
            logger.error(f"Error listing papers: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        if not self._registry:
            return {
                'total_papers': 0,
                'loaded_papers': 0,
                'available': False
            }
        
        try:
            stats = self._registry.get_statistics()
            stats['available'] = True
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'available': False, 'error': str(e)}
    
    def _convert_metadata(self, source_meta: Any) -> PaperMetadata:
        """Convert source metadata to our format."""
        if isinstance(source_meta, PaperMetadata):
            return source_meta
        
        try:
            # Convert from source format
            return PaperMetadata(
                paper_id=getattr(source_meta, 'paper_id', 'unknown'),
                paper_name=getattr(source_meta, 'paper_name', 'Unknown Paper'),
                module_path=Path(getattr(source_meta, 'module_path', '')),
                module_name=getattr(source_meta, 'module_name', 'unknown'),
                category=getattr(source_meta, 'category', 'unknown'),
                config_class=getattr(source_meta, 'config_class', None),
                module_class=getattr(source_meta, 'module_class', None),
                dependencies=getattr(source_meta, 'dependencies', []),
                enabled_by_default=getattr(source_meta, 'enabled_by_default', False),
                performance_impact=getattr(source_meta, 'performance_impact', 'medium'),
                memory_impact=getattr(source_meta, 'memory_impact', 'medium'),
                speedup=getattr(source_meta, 'speedup', None),
                accuracy_improvement=getattr(source_meta, 'accuracy_improvement', None),
                arxiv_id=getattr(source_meta, 'arxiv_id', None),
                year=getattr(source_meta, 'year', None),
                authors=getattr(source_meta, 'authors', []),
                benchmarks=getattr(source_meta, 'benchmarks', {}),
                key_techniques=getattr(source_meta, 'key_techniques', []),
                usage_count=getattr(source_meta, 'usage_count', 0),
                last_used=getattr(source_meta, 'last_used', None),
                load_count=getattr(source_meta, 'load_count', 0),
                error_count=getattr(source_meta, 'error_count', 0),
            )
        except Exception as e:
            logger.error(f"Error converting metadata: {e}", exc_info=True)
            # Return minimal metadata on error
            return PaperMetadata(
                paper_id='unknown',
                paper_name='Unknown Paper',
                module_path=Path('.'),
                module_name='unknown',
                category='unknown'
            )


# Global registry instance
_global_registry: Optional[PaperRegistry] = None


def get_paper_registry(**kwargs) -> PaperRegistry:
    """
    Get or create global paper registry instance.
    
    Args:
        **kwargs: Arguments to pass to PaperRegistry constructor
    
    Returns:
        PaperRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PaperRegistry(**kwargs)
    return _global_registry


