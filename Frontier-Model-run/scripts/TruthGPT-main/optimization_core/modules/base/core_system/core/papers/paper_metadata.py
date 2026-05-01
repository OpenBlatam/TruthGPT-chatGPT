"""
Paper Metadata - Metadata classes for research papers
=====================================================

Provides metadata structures for papers integrated into the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Type

__all__ = ['PaperMetadata', 'PaperModule']


@dataclass
class PaperMetadata:
    """Metadata for a research paper."""
    paper_id: str
    paper_name: str
    module_path: Path
    module_name: str
    category: str
    config_class: Optional[str] = None
    module_class: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    enabled_by_default: bool = False
    performance_impact: str = "medium"
    memory_impact: str = "medium"
    speedup: Optional[float] = None
    accuracy_improvement: Optional[float] = None
    arxiv_id: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    key_techniques: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[float] = None
    load_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'paper_id': self.paper_id,
            'paper_name': self.paper_name,
            'module_path': str(self.module_path),
            'module_name': self.module_name,
            'category': self.category,
            'config_class': self.config_class,
            'module_class': self.module_class,
            'dependencies': self.dependencies,
            'enabled_by_default': self.enabled_by_default,
            'performance_impact': self.performance_impact,
            'memory_impact': self.memory_impact,
            'speedup': self.speedup,
            'accuracy_improvement': self.accuracy_improvement,
            'arxiv_id': self.arxiv_id,
            'year': self.year,
            'authors': self.authors,
            'benchmarks': self.benchmarks,
            'key_techniques': self.key_techniques,
            'usage_count': self.usage_count,
            'load_count': self.load_count,
            'error_count': self.error_count,
        }


@dataclass
class PaperModule:
    """Loaded paper module with metadata."""
    metadata: PaperMetadata
    config_class: Optional[Type] = None
    module_class: Optional[Type] = None
    module: Optional[Any] = None
    loaded: bool = False
    load_time: float = 0.0
    error: Optional[str] = None
    cache_key: Optional[str] = None
    
    def is_available(self) -> bool:
        """Check if module is loaded and available."""
        return self.loaded and self.module_class is not None
    
    def create_config(self, **kwargs) -> Any:
        """Create config instance for this paper."""
        if not self.config_class:
            raise ValueError(f"No config class available for {self.metadata.paper_id}")
        return self.config_class(**kwargs)
    
    def create_module(self, config: Any) -> Any:
        """Create module instance for this paper."""
        if not self.module_class:
            raise ValueError(f"No module class available for {self.metadata.paper_id}")
        return self.module_class(config)





