"""
Papers Module - Integration of Research Papers into TruthGPT Core
==================================================================

This module integrates research papers from truthgpt_collected into the core framework,
allowing small models to leverage advanced techniques and become more capable when
integrated into the framework.

The goal is: "Any person can create a small model, and when integrated into the framework,
it becomes a larger, more capable AI."

Features:
- Automatic paper discovery and registration
- Paper registry with metadata extraction
- Paper adapters for model integration
- Factory pattern for paper-based components
- Search and filtering capabilities
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add papers source directory to path if needed
_papers_source = Path(__file__).parent.parent.parent / "truthgpt_collected" / "integration_code" / "papers"
if _papers_source.exists() and str(_papers_source) not in sys.path:
    sys.path.insert(0, str(_papers_source))

# Core paper components
from .paper_registry import PaperRegistry, get_paper_registry
from .paper_adapter import PaperAdapter, ModelEnhancer, EnhancementConfig
from .paper_factory import PaperFactory, create_paper_component
from .paper_metadata import PaperMetadata, PaperModule
from .paper_component import PaperComponent, PaperComponentMetrics
from .paper_validator import (
    PaperValidator,
    validate_paper_id,
    validate_paper_config
)

# Re-export key classes
__all__ = [
    # Registry
    'PaperRegistry',
    'get_paper_registry',
    # Adapter
    'PaperAdapter',
    'ModelEnhancer',
    'EnhancementConfig',
    # Factory
    'PaperFactory',
    'create_paper_component',
    # Metadata
    'PaperMetadata',
    'PaperModule',
    # Component
    'PaperComponent',
    'PaperComponentMetrics',
    # Validator
    'PaperValidator',
    'validate_paper_id',
    'validate_paper_config',
]


