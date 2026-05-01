"""
Core Module - Módulos Base Compartidos
======================================

Módulos base para el sistema de papers:
- Metadata extraction
- Common utilities
- Base classes
"""

from .metadata_extractor import MetadataExtractor, PaperMetadata
from .paper_base import BasePaperModule, BasePaperConfig

__all__ = [
    'MetadataExtractor',
    'PaperMetadata',
    'BasePaperModule',
    'BasePaperConfig'
]



