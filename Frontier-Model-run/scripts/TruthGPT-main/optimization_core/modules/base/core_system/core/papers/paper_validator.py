"""
Paper Validator - Validation utilities for papers
=================================================

Provides validation following architecture specifications.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..validators import ValidationError, validate_non_empty_string, validate_positive_int
from .paper_metadata import PaperMetadata, PaperModule

logger = logging.getLogger(__name__)

__all__ = ['PaperValidator', 'validate_paper_id', 'validate_paper_config']


def validate_paper_id(paper_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate paper ID format.
    
    Args:
        paper_id: Paper identifier
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not paper_id:
        return False, "Paper ID cannot be empty"
    
    if not isinstance(paper_id, str):
        return False, "Paper ID must be a string"
    
    # Check format (e.g., "2503.00735v3" or "paper-name")
    if len(paper_id) < 3:
        return False, "Paper ID too short (minimum 3 characters)"
    
    if len(paper_id) > 100:
        return False, "Paper ID too long (maximum 100 characters)"
    
    return True, None


def validate_paper_config(config: Dict[str, Any], required_keys: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate paper configuration.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "Config must be a dictionary"
    
    if required_keys:
        for key in required_keys:
            if key not in config:
                return False, f"Required key '{key}' missing from config"
    
    # Validate common config keys
    if 'hidden_dim' in config:
        if not isinstance(config['hidden_dim'], int) or config['hidden_dim'] <= 0:
            return False, "hidden_dim must be a positive integer"
    
    if 'batch_size' in config:
        if not isinstance(config['batch_size'], int) or config['batch_size'] <= 0:
            return False, "batch_size must be a positive integer"
    
    return True, None


class PaperValidator:
    """
    Validator for paper operations.
    
    Follows architecture specifications for validation.
    """
    
    @staticmethod
    def validate_paper_metadata(metadata: PaperMetadata) -> Tuple[bool, Optional[str]]:
        """
        Validate paper metadata.
        
        Args:
            metadata: Paper metadata
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(metadata, PaperMetadata):
            return False, "Metadata must be a PaperMetadata instance"
        
        # Validate paper_id
        is_valid, error = validate_paper_id(metadata.paper_id)
        if not is_valid:
            return False, error
        
        # Validate module_path
        if not metadata.module_path or not Path(metadata.module_path).exists():
            return False, f"Module path does not exist: {metadata.module_path}"
        
        # Validate category
        valid_categories = [
            'research', 'architecture', 'inference', 'memory',
            'redundancy', 'techniques', 'code', 'best'
        ]
        if metadata.category not in valid_categories:
            return False, f"Invalid category: {metadata.category}. Must be one of {valid_categories}"
        
        return True, None
    
    @staticmethod
    def validate_paper_module(paper_module: PaperModule) -> Tuple[bool, Optional[str]]:
        """
        Validate paper module.
        
        Args:
            paper_module: Paper module
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(paper_module, PaperModule):
            return False, "Must be a PaperModule instance"
        
        # Validate metadata
        is_valid, error = PaperValidator.validate_paper_metadata(paper_module.metadata)
        if not is_valid:
            return False, f"Invalid metadata: {error}"
        
        # Check if module is available
        if not paper_module.is_available():
            return False, f"Paper module not available: {paper_module.error or 'Unknown error'}"
        
        return True, None
    
    @staticmethod
    def validate_enhancement_config(
        paper_ids: List[str],
        max_papers: int = 10
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate enhancement configuration.
        
        Args:
            paper_ids: List of paper IDs
            max_papers: Maximum number of papers allowed
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(paper_ids, list):
            return False, "paper_ids must be a list"
        
        if len(paper_ids) == 0:
            return False, "At least one paper ID is required"
        
        if len(paper_ids) > max_papers:
            return False, f"Too many papers (maximum {max_papers})"
        
        # Validate each paper ID
        for paper_id in paper_ids:
            is_valid, error = validate_paper_id(paper_id)
            if not is_valid:
                return False, f"Invalid paper ID '{paper_id}': {error}"
        
        # Check for duplicates
        if len(paper_ids) != len(set(paper_ids)):
            return False, "Duplicate paper IDs found"
        
        return True, None
    
    @staticmethod
    def validate_model_for_enhancement(model: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate model for enhancement.
        
        Args:
            model: Model to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import torch.nn as nn
        except ImportError:
            return False, "PyTorch not available"
        
        if not isinstance(model, nn.Module):
            return False, "Model must be a torch.nn.Module instance"
        
        # Check if model has forward method
        if not hasattr(model, 'forward'):
            return False, "Model must have a forward method"
        
        return True, None





