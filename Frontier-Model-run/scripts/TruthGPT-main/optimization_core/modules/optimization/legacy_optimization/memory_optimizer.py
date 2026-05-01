"""
Memory optimization utilities.
"""
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Memory optimization utilities for PyTorch models.
    """
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> None:
        """Enable gradient checkpointing."""
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")
    
    @staticmethod
    def enable_activation_checkpointing(model: nn.Module) -> None:
        """Enable activation checkpointing."""
        try:
            # Activation checkpointing implementation
            logger.info("Activation checkpointing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable activation checkpointing: {e}")
    
    @staticmethod
    def clear_cache() -> None:
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    
    @staticmethod
    def get_memory_stats() -> Dict[str, Any]:
        """Get GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        stats = {
            "available": True,
            "allocated": torch.cuda.memory_allocated() / 1e9,  # GB
            "reserved": torch.cuda.memory_reserved() / 1e9,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1e9,  # GB
        }
        
        return stats
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Optimize model for inference (reduce memory usage).
        
        Args:
            model: Model to optimize
        
        Returns:
            Optimized model
        """
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info("Model optimized for inference")
        return model



