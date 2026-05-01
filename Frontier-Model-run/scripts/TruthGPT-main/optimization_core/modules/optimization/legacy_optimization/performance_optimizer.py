"""
Comprehensive performance optimization utilities.
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Comprehensive performance optimizer for PyTorch models.
    Implements various optimization techniques.
    """
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.optimizations_applied: List[str] = []
    
    def optimize_model(
        self,
        model: nn.Module,
        optimizations: Optional[List[str]] = None,
        **kwargs
    ) -> nn.Module:
        """
        Apply multiple optimizations to model.
        
        Args:
            model: Model to optimize
            optimizations: List of optimizations to apply (all if None)
            **kwargs: Optimization-specific arguments
        
        Returns:
            Optimized model
        """
        if optimizations is None:
            optimizations = [
                "torch_compile",
                "fuse_conv_bn",
                "quantize",
            ]
        
        optimized_model = model
        
        for opt in optimizations:
            try:
                if opt == "torch_compile":
                    optimized_model = self.apply_torch_compile(
                        optimized_model,
                        mode=kwargs.get("compile_mode", "default")
                    )
                elif opt == "fuse_conv_bn":
                    optimized_model = self.fuse_conv_bn(optimized_model)
                elif opt == "quantize":
                    if kwargs.get("quantize", False):
                        optimized_model = self.quantize_model(optimized_model)
                elif opt == "gradient_checkpointing":
                    self.enable_gradient_checkpointing(optimized_model)
                
            except Exception as e:
                logger.warning(f"Failed to apply optimization {opt}: {e}")
        
        return optimized_model
    
    def apply_torch_compile(
        self,
        model: nn.Module,
        mode: str = "default"
    ) -> nn.Module:
        """
        Apply torch.compile optimization.
        
        Args:
            model: Model to compile
            mode: Compilation mode
        
        Returns:
            Compiled model
        """
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available")
            return model
        
        try:
            compiled = torch.compile(model, mode=mode)
            self.optimizations_applied.append(f"torch_compile_{mode}")
            logger.info(f"Model compiled with mode: {mode}")
            return compiled
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
            return model
    
    def fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """
        Fuse Conv-BN layers for faster inference.
        
        Args:
            model: Model to optimize
        
        Returns:
            Model with fused layers
        """
        try:
            fused = torch.quantization.fuse_modules(
                model,
                [["conv", "bn"], ["conv", "bn", "relu"]],
            )
            self.optimizations_applied.append("fuse_conv_bn")
            logger.info("Conv-BN layers fused")
            return fused
        except Exception as e:
            logger.warning(f"Failed to fuse Conv-BN: {e}")
            return model
    
    def quantize_model(
        self,
        model: nn.Module,
        quantization_type: str = "dynamic"
    ) -> nn.Module:
        """
        Quantize model for faster inference.
        
        Args:
            model: Model to quantize
            quantization_type: Type of quantization (dynamic|static|qat)
        
        Returns:
            Quantized model
        """
        try:
            if quantization_type == "dynamic":
                quantized = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            else:
                # Static quantization requires calibration data
                logger.warning(f"{quantization_type} quantization not fully implemented")
                return model
            
            self.optimizations_applied.append(f"quantize_{quantization_type}")
            logger.info(f"Model quantized with {quantization_type} quantization")
            return quantized
        except Exception as e:
            logger.warning(f"Failed to quantize model: {e}")
            return model
    
    def enable_gradient_checkpointing(self, model: nn.Module) -> None:
        """
        Enable gradient checkpointing for memory efficiency.
        
        Args:
            model: Model to optimize
        """
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                self.optimizations_applied.append("gradient_checkpointing")
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Model does not support gradient checkpointing")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")
    
    def optimize_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        pin_memory: bool = True,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        """
        Optimize DataLoader settings.
        
        Args:
            dataloader: DataLoader to optimize
            pin_memory: Pin memory for faster GPU transfer
            num_workers: Number of worker processes
            prefetch_factor: Prefetch factor
        
        Returns:
            Optimized DataLoader
        """
        # DataLoader optimization is typically done at creation time
        # This method serves as documentation
        logger.info("DataLoader optimization should be done at creation")
        return dataloader
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of applied optimizations.
        
        Returns:
            Dictionary with optimization summary
        """
        return {
            "optimizations_applied": self.optimizations_applied,
            "count": len(self.optimizations_applied),
        }



