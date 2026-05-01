"""
Optimization Techniques
=======================
Modular, reusable optimization techniques using Strategy pattern.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class OptimizationTechnique(ABC):
    """Base class for optimization techniques."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def apply(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """
        Apply optimization technique to model.
        
        Returns:
            Tuple of (optimized_model, success)
        """
        pass
    
    def can_apply(self, model: nn.Module) -> bool:
        """Check if technique can be applied to model."""
        return True


class GradientCheckpointingTechnique(OptimizationTechnique):
    """Gradient checkpointing optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("gradient_checkpointing", config)
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """Apply gradient checkpointing."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            try:
                model.gradient_checkpointing_enable()
                return model, True
            except Exception as e:
                self.logger.warning(f"Failed to enable gradient checkpointing: {e}")
                return model, False
        return model, False


class MixedPrecisionTechnique(OptimizationTechnique):
    """Mixed precision optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("mixed_precision", config)
        self.dtype = self.config.get('dtype', 'bf16')
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """Apply mixed precision (marker technique)."""
        return model, True
    
    def can_apply(self, model: nn.Module) -> bool:
        """Check if CUDA is available for mixed precision."""
        return torch.cuda.is_available()


class TorchCompileTechnique(OptimizationTechnique):
    """Torch compile optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("torch_compile", config)
        self.mode = self.config.get('compile_mode', 'default')
        self.backend = self.config.get('compile_backend', 'inductor')
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """Apply torch.compile."""
        try:
            compiled = torch.compile(model, mode=self.mode, backend=self.backend)
            return compiled, True
        except Exception as e:
            self.logger.warning(f"torch.compile failed: {e}")
            return model, False


class TF32Technique(OptimizationTechnique):
    """TF32 acceleration for Ampere+ GPUs."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("tf32", config)
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """Enable TF32."""
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                return model, True
            except Exception as e:
                self.logger.warning(f"Failed to enable TF32: {e}")
                return model, False
        return model, False


class FusedAdamWTechnique(OptimizationTechnique):
    """Fused AdamW optimizer technique."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("fused_adamw", config)
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """Mark model for fused AdamW (actual optimization happens in optimizer setup)."""
        return model, True


class QuantizationTechnique(OptimizationTechnique):
    """Quantization optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("quantization", config)
        self.bits = self.config.get('bits', 8)
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """Apply quantization."""
        try:
            if self.bits == 8:
                quantized = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                return quantized, True
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
        return model, False


class PruningTechnique(OptimizationTechnique):
    """Model pruning optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("pruning", config)
        self.amount = self.config.get('amount', 0.1)
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, bool]:
        """Apply pruning."""
        try:
            import torch.nn.utils.prune as prune
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=self.amount)
            return model, True
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
        return model, False


class TechniqueRegistry:
    """Registry for optimization techniques."""
    
    def __init__(self):
        self._techniques: Dict[str, type] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default techniques."""
        self.register('gradient_checkpointing', GradientCheckpointingTechnique)
        self.register('mixed_precision', MixedPrecisionTechnique)
        self.register('torch_compile', TorchCompileTechnique)
        self.register('tf32', TF32Technique)
        self.register('fused_adamw', FusedAdamWTechnique)
        self.register('quantization', QuantizationTechnique)
        self.register('pruning', PruningTechnique)
    
    def register(self, name: str, technique_class: type):
        """Register a new optimization technique."""
        if not issubclass(technique_class, OptimizationTechnique):
            raise TypeError(f"Technique must subclass OptimizationTechnique")
        self._techniques[name] = technique_class
    
    def create(self, name: str, config: Dict[str, Any] = None) -> OptimizationTechnique:
        """Create an optimization technique instance."""
        if name not in self._techniques:
            raise ValueError(f"Unknown technique: {name}")
        return self._techniques[name](config or {})
    
    def list_techniques(self) -> List[str]:
        """List all registered techniques."""
        return list(self._techniques.keys())


_global_registry = TechniqueRegistry()


def get_technique(name: str, config: Dict[str, Any] = None) -> OptimizationTechnique:
    """Get an optimization technique from the global registry."""
    return _global_registry.create(name, config)


def register_technique(name: str, technique_class: type):
    """Register a new optimization technique globally."""
    _global_registry.register(name, technique_class)








