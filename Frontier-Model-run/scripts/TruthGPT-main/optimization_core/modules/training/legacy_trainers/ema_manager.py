"""
EMA Manager - Handles Exponential Moving Average weights.

Separated from trainer for better modularity.
"""
import logging
from typing import Dict, Optional
import torch
import torch.nn as nn

from optimization_core.trainers.config import EMAConfig

logger = logging.getLogger(__name__)


class EMAManager:
    """
    Manages Exponential Moving Average (EMA) of model weights.
    
    Responsibilities:
    - Initialize EMA shadow parameters
    - Update EMA on each optimizer step
    - Apply/restore EMA weights for evaluation
    """
    
    def __init__(self, ema_config: EMAConfig, model: nn.Module) -> None:
        """
        Initialize EMAManager.
        
        Args:
            ema_config: EMA configuration (Pydantic instance)
            model: Model to track
        """
        if not isinstance(ema_config, EMAConfig):
            raise TypeError(f"ema_config must be an instance of EMAConfig, got {type(ema_config)}")
            
        self.ema_config = ema_config
        self.model = model
        self._ema_shadow: Optional[Dict[str, torch.Tensor]] = None
        self._ema_backup: Optional[Dict[str, torch.Tensor]] = None
        
        if self.ema_config.enabled:
            self._init_ema()
            logger.info(f"EMA initialized with decay={self.ema_config.decay}")
    
    def _get_base_model(self) -> nn.Module:
        """Get base model (handles parallel wrappers)."""
        model = self.model
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module
        if hasattr(model, "module"):
            model = model.module
        return model
    
    def _init_ema(self) -> None:
        """Initialize EMA shadow parameters."""
        self._ema_shadow = {}
        model = self._get_base_model()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._ema_shadow[name] = param.detach().clone().to(device=param.device)
    
    @torch.no_grad()
    def update(self) -> None:
        """Update EMA shadow parameters."""
        if not self.ema_config.enabled or self._ema_shadow is None:
            return
        
        decay = self.ema_config.decay
        model = self._get_base_model()
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self._ema_shadow:
                self._ema_shadow[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    
    def apply_ema(self) -> None:
        """Apply EMA weights to model."""
        if not self.ema_config.enabled or self._ema_shadow is None:
            return
        
        self._ema_backup = {}
        model = self._get_base_model()
        
        for name, param in model.named_parameters():
            if name in self._ema_shadow and param.requires_grad:
                # Backup current weights
                self._ema_backup[name] = param.detach().clone()
                # Apply EMA weights
                param.data.copy_(self._ema_shadow[name].data)
    
    def restore_from_ema(self) -> None:
        """Restore model weights from EMA backup."""
        if not self.ema_config.enabled or self._ema_backup is None:
            return
        
        model = self._get_base_model()
        
        for name, param in model.named_parameters():
            if name in self._ema_backup and param.requires_grad:
                param.data.copy_(self._ema_backup[name].data)
        
        self._ema_backup = {}
    
    def get_ema_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state dict."""
        if self._ema_shadow is None:
            return {}
        return {k: v.clone() for k, v in self._ema_shadow.items()}




