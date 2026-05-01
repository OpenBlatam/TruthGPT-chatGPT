#!/usr/bin/env python3
"""
Base Paper Classes - Clases Base para Papers
============================================

Clases base para Config y Module de papers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class BasePaperConfig:
    """Clase base para configuraciones de papers."""
    hidden_dim: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte config a dict."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class BasePaperModule(nn.Module, ABC):
    """
    Clase base para módulos de papers.
    
    Todos los papers deben heredar de esta clase.
    """
    
    def __init__(self, config: BasePaperConfig):
        super().__init__()
        self.config = config
        self._metrics: Dict[str, Any] = {}
        self._forward_count = 0
    
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass del paper.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            **kwargs: Argumentos adicionales
        
        Returns:
            Tuple (output, metadata)
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del módulo."""
        return {
            **self._metrics,
            'forward_count': self._forward_count
        }
    
    def reset_metrics(self):
        """Resetea métricas."""
        self._metrics.clear()
        self._forward_count = 0
    
    def _update_metrics(self, **kwargs):
        """Actualiza métricas."""
        self._metrics.update(kwargs)
        self._forward_count += 1



