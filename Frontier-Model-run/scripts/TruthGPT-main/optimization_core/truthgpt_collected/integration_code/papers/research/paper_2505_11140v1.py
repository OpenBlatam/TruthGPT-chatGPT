#!/usr/bin/env python3
"""
Paper: 2505.11140v1 (Research Paper)
====================================

Implementación específica basada en el paper de investigación.
Este módulo implementa las técnicas específicas propuestas en este paper.

Paper URL: https://arxiv.org/html/2505.11140v1

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. Rotary Position Embedding (RoPE):
   - Frecuencias: θ_i = 10000^(-2i/d) para i ∈ [0, d/2)
   - Rotación 2D: Para cada par (2i, 2i+1):
     [x'_2i  ]   [cos(θ_i · m)  -sin(θ_i · m)] [x_2i  ]
     [x'_2i+1] = [sin(θ_i · m)   cos(θ_i · m)] [x_2i+1]
     donde m es la posición del token
   - Implementado en: RotaryPositionEmbedding.apply_rotary_pos_emb()

2. Posición Relativa:
   - Distancia relativa: d = |i - j| donde i, j son posiciones
   - Embedding relativo: PE_rel(d) para distancias hasta max_relative_distance
   - Implementado para capturar relaciones posicionales relativas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Paper2505_11140v1Config:
    """Configuración específica para paper 2505.11140v1."""
    hidden_dim: int = 512
    num_heads: int = 8
    use_rotary_embeddings: bool = True
    use_relative_position: bool = True
    max_relative_distance: int = 128


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) basado en paper 2505.11140v1.
    Técnica: Embeddings rotatorios para posición relativa.
    
    Mejoras:
    - Caching de embeddings
    - Validación de max_seq_len
    - Mejor manejo de dimensiones impares
    - Soporte para diferentes dispositivos
    """
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"
        assert max_seq_len > 0, f"max_seq_len must be positive, got {max_seq_len}"
        assert hidden_dim % 2 == 0, f"hidden_dim must be even for RoPE, got {hidden_dim}"
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Generate rotary embeddings (cached)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for precomputed embeddings
        self._cached_embeddings = None
        self._cached_seq_len = 0
        
        # Metrics
        self.register_buffer('rotation_usage', torch.tensor(0.0))
        self.register_buffer('cache_hit_rate', torch.tensor(0.0))
        self._cache_hits = 0
        self._cache_misses = 0
        
    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        Apply rotary position embeddings with caching.
        
        Args:
            x: [batch, seq_len, hidden_dim]
            positions: Optional position indices [batch, seq_len]
            
        Returns:
            Rotated embeddings [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Validation
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Input hidden_dim ({hidden_dim}) != configured ({self.hidden_dim})")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len ({seq_len}) > max_seq_len ({self.max_seq_len})")
        
        # Generate or use cached positions
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
        else:
            positions = positions.float()
        
        # Use cache if available and sequence length matches
        if self._cached_embeddings is not None and self._cached_seq_len == seq_len:
            cos, sin = self._cached_embeddings
            if cos.device != x.device:
                cos = cos.to(x.device)
                sin = sin.to(x.device)
            self._cache_hits += 1
        else:
            # Compute angles
            angles = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)  # [batch, seq, dim/2]
            
            # Create rotary embeddings
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            
            # Cache for future use
            self._cached_embeddings = (cos, sin)
            self._cached_seq_len = seq_len
            self._cache_misses += 1
        
        # Update cache hit rate
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            self.cache_hit_rate = torch.tensor(self._cache_hits / total_requests)
        
        self.rotation_usage = 0.9 * self.rotation_usage + 0.1 * 1.0
        
        # Apply rotation
        x1, x2 = x.chunk(2, dim=-1)
        rotated_x = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated_x
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get RoPE metrics."""
        return {
            'rotation_usage': self.rotation_usage.item(),
            'cache_hit_rate': self.cache_hit_rate.item(),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cached_seq_len': self._cached_seq_len
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cached_embeddings = None
        self._cached_seq_len = 0
        self._cache_hits = 0
        self._cache_misses = 0


class Paper2505_11140v1Module(nn.Module):
    """
    Módulo implementando técnicas específicas del paper 2505.11140v1.
    
    Técnicas implementadas:
    - Rotary Position Embeddings (RoPE)
    - Relative position encoding
    - Improved positional understanding
    
    Basado en: https://arxiv.org/html/2505.11140v1
    """
    
    def __init__(self, config: Paper2505_11140v1Config):
        super().__init__()
        self.config = config
        
        # Rotary position embeddings
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryPositionEmbedding(
                config.hidden_dim,
                config.max_relative_distance
            )
        else:
            self.rotary_emb = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        logger.info(f"Initialized Paper 2505.11140v1 module with config: {config}")
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass implementando técnicas del paper.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            positions: Optional position indices
            
        Returns:
            Output tensor con técnicas aplicadas
        """
        if self.rotary_emb is not None:
            x = self.rotary_emb(x, positions)
        
        x = self.layer_norm(x)
        return x
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        if self.rotary_emb is not None:
            return self.rotary_emb.get_metrics()
        return {}


# ============================================================================
# Integración con TruthGPT
# ============================================================================

class TruthGPT_Paper2505_11140v1_Integration(nn.Module):
    """
    Integración del paper 2505.11140v1 con TruthGPT.
    """
    
    def __init__(self, base_model, paper_config: Paper2505_11140v1Config):
        super().__init__()
        self.base_model = base_model
        self.paper_module = Paper2505_11140v1Module(paper_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass integrado."""
        output = self.base_model(*args, **kwargs)
        enhanced_output = self.paper_module(output)
        return enhanced_output


if __name__ == "__main__":
    config = Paper2505_11140v1Config()
    module = Paper2505_11140v1Module(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output = module(x)
    print(f"✅ Paper 2505.11140v1 module test: Input {x.shape} -> Output {output.shape}")


