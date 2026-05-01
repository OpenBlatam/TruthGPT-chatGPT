#!/usr/bin/env python3
"""
Paper: 2506.10848v2 (Best Techniques Paper)
============================================

Implementación específica basada en las mejores técnicas.
Este módulo implementa las técnicas específicas propuestas en este paper.

Basado en: https://arxiv.org/html/2506.10848v2
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
class Paper2506_10848v2Config:
    """Configuración específica para paper 2506.10848v2 (Best Techniques)."""
    hidden_dim: int = 512
    num_heads: int = 8
    best_technique_param: float = 1.0
    use_advanced_techniques: bool = True
    use_adaptive_layer_norm: bool = True
    use_gated_attention: bool = True


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization basado en paper 2506.10848v2.
    Técnica: Normalización adaptativa con parámetros aprendibles.
    
    Mejoras:
    - Validación de hidden_dim
    - Clamping de parámetros adaptativos
    - Métricas de adaptación
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"
        
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Adaptive parameters con mejor inicialización
        self.adaptive_scale = nn.Parameter(torch.ones(hidden_dim))
        self.adaptive_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Standard layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=eps)
        
        # Metrics
        self.register_buffer('scale_variance', torch.tensor(0.0))
        self.register_buffer('bias_mean', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptive layer normalization con mejoras.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
        """
        # Validation
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"Input hidden_dim ({x.size(-1)}) != configured ({self.hidden_dim})")
        
        # Standard normalization
        normalized = self.layer_norm(x)
        
        # Clamp adaptive parameters for stability
        scale = torch.clamp(self.adaptive_scale, min=0.1, max=10.0)
        bias = torch.clamp(self.adaptive_bias, min=-10.0, max=10.0)
        
        # Apply adaptive scaling
        output = normalized * scale + bias
        
        # Update metrics
        self.scale_variance = 0.9 * self.scale_variance + 0.1 * scale.var().item()
        self.bias_mean = 0.9 * self.bias_mean + 0.1 * bias.mean().item()
        
        return output
    
    def get_metrics(self) -> Dict[str, float]:
        """Get adaptive normalization metrics."""
        return {
            'scale_variance': self.scale_variance.item(),
            'bias_mean': self.bias_mean.item(),
            'scale_mean': self.adaptive_scale.mean().item(),
            'bias_std': self.adaptive_bias.std().item()
        }


class GatedAttention(nn.Module):
    """
    Gated Attention basado en paper 2506.10848v2.
    Técnica: Atención con gating mechanism.
    
    Mejoras:
    - Validación mejorada
    - Mejor inicialización
    - Soporte para attention masks
    - Métricas de gating
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projections con mejor inicialización
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Metrics
        self.register_buffer('gate_activation_rate', torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gated attention forward con mejoras.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
        """
        # Validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"Input hidden_dim ({x.size(-1)}) != configured ({self.hidden_dim})")
        
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Gating
        gate_values = torch.sigmoid(self.gate(x))
        gated_output = attn_output * gate_values
        
        # Update metrics
        gate_rate = gate_values.mean().item()
        self.gate_activation_rate = 0.9 * self.gate_activation_rate + 0.1 * gate_rate
        
        return self.out_proj(gated_output)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get gating metrics."""
        return {
            'gate_activation_rate': self.gate_activation_rate.item()
        }


class Paper2506_10848v2_BestTechniques(nn.Module):
    """
    Módulo implementando las mejores técnicas del paper 2506.10848v2.
    
    Técnicas implementadas:
    - Adaptive Layer Normalization
    - Gated Attention
    - Best practices optimizadas
    
    Basado en: https://arxiv.org/html/2506.10848v2
    """
    
    def __init__(self, config: Paper2506_10848v2Config):
        super().__init__()
        self.config = config
        
        # Adaptive layer norm
        if config.use_adaptive_layer_norm:
            self.adaptive_norm = AdaptiveLayerNorm(config.hidden_dim)
        else:
            self.adaptive_norm = nn.LayerNorm(config.hidden_dim)
        
        # Gated attention
        if config.use_gated_attention:
            self.gated_attn = GatedAttention(config.hidden_dim, config.num_heads)
        else:
            self.gated_attn = None
        
        # Feed-forward con mejor inicialización
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Initialize FFN weights
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        logger.info(f"Initialized Paper 2506.10848v2 Best Techniques with config: {config}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementando las mejores técnicas del paper.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Output tensor con mejores técnicas aplicadas
        """
        # Adaptive normalization
        x = self.adaptive_norm(x)
        
        # Gated attention
        residual = x
        if self.gated_attn is not None:
            x = self.gated_attn(x)
            x = residual + x
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = residual + x
        
        return x
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from all components."""
        metrics = {}
        
        if isinstance(self.adaptive_norm, AdaptiveLayerNorm):
            metrics.update(self.adaptive_norm.get_metrics())
        
        if self.gated_attn is not None:
            metrics.update(self.gated_attn.get_metrics())
        
        return metrics


class TruthGPT_Paper2506_10848v2_Integration(nn.Module):
    """Integración del paper 2506.10848v2 con TruthGPT."""
    
    def __init__(self, base_model, paper_config: Paper2506_10848v2Config):
        super().__init__()
        self.base_model = base_model
        self.best_techniques = Paper2506_10848v2_BestTechniques(paper_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass con mejores técnicas del paper."""
        output = self.base_model(*args, **kwargs)
        enhanced_output = self.best_techniques(output)
        return enhanced_output


if __name__ == "__main__":
    config = Paper2506_10848v2Config()
    module = Paper2506_10848v2_BestTechniques(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output = module(x)
    print(f"✅ Paper 2506.10848v2 Best Techniques test: Input {x.shape} -> Output {output.shape}")


