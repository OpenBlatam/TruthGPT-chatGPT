#!/usr/bin/env python3
"""
Long-Context Language Modeling via Semantic Compression
========================================================

Fei, Niu, Zhou, et al. (2023)
Extending Context Window of Large Language Models via Semantic Compression

Técnica principal:
- Comprime semánticamente inputs redundantes
- Usa modelo previo para identificar contenido relevante
- LLM "ve" más contenido relevante sin aumentar tokens directamente
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
import logging

from ..core.paper_base import BasePaperModule, BasePaperConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SemanticCompressionConfig(BasePaperConfig):
    """Configuración para Semantic Compression."""
    base_context_length: int = 2048
    extended_context_length: int = 16384
    compression_ratio: float = 0.25  # Ratio de compresión
    semantic_model_dim: int = 256  # Dimensión del modelo semántico
    use_redundancy_detection: bool = True
    redundancy_threshold: float = 0.7  # Umbral de redundancia
    num_compression_layers: int = 2


class RedundancyDetector(nn.Module):
    """Detector de redundancia semántica."""
    
    def __init__(self, config: SemanticCompressionConfig):
        super().__init__()
        self.config = config
        
        # Modelo para detectar redundancia
        self.redundancy_model = nn.Sequential(
            nn.Linear(config.hidden_dim, config.semantic_model_dim),
            nn.LayerNorm(config.semantic_model_dim),
            nn.GELU(),
            nn.Linear(config.semantic_model_dim, config.semantic_model_dim),
            nn.LayerNorm(config.semantic_model_dim),
            nn.GELU(),
            nn.Linear(config.semantic_model_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Detecta redundancia en hidden_states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            redundancy_scores: [batch, seq_len] scores de redundancia
        """
        scores = self.redundancy_model(hidden_states).squeeze(-1)  # [batch, seq_len]
        return scores


class SemanticCompressor(nn.Module):
    """Compresor semántico."""
    
    def __init__(self, config: SemanticCompressionConfig):
        super().__init__()
        self.config = config
        
        # Capas de compresión
        compression_layers = []
        for i in range(config.num_compression_layers):
            compression_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=8,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
            )
        self.compressor = nn.Sequential(*compression_layers)
        
        # Proyección final
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(self, hidden_states: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """
        Comprime hidden_states manteniendo solo contenido relevante.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            keep_mask: [batch, seq_len] máscara de tokens a mantener
        
        Returns:
            compressed: [batch, compressed_len, hidden_dim]
        """
        # Filtrar tokens redundantes
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Aplicar máscara
        keep_indices = keep_mask.nonzero(as_tuple=True)[1]  # Índices a mantener
        
        if len(keep_indices) == 0:
            # Si no hay tokens a mantener, usar promedio
            compressed = hidden_states.mean(dim=1, keepdim=True)
        else:
            # Seleccionar tokens relevantes
            relevant_tokens = hidden_states[:, keep_indices, :]  # [batch, num_relevant, hidden_dim]
            
            # Comprimir
            compressed = self.compressor(relevant_tokens)
            compressed = self.output_proj(compressed)
        
        return compressed


class SemanticCompressionModule(BasePaperModule):
    """
    Semantic Compression para contexto largo.
    
    Características:
    - Detecta y comprime redundancia semántica
    - Mantiene solo contenido relevante
    - Extiende contexto efectivo
    """
    
    def __init__(self, config: SemanticCompressionConfig):
        super().__init__(config)
        self.config = config
        
        # Detector de redundancia
        if config.use_redundancy_detection:
            self.redundancy_detector = RedundancyDetector(config)
        else:
            self.redundancy_detector = None
        
        # Compresor semántico
        self.semantic_compressor = SemanticCompressor(config)
        
        logger.info(f"Semantic Compression initialized: {config.base_context_length} → {config.extended_context_length} tokens")
    
    def detect_and_compress(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detecta redundancia y comprime.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            (compressed, keep_mask)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if self.redundancy_detector is None:
            # Sin detección: comprimir uniformemente
            target_len = int(seq_len * self.config.compression_ratio)
            keep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
            step = seq_len / target_len
            for i in range(target_len):
                idx = int(i * step)
                keep_mask[:, idx] = True
        else:
            # Detectar redundancia
            redundancy_scores = self.redundancy_detector(hidden_states)  # [batch, seq_len]
            
            # Crear máscara: mantener tokens con baja redundancia
            threshold = self.config.redundancy_threshold
            keep_mask = redundancy_scores < threshold  # [batch, seq_len]
            
            # Asegurar mínimo de tokens
            min_tokens = max(1, int(seq_len * self.config.compression_ratio))
            if keep_mask.sum() < min_tokens:
                # Mantener top-k tokens menos redundantes
                _, top_indices = redundancy_scores.topk(min_tokens, dim=1, largest=False)
                keep_mask = torch.zeros_like(keep_mask)
                keep_mask.scatter_(1, top_indices, True)
        
        # Comprimir
        compressed = self.semantic_compressor(hidden_states, keep_mask)
        
        return compressed, keep_mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con Semantic Compression.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            (output, metadata)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if seq_len <= self.config.base_context_length:
            # Contexto corto: usar directamente
            output = hidden_states
            compression_ratio = 1.0
        else:
            # Contexto largo: comprimir semánticamente
            compressed, keep_mask = self.detect_and_compress(hidden_states)
            
            # Combinar comprimido con tokens originales si es necesario
            compressed_len = compressed.size(1)
            
            if compressed_len < self.config.base_context_length:
                # Usar comprimido directamente
                output = compressed
                # Padding si es necesario
                if compressed_len < self.config.base_context_length:
                    padding = torch.zeros(
                        batch_size,
                        self.config.base_context_length - compressed_len,
                        hidden_dim,
                        device=compressed.device,
                        dtype=compressed.dtype
                    )
                    output = torch.cat([compressed, padding], dim=1)
            else:
                # Truncar a base_context_length
                output = compressed[:, :self.config.base_context_length, :]
            
            compression_ratio = compressed_len / seq_len
        
        metadata = {
            'context_length': seq_len,
            'extended': seq_len > self.config.base_context_length,
            'compression_ratio': compression_ratio,
            'redundancy_detection': self.config.use_redundancy_detection,
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=seq_len,
            compression_ratio=compression_ratio
        )
        
        return output, metadata


if __name__ == "__main__":
    config = SemanticCompressionConfig(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=16384
    )
    
    module = SemanticCompressionModule(config)
    
    # Test
    hidden_states = torch.randn(2, 4096, config.hidden_dim)
    output, metadata = module(hidden_states)
    
    print(f"✅ Semantic Compression test:")
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Compression ratio: {metadata['compression_ratio']:.4f}")



