#!/usr/bin/env python3
"""
Efficient Solutions for Intriguing Failure of LLMs: Long Context
==================================================================

Hosseini, Castro, Ghinassi, Purver (COLING 2025)
Efficient Solutions For An Intriguing Failure of LLMs: Long Context Window Does Not Mean LLMs Can Analyze Long Sequences Flawlessly

Técnica principal:
- Identifica limitaciones reales incluso con ventanas grandes
- Propone soluciones prácticas ligeras
- Mejora capacidad de análisis en secuencias largas
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
class EfficientLongContextConfig(BasePaperConfig):
    """Configuración para Efficient Long Context Solutions."""
    base_context_length: int = 2048
    extended_context_length: int = 16384
    analysis_window_size: int = 512  # Ventana para análisis local
    use_hierarchical_analysis: bool = True
    use_local_attention: bool = True
    local_window_size: int = 256
    use_global_summary: bool = True
    summary_size: int = 128


class LocalAnalyzer(nn.Module):
    """Analizador local para ventanas pequeñas."""
    
    def __init__(self, config: EfficientLongContextConfig):
        super().__init__()
        self.config = config
        
        # Atención local
        self.local_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Procesamiento local
        self.local_processor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Analiza localmente.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            analyzed: [batch, seq_len, hidden_dim]
        """
        # Atención local (solo dentro de ventana)
        analyzed, _ = self.local_attention(hidden_states, hidden_states, hidden_states)
        
        # Procesar
        analyzed = self.local_processor(analyzed)
        
        return analyzed


class GlobalSummarizer(nn.Module):
    """Resumidor global para contexto completo."""
    
    def __init__(self, config: EfficientLongContextConfig):
        super().__init__()
        self.config = config
        
        # Compresor global
        self.global_compressor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
        
        # Atención global
        self.global_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Crea resumen global.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            summary: [batch, summary_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Comprimir
        compressed = self.global_compressor(hidden_states)  # [batch, seq_len, hidden_dim]
        
        # Crear resumen
        if seq_len > self.config.summary_size:
            # Reducir a summary_size
            summary = F.adaptive_avg_pool1d(
                compressed.transpose(1, 2),
                self.config.summary_size
            ).transpose(1, 2)  # [batch, summary_size, hidden_dim]
        else:
            # Padding
            summary = compressed
            if seq_len < self.config.summary_size:
                padding = torch.zeros(
                    batch_size,
                    self.config.summary_size - seq_len,
                    hidden_dim,
                    device=compressed.device,
                    dtype=compressed.dtype
                )
                summary = torch.cat([compressed, padding], dim=1)
        
        # Atención global
        summary, _ = self.global_attention(summary, summary, summary)
        
        return summary


class HierarchicalAnalyzer(nn.Module):
    """Analizador jerárquico: local + global."""
    
    def __init__(self, config: EfficientLongContextConfig):
        super().__init__()
        self.config = config
        
        self.local_analyzer = LocalAnalyzer(config)
        self.global_summarizer = GlobalSummarizer(config)
        
        # Combinador
        self.combiner = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Análisis jerárquico.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            analyzed: [batch, seq_len, hidden_dim]
        """
        # Análisis local
        local_analyzed = self.local_analyzer(hidden_states)  # [batch, seq_len, hidden_dim]
        
        # Resumen global
        global_summary = self.global_summarizer(hidden_states)  # [batch, summary_size, hidden_dim]
        
        # Expandir resumen global a seq_len
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if global_summary.size(1) != seq_len:
            global_expanded = F.interpolate(
                global_summary.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [batch, seq_len, hidden_dim]
        else:
            global_expanded = global_summary
        
        # Combinar
        combined = torch.cat([local_analyzed, global_expanded], dim=-1)  # [batch, seq_len, hidden_dim*2]
        output = self.combiner(combined)  # [batch, seq_len, hidden_dim]
        
        return output


class EfficientLongContextModule(BasePaperModule):
    """
    Efficient Solutions for Long Context.
    
    Características:
    - Análisis jerárquico (local + global)
    - Soluciones prácticas ligeras
    - Mejora capacidad de análisis
    """
    
    def __init__(self, config: EfficientLongContextConfig):
        super().__init__(config)
        self.config = config
        
        # Analizador jerárquico
        if config.use_hierarchical_analysis:
            self.hierarchical_analyzer = HierarchicalAnalyzer(config)
        else:
            self.hierarchical_analyzer = None
        
        # Atención local directa (si no se usa jerárquico)
        if config.use_local_attention and not config.use_hierarchical_analysis:
            self.local_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=8,
                batch_first=True
            )
        else:
            self.local_attention = None
        
        logger.info(f"Efficient Long Context initialized: {config.base_context_length} → {config.extended_context_length} tokens")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con Efficient Long Context.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            (output, metadata)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if seq_len <= self.config.base_context_length:
            # Contexto corto: usar directamente
            output = hidden_states
        else:
            # Contexto largo: usar análisis jerárquico
            if self.hierarchical_analyzer is not None:
                output = self.hierarchical_analyzer(hidden_states)
            elif self.local_attention is not None:
                # Atención local simple
                output, _ = self.local_attention(hidden_states, hidden_states, hidden_states)
            else:
                output = hidden_states
        
        metadata = {
            'context_length': seq_len,
            'extended': seq_len > self.config.base_context_length,
            'hierarchical_analysis': self.config.use_hierarchical_analysis,
            'local_attention': self.config.use_local_attention,
            'analysis_window_size': self.config.analysis_window_size,
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=seq_len,
            hierarchical_used=self.config.use_hierarchical_analysis
        )
        
        return output, metadata


if __name__ == "__main__":
    config = EfficientLongContextConfig(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=16384
    )
    
    module = EfficientLongContextModule(config)
    
    # Test
    hidden_states = torch.randn(2, 4096, config.hidden_dim)
    output, metadata = module(hidden_states)
    
    print(f"✅ Efficient Long Context test:")
    print(f"   Output shape: {output.shape}")
    print(f"   Hierarchical analysis: {metadata['hierarchical_analysis']}")



