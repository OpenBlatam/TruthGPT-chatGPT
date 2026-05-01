#!/usr/bin/env python3
"""
LIFT: Long Input Fine-Tuning
==============================

Mao, Xu, Li, et al. (2025)
LIFT: Improving Long Context Understanding of Large Language Models through Long Input Fine-Tuning
Paper URL: https://arxiv.org/abs/[ID_PENDIENTE]
Nota: Paper de 2025, buscar en arXiv o repositorio del autor

Técnica principal:
- Almacena información del input largo en parámetros mediante fine-tuning
- Módulo "Gated Memory" para balancear memoria a largo plazo vs ICL
- Mejora comprensión de contexto largo
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
class LIFTConfig(BasePaperConfig):
    """Configuración para LIFT."""
    base_context_length: int = 2048
    extended_context_length: int = 32768
    memory_size: int = 1024  # Tamaño de memoria gated
    memory_dim: int = 512  # Dimensión de memoria
    num_memory_layers: int = 2
    use_gated_memory: bool = True
    icl_balance_factor: float = 0.5  # Balance entre memoria y ICL
    fine_tune_steps: int = 5000


class GatedMemory(nn.Module):
    """Módulo de memoria gated para LIFT."""
    
    def __init__(self, config: LIFTConfig):
        super().__init__()
        self.config = config
        
        # Memoria a largo plazo
        self.long_term_memory = nn.Parameter(
            torch.randn(config.memory_size, config.memory_dim) * 0.02
        )
        
        # Gate para balancear memoria vs ICL
        self.memory_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.memory_dim),
            nn.Sigmoid()
        )
        
        # Proyección de memoria
        self.memory_proj = nn.Linear(config.memory_dim, config.hidden_dim)
        
        # Capas de procesamiento de memoria
        memory_layers = []
        for _ in range(config.num_memory_layers):
            memory_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=config.memory_dim,
                    nhead=8,
                    dim_feedforward=config.memory_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
            )
        self.memory_processor = nn.Sequential(*memory_layers) if memory_layers else nn.Identity()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de memoria gated.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            enhanced_states: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Calcular gate
        gate = self.memory_gate(hidden_states.mean(dim=1, keepdim=True))  # [batch, 1, memory_dim]
        
        # Procesar memoria
        memory = self.long_term_memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, memory_size, memory_dim]
        processed_memory = self.memory_processor(memory)  # [batch, memory_size, memory_dim]
        
        # Agregar memoria gated
        memory_context = (processed_memory * gate).sum(dim=1, keepdim=True)  # [batch, 1, memory_dim]
        memory_context = self.memory_proj(memory_context)  # [batch, 1, hidden_dim]
        
        # Combinar con hidden_states
        enhanced_states = hidden_states + memory_context.expand(-1, seq_len, -1)
        
        return enhanced_states


class LIFTModule(BasePaperModule):
    """
    LIFT: Long Input Fine-Tuning.
    
    Características:
    - Almacena información en parámetros
    - Gated Memory para balancear memoria vs ICL
    - Fine-tuning para contexto largo
    """
    
    def __init__(self, config: LIFTConfig):
        super().__init__(config)
        self.config = config
        
        # Gated Memory
        if config.use_gated_memory:
            self.gated_memory = GatedMemory(config)
        else:
            self.gated_memory = None
        
        # Compresor de contexto largo
        self.context_compressor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.memory_dim)
        )
        
        # Decompresor
        self.context_decompressor = nn.Sequential(
            nn.Linear(config.memory_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
        
        # Balanceador ICL vs Memoria
        self.icl_balance = nn.Parameter(torch.tensor(config.icl_balance_factor))
        
        logger.info(f"LIFT initialized: {config.base_context_length} → {config.extended_context_length} tokens")
    
    def compress_long_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Comprime contexto largo para almacenar en memoria.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            compressed: [batch, memory_size, memory_dim]
        """
        # Comprimir
        compressed = self.context_compressor(hidden_states)  # [batch, seq_len, memory_dim]
        
        # Agregar a memoria (promediar o max pooling)
        if compressed.size(1) > self.config.memory_size:
            # Reducir a memory_size
            compressed = F.adaptive_avg_pool1d(
                compressed.transpose(1, 2),
                self.config.memory_size
            ).transpose(1, 2)
        elif compressed.size(1) < self.config.memory_size:
            # Padding
            padding = torch.zeros(
                compressed.size(0),
                self.config.memory_size - compressed.size(1),
                compressed.size(2),
                device=compressed.device,
                dtype=compressed.dtype
            )
            compressed = torch.cat([compressed, padding], dim=1)
        
        return compressed
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con LIFT.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            (output, metadata)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if seq_len <= self.config.base_context_length:
            # Contexto corto: usar directamente
            output = hidden_states
            if self.gated_memory:
                output = self.gated_memory(output)
        else:
            # Contexto largo: usar LIFT
            # 1. Comprimir contexto largo
            compressed_context = self.compress_long_context(hidden_states)
            
            # 2. Actualizar memoria gated (si está habilitada)
            if self.gated_memory:
                # Actualizar memoria con contexto comprimido
                with torch.no_grad():
                    # Promediar contexto comprimido en memoria
                    memory_update = compressed_context.mean(dim=1, keepdim=True)  # [batch, 1, memory_dim]
                    # Actualizar memoria (simulado, en realidad se haría durante fine-tuning)
                    pass
                
                # Usar memoria gated
                output = self.gated_memory(hidden_states)
            else:
                output = hidden_states
            
            # 3. Descomprimir y combinar
            decompressed = self.context_decompressor(compressed_context)  # [batch, memory_size, hidden_dim]
            
            # Combinar con ICL balance
            icl_weight = torch.sigmoid(self.icl_balance)
            memory_weight = 1.0 - icl_weight
            
            # Usar primeros tokens como ICL
            icl_context = hidden_states[:, :self.config.base_context_length, :]
            
            # Combinar
            if decompressed.size(1) > 0:
                memory_context = decompressed.mean(dim=1, keepdim=True)  # [batch, 1, hidden_dim]
                output = icl_weight * icl_context + memory_weight * memory_context.expand(-1, icl_context.size(1), -1)
            else:
                output = icl_context
        
        metadata = {
            'context_length': seq_len,
            'extended': seq_len > self.config.base_context_length,
            'memory_used': self.config.use_gated_memory,
            'icl_balance': torch.sigmoid(self.icl_balance).item(),
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=seq_len,
            memory_used=self.config.use_gated_memory
        )
        
        return output, metadata


if __name__ == "__main__":
    config = LIFTConfig(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=32768
    )
    
    module = LIFTModule(config)
    
    # Test
    hidden_states = torch.randn(2, 4096, config.hidden_dim)
    output, metadata = module(hidden_states)
    
    print(f"✅ LIFT test:")
    print(f"   Output shape: {output.shape}")
    print(f"   Memory used: {metadata['memory_used']}")
    print(f"   ICL balance: {metadata['icl_balance']:.4f}")



