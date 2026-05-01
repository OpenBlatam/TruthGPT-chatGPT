#!/usr/bin/env python3
"""
LongRoPE2: Near-Lossless LLM Context Window Scaling
====================================================

Shang, Zhang, Wang, et al. (2025)
Paper URL: https://arxiv.org/abs/2502.20082

Mejoras sobre LongRoPE:
- Búsqueda evolutiva ("needle-driven") para posicionamientos RoPE
- Entrenamiento mixto para mantener rendimiento en contextos cortos
- Mejor calidad que LongRoPE original
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
import numpy as np
import logging

try:
    from ..core.paper_base import BasePaperModule, BasePaperConfig
except (ImportError, ValueError):
    from paper_base import BasePaperModule, BasePaperConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LongRoPE2Config(BasePaperConfig):
    """Configuración para LongRoPE2."""
    base_context_length: int = 2048
    extended_context_length: int = 2048000
    rope_dim: int = 64
    use_evolutionary_search: bool = True
    evolutionary_population_size: int = 50
    evolutionary_generations: int = 20
    use_mixed_training: bool = True
    short_context_ratio: float = 0.5  # 50% contextos cortos en entrenamiento
    needle_positions: List[int] = None  # Posiciones "needle" para búsqueda
    
    def __post_init__(self):
        if self.needle_positions is None:
            # Posiciones estratégicas para búsqueda evolutiva
            self.needle_positions = [
                self.base_context_length // 4,
                self.base_context_length // 2,
                self.base_context_length * 3 // 4,
                self.base_context_length,
                self.extended_context_length // 2,
                self.extended_context_length
            ]


class LongRoPE2Module(BasePaperModule):
    """
    LongRoPE2: Versión mejorada con búsqueda evolutiva.
    
    Mejoras sobre LongRoPE:
    - Búsqueda evolutiva para optimizar posicionamientos
    - Entrenamiento mixto (cortos + largos)
    - Mejor calidad en contextos cortos
    """
    
    def __init__(self, config: LongRoPE2Config):
        super().__init__(config)
        self.config = config
        
        # Búsqueda evolutiva de posicionamientos
        if config.use_evolutionary_search:
            self.optimal_positions = self._evolutionary_search_positions()
        else:
            self.optimal_positions = self._compute_default_positions()
        
        # RoPE base
        self.rope_dim = config.rope_dim
        self.base_freqs = self._compute_base_frequencies()
        
        # Parámetros adaptativos
        self.position_embeddings = nn.Parameter(
            torch.randn(config.extended_context_length, config.hidden_dim) * 0.02
        )
        
        # Módulo de adaptación para contextos cortos
        if config.use_mixed_training:
            self.short_context_adapter = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
        
        logger.info(f"LongRoPE2 initialized with evolutionary search: {config.use_evolutionary_search}")
    
    def _compute_base_frequencies(self) -> torch.Tensor:
        """Calcula frecuencias base para RoPE."""
        freqs = 1.0 / (10000 ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        return freqs
    
    def _evolutionary_search_positions(self) -> torch.Tensor:
        """
        Búsqueda evolutiva ("needle-driven") para encontrar posicionamientos óptimos.
        
        Simula búsqueda evolutiva para optimizar posicionamientos RoPE.
        """
        logger.info("🔍 Running evolutionary search for optimal positions...")
        
        # Población inicial de posicionamientos
        population_size = self.config.evolutionary_population_size
        positions = torch.randn(population_size, self.config.extended_context_length) * 0.1
        
        # Función de fitness: minimizar pérdida de información en posiciones "needle"
        def fitness(pos_config):
            # Simular pérdida de información en posiciones clave
            needle_loss = 0.0
            for needle_pos in self.config.needle_positions:
                if needle_pos < len(pos_config):
                    # Pérdida basada en distancia del óptimo
                    optimal = needle_pos / self.config.extended_context_length
                    actual = pos_config[needle_pos] if needle_pos < len(pos_config) else 0
                    needle_loss += abs(optimal - actual)
            return -needle_loss  # Maximizar = minimizar pérdida
        
        # Evolución
        for generation in range(self.config.evolutionary_generations):
            # Evaluar fitness
            fitness_scores = [fitness(p) for p in positions]
            
            # Seleccionar mejores
            top_k = int(population_size * 0.3)
            top_indices = torch.tensor(fitness_scores).topk(top_k).indices
            
            # Crossover y mutación
            new_population = []
            for _ in range(population_size):
                parent1 = positions[top_indices[torch.randint(0, top_k, (1,))]]
                parent2 = positions[top_indices[torch.randint(0, top_k, (1,))]]
                
                # Crossover
                child = (parent1 + parent2) / 2
                
                # Mutación
                mutation_mask = torch.rand_like(child) < 0.1
                child[mutation_mask] += torch.randn_like(child[mutation_mask]) * 0.05
                
                new_population.append(child)
            
            positions = torch.stack(new_population)
        
        # Mejor configuración
        final_fitness = [fitness(p) for p in positions]
        best_idx = torch.tensor(final_fitness).argmax()
        optimal_positions = positions[best_idx]
        
        logger.info("✅ Evolutionary search completed")
        return optimal_positions
    
    def _compute_default_positions(self) -> torch.Tensor:
        """Calcula posicionamientos por defecto."""
        positions = torch.arange(self.config.extended_context_length, dtype=torch.float32)
        # Escalado suave
        ratio = self.config.extended_context_length / self.config.base_context_length
        return positions / ratio
    
    def apply_rope2(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Aplica RoPE con posicionamientos evolutivos."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Usar posicionamientos evolutivos
        if len(positions) <= len(self.optimal_positions):
            scaled_positions = self.optimal_positions[:len(positions)]
        else:
            # Interpolar para secuencias más largas
            scaled_positions = F.interpolate(
                self.optimal_positions.unsqueeze(0).unsqueeze(0),
                size=len(positions),
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Calcular frecuencias
        freqs = scaled_positions.unsqueeze(-1) * self.base_freqs.unsqueeze(0)
        
        # Aplicar rotación (similar a LongRoPE pero con posicionamientos optimizados)
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        
        # Aplicar rotación a hidden_states
        hidden_states_reshaped = hidden_states.view(batch_size, seq_len, hidden_dim // self.rope_dim, self.rope_dim)
        rotated = torch.zeros_like(hidden_states_reshaped)
        
        for i in range(self.rope_dim // 2):
            idx1, idx2 = i * 2, i * 2 + 1
            cos_val = cos_freqs[:, i].unsqueeze(-1)
            sin_val = sin_freqs[:, i].unsqueeze(-1)
            
            rotated[:, :, :, idx1] = (
                hidden_states_reshaped[:, :, :, idx1] * cos_val -
                hidden_states_reshaped[:, :, :, idx2] * sin_val
            )
            rotated[:, :, :, idx2] = (
                hidden_states_reshaped[:, :, :, idx1] * sin_val +
                hidden_states_reshaped[:, :, :, idx2] * cos_val
            )
        
        return rotated.view(batch_size, seq_len, hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        is_short_context: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con LongRoPE2.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            position_ids: [batch, seq_len] posiciones
            is_short_context: Si es contexto corto (aplicar adaptador)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # Aplicar LongRoPE2
        output = self.apply_rope2(hidden_states, position_ids[0])
        
        # Adaptador para contextos cortos (mantener rendimiento)
        if self.config.use_mixed_training and is_short_context:
            output = output + self.short_context_adapter(output)
        
        metadata = {
            'context_length': seq_len,
            'extended': seq_len > self.config.base_context_length,
            'is_short_context': is_short_context,
            'evolutionary_optimized': self.config.use_evolutionary_search,
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=seq_len,
            short_context_used=is_short_context
        )
        
        return output, metadata


if __name__ == "__main__":
    config = LongRoPE2Config(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=2048000,
        use_evolutionary_search=True
    )
    
    module = LongRoPE2Module(config)
    
    # Test
    hidden_states = torch.randn(2, 4096, config.hidden_dim)
    output, metadata = module(hidden_states, is_short_context=False)
    
    print(f"✅ LongRoPE2 test:")
    print(f"   Output shape: {output.shape}")
    print(f"   Evolutionary optimized: {metadata['evolutionary_optimized']}")



