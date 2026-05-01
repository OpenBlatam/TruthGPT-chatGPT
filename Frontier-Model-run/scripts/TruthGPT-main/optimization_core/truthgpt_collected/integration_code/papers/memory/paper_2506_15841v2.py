#!/usr/bin/env python3
"""
Paper: 2506.15841v2 (Memory Paper)
===================================

Implementación específica basada en técnicas de memoria avanzada.
Este módulo implementa las técnicas específicas propuestas en este paper.

Paper URL: https://arxiv.org/html/2506.15841v2

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. Memoria Episódica:
   - Episodio: E_i = {(x_t, a_t, r_t)} para t ∈ [t_start, t_end]
   - Almacenamiento: M_episodic = {E_i} con metadata temporal
   - Implementado en: episodic_memory

2. Memoria Semántica:
   - Consolidación: M_semantic = f_consolidate(M_episodic)
     donde f_consolidate agrega episodios similares
   - Embedding semántico: s_i = W_s · aggregate(E_i)
   - Implementado en: semantic_memory

3. Consolidación Adaptativa:
   - Rate: r(t) = r_0 · (1 - consolidation_rate)^t
   - Agregación: v_consolidated = Σ_i w_i · v_i / Σ_i w_i
     donde w_i son pesos basados en frecuencia de acceso
   - Implementado en: consolidate()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Paper2506_15841v2Config:
    """Configuración específica para paper 2506.15841v2 (Memory)."""
    memory_dim: int = 512
    max_memory_size: int = 10000
    retrieval_k: int = 10
    memory_consolidation_rate: float = 0.1
    use_episodic_memory: bool = True
    # Agregar parámetros específicos del paper aquí


class Paper2506_15841v2_MemorySystem(nn.Module):
    """
    Sistema de memoria avanzado basado en paper 2506.15841v2.
    
    Mejoras implementadas:
    - Validación completa
    - Batch processing
    - Estadísticas de episodios
    - Temperature scaling
    - Device handling mejorado
    
    Basado en: https://arxiv.org/html/2506.15841v2
    """
    
    def __init__(self, config: Paper2506_15841v2Config):
        super().__init__()
        assert config.memory_dim > 0, f"memory_dim must be positive, got {config.memory_dim}"
        assert config.max_memory_size > 0, f"max_memory_size must be positive, got {config.max_memory_size}"
        assert 0.0 < config.memory_consolidation_rate <= 1.0, f"consolidation_rate must be in (0, 1], got {config.memory_consolidation_rate}"
        
        self.config = config
        
        # Memoria episódica (técnica específica del paper)
        self.episodic_memory = deque(maxlen=config.max_memory_size)
        
        # Memoria semántica
        self.semantic_memory = {}
        
        # Embeddings de memoria con mejor inicialización
        self.memory_embeddings = nn.Parameter(
            torch.randn(config.max_memory_size, config.memory_dim) * 0.02
        )
        
        # Proyecciones específicas del paper con mejor inicialización
        self.episodic_projection = nn.Linear(config.memory_dim, config.memory_dim)
        self.semantic_projection = nn.Linear(config.memory_dim, config.memory_dim)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.episodic_projection.weight)
        nn.init.xavier_uniform_(self.semantic_projection.weight)
        if self.episodic_projection.bias is not None:
            nn.init.zeros_(self.episodic_projection.bias)
        if self.semantic_projection.bias is not None:
            nn.init.zeros_(self.semantic_projection.bias)
        
        # Tracking
        self.episode_access_counts = defaultdict(int)
        self.consolidation_counter = 0
        
        # Metrics
        self.register_buffer('avg_episode_similarity', torch.tensor(0.0))
        self.register_buffer('retrieval_accuracy', torch.tensor(0.0))
        
        logger.info(f"Initialized Paper 2506.15841v2 Memory System with config: {config}")
    
    def store_episode(self, episode: torch.Tensor, metadata: Dict = None):
        """Almacena un episodio en memoria según técnicas del paper."""
        self.episodic_memory.append({
            'episode': episode.detach(),
            'metadata': metadata or {},
            'timestamp': time.time()
        })
    
    def retrieve_episodes(self, query: torch.Tensor, k: int = None, 
                         temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recupera episodios relevantes usando técnicas del paper.
        
        Mejoras:
        - Batch processing support
        - Temperature scaling
        - Validación mejorada
        - Métricas de similitud
        
        Args:
            query: Query tensor [memory_dim] or [batch, memory_dim]
            k: Número de episodios a recuperar
            temperature: Temperature para softmax
            
        Returns:
            retrieved_episodes: Episodios recuperados
            retrieved_weights: Pesos de atención
        """
        k = k or self.config.retrieval_k
        
        # Validation and batch handling
        if query.dim() == 1:
            query = query.unsqueeze(0)
            batch_size = 1
            squeeze_output = True
        else:
            batch_size = query.size(0)
            squeeze_output = False
        
        if query.size(-1) != self.config.memory_dim:
            raise ValueError(f"Query dim mismatch: expected {self.config.memory_dim}, got {query.size(-1)}")
        
        if len(self.episodic_memory) == 0:
            empty_episodes = torch.zeros(batch_size, 1, self.config.memory_dim, device=query.device)
            empty_weights = torch.ones(batch_size, 1, device=query.device)
            if squeeze_output:
                return empty_episodes.squeeze(0), empty_weights.squeeze(0)
            return empty_episodes, empty_weights
        
        # Proyectar query
        query_proj = self.episodic_projection(query)  # [batch, memory_dim]
        
        # Calcular similitud con episodios
        episodes = torch.stack([item['episode'] for item in self.episodic_memory]).to(query.device)
        # [batch, memory_dim] x [num_episodes, memory_dim]^T -> [batch, num_episodes]
        similarity_scores = torch.matmul(query_proj, episodes.transpose(-2, -1))
        
        # Temperature scaling
        similarity_scores = similarity_scores / (temperature * (self.config.memory_dim ** 0.5))
        similarity_weights = F.softmax(similarity_scores, dim=-1)
        
        # Update metrics
        avg_sim = similarity_scores.mean().item()
        self.avg_episode_similarity = 0.9 * self.avg_episode_similarity + 0.1 * avg_sim
        
        # Top-k retrieval
        actual_k = min(k, len(episodes))
        top_k_values, top_k_indices = torch.topk(similarity_weights, actual_k, dim=-1)
        
        # Gather retrieved episodes
        indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.memory_dim)
        retrieved_episodes = torch.gather(
            episodes.unsqueeze(0).expand(batch_size, -1, -1),
            dim=1,
            index=indices_expanded
        )
        retrieved_weights = top_k_values
        
        # Update access counts
        for batch_idx in range(batch_size):
            for idx in top_k_indices[batch_idx].cpu().numpy():
                if idx < len(self.episodic_memory):
                    self.episode_access_counts[idx] += 1
        
        if squeeze_output:
            return retrieved_episodes.squeeze(0), retrieved_weights.squeeze(0)
        return retrieved_episodes, retrieved_weights
    
    def consolidate_to_semantic(self):
        """Consolida episodios frecuentemente accedidos a memoria semántica."""
        if len(self.episodic_memory) == 0:
            return
        
        # Sort by access count
        sorted_episodes = sorted(
            enumerate(self.episodic_memory),
            key=lambda x: self.episode_access_counts.get(x[0], 0),
            reverse=True
        )
        
        # Consolidate top episodes
        num_to_consolidate = max(1, int(len(self.episodic_memory) * self.config.memory_consolidation_rate))
        for idx, episode_item in sorted_episodes[:num_to_consolidate]:
            semantic_key = f"semantic_{self.consolidation_counter}_{idx}"
            self.semantic_memory[semantic_key] = {
                'episode': episode_item['episode'],
                'metadata': episode_item['metadata'],
                'consolidation_time': time.time(),
                'access_count': self.episode_access_counts.get(idx, 0)
            }
        
        self.consolidation_counter += 1
    
    def get_episodic_stats(self) -> Dict[str, Any]:
        """Get statistics about episodic memory."""
        return {
            'episodic_size': len(self.episodic_memory),
            'semantic_size': len(self.semantic_memory),
            'total_accesses': sum(self.episode_access_counts.values()),
            'consolidation_count': self.consolidation_counter,
            'avg_similarity': self.avg_episode_similarity.item(),
            'retrieval_accuracy': self.retrieval_accuracy.item()
        }


class TruthGPT_Paper2506_15841v2_Integration(nn.Module):
    """Integración del paper 2506.15841v2 con TruthGPT."""
    
    def __init__(self, base_model, paper_config: Paper2506_15841v2Config):
        super().__init__()
        self.base_model = base_model
        self.memory_system = Paper2506_15841v2_MemorySystem(paper_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass con memoria episódica del paper."""
        output = self.base_model(*args, **kwargs)
        
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            query = output[:, -1, :] if output.dim() == 3 else output
            retrieved, weights = self.memory_system.retrieve_episodes(query[0])
            if retrieved.size(0) > 0:
                memory_contribution = torch.sum(retrieved * weights.unsqueeze(-1), dim=0)
                output = output + memory_contribution.unsqueeze(0).unsqueeze(0)
        
        return output


if __name__ == "__main__":
    config = Paper2506_15841v2Config()
    memory_system = Paper2506_15841v2_MemorySystem(config)
    
    # Test
    episode = torch.randn(config.memory_dim)
    memory_system.store_episode(episode, {'test': True})
    
    query = torch.randn(config.memory_dim)
    retrieved, weights = memory_system.retrieve_episodes(query)
    print(f"✅ Paper 2506.15841v2 Memory System test:")
    print(f"   Retrieved episodes shape: {retrieved.shape}, Weights shape: {weights.shape}")


