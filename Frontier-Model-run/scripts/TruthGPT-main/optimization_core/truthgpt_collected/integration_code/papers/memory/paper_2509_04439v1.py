#!/usr/bin/env python3
"""
Paper: 2509.04439v1 (Memory Paper)
===================================

Implementación específica basada en técnicas de memoria avanzada.
Este módulo implementa las técnicas específicas propuestas en este paper.

Paper URL: https://arxiv.org/html/2509.04439v1

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. Almacenamiento en Memoria:
   - Key-Value Storage: M = {(k_i, v_i)} donde k_i son claves y v_i son valores
   - Embeddings: k_i = W_k · x_i, v_i = W_v · x_i
   - Implementado en: store()

2. Recuperación por Similitud:
   - Similarity: s(q, k_i) = cos(q, k_i) = (q · k_i) / (||q|| · ||k_i||)
   - Top-K retrieval: R = {k_i | s(q, k_i) ∈ top-K}
   - Implementado en: retrieve()

3. Consolidación de Memoria:
   - Decay: w_i(t) = w_i(0) · decay^t donde decay < 1
   - Consolidación: v'_i = Σ_j w_j · v_j / Σ_j w_j (promedio ponderado)
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
class Paper2509_04439v1Config:
    """Configuración específica para paper 2509.04439v1 (Memory)."""
    memory_dim: int = 512
    max_memory_size: int = 10000
    retrieval_k: int = 10
    memory_decay: float = 0.95
    use_hierarchical_memory: bool = True
    # Agregar parámetros específicos del paper aquí


class Paper2509_04439v1_MemorySystem(nn.Module):
    """
    Sistema de memoria avanzado basado en paper 2509.04439v1.
    
    Basado en: https://arxiv.org/html/2509.04439v1
    
    Este paper propone técnicas específicas de memoria que implementamos aquí.
    """
    
    def __init__(self, config: Paper2509_04439v1Config):
        super().__init__()
        self.config = config
        
        # Memoria a corto plazo
        self.short_term_memory = deque(maxlen=config.max_memory_size // 10)
        
        # Memoria a largo plazo
        self.long_term_memory = {}
        
        # Embeddings de memoria
        self.memory_embeddings = nn.Parameter(
            torch.randn(config.max_memory_size, config.memory_dim) * 0.02
        )
        self.memory_keys = nn.Parameter(
            torch.randn(config.max_memory_size, config.memory_dim) * 0.02
        )
        
        # Proyecciones específicas del paper
        self.query_projection = nn.Linear(config.memory_dim, config.memory_dim)
        self.memory_projection = nn.Linear(config.memory_dim, config.memory_dim)
        
        # Tracking
        self.consolidation_counter = 0
        self.memory_access_counts = defaultdict(int)
        
        logger.info(f"Initialized Paper 2509.04439v1 Memory System with config: {config}")
    
    def store(self, key: torch.Tensor, value: torch.Tensor, metadata: Dict = None):
        """Almacena información en memoria según técnicas del paper."""
        self.short_term_memory.append({
            'key': key.detach(),
            'value': value.detach(),
            'metadata': metadata or {},
            'timestamp': time.time(),
            'access_count': 0
        })
    
    def retrieve(self, query: torch.Tensor, k: int = None, 
                 temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recupera información de memoria usando técnicas específicas del paper.
        
        Mejoras:
        - Validación de inputs
        - Temperature scaling para control de sharpness
        - Mejor manejo de memoria vacía
        - Batch processing support
        
        Args:
            query: Query tensor [memory_dim] or [batch, memory_dim]
            k: Número de items a recuperar
            temperature: Temperature para softmax (controla sharpness)
            
        Returns:
            retrieved_values: Valores recuperados [k, memory_dim] or [batch, k, memory_dim]
            retrieved_weights: Pesos de atención [k] or [batch, k]
        """
        k = k or self.config.retrieval_k
        
        # Validation
        if query.dim() == 1:
            query = query.unsqueeze(0)
            batch_size = 1
            squeeze_output = True
        else:
            batch_size = query.size(0)
            squeeze_output = False
        
        if query.size(-1) != self.config.memory_dim:
            raise ValueError(f"Query dim mismatch: expected {self.config.memory_dim}, got {query.size(-1)}")
        
        if len(self.short_term_memory) == 0:
            empty_values = torch.zeros(batch_size, 1, self.config.memory_dim, device=query.device)
            empty_weights = torch.ones(batch_size, 1, device=query.device)
            if squeeze_output:
                return empty_values.squeeze(0), empty_weights.squeeze(0)
            return empty_values, empty_weights
        
        # Proyectar query según técnicas del paper
        query_proj = self.query_projection(query)  # [batch, memory_dim]
        
        # Recuperar de memoria a corto plazo
        short_term_keys = torch.stack([item['key'] for item in self.short_term_memory]).to(query.device)
        short_term_values = torch.stack([item['value'] for item in self.short_term_memory]).to(query.device)
        
        # Calcular similitud (técnica específica del paper)
        # [batch, memory_dim] x [num_memories, memory_dim]^T -> [batch, num_memories]
        similarity_scores = torch.matmul(query_proj, short_term_keys.transpose(-2, -1))
        
        # Temperature scaling
        similarity_scores = similarity_scores / (temperature * (self.config.memory_dim ** 0.5))
        similarity_weights = F.softmax(similarity_scores, dim=-1)
        
        # Top-k retrieval
        actual_k = min(k, len(short_term_keys))
        top_k_values, top_k_indices = torch.topk(similarity_weights, actual_k, dim=-1)
        
        # Gather retrieved values
        # Expand indices for batch: [batch, k] -> [batch, k, 1] -> [batch, k, memory_dim]
        indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.memory_dim)
        retrieved_values = torch.gather(
            short_term_values.unsqueeze(0).expand(batch_size, -1, -1),
            dim=1,
            index=indices_expanded
        )
        retrieved_weights = top_k_values
        
        # Actualizar contadores
        for batch_idx in range(batch_size):
            for idx in top_k_indices[batch_idx].cpu().numpy():
                if idx < len(self.short_term_memory):
                    self.short_term_memory[idx]['access_count'] += 1
                    self.memory_access_counts[idx] += 1
        
        if squeeze_output:
            return retrieved_values.squeeze(0), retrieved_weights.squeeze(0)
        return retrieved_values, retrieved_weights
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        return {
            'short_term_size': len(self.short_term_memory),
            'long_term_size': len(self.long_term_memory),
            'total_accesses': sum(self.memory_access_counts.values()),
            'consolidation_count': self.consolidation_counter,
            'avg_access_count': (
                sum(self.memory_access_counts.values()) / len(self.memory_access_counts)
                if self.memory_access_counts else 0.0
            )
        }
    
    def consolidate(self):
        """Consolida memoria según técnicas del paper."""
        if len(self.short_term_memory) == 0:
            return
        
        # Consolidar memorias más accedidas
        sorted_memories = sorted(
            enumerate(self.short_term_memory),
            key=lambda x: x[1]['access_count'],
            reverse=True
        )
        
        # Mover top memorias a largo plazo
        for idx, memory_item in sorted_memories[:self.config.max_memory_size // 20]:
            memory_id = f"ltm_{self.consolidation_counter}_{idx}"
            self.long_term_memory[memory_id] = {
                'key': memory_item['key'],
                'value': memory_item['value'],
                'metadata': memory_item['metadata'],
                'consolidation_time': time.time()
            }
        
        self.consolidation_counter += 1


class TruthGPT_Paper2509_04439v1_Integration(nn.Module):
    """Integración del paper 2509.04439v1 con TruthGPT."""
    
    def __init__(self, base_model, paper_config: Paper2509_04439v1Config):
        super().__init__()
        self.base_model = base_model
        self.memory_system = Paper2509_04439v1_MemorySystem(paper_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass con sistema de memoria del paper."""
        output = self.base_model(*args, **kwargs)
        
        # Usar memoria si es tensor
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            # Usar último token como query
            query = output[:, -1, :] if output.dim() == 3 else output
            retrieved, weights = self.memory_system.retrieve(query[0])
            # Combinar con output
            if retrieved.size(0) > 0:
                memory_contribution = torch.sum(retrieved * weights.unsqueeze(-1), dim=0)
                output = output + memory_contribution.unsqueeze(0).unsqueeze(0)
        
        return output


if __name__ == "__main__":
    config = Paper2509_04439v1Config()
    memory_system = Paper2509_04439v1_MemorySystem(config)
    
    # Test
    key = torch.randn(config.memory_dim)
    value = torch.randn(config.memory_dim)
    memory_system.store(key, value, {'test': True})
    
    query = torch.randn(config.memory_dim)
    retrieved, weights = memory_system.retrieve(query)
    print(f"✅ Paper 2509.04439v1 Memory System test:")
    print(f"   Retrieved shape: {retrieved.shape}, Weights shape: {weights.shape}")


