#!/usr/bin/env python3
"""
LongEmbed: Extending Embedding Models for Long Context Retrieval
==================================================================

Zhu, Wang, Yang, Song, Wu, Wei, Li (2024)
LongEmbed: Extending Embedding Models for Long Context Retrieval
Paper URL: https://arxiv.org/abs/2404.12096

Técnica principal:
- No es un LLM per se, sino modelo de embeddings para IR/RAG
- Extiende embeddings a ventanas largas (hasta 32K)
- Permite trabajar con documentos muy grandes en búsqueda + recuperación
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
import logging

try:
    from ..core.paper_base import BasePaperModule, BasePaperConfig
except (ImportError, ValueError):
    from paper_base import BasePaperModule, BasePaperConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LongEmbedConfig(BasePaperConfig):
    """Configuración para LongEmbed."""
    base_context_length: int = 512  # Típicamente más corto para embeddings
    extended_context_length: int = 32768  # 32K
    embedding_dim: int = 768  # Dimensión de embedding
    use_hierarchical_embedding: bool = True
    chunk_size: int = 512
    aggregation_method: str = "mean"  # "mean", "max", "attention"
    use_position_encoding: bool = True
    num_aggregation_layers: int = 2


class HierarchicalEmbedder(nn.Module):
    """Embedder jerárquico para documentos largos."""
    
    def __init__(self, config: LongEmbedConfig):
        super().__init__()
        self.config = config
        
        # Encoder por chunk
        chunk_encoder = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.chunk_encoder = nn.TransformerEncoder(chunk_encoder, num_layers=2)
        
        # Agregador
        if config.aggregation_method == "attention":
            self.aggregator = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=8,
                batch_first=True
            )
        else:
            self.aggregator = None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Crea embeddings jerárquicos.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            embeddings: [batch, embedding_dim] o [batch, num_chunks, embedding_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        chunk_size = self.config.chunk_size
        
        # Dividir en chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        chunk_embeddings = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk = hidden_states[:, start:end, :]
            
            # Padding si es necesario
            if chunk.size(1) < chunk_size:
                padding = torch.zeros(
                    batch_size, chunk_size - chunk.size(1), hidden_dim,
                    device=chunk.device, dtype=chunk.dtype
                )
                chunk = torch.cat([chunk, padding], dim=1)
            
            # Codificar chunk
            chunk_encoded = self.chunk_encoder(chunk)  # [batch, chunk_size, hidden_dim]
            
            # Agregar chunk
            if self.config.aggregation_method == "mean":
                chunk_embedding = chunk_encoded.mean(dim=1)  # [batch, hidden_dim]
            elif self.config.aggregation_method == "max":
                chunk_embedding = chunk_encoded.max(dim=1)[0]  # [batch, hidden_dim]
            else:  # attention
                # Usar primer token como query
                query = chunk_encoded[:, :1, :]  # [batch, 1, hidden_dim]
                aggregated, _ = self.aggregator(query, chunk_encoded, chunk_encoded)
                chunk_embedding = aggregated.squeeze(1)  # [batch, hidden_dim]
            
            chunk_embeddings.append(chunk_embedding)
        
        # Concatenar embeddings de chunks
        if len(chunk_embeddings) > 1:
            embeddings = torch.stack(chunk_embeddings, dim=1)  # [batch, num_chunks, hidden_dim]
        else:
            embeddings = chunk_embeddings[0].unsqueeze(1)  # [batch, 1, hidden_dim]
        
        return embeddings


class LongEmbedModule(BasePaperModule):
    """
    LongEmbed: Embedding Model para Long Context Retrieval.
    
    Características:
    - Embeddings para IR/RAG
    - Ventanas largas (hasta 32K)
    - Agregación jerárquica
    - Optimizado para recuperación
    """
    
    def __init__(self, config: LongEmbedConfig):
        super().__init__(config)
        self.config = config
        
        # Embedder jerárquico
        if config.use_hierarchical_embedding:
            self.hierarchical_embedder = HierarchicalEmbedder(config)
        else:
            self.hierarchical_embedder = None
        
        # Proyección final a embedding_dim
        self.embedding_proj = nn.Linear(config.hidden_dim, config.embedding_dim)
        
        # Encoding posicional (si se usa)
        if config.use_position_encoding:
            self.position_encoding = nn.Parameter(
                torch.randn(config.extended_context_length, config.hidden_dim) * 0.02
            )
        else:
            self.position_encoding = None
        
        logger.info(f"LongEmbed initialized: {config.base_context_length} → {config.extended_context_length} tokens")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        return_chunk_embeddings: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con LongEmbed.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            position_ids: [batch, seq_len] posiciones
            return_chunk_embeddings: Si retornar embeddings por chunk
        
        Returns:
            (embeddings, metadata)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Aplicar encoding posicional si está habilitado
        if self.position_encoding is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            
            # Obtener encodings posicionales
            pos_encodings = self.position_encoding[position_ids.clamp(0, len(self.position_encoding) - 1)]
            hidden_states = hidden_states + pos_encodings
        
        if seq_len <= self.config.base_context_length:
            # Contexto corto: embedding directo
            if self.hierarchical_embedder is None:
                # Agregación simple
                if self.config.aggregation_method == "mean":
                    embedding = hidden_states.mean(dim=1)  # [batch, hidden_dim]
                elif self.config.aggregation_method == "max":
                    embedding = hidden_states.max(dim=1)[0]
                else:
                    embedding = hidden_states[:, 0, :]  # Primer token
            else:
                # Usar embedder jerárquico
                chunk_embeddings = self.hierarchical_embedder(hidden_states)
                if return_chunk_embeddings:
                    embedding = chunk_embeddings
                else:
                    # Agregar chunks
                    embedding = chunk_embeddings.mean(dim=1)  # [batch, hidden_dim]
        else:
            # Contexto largo: usar embedder jerárquico
            if self.hierarchical_embedder is not None:
                chunk_embeddings = self.hierarchical_embedder(hidden_states)
                if return_chunk_embeddings:
                    embedding = chunk_embeddings
                else:
                    # Agregar chunks
                    embedding = chunk_embeddings.mean(dim=1)  # [batch, hidden_dim]
            else:
                # Fallback: agregación simple
                embedding = hidden_states.mean(dim=1)
        
        # Proyectar a embedding_dim
        if embedding.dim() == 2:
            # [batch, hidden_dim] -> [batch, embedding_dim]
            embedding = self.embedding_proj(embedding)
        else:
            # [batch, num_chunks, hidden_dim] -> [batch, num_chunks, embedding_dim]
            embedding = self.embedding_proj(embedding)
        
        metadata = {
            'context_length': seq_len,
            'extended': seq_len > self.config.base_context_length,
            'embedding_dim': self.config.embedding_dim,
            'hierarchical': self.config.use_hierarchical_embedding,
            'aggregation_method': self.config.aggregation_method,
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=seq_len,
            embedding_dim=self.config.embedding_dim
        )
        
        return embedding, metadata


if __name__ == "__main__":
    config = LongEmbedConfig(
        hidden_dim=768,
        embedding_dim=768,
        base_context_length=512,
        extended_context_length=32768
    )
    
    module = LongEmbedModule(config)
    
    # Test
    hidden_states = torch.randn(2, 4096, config.hidden_dim)
    embedding, metadata = module(hidden_states)
    
    print(f"✅ LongEmbed test:")
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Hierarchical: {metadata['hierarchical']}")



