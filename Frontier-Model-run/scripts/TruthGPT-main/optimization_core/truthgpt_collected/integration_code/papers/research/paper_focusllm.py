#!/usr/bin/env python3
"""
FocusLLM: Scaling LLM's Context by Parallel Decoding
======================================================

(2024) FocusLLM: a framework to extend the context length via parallel decoding
Lee, Thu, etc.

Paper URL: https://arxiv.org/abs/[ID_PENDIENTE]
Nota: Paper de 2024, buscar en arXiv o repositorio del autor

Técnica principal:
- Divide input largo en "chunks" según ventana original
- Extrae información relevante de cada chunk con paralelismo
- Agrega resúmenes al contexto
- Permite contextos muy largos con bajo costo de entrenamiento

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. División en Chunks:
   - N_chunks = ceil(L / C) donde L es la longitud total y C es chunk_size
   - Chunk i: C_i = [x_{i·C}, ..., x_{(i+1)·C-1}]
   - Implementado en: FocusLLMModule.forward()

2. Extracción Paralela:
   - Para cada chunk C_i: S_i = Extract(C_i)
     donde Extract es una red que comprime el chunk
   - S_i tiene tamaño summary_size_per_chunk << chunk_size
   - Implementado en: ChunkExtractor.forward()

3. Contexto Extendido:
   - Contexto final: [x_0, ..., x_{C-1}, S_0, S_1, ..., S_{N-1}]
   - Longitud: C + N · summary_size donde N es el número de chunks
   - Permite procesar secuencias mucho más largas que C
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
class FocusLLMConfig(BasePaperConfig):
    """Configuración para FocusLLM."""
    base_context_length: int = 2048
    extended_context_length: int = 65536  # Puede ser muy largo
    chunk_size: int = 2048  # Tamaño de chunk (igual a base_context_length)
    use_parallel_decoding: bool = True
    extraction_dim: int = 256  # Dimensión para extracción
    num_extraction_layers: int = 2
    summary_size_per_chunk: int = 64  # Tamaño de resumen por chunk


class ChunkExtractor(nn.Module):
    """Extractor de información relevante de chunks."""
    
    def __init__(self, config: FocusLLMConfig):
        super().__init__()
        self.config = config
        
        # Capas de extracción
        extraction_layers = []
        for i in range(config.num_extraction_layers):
            extraction_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=8,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
            )
        self.extractor = nn.Sequential(*extraction_layers)
        
        # Proyección de resumen
        self.summary_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Extrae información relevante de un chunk.
        
        Args:
            chunk: [batch, chunk_size, hidden_dim]
        
        Returns:
            summary: [batch, summary_size, hidden_dim]
        """
        # Extraer
        extracted = self.extractor(chunk)  # [batch, chunk_size, hidden_dim]
        
        # Crear resumen
        summary_size = self.config.summary_size_per_chunk
        if extracted.size(1) > summary_size:
            # Reducir a summary_size
            summary = F.adaptive_avg_pool1d(
                extracted.transpose(1, 2),
                summary_size
            ).transpose(1, 2)  # [batch, summary_size, hidden_dim]
        else:
            # Padding
            summary = extracted
            if extracted.size(1) < summary_size:
                padding = torch.zeros(
                    extracted.size(0),
                    summary_size - extracted.size(1),
                    extracted.size(2),
                    device=extracted.device,
                    dtype=extracted.dtype
                )
                summary = torch.cat([extracted, padding], dim=1)
        
        # Proyectar
        summary = self.summary_proj(summary)
        
        return summary


class ParallelDecoder(nn.Module):
    """Decodificador paralelo para procesar múltiples chunks."""
    
    def __init__(self, config: FocusLLMConfig):
        super().__init__()
        self.config = config
        
        # Decodificador
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
    
    def forward(self, query: torch.Tensor, chunk_summaries: torch.Tensor) -> torch.Tensor:
        """
        Decodifica usando resúmenes de chunks.
        
        Args:
            query: [batch, query_len, hidden_dim]
            chunk_summaries: [batch, num_chunks * summary_size, hidden_dim]
        
        Returns:
            decoded: [batch, query_len, hidden_dim]
        """
        decoded = self.decoder(query, chunk_summaries)
        return decoded


class FocusLLMModule(BasePaperModule):
    """
    FocusLLM: Context Extension via Parallel Decoding.
    
    Características:
    - División en chunks
    - Extracción paralela de información
    - Agregación de resúmenes
    - Bajo costo de entrenamiento
    """
    
    def __init__(self, config: FocusLLMConfig):
        super().__init__(config)
        self.config = config
        
        # Extractor de chunks
        self.chunk_extractor = ChunkExtractor(config)
        
        # Decodificador paralelo
        if config.use_parallel_decoding:
            self.parallel_decoder = ParallelDecoder(config)
        else:
            self.parallel_decoder = None
        
        logger.info(f"FocusLLM initialized: {config.base_context_length} → {config.extended_context_length} tokens")
    
    def extract_chunks_parallel(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extrae información de chunks en paralelo.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            chunk_summaries: [batch, total_summary_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        chunk_size = self.config.chunk_size
        
        # Dividir en chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        chunk_summaries = []
        
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
            
            # Extraer información relevante
            summary = self.chunk_extractor(chunk)  # [batch, summary_size, hidden_dim]
            chunk_summaries.append(summary)
        
        # Concatenar todos los resúmenes
        all_summaries = torch.cat(chunk_summaries, dim=1)  # [batch, num_chunks * summary_size, hidden_dim]
        
        return all_summaries
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con FocusLLM.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            (output, metadata)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if seq_len <= self.config.base_context_length:
            # Contexto corto: usar directamente
            output = hidden_states
            num_chunks = 0
        else:
            # Contexto largo: usar FocusLLM
            # 1. Extraer información de chunks en paralelo
            chunk_summaries = self.extract_chunks_parallel(hidden_states)
            
            # 2. Usar primeros tokens como query
            query = hidden_states[:, :self.config.base_context_length, :]
            
            # 3. Decodificar con resúmenes de chunks
            if self.parallel_decoder is not None:
                decoded = self.parallel_decoder(query, chunk_summaries)
            else:
                # Sin decodificador: simplemente concatenar
                decoded = query
            
            # 4. Combinar query decodificada con resúmenes
            # Truncar resúmenes si es necesario para caber en base_context_length
            max_summary_len = self.config.base_context_length - decoded.size(1)
            if max_summary_len > 0 and chunk_summaries.size(1) > max_summary_len:
                chunk_summaries = chunk_summaries[:, :max_summary_len, :]
            
            # Concatenar
            if decoded.size(1) + chunk_summaries.size(1) <= self.config.base_context_length:
                output = torch.cat([decoded, chunk_summaries], dim=1)
            else:
                output = decoded
            
            num_chunks = (seq_len + self.config.chunk_size - 1) // self.config.chunk_size
        
        metadata = {
            'context_length': seq_len,
            'extended': seq_len > self.config.base_context_length,
            'num_chunks': num_chunks,
            'parallel_decoding': self.config.use_parallel_decoding,
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=seq_len,
            num_chunks=num_chunks
        )
        
        return output, metadata


if __name__ == "__main__":
    config = FocusLLMConfig(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=65536
    )
    
    module = FocusLLMModule(config)
    
    # Test
    hidden_states = torch.randn(2, 8192, config.hidden_dim)
    output, metadata = module(hidden_states)
    
    print(f"✅ FocusLLM test:")
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Num chunks: {metadata['num_chunks']}")



