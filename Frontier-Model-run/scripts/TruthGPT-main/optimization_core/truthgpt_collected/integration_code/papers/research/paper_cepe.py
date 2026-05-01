#!/usr/bin/env python3
"""
CEPE: Context Expansion with Parallel Encoding
===============================================

Yen, Gao, Chen (2024). Long-Context Language Modeling with Parallel Context Encoding.

Paper URL: https://arxiv.org/abs/2402.16617

Técnica principal:
- Encoder pequeño que procesa tokens en "chunks" largos
- Decodificador con cross-attention para incorporar contexto extendido
- Extiende ventana hasta 128K en LLaMA-2 sin elevar mucho costo de memoria

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. Codificación Paralela de Chunks:
   - Para cada chunk C_i de tamaño chunk_size:
     E_i = Encoder_small(W_e · C_i)
     donde W_e es proyección de hidden_dim a encoder_hidden_dim
   - Implementado en: encode_chunks()
   - Ecuación: encoded = TransformerEncoder(Linear(hidden_states))

2. Cross-Attention para Incorporar Contexto:
   - H' = Decoder(H, E)
     donde H son los hidden states originales,
     E son los chunks codificados
   - Implementado en: forward() usando TransformerDecoder
   - Ecuación: output = TransformerDecoder(query=H, memory=E)

3. Compresión de Memoria:
   - Ratio de compresión: r = encoder_hidden_dim / hidden_dim
   - Memoria reducida: O(N · d_enc) vs O(N · d) donde d_enc << d
   - Permite procesar secuencias mucho más largas
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
class CEPEConfig(BasePaperConfig):
    """Configuración para CEPE."""
    base_context_length: int = 2048
    extended_context_length: int = 131072  # 128K
    chunk_size: int = 4096  # Tamaño de chunk para procesamiento paralelo
    encoder_hidden_dim: int = 256  # Encoder más pequeño
    num_encoder_layers: int = 2
    num_decoder_layers: int = 1
    num_attention_heads: int = 8
    use_parallel_encoding: bool = True
    compression_ratio: float = 0.25  # Ratio de compresión del encoder


class CEPEModule(BasePaperModule):
    """
    CEPE: Context Expansion with Parallel Encoding.
    
    Características:
    - Encoder pequeño para chunks largos
    - Cross-attention para incorporar contexto
    - Bajo costo de memoria
    """
    
    def __init__(self, config: CEPEConfig):
        """
        Inicialización del módulo CEPE.
        
        EN EL PAPER: Sección 3 - Parallel Context Encoding
        - El paper propone un encoder pequeño que procesa chunks largos en paralelo
        - FÓRMULA: E_i = Encoder_small(W_e · C_i)
          donde C_i es un chunk, W_e es proyección, y Encoder_small es pequeño
        - Luego usa cross-attention para incorporar el contexto extendido
        - Esto permite hasta 128K tokens en LLaMA-2 sin mucho costo de memoria
        
        CÓDIGO: Inicializamos:
        1. Encoder pequeño (dimensión reducida)
        2. Proyecciones de entrada/salida
        3. Decodificador con cross-attention
        """
        super().__init__(config)
        self.config = config
        
        # EN EL PAPER: Sección 3.1 - Small Encoder Architecture
        # El paper usa un encoder pequeño para reducir costo de memoria
        # FÓRMULA: encoder_hidden_dim << hidden_dim (ej: 256 vs 768)
        # FÓRMULA: Memoria = O(N · d_enc) vs O(N · d) donde d_enc << d
        # CÓDIGO: TransformerEncoder con dimensión reducida
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.encoder_hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.chunk_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # EN EL PAPER: Sección 3.1.1 - Input Projection
        # El paper proyecta hidden states a dimensión del encoder
        # FÓRMULA: h_enc = W_e · h donde W_e ∈ R^(d × d_enc)
        # CÓDIGO: Capa lineal que reduce dimensión
        self.encoder_input_proj = nn.Linear(config.hidden_dim, config.encoder_hidden_dim)
        
        # EN EL PAPER: Sección 3.2 - Cross-Attention Decoder
        # El paper usa decodificador con cross-attention para incorporar contexto
        # FÓRMULA: H' = Decoder(H_query, E_memory)
        # donde H_query son tokens originales y E_memory son chunks codificados
        # CÓDIGO: TransformerDecoder con cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.context_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers
        )
        
        # EN EL PAPER: Sección 3.1.2 - Output Projection
        # El paper proyecta de vuelta a dimensión original
        # FÓRMULA: h_out = W_o · h_enc donde W_o ∈ R^(d_enc × d)
        # CÓDIGO: Capa lineal que restaura dimensión
        self.encoder_output_proj = nn.Linear(config.encoder_hidden_dim, config.hidden_dim)
        
        logger.info(f"CEPE initialized: {config.base_context_length} → {config.extended_context_length} tokens")
    
    def encode_chunks(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Codifica chunks en paralelo.
        
        EN EL PAPER: Sección 3.1 - Parallel Chunk Encoding
        - El paper divide la secuencia en chunks y los procesa en paralelo
        - FÓRMULA: Para cada chunk C_i: E_i = Encoder_small(W_e · C_i)
        - FÓRMULA: Summary_i = mean(E_i) (promedio del chunk codificado)
        - Esto reduce memoria de O(N²) a O(N · chunk_size²)
        
        CÓDIGO: Implementamos el método del paper:
        1. Dividir en chunks de tamaño chunk_size
        2. Codificar cada chunk con encoder pequeño
        3. Promediar para obtener resumen comprimido
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            encoded_chunks: [batch, num_chunks, hidden_dim]
        """
        # NOTACIÓN DEL PAPER: H ∈ R^(B×N×d) donde B=batch, N=seq_len, d=hidden_dim
        # NOTACIÓN EN CÓDIGO: hidden_states ∈ R^(B×N×d)
        B, N, d = hidden_states.shape
        C = self.config.chunk_size  # tamaño de chunk
        
        # PASO 1: Dividir secuencia en chunks
        # NOTACIÓN DEL PAPER: N_chunks = ⌈N / C⌉
        # NOTACIÓN EN CÓDIGO: num_chunks = ⌈N / C⌉
        N_chunks = (N + C - 1) // C
        chunks_list = []
        
        for i in range(N_chunks):
            start_idx = i * C
            end_idx = min(start_idx + C, N)
            C_i = hidden_states[:, start_idx:end_idx, :]  # C_i ∈ R^(B×C_i×d)
            
            # NOTACIÓN: Padding para chunks incompletos
            if C_i.size(1) < C:
                padding = torch.zeros(
                    B, C - C_i.size(1), d,
                    device=C_i.device, dtype=C_i.dtype
                )
                C_i = torch.cat([C_i, padding], dim=1)  # C_i ∈ R^(B×C×d)
            
            chunks_list.append(C_i)
        
        # PASO 2: Procesar chunks en paralelo
        # NOTACIÓN DEL PAPER: Para cada chunk C_i:
        #   E_i = Encoder_small(W_e · C_i) donde W_e ∈ R^(d×d_enc)
        #   Summary_i = (1/C) Σ_{j=0}^{C-1} E_i[j] donde E_i[j] ∈ R^d
        # NOTACIÓN EN CÓDIGO: 
        #   C_i ∈ R^(B×C×d), E_i ∈ R^(B×C×d_enc), Summary_i ∈ R^(B×1×d)
        summaries = []
        for C_i in chunks_list:
            # NOTACIÓN: Proyección de entrada h_enc = C_i · W_e^T
            h_enc = self.encoder_input_proj(C_i)  # h_enc ∈ R^(B×C×d_enc)
            
            # NOTACIÓN: Encoder pequeño E_i = TransformerEncoder(h_enc)
            E_i = self.chunk_encoder(h_enc)  # E_i ∈ R^(B×C×d_enc)
            
            # NOTACIÓN: Proyección de salida h_out = E_i · W_o^T
            h_out = self.encoder_output_proj(E_i)  # h_out ∈ R^(B×C×d)
            
            # NOTACIÓN: Resumen por promedio Summary_i = (1/C) Σ_j h_out[j]
            Summary_i = h_out.mean(dim=1, keepdim=True)  # Summary_i ∈ R^(B×1×d)
            summaries.append(Summary_i)
        
        # PASO 3: Concatenar resúmenes
        # NOTACIÓN DEL PAPER: Context = [Summary_0, Summary_1, ..., Summary_{N_chunks-1}]
        # NOTACIÓN EN CÓDIGO: context_summary[b, i, :] = Summary_i
        #   context_summary ∈ R^(B×N_chunks×d)
        context_summary = torch.cat(summaries, dim=1)  # [B, N_chunks, d]
        
        return context_summary
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con CEPE.
        
        EN EL PAPER: Sección 3 - Method Overview
        - El paper procesa contextos largos dividiéndolos en chunks
        - FÓRMULA: Si seq_len <= L_base: usar directamente
        - FÓRMULA: Si seq_len > L_base:
          1. E = encode_chunks(H)  (codificar chunks)
          2. H' = Decoder(H_query, E)  (cross-attention)
        - Esto permite hasta 128K tokens sin mucho costo
        
        CÓDIGO: Implementamos el método completo del paper
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            (output, metadata)
        """
        # NOTACIÓN DEL PAPER: H ∈ R^(B×N×d) donde B=batch, N=seq_len, d=hidden_dim
        B, N, d = hidden_states.shape
        H = hidden_states  # H ∈ R^(B×N×d)
        L_base = self.config.base_context_length
        
        # PASO 1: Verificar si necesitamos CEPE
        # NOTACIÓN: Si N ≤ L_base, no necesitamos procesamiento adicional
        if N <= L_base:
            # NOTACIÓN: h' = H (sin cambios)
            h_prime = H  # h' ∈ R^(B×N×d)
            E = None  # No hay contexto extendido
        else:
            # PASO 2: Codificar chunks en paralelo
            # NOTACIÓN DEL PAPER: E = encode_chunks(H) donde E ∈ R^(B×N_chunks×d)
            #   E = [Summary_0, Summary_1, ..., Summary_{N_chunks-1}]
            # NOTACIÓN EN CÓDIGO: E[b, i, :] = Summary_i
            E = self.encode_chunks(H)  # E ∈ R^(B×N_chunks×d)
            
            # PASO 3: Usar primeros tokens como query
            # NOTACIÓN DEL PAPER: H_query = H[:, 0:L_base, :] (primeros L_base tokens)
            # NOTACIÓN EN CÓDIGO: H_query ∈ R^(B×L_base×d)
            H_query = H[:, :L_base, :]  # H_query ∈ R^(B×L_base×d)
            
            # PASO 4: Cross-Attention Decoder
            # NOTACIÓN DEL PAPER: h' = Decoder(H_query, E)
            #   donde Decoder usa cross-attention: Attention(Q=H_query, K=E, V=E)
            # NOTACIÓN EN CÓDIGO: 
            #   h_prime[b, i, :] = Decoder(H_query[b, i, :], E[b, :, :])
            #   h_prime ∈ R^(B×L_base×d)
            h_prime = self.context_decoder(H_query, E)  # h' = Decoder(H_query, E)
            
            # PASO 5: Opcional - mantener tokens restantes
            # NOTACIÓN: Si N > L_base, concatenamos tokens restantes
            if N > L_base:
                H_remaining = H[:, L_base:, :]  # H_remaining ∈ R^(B×(N-L_base)×d)
                h_prime = torch.cat([h_prime, H_remaining], dim=1)  # h' ∈ R^(B×N×d)
        
        # Metadata para tracking
        N_chunks = (N + self.config.chunk_size - 1) // self.config.chunk_size if N > L_base else 0
        metadata = {
            'context_length': N,
            'extended': N > L_base,
            'num_chunks': N_chunks,
            'compression_ratio': self.config.compression_ratio,
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=N,
            num_chunks=N_chunks
        )
        
        return h_prime, metadata  # h' ∈ R^(B×N×d)


if __name__ == "__main__":
    config = CEPEConfig(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=131072
    )
    
    module = CEPEModule(config)
    
    # Test con contexto largo
    hidden_states = torch.randn(2, 8192, config.hidden_dim)
    output, metadata = module(hidden_states)
    
    print(f"✅ CEPE test:")
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Num chunks: {metadata['num_chunks']}")



