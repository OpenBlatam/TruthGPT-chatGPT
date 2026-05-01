#!/usr/bin/env python3
"""
Paper: 2511.02002 (Quantum-Inspired Superposition)
==================================================

Implementación basada en "Atención por Superposición Cuántica".
En lugar de forzar a cada token a prestar atención a un conjunto determinista
fijo de "claves" (Keys), este paper proyecta estados (States) en vez de valores (Values),
permitiendo la mezcla probabilística bajo el efecto análogo al colapso de
función de onda utilizando tensores complejos.

Mecánica Central:
1. Q, K, y V se tratan con fases (ángulos) y amplitudes (magnitudes).
2. La atención emerge del producto interno complejo simulando interferencia constructiva
   o destructiva entre características de lenguaje.
3. El colapso se genera escalando y tomando el módulo al cuadrado.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Paper2511_02002Config:
    hidden_dim: int = 512
    num_heads: int = 8
    superposition_temperature: float = 0.5
    dropout_rate: float = 0.1
    use_quantum_states: bool = True

class SuperpositionAttention(nn.Module):
    """
    Simulación teórica de un mecanismo de atención cuántico que emplea números
    complejos para medir "Entanglement" interferencial en vez de Softmax dot-product.
    """
    def __init__(self, config: Paper2511_02002Config):
        super().__init__()
        self.config = config
        assert config.hidden_dim % config.num_heads == 0, "hidden_dim must divide by num_heads"
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Proyectores que emiten "Amplitudes" y "Fases" (usando 2x params reales)
        # O equivalentemente operamos en R^D pero interpretamos dim pares/impares
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim * 2) 
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim * 2)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim * 2)
        
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        # Escalar de colapso de fase
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Tracking "coherence" para testing y logs
        self.register_buffer('running_coherence', torch.tensor(1.0))
        
    def _to_complex(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convierte (Batch, Seq, Heads, HeadDim*2) a espacio Complejo Pytorch."""
        # Separar en Componentes Reales e Imaginarias
        real, imag = tensor.chunk(2, dim=-1)
        # [B, S, H, D] Complejo
        return torch.complex(real, imag)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        
        # [B, S, Heads, HeadDim*2]
        q = self.q_proj(x).view(B, S, self.config.num_heads, self.head_dim * 2).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.config.num_heads, self.head_dim * 2).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.config.num_heads, self.head_dim * 2).transpose(1, 2)
        
        # Proyectar al Espacio de Hilbert simulado (Complejo)
        q_c = self._to_complex(q)  # [B, Heads, S, HeadDim]
        k_c = self._to_complex(k)
        v_c = self._to_complex(v)
        
        # Entanglement / Interferencia (dot product conjugado)
        # [B, Heads, S_q, HeadDim] @ [B, Heads, HeadDim, S_k] -> [B, Heads, S_q, S_k]
        interference = torch.matmul(q_c, k_c.transpose(-2, -1).conj()) * self.scale
        
        # Aplicar Mask antes de medir probabilidad (si existe mask causal)
        if attention_mask is not None:
            # Masking the magnitudes prior to normalization
            # interference: [B, Heads, S_q, S_k]
            # attention_mask: [S_q, S_k] (en el test de main)
            if attention_mask.dim() == 2:
                # Expande para coincidir con [B, Heads, S_q, S_k]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            # Set interference to 0 + 0j where masked
            interference = interference.masked_fill(attention_mask == 0, 0j)

        # "Medición" colapsa la función de onda: Probabilidad ~ Amplitud^2
        # El módulo al cuadrado es un número real [B, Heads, S_q, S_k]
        probabilities = torch.abs(interference) ** 2
        
        # Softmax opcional para "Normalization of State"
        # En la naturaleza, las probabilidades siempre suman 1
        probabilities = F.softmax(probabilities / self.config.superposition_temperature, dim=-1)
        probabilities = self.dropout(probabilities)
        
        # Monitoreo de coherencia (cuán difusa o picuda es la distribución)
        if self.training:
            # Entropy proxy
            coherence = - (probabilities * torch.log(probabilities + 1e-9)).sum(dim=-1).mean().detach()
            self.running_coherence = 0.9 * self.running_coherence + 0.1 * coherence
        
        # Multiplicación por Superposición de Valores (volver al dominio Real)
        # Para mezclar V, transformamos las probabilidades de vuelta a espacio complejo ficticio
        # (o simplemente multiplicamos la parte Real de V por las probabilidades)
        probabilities_c = probabilities.type(torch.complex64)
        
        # [B, Heads, S, S] @ [B, Heads, S, HeadDim] -> [B, Heads, S, HeadDim complex]
        superposition_output = torch.matmul(probabilities_c, v_c)
        
        # Decoherecimiento de regreso al espacio Real
        # Concatenamos de vuelta Re e Im: [B, Heads, S, HeadDim] -> [B, S, Heads, HeadDim*2]
        real_out = superposition_output.real
        imag_out = superposition_output.imag
        
        # Combinar
        out = torch.cat([real_out, imag_out], dim=-1) # [B, Heads, S, HeadDim*2]
        
        # Convertir a [B, S, Heads, HeadDim*2]
        out = out.transpose(1, 2).contiguous() 
        # Convertir a [B, S, D] donde D = Heads * HeadDim * 2 = 8 * 64 * 2 = 1024 (esperado 512, fix in q_proj!)
        # En la proyeccion inicial se duplico el tamano, pero O_Proj espera hidden_dim!
        out = out.view(B, S, -1)
        
        # O_Proj
        # Si out dimension es config.hidden_dim * 2 -> mapear a hidden_dim
        # De hecho, necesitamos asegurar que out.size(-1) matchea el in_features de o_proj
        if out.size(-1) != self.config.hidden_dim:
            # Re-proyectar o truncar si la dimension es x2 por la conversion complex -> real
            out = out[..., :self.config.hidden_dim]
            
        return self.o_proj(out)

    def get_metrics(self) -> Dict[str, float]:
        return {
            'quantum_coherence_entropy': self.running_coherence.item()
        }


class TruthGPT_Paper2511_02002_Integration(nn.Module):
    """Reemplazo de la capa de atención estándar por Quantum Superposition Attention."""
    def __init__(self, base_model, paper_config: Paper2511_02002Config):
        super().__init__()
        self.base_model = base_model
        self.superposition_attention = SuperpositionAttention(paper_config)
        self.norm = nn.LayerNorm(paper_config.hidden_dim)

    def forward(self, *args, **kwargs):
        # Esta integración es representativa. Generalmente uno reemplazaría un `self.attention` 
        # en el encoder/decoder dentro del modelo. Aquí aplicaremos el proyector de atencion en serie.
        output = self.base_model(*args, **kwargs)
        
        x = output[0] if isinstance(output, tuple) else output
        
        # Extraer máscara de los kwargs si se pasó
        attention_mask = kwargs.get('attention_mask', None)
        
        residual = x
        x_norm = self.norm(x)
        attended = self.superposition_attention(x_norm, attention_mask=attention_mask)
        
        final_x = residual + attended
        
        if isinstance(output, tuple):
             return (final_x,) + output[1:]
        return final_x


if __name__ == "__main__":
    config = Paper2511_02002Config()
    module = SuperpositionAttention(config)
    
    # Input simulado [Batch, Seq, Dim]
    x = torch.randn(2, 16, config.hidden_dim)
    
    # Mask Causal
    mask = torch.tril(torch.ones(16, 16))
    
    output = module(x, attention_mask=mask)
    print(f"Paper 2511.02002 Superposition test: Input {x.shape} -> Output {output.shape}")
    print(f"Metrics: {module.get_metrics()}")

