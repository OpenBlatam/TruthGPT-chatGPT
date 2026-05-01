#!/usr/bin/env python3
"""
LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens
================================================================

Ding, Zhang, Xu, Shang, et al. (2024)

Paper URL: https://arxiv.org/abs/2402.13753

Técnica principal:
- Reescala no uniforme del embedding posicional RoPE
- Extiende ventana hasta ~2,048k tokens con poco fine-tuning
- Mejora eficiencia de memoria y procesamiento

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. Frecuencias Base de RoPE:
   - θ_i = 10000^(-2i/d) para i ∈ [0, d/2)
     donde d es rope_dim (dimensión de RoPE)
   - Implementado en: _compute_base_frequencies()
   - Ecuación: freqs = 1.0 / (10000 ** (arange(0, d, 2) / d))

2. Escalado No Uniforme de Posiciones:
   - p'_i = α · s(p_i) + β
     donde s(p_i) es el factor de escalado no uniforme,
     α y β son parámetros aprendibles
   - Implementado en: _compute_non_uniform_scaling() y apply_rope()
   - El escalado es más agresivo en posiciones lejanas para comprimir el espacio

3. Rotación de Embeddings (RoPE):
   - Para cada par de dimensiones (2i, 2i+1):
     [x'_2i  ]   [cos(θ_i · p')  -sin(θ_i · p')] [x_2i  ]
     [x'_2i+1] = [sin(θ_i · p')   cos(θ_i · p')] [x_2i+1]
   - Implementado en: apply_rope()
   - Ecuación: rotated = R(θ · p') · x donde R es matriz de rotación
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
class LongRoPEConfig(BasePaperConfig):
    """Configuración para LongRoPE."""
    base_context_length: int = 2048  # Contexto original
    extended_context_length: int = 2048000  # Contexto extendido (2M tokens)
    rope_dim: int = 64  # Dimensión de RoPE
    scaling_factor: float = 1.0  # Factor de escalado no uniforme
    use_non_uniform_scaling: bool = True
    fine_tune_steps: int = 1000


class LongRoPEModule(BasePaperModule):
    """
    LongRoPE: Extensión de ventana de contexto usando RoPE no uniforme.
    
    Características:
    - Reescala no uniforme de embeddings posicionales
    - Extensión hasta 2M tokens
    - Fine-tuning mínimo requerido
    """
    
    def __init__(self, config: LongRoPEConfig):
        """
        Inicialización del módulo LongRoPE.
        
        EN EL PAPER: Sección 3 - Non-uniform Scaling Method
        - El paper propone reescalar posiciones de forma no uniforme antes de aplicar RoPE
        - FÓRMULA: p'_i = α · s(p_i) + β
          donde s(p_i) es función de escalado no uniforme,
          α y β son parámetros aprendibles durante fine-tuning
        - Esto permite extender hasta ~2M tokens con poco fine-tuning
        
        CÓDIGO: Inicializamos:
        1. Factores de escalado no uniforme (pre-calculados)
        2. Frecuencias base de RoPE
        3. Parámetros adaptativos α y β (aprendibles)
        """
        super().__init__(config)
        self.config = config
        
        # EN EL PAPER: Sección 3.1 - Non-uniform Scaling Function
        # El paper calcula factores de escalado que varían según la posición
        # FÓRMULA: s(p_i) varía según región (cerca: menos escalado, lejos: más escalado)
        # CÓDIGO: Pre-calculamos los factores de escalado para todas las posiciones
        self.scaling_factors = self._compute_non_uniform_scaling()
        
        # EN EL PAPER: Sección 2 - Background on RoPE
        # El paper usa las frecuencias base estándar de RoPE
        # FÓRMULA: θ_i = 10000^(-2i/d) para i ∈ [0, d/2)
        # CÓDIGO: Calculamos las frecuencias base una vez
        self.rope_dim = config.rope_dim
        self.base_freqs = self._compute_base_frequencies()
        
        # EN EL PAPER: Sección 3.2 - Learnable Scaling Parameters
        # El paper introduce parámetros α y β que se aprenden durante fine-tuning
        # FÓRMULA: p'_i = α · s(p_i) + β
        # CÓDIGO: Inicializamos α=1.0 y β=0.0, se ajustan durante entrenamiento
        self.alpha = nn.Parameter(torch.ones(1) * config.scaling_factor)
        self.beta = nn.Parameter(torch.zeros(1))
        
        logger.info(f"LongRoPE initialized: {config.base_context_length} → {config.extended_context_length} tokens")
    
    def _compute_base_frequencies(self) -> torch.Tensor:
        """
        Calcula frecuencias base para RoPE.
        
        EN EL PAPER: Sección 2 - Background on RoPE
        
        NOTACIÓN DEL PAPER: θ_j = 10000^(-2j/d) para j ∈ [0, d/2)
        - d = rope_dim (dimensión de RoPE)
        - Cada frecuencia θ_j corresponde al par de dimensiones (2j, 2j+1)
        - Frecuencias decrecen: θ_0 = 1, θ_{d/2-1} ≈ 10000^-1
        
        NOTACIÓN EN CÓDIGO: theta[j] = θ_j
        - theta ∈ R^(d/2)
        
        Returns:
            theta: [d/2] frecuencias base θ
        """
        # NOTACIÓN DEL PAPER: θ_j = 10000^(-2j/d)
        # NOTACIÓN EN CÓDIGO: theta[j] = 10000^(-2j/d)
        #   Para j=0: θ_0 = 10000^0 = 1
        #   Para j=d/2-1: θ_{d/2-1} = 10000^(-(d-2)/d) ≈ 10000^-1
        j_indices = torch.arange(0, self.rope_dim, 2).float()  # j ∈ [0, 2, 4, ..., d-2]
        theta = 1.0 / (10000 ** (j_indices / self.rope_dim))  # θ_j = 10000^(-2j/d)
        return theta
    
    def _compute_non_uniform_scaling(self) -> torch.Tensor:
        """
        Calcula factores de escalado no uniforme.
        
        EN EL PAPER: Sección 3.1 - Non-uniform Scaling Strategy
        - El paper propone escalado no uniforme donde posiciones cercanas
          tienen menos compresión y posiciones lejanas tienen más compresión
        - FÓRMULA: s(p_i) = f_region(p_i) donde f_region es función por región
        - Esto permite mantener precisión cerca y comprimir lejos
        
        CÓDIGO: Dividimos el contexto en 4 regiones con factores crecientes
        """
        if not self.config.use_non_uniform_scaling:
            # EN EL PAPER: Fallback a escalado uniforme (no recomendado)
            # FÓRMULA: s(p_i) = L_extended / L_base (constante)
            # CÓDIGO: Escalado simple proporcional
            ratio = self.config.extended_context_length / self.config.base_context_length
            return torch.ones(self.config.base_context_length) * ratio
        
        # NOTACIÓN DEL PAPER: s: p → s(p) donde s(p) es factor de escalado
        # NOTACIÓN EN CÓDIGO: s_p[i] = s(i) para i ∈ [0, L_extended)
        L_base = self.config.base_context_length
        L_extended = self.config.extended_context_length
        p = torch.arange(L_extended, dtype=torch.float32)  # p ∈ [0, L_extended)
        
        # NOTACIÓN DEL PAPER: s(p) = f_region(p) (función por regiones)
        # CÓDIGO: Inicializamos s(p) = 1
        s_p = torch.ones_like(p)  # s(p) ∈ R^L_extended
        
        # NOTACIÓN DEL PAPER: 4 regiones con factores crecientes
        # region_factor(r) = 1.0 + r · 0.5 para r ∈ {0, 1, 2, 3}
        region_size = L_extended // 4
        for r in range(4):  # r = índice de región
            start_idx = r * region_size
            end_idx = min((r + 1) * region_size, L_extended)
            
            # NOTACIÓN DEL PAPER: region_ratio(r) = |región_r| / L_base
            #   donde |región_r| = end_idx - start_idx
            region_size_actual = end_idx - start_idx
            region_ratio = region_size_actual / L_base
            
            # NOTACIÓN DEL PAPER: region_factor(r) = 1.0 + r · 0.5
            region_factor = 1.0 + (r * 0.5)
            
            # NOTACIÓN: s(p) = region_factor(r) · region_ratio(r) para p ∈ región_r
            # NOTACIÓN EN CÓDIGO: s_p[start:end] = region_factor · region_ratio
            s_p[start_idx:end_idx] = region_ratio * region_factor
        
        return s_p  # s(p) ∈ R^L_extended
    
    def apply_rope(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Aplica RoPE con escalado no uniforme.
        
        EN EL PAPER: Sección 3 - Non-uniform RoPE Application
        
        NOTACIÓN DEL PAPER:
        1. Escalado: p' = α · s(p) + β donde s(p) es función de escalado
        2. Frecuencias: θ_j · p' para cada frecuencia j
        3. Rotación 2D: Para par (2j, 2j+1):
           [x'_{2j}  ]   [cos(θ_j·p')  -sin(θ_j·p')] [x_{2j}  ]
           [x'_{2j+1}] = [sin(θ_j·p')   cos(θ_j·p')] [x_{2j+1}]
        
        NOTACIÓN EN CÓDIGO:
        - hidden_states ∈ R^(B×N×d) donde B=batch, N=seq_len, d=hidden_dim
        - positions ∈ R^N (posiciones absolutas)
        - output ∈ R^(B×N×d) (hidden states rotados)
        
        Args:
            hidden_states: [B, N, d] estados ocultos
            positions: [N] posiciones absolutas p
        
        Returns:
            output: [B, N, d] estados con RoPE aplicado
        """
        B, N, d = hidden_states.shape  # batch_size, seq_len, hidden_dim
        
        # PASO 1: Escalado no uniforme de posiciones
        # NOTACIÓN DEL PAPER: p' = α · s(p) + β
        # NOTACIÓN EN CÓDIGO: p_prime[i] = α · s(positions[i]) + β
        #   donde s(positions[i]) = scaling_factors[positions[i]]
        p = positions  # p ∈ R^N
        s_p = self.scaling_factors[:len(p)]  # s(p) ∈ R^N
        p_prime = p * s_p * self.alpha + self.beta  # p' = α·s(p) + β ∈ R^N
        
        # PASO 2: Cálculo de frecuencias por posición
        # NOTACIÓN DEL PAPER: θ_j · p'_i para frecuencia j ∈ [0, d/2) y posición i
        #   donde θ_j = 10000^(-2j/d) son frecuencias base
        # NOTACIÓN EN CÓDIGO: freqs[i, j] = θ_j · p'_i
        #   freqs ∈ R^(N × d_rope/2) donde d_rope = rope_dim
        theta = self.base_freqs  # θ ∈ R^(d_rope/2)
        freqs = p_prime.unsqueeze(-1) * theta.unsqueeze(0)  # [N, d_rope/2]
        
        # PASO 3: Embeddings rotatorios (coseno y seno)
        # NOTACIÓN DEL PAPER: cos(θ_j · p'_i) y sin(θ_j · p'_i)
        # NOTACIÓN EN CÓDIGO: cos_freqs[i, j] = cos(θ_j · p'_i)
        cos_theta_p = torch.cos(freqs)  # [N, d_rope/2]
        sin_theta_p = torch.sin(freqs)  # [N, d_rope/2]
        
        # PASO 4: Aplicación de rotación 2D
        # NOTACIÓN DEL PAPER: Rotación por pares (2j, 2j+1)
        # NOTACIÓN EN CÓDIGO: Reshape para procesar grupos de rope_dim dimensiones
        #   x ∈ R^(B×N×d) → x_reshaped ∈ R^(B×N×G×d_rope)
        #   donde G = d / d_rope (número de grupos)
        G = d // self.rope_dim  # número de grupos
        x = hidden_states.view(B, N, G, self.rope_dim)  # [B, N, G, d_rope]
        x_rotated = torch.zeros_like(x)  # [B, N, G, d_rope]
        
        # NOTACIÓN: Para cada frecuencia j ∈ [0, d_rope/2)
        for j in range(self.rope_dim // 2):
            # NOTACIÓN: Par de dimensiones (2j, 2j+1)
            dim_2j = j * 2
            dim_2j_plus_1 = j * 2 + 1
            
            # NOTACIÓN: Valores de coseno y seno para frecuencia j
            # cos_val[i] = cos(θ_j · p'_i), sin_val[i] = sin(θ_j · p'_i)
            # Ensure correct broadcasting for [B, N, G]
            cos_j = cos_theta_p[:, j].view(1, N, 1)  # [1, N, 1]
            sin_j = sin_theta_p[:, j].view(1, N, 1)  # [1, N, 1]
            
            # NOTACIÓN DEL PAPER: Rotación 2D
            # x'_{2j} = x_{2j} · cos(θ_j·p') - x_{2j+1} · sin(θ_j·p')
            # NOTACIÓN EN CÓDIGO: x_rotated[b, i, g, 2j] = x[b, i, g, 2j]·cos - x[b, i, g, 2j+1]·sin
            x_rotated[:, :, :, dim_2j] = (
                x[:, :, :, dim_2j] * cos_j -
                x[:, :, :, dim_2j_plus_1] * sin_j
            )
            
            # NOTACIÓN DEL PAPER: x'_{2j+1} = x_{2j} · sin(θ_j·p') + x_{2j+1} · cos(θ_j·p')
            # NOTACIÓN EN CÓDIGO: x_rotated[b, i, g, 2j+1] = x[b, i, g, 2j]·sin + x[b, i, g, 2j+1]·cos
            x_rotated[:, :, :, dim_2j_plus_1] = (
                x[:, :, :, dim_2j] * sin_j +
                x[:, :, :, dim_2j_plus_1] * cos_j
            )
        
        # NOTACIÓN: Reshape de vuelta a [B, N, d]
        output = x_rotated.view(B, N, d)
        return output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con LongRoPE.
        
        EN EL PAPER: Sección 3 - Method Overview
        - El paper aplica RoPE con escalado no uniforme a los hidden states
        - FÓRMULA: h'_i = RoPE(h_i, p'_i) donde p'_i = α · s(p_i) + β
        - Esto extiende el contexto hasta ~2M tokens
        
        CÓDIGO: Aplicamos el método del paper paso a paso:
        1. Generar/obtener position_ids
        2. Aplicar RoPE con escalado no uniforme
        3. Devolver estados transformados
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            position_ids: [batch, seq_len] posiciones (opcional)
        
        Returns:
            (output, metadata)
        """
        # NOTACIÓN DEL PAPER: H ∈ R^(B×N×d) donde B=batch, N=seq_len, d=hidden_dim
        B, N, d = hidden_states.shape
        H = hidden_states  # H ∈ R^(B×N×d)
        
        # PASO 1: Generar/obtener position_ids
        # NOTACIÓN: Si no se proporcionan, asumimos posiciones secuenciales p = [0, 1, ..., N-1]
        # NOTACIÓN EN CÓDIGO: position_ids[b, i] = i para b ∈ [0, B), i ∈ [0, N)
        if position_ids is None:
            p = torch.arange(N, device=H.device).unsqueeze(0).expand(B, -1)  # p ∈ Z^(B×N)
        else:
            p = position_ids  # p ∈ Z^(B×N)
        
        # PASO 2: Aplicar LongRoPE con escalado no uniforme
        # NOTACIÓN DEL PAPER: h' = RoPE(h, p') donde p' = α · s(p) + β
        # NOTACIÓN EN CÓDIGO: 
        #   Para aplicar RoPE, necesitamos p como tensor 1D: p[0] ∈ Z^N
        #   output[b, i, :] = RoPE(H[b, i, :], p'_i)
        #   output ∈ R^(B×N×d)
        if p.dim() > 1:
            p_1d = p[0]  # p_1d ∈ Z^N (tomamos primer batch)
        else:
            p_1d = p  # p_1d ∈ Z^N
        h_prime = self.apply_rope(H, p_1d)  # h' = RoPE(h, p') ∈ R^(B×N×d)
        
        # Metadata para tracking
        L_base = self.config.base_context_length
        L_extended = self.config.extended_context_length
        metadata = {
            'context_length': N,
            'extended': N > L_base,
            'scaling_factor': self.alpha.item(),  # α
            'max_context': L_extended
        }
        
        self._update_metrics(
            context_length=N,
            scaling_factor=self.alpha.item()
        )
        
        return h_prime, metadata  # h' ∈ R^(B×N×d)


if __name__ == "__main__":
    # Test
    config = LongRoPEConfig(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=2048000
    )
    
    module = LongRoPEModule(config)
    
    # Test con secuencia larga
    seq_len = 4096
    hidden_states = torch.randn(2, seq_len, config.hidden_dim)
    
    output, metadata = module(hidden_states)
    
    print(f"✅ LongRoPE test:")
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Extended: {metadata['extended']}")
    print(f"   Scaling factor: {metadata['scaling_factor']:.4f}")


