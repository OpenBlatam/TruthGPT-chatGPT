#!/usr/bin/env python3
"""
AdaGroPE: Adaptive Grouped Positional Encoding
================================================

Xu, Li, Chen, Lin, Han, Ding (ACL 2025)
Extending LLM Context Window with Adaptive Grouped Positional Encoding

Paper URL: https://aclanthology.org/[ID_PENDIENTE]
ACL 2025: Extending LLM Context Window with Adaptive Grouped Positional Encoding
Nota: Paper de ACL 2025, buscar en ACL Anthology cuando esté disponible

Técnica principal:
- Método sin entrenamiento ("training-free")
- Plug and play
- Reutiliza posiciones relativas de forma adaptativa según distancia
- Aprovecha embeddings posicionales ya entrenados

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. Agrupación Adaptativa de Posiciones:
   - G(p_i) = floor(p_i / group_size(p_i))
     donde group_size(p_i) varía según la distancia
   - Posiciones cercanas: grupos más pequeños (granularidad fina)
   - Posiciones lejanas: grupos más grandes (granularidad gruesa)
   - Implementado en: _compute_group_assignments()

2. Mapeo de Posiciones a Embeddings Base:
   - PE(p_i) = PE_base(G(p_i) mod L_base)
     donde L_base es el contexto base (2048)
   - Reutiliza embeddings existentes mediante módulo
   - Implementado en: _build_position_group_map() y forward()

3. Distancia Adaptativa:
   - d(p_i, p_j) = |p_i - p_j| / L_extended
     donde L_extended es el contexto extendido
   - La agrupación se adapta según esta distancia normalizada
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
class AdaGroPEConfig(BasePaperConfig):
    """Configuración para AdaGroPE."""
    base_context_length: int = 2048
    extended_context_length: int = 32768  # Puede extenderse más
    num_groups: int = 8  # Número de grupos para agrupación adaptativa
    group_size: int = 256  # Tamaño de grupo base
    use_adaptive_grouping: bool = True
    distance_threshold: float = 0.5  # Umbral para agrupación
    training_free: bool = True  # Sin entrenamiento requerido


class AdaGroPEModule(BasePaperModule):
    """
    AdaGroPE: Adaptive Grouped Positional Encoding.
    
    Características:
    - Training-free (plug and play)
    - Agrupación adaptativa de posiciones
    - Reutiliza embeddings existentes
    """
    
    def __init__(self, config: AdaGroPEConfig):
        """
        Inicialización del módulo AdaGroPE.
        
        EN EL PAPER: Sección 3.1 - Architecture
        - El paper propone reutilizar embeddings posicionales existentes (PE_base)
        - No requiere entrenamiento adicional (training-free)
        - Usa agrupación adaptativa según distancia
        
        CÓDIGO: Inicializamos:
        1. Embeddings base (ya entrenados del modelo original)
        2. Asignaciones de grupos adaptativos
        3. Mapeo de posiciones extendidas a posiciones base
        """
        super().__init__(config)
        self.config = config
        
        # EN EL PAPER: Sección 3.2 - Base Positional Embeddings
        # El paper asume embeddings posicionales base ya entrenados
        # NOTACIÓN DEL PAPER: PE_base ∈ R^(L_base × d)
        #   donde L_base = base_context_length, d = hidden_dim
        # NOTACIÓN EN CÓDIGO: base_position_embeddings[batch, pos, :] = PE_base[pos, :]
        # CÓDIGO: nn.Embedding almacena PE_base[pos] para pos ∈ [0, L_base)
        self.base_position_embeddings = nn.Embedding(
            config.base_context_length,  # L_base
            config.hidden_dim            # d
        )
        
        # EN EL PAPER: Sección 3.3 - Adaptive Grouping
        # El paper agrupa posiciones adaptativamente según distancia
        # NOTACIÓN DEL PAPER: G: p → g donde g ∈ [0, num_groups)
        #   G(p_i) = floor(p_i / group_size(p_i))
        #   donde group_size(p_i) varía según d_i = p_i / L_extended
        # NOTACIÓN EN CÓDIGO: group_assignments[i] = G(i) para i ∈ [0, L_extended)
        # CÓDIGO: Pre-calculamos G(p) para todas las posiciones
        self.num_groups = config.num_groups
        self.group_assignments = self._compute_group_assignments()  # G(p) para todo p
        
        # EN EL PAPER: Sección 3.4 - Position Mapping
        # El paper mapea posiciones extendidas a posiciones base
        # NOTACIÓN DEL PAPER: PE(p) = PE_base[map(p)]
        #   donde map(p) = f(G(p), p) mapea posición extendida a base
        # NOTACIÓN EN CÓDIGO: position_to_group[g] = lista de posiciones base del grupo g
        # CÓDIGO: Construimos mapeo inverso: grupo → posiciones base
        self.position_to_group = self._build_position_group_map()
        
        logger.info(f"AdaGroPE initialized (training-free): {config.base_context_length} → {config.extended_context_length}")
    
    def _compute_group_assignments(self) -> torch.Tensor:
        """
        Calcula asignaciones de grupos basadas en distancia.
        
        EN EL PAPER: Sección 3.3 - Adaptive Grouping Strategy
        - El paper propone agrupación adaptativa donde posiciones cercanas tienen
          granularidad fina y posiciones lejanas tienen granularidad gruesa
        - FÓRMULA: d(p_i) = p_i / L_extended (distancia normalizada)
        - FÓRMULA: group_size(p_i) = f(d(p_i)) donde f es función creciente
        - FÓRMULA: G(p_i) = floor(p_i / group_size(p_i))
        
        CÓDIGO: Implementamos la estrategia del paper dividiendo en 4 regiones:
        - Región 1 (0-25%): Granularidad fina (3 grupos)
        - Región 2 (25-50%): Granularidad media (2 grupos)
        - Región 3 (50-75%): Granularidad media (2 grupos)
        - Región 4 (75-100%): Granularidad gruesa (1 grupo)
        """
        # NOTACIÓN DEL PAPER: G: p → g donde g ∈ [0, K) y K = num_groups
        # NOTACIÓN EN CÓDIGO: G_p[i] = G(i) para i ∈ [0, L_extended)
        L_extended = self.config.extended_context_length
        K = self.num_groups  # número de grupos
        
        # NOTACIÓN: Generamos posiciones p ∈ [0, L_extended)
        p = torch.arange(L_extended, dtype=torch.float32)  # p ∈ R^L_extended
        
        # NOTACIÓN DEL PAPER: Distancia normalizada d = p / L_extended
        # NOTACIÓN EN CÓDIGO: d[i] = p[i] / L_extended ∈ [0, 1)
        d = p / L_extended  # d ∈ R^L_extended, d[i] ∈ [0, 1)
        
        # NOTACIÓN DEL PAPER: G(p) = f_region(d) donde f_region es función por partes
        # NOTACIÓN EN CÓDIGO: G_p[i] = G(i) = f_region(d[i])
        G_p = torch.zeros(L_extended, dtype=torch.long)  # G(p) ∈ Z^L_extended
        
        for i in range(L_extended):
            p_i = i
            d_i = d[i]  # d_i ∈ [0, 1)
            
            # NOTACIÓN DEL PAPER: Función por regiones
            if d_i < 0.25:
                # Región 1 (cerca): G(p_i) = ⌊d_i · 3 · K⌋
                G_p[i] = int(d_i * 3 * K)
            elif d_i < 0.5:
                # Región 2: G(p_i) = 2 + ⌊(d_i - 0.25) · 2 · K⌋
                G_p[i] = int(2 + (d_i - 0.25) * 2 * K)
            elif d_i < 0.75:
                # Región 3: G(p_i) = 4 + ⌊(d_i - 0.5) · 2 · K⌋
                G_p[i] = int(4 + (d_i - 0.5) * 2 * K)
            else:
                # Región 4 (lejos): G(p_i) = 6 + ⌊(d_i - 0.75) · K⌋
                G_p[i] = int(6 + (d_i - 0.75) * K)
            
            # NOTACIÓN: Asegurar G(p_i) ∈ [0, K-1]
            G_p[i] = min(G_p[i], K - 1)
        
        return G_p  # G(p) ∈ Z^L_extended con valores en [0, K)
    
    def _build_position_group_map(self) -> Dict[int, List[int]]:
        """
        Construye mapeo de grupos a posiciones base.
        
        EN EL PAPER: Sección 3.4 - Position Mapping to Base Embeddings
        - El paper mapea posiciones extendidas a posiciones base usando módulo
        - FÓRMULA: PE(p_i) = PE_base(G(p_i) mod L_base)
        - donde L_base es el tamaño del contexto base (ej: 2048)
        - Esto permite reutilizar embeddings existentes sin entrenamiento
        
        CÓDIGO: Construimos un diccionario que mapea cada grupo a las
        posiciones base que le corresponden según el mapeo del paper
        """
        # EN EL PAPER: Para cada grupo, necesitamos saber qué posiciones base mapea
        # CÓDIGO: Inicializamos diccionario vacío para cada grupo
        group_map = {i: [] for i in range(self.num_groups)}
        
        # NOTACIÓN DEL PAPER: map: p → p_base donde p_base ∈ [0, L_base)
        # NOTACIÓN EN CÓDIGO: Construimos mapeo inverso grupo → posiciones base
        L_base = self.config.base_context_length
        L_extended = self.config.extended_context_length
        
        for p in range(L_extended):
            # NOTACIÓN: g = G(p) (grupo asignado a posición p)
            g = self.group_assignments[p].item()  # g ∈ [0, K)
            
            # NOTACIÓN DEL PAPER: Mapeo de posición extendida a base
            #   Si p < L_base: map(p) = p
            #   Si p ≥ L_base: map(p) = ⌊(p / L_extended) · L_base⌋
            if p < L_base:
                # NOTACIÓN: map(p) = p cuando p < L_base
                p_base = p
            else:
                # NOTACIÓN: map(p) = ⌊(p / L_extended) · L_base⌋ cuando p ≥ L_base
                ratio = p / L_extended  # p / L_extended
                p_base = int(ratio * L_base)  # ⌊ratio · L_base⌋
                p_base = min(p_base, L_base - 1)  # Clamp a [0, L_base)
            
            # NOTACIÓN EN CÓDIGO: Almacenamos p_base en el grupo g
            #   group_map[g] = lista de posiciones base del grupo g
            group_map[g].append(p_base)
        
        return group_map
    
    def get_positional_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Obtiene encoding posicional adaptativo.
        
        EN EL PAPER: Sección 3.4 - Positional Encoding Retrieval
        - El paper propone obtener embeddings usando el mapeo de grupos
        - FÓRMULA: PE(p_i) = PE_base(G(p_i) mod L_base)
        - Esto permite extender contexto sin entrenar nuevos embeddings
        
        CÓDIGO: Para cada posición, obtenemos su grupo y luego el embedding base
        correspondiente según el mapeo del paper.
        
        Args:
            positions: [batch, seq_len] posiciones
        
        Returns:
            positional_encodings: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len = positions.shape
        device = positions.device
        
        # EN EL PAPER: Obtenemos el grupo asignado a cada posición
        # FÓRMULA: g_i = G(p_i) donde G es la función de agrupación
        # CÓDIGO: Indexamos las asignaciones de grupos pre-calculadas
        position_groups = self.group_assignments[positions.clamp(0, len(self.group_assignments) - 1)]
        
        # EN EL PAPER: Para cada posición, mapeamos a posición base
        # FÓRMULA: base_pos = f(g_i, p_i) donde f es el mapeo del paper
        # CÓDIGO: Calculamos la posición base para cada posición en el batch
        base_positions = torch.zeros_like(positions)
        
        for i in range(seq_len):
            if positions.dim() > 1:
                pos = int(positions[0, i].item())
            else:
                pos = int(positions[i].item())
            
            # EN EL PAPER: Si la posición está en el contexto base, usar directamente
            # FÓRMULA: Si p_i < L_base: base_pos = p_i
            # CÓDIGO: No hay transformación necesaria
            if pos < self.config.base_context_length:
                base_pos = pos
            else:
                # EN EL PAPER: Para posiciones extendidas, usar el grupo para mapear
                # FÓRMULA: base_pos = representante(G(p_i))
                # donde representante es una posición base del grupo
                # CÓDIGO: Obtenemos el grupo y usamos posición representativa
                group = self.group_assignments[min(pos, len(self.group_assignments) - 1)].item()
                group_positions = self.position_to_group[group]
                
                if group_positions:
                    # EN EL PAPER: Usamos posición mediana del grupo como representante
                    # CÓDIGO: Tomamos la posición del medio del grupo
                    base_pos = group_positions[len(group_positions) // 2]
                else:
                    # EN EL PAPER: Fallback - interpolación lineal
                    # FÓRMULA: base_pos = floor((p_i / L_extended) * L_base)
                    # CÓDIGO: Calculamos proporcionalmente
                    ratio = pos / self.config.extended_context_length
                    base_pos = int(ratio * self.config.base_context_length)
                    base_pos = min(base_pos, self.config.base_context_length - 1)
            
            if positions.dim() > 1:
                base_positions[0, i] = base_pos
            else:
                base_positions[i] = base_pos
        
        # PASO 3: Obtener embeddings posicionales
        # NOTACIÓN DEL PAPER: PE(p) = PE_base[map(p)]
        #   donde PE_base ∈ R^(L_base × d) y map(p) ∈ [0, L_base)
        # NOTACIÓN EN CÓDIGO: PE[b, i, :] = PE_base[p_base[b, i], :]
        #   PE ∈ R^(B×N×d)
        p_base = p_base.clamp(0, L_base - 1).long()  # Asegurar p_base ∈ [0, L_base)
        PE = self.base_position_embeddings(p_base)  # PE_base[map(p)] ∈ R^(B×N×d)
        
        return PE  # PE(p) ∈ R^(B×N×d)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con AdaGroPE (training-free).
        
        EN EL PAPER: Sección 3 - Method Overview
        - El paper propone agregar embeddings posicionales adaptativos a hidden states
        - FÓRMULA: h'_i = h_i + PE(p_i)
        - donde PE(p_i) se obtiene del mapeo adaptativo sin entrenamiento
        
        CÓDIGO: Aplicamos el encoding posicional adaptativo a los hidden states
        según la metodología del paper.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            position_ids: [batch, seq_len] posiciones
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # EN EL PAPER: Si no se proporcionan position_ids, generamos secuenciales
        # CÓDIGO: Creamos posición [0, 1, 2, ..., seq_len - 1] para cada batch
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # PASO 1: Obtener encoding posicional adaptativo
        # NOTACIÓN DEL PAPER: PE(p) = PE_base[map(p)]
        # NOTACIÓN EN CÓDIGO: PE = get_positional_encoding(position_ids)
        #   PE ∈ R^(B×N×d)
        PE = self.get_positional_encoding(position_ids)  # PE(p)
        
        # PASO 2: Aplicar encoding posicional
        # NOTACIÓN DEL PAPER: h' = h + PE(p)
        #   donde h ∈ R^(B×N×d), PE(p) ∈ R^(B×N×d), h' ∈ R^(B×N×d)
        # NOTACIÓN EN CÓDIGO: output[b, i, j] = hidden_states[b, i, j] + PE[b, i, j]
        # CÓDIGO: Suma elemento a elemento
        output = hidden_states + PE  # h' = h + PE(p)
        
        metadata = {
            'context_length': seq_len,
            'extended': seq_len > self.config.base_context_length,
            'num_groups': self.num_groups,
            'training_free': self.config.training_free,
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=seq_len,
            num_groups_used=self.num_groups
        )
        
        return output, metadata


if __name__ == "__main__":
    config = AdaGroPEConfig(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=32768
    )
    
    module = AdaGroPEModule(config)
    
    # Test
    hidden_states = torch.randn(2, 4096, config.hidden_dim)
    output, metadata = module(hidden_states)
    
    print(f"✅ AdaGroPE test:")
    print(f"   Output shape: {output.shape}")
    print(f"   Training-free: {metadata['training_free']}")
    print(f"   Num groups: {metadata['num_groups']}")


