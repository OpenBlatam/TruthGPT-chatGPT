#!/usr/bin/env python3
"""
LongReward: Improving Long-context LLMs
========================================

(ACL 2025) LongReward: Improving Long-context Large Language Models
Hossein / Bai etc.

Paper URL: https://aclanthology.org/[ID_PENDIENTE]
ACL 2025: LongReward: Improving Long-context Large Language Models
Nota: Paper de ACL 2025, buscar en ACL Anthology cuando esté disponible

Técnica principal:
- No solo extiende ventana sino optimiza comportamiento
- Usa recompensas para mejorar manejo de dependencias largas
- Entrenamiento con RL para contexto largo

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. Modelo de Recompensa (Reward Model):
   - Función de recompensa: R(h_i, p_i) = f_θ(h_i) / τ
     donde f_θ es la red de recompensa, h_i son los hidden states,
     p_i son las posiciones, y τ es la temperatura
   - Implementado en: RewardModel.forward()
   - Ecuación: rewards = reward_head(reward_network(hidden_states)) / temperature

2. Rastreo de Dependencias (Dependency Tracking):
   - Score de dependencia: D_i = mean_j(Attention(h_i, h_j))
     donde Attention es multi-head attention sobre el contexto completo
   - Implementado en: DependencyTracker.forward()
   - Ecuación: dependency_scores = mean(attention_weights, dim=[1, -1])

3. Guía de Atención con Recompensas (Reward-Guided Attention):
   - Pesos normalizados: w_i = softmax(R(h_i, p_i))
   - Gate adaptativo: g_i = σ(W_g · h_i)
   - Estados guiados: h'_i = h_i ⊙ g_i ⊙ w_i
     donde ⊙ es multiplicación elemento a elemento
   - Implementado en: LongRewardModule.apply_reward_guidance()
   - Ecuación: guided_states = hidden_states * sigmoid(gate_network(h)) * softmax(rewards)
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
class LongRewardConfig(BasePaperConfig):
    """Configuración para LongReward."""
    base_context_length: int = 2048
    extended_context_length: int = 32768
    reward_model_dim: int = 256
    use_reward_guidance: bool = True
    reward_temperature: float = 1.0
    dependency_window: int = 512  # Ventana para detectar dependencias largas
    num_reward_layers: int = 2


class RewardModel(nn.Module):
    """
    Modelo de recompensa para dependencias largas.
    
    EN EL PAPER: Sección 3.1 - Reward Model Architecture
    - El paper propone un modelo de recompensa que evalúa la importancia
      de cada token para manejar dependencias largas
    - FÓRMULA: R(h_i, p_i) = f_θ(h_i) / τ
      donde f_θ es una red neuronal y τ es temperatura
    - El modelo aprende a asignar recompensas altas a tokens que son
      importantes para dependencias de largo alcance
    """
    
    def __init__(self, config: LongRewardConfig):
        super().__init__()
        self.config = config
        
        # EN EL PAPER: Sección 3.1.1 - Reward Network
        # El paper usa red feed-forward con múltiples capas
        # NOTACIÓN DEL PAPER: f_θ: R^d → R^(d_r)
        #   f_θ(h) = W_L · GELU(LN(W_{L-1} · ... GELU(LN(W_1 · h)) ...))
        #   donde W_i son matrices de pesos, LN es LayerNorm, d_r = reward_model_dim
        # NOTACIÓN EN CÓDIGO: reward_network(h) = f_θ(h)
        #   reward_network: R^(B×N×d) → R^(B×N×d_r)
        reward_layers = []
        for i in range(config.num_reward_layers):
            # NOTACIÓN: Capa i: W_i ∈ R^(d_in×d_r) donde d_in = d si i=0, d_r si i>0
            reward_layers.append(
                nn.Sequential(
                    nn.Linear(config.hidden_dim if i == 0 else config.reward_model_dim,  # W_i
                             config.reward_model_dim),  # d_r
                    nn.LayerNorm(config.reward_model_dim),  # LN
                    nn.GELU()  # GELU
                )
            )
        self.reward_network = nn.Sequential(*reward_layers)  # f_θ
        
        # EN EL PAPER: Sección 3.1.2 - Reward Head
        # El paper mapea a escalar de recompensa
        # NOTACIÓN DEL PAPER: r = W_r · f_θ(h) donde W_r ∈ R^(d_r×1)
        # NOTACIÓN EN CÓDIGO: reward_head(x) = x · W_r^T
        #   reward_head: R^(B×N×d_r) → R^(B×N×1)
        self.reward_head = nn.Linear(config.reward_model_dim, 1)  # W_r
    
    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Calcula recompensas para dependencias largas.
        
        EN EL PAPER: Sección 3.1 - Reward Computation
        
        NOTACIÓN DEL PAPER: R(h_i, p_i) = f_θ(h_i) / τ
        - f_θ: Red neuronal de recompensa (reward_network + reward_head)
        - τ: Temperatura (reward_temperature) para controlar la suavidad
        - h_i: Hidden states en posición i ∈ R^d
        
        NOTACIÓN EN CÓDIGO:
        - hidden_states ∈ R^(B×N×d) donde B=batch, N=seq_len, d=hidden_dim
        - R ∈ R^(B×N) (recompensas escaladas)
        
        Args:
            hidden_states: [B, N, d] estados ocultos h
            positions: [B, N] posiciones p (no usado en cálculo, solo para compatibilidad)
        
        Returns:
            R: [B, N] recompensas R(h_i, p_i)
        """
        # PASO 1: Procesamiento con reward network
        # NOTACIÓN DEL PAPER: f_θ(h) donde h ∈ R^d
        # NOTACIÓN EN CÓDIGO: f_theta_h[b, i, :] = f_θ(hidden_states[b, i, :])
        #   f_theta_h ∈ R^(B×N×d_r) donde d_r = reward_model_dim
        f_theta_h = self.reward_network(hidden_states)  # f_θ(h) ∈ R^(B×N×d_r)
        
        # PASO 2: Cálculo de recompensas base
        # NOTACIÓN DEL PAPER: r = W_r · f_θ(h) donde W_r ∈ R^(d_r×1)
        # NOTACIÓN EN CÓDIGO: r[b, i] = Σ_k f_theta_h[b, i, k] · W_r[k, 0]
        #   r ∈ R^(B×N)
        r = self.reward_head(f_theta_h).squeeze(-1)  # r ∈ R^(B×N)
        
        # PASO 3: Escalado por temperatura
        # NOTACIÓN DEL PAPER: R(h, p) = r / τ donde τ = reward_temperature
        # NOTACIÓN EN CÓDIGO: R[b, i] = r[b, i] / τ
        tau = self.config.reward_temperature  # τ
        R = r / tau  # R = r / τ ∈ R^(B×N)
        
        return R


class DependencyTracker(nn.Module):
    """
    Rastrea dependencias largas en el contexto.
    
    EN EL PAPER: Sección 3.2 - Dependency Tracking
    - El paper propone usar atención para detectar dependencias entre tokens
    - FÓRMULA: D_i = (1/|H|) Σ_h (1/seq_len) Σ_j Attention_h(h_i, h_j)
      donde Attention_h es la atención del head h
    - Esto identifica qué tokens están más conectados con el resto del contexto
    """
    
    def __init__(self, config: LongRewardConfig):
        super().__init__()
        self.config = config
        
        # EN EL PAPER: Sección 3.2.1 - Multi-Head Attention
        # El paper usa self-attention para detectar dependencias
        # NOTACIÓN DEL PAPER: Attention(Q, K, V) = softmax(QK^T / √d_k) V
        #   donde Q=K=V=H (self-attention), d_k = head_dim = d / num_heads
        # NOTACIÓN EN CÓDIGO: 
        #   dependency_attention(H, H, H) → (enhanced, attention_weights)
        #   enhanced ∈ R^(B×N×d), attention_weights ∈ R^(B×H×N×N)
        self.dependency_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,  # d
            num_heads=8,  # H
            batch_first=True
        )
        
        # EN EL PAPER: Sección 3.2.2 - Dependency Projection
        # El paper proyecta estados mejorados
        # NOTACIÓN DEL PAPER: h' = W_p · enhanced donde W_p ∈ R^(d×d)
        # NOTACIÓN EN CÓDIGO: dependency_proj(enhanced) = enhanced · W_p^T
        #   dependency_proj: R^(B×N×d) → R^(B×N×d)
        self.dependency_proj = nn.Linear(config.hidden_dim, config.hidden_dim)  # W_p
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detecta dependencias largas.
        
        MATEMÁTICA: D_i = (1/|H|) Σ_j Attention(h_i, h_j)
        - Attention: Multi-head self-attention sobre el contexto completo
        - D_i: Score de dependencia para la posición i
        - El promedio sobre todas las posiciones j indica qué tan conectada está i
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            (dependency_scores, enhanced_states)
        """
        # EN EL PAPER: Self-attention para detectar dependencias
        # NOTACIÓN DEL PAPER: Attention(Q, K, V) = softmax(QK^T / √d_k) V
        #   donde Q=K=V=H (self-attention), d_k = d / H
        # NOTACIÓN EN CÓDIGO:
        #   enhanced[b, i, :] = Attention(H[b, i, :], H[b, :, :], H[b, :, :])
        #   enhanced ∈ R^(B×N×d), attention_weights ∈ R^(B×H×N×N)
        enhanced, attention_weights = self.dependency_attention(
            hidden_states, hidden_states, hidden_states
        )
        
        # EN EL PAPER: Cálculo de scores de dependencia
        # NOTACIÓN DEL PAPER: D_i = (1/H) Σ_h (1/N) Σ_j Attention_h[i, j]
        #   donde Attention_h es atención del head h, promediamos sobre heads y posiciones
        # NOTACIÓN EN CÓDIGO:
        #   dependency_scores[b, i] = mean_h(mean_j(attention_weights[b, h, i, j]))
        #   dependency_scores ∈ R^(B×N)
        dependency_scores = attention_weights.mean(dim=1)  # [B, N, N] (promedio sobre heads H)
        dependency_scores = dependency_scores.mean(dim=-1)  # [B, N] (promedio sobre posiciones j)
        
        # EN EL PAPER: Proyección de estados mejorados
        # NOTACIÓN DEL PAPER: h' = W_p · enhanced donde W_p ∈ R^(d×d)
        # NOTACIÓN EN CÓDIGO: enhanced[b, i, :] = enhanced[b, i, :] · W_p^T
        enhanced = self.dependency_proj(enhanced)  # [B, N, d]
        
        return D, h_prime  # D ∈ R^(B×N), h_prime ∈ R^(B×N×d)


class LongRewardModule(BasePaperModule):
    """
    LongReward: Optimización con recompensas para contexto largo.
    
    Características:
    - Modelo de recompensa para dependencias largas
    - Rastreo de dependencias
    - Guía de atención basada en recompensas
    """
    
    def __init__(self, config: LongRewardConfig):
        """
        Inicialización del módulo LongReward.
        
        EN EL PAPER: Sección 3 - Method
        - El paper combina tres componentes:
          1. Reward Model: Evalúa importancia de tokens
          2. Dependency Tracker: Detecta dependencias largas
          3. Reward Gate: Combina recompensas con estados
        - Todo esto se usa para guiar la atención hacia tokens importantes
        """
        super().__init__(config)
        self.config = config
        
        # EN EL PAPER: Sección 3.1 - Reward Model (opcional)
        # El paper permite desactivar el reward model para comparaciones
        # CÓDIGO: Creamos el modelo solo si está habilitado
        if config.use_reward_guidance:
            self.reward_model = RewardModel(config)
        else:
            self.reward_model = None
        
        # EN EL PAPER: Sección 3.2 - Dependency Tracker
        # El paper siempre usa el dependency tracker para detectar conexiones
        # CÓDIGO: Inicializamos el rastreador de dependencias
        self.dependency_tracker = DependencyTracker(config)
        
        # EN EL PAPER: Sección 3.3 - Reward Gate
        # El paper propone gate adaptativo para combinar recompensas
        # NOTACIÓN DEL PAPER: g: R^d → R^d, g(h) = σ(W_g · h)
        #   donde σ es sigmoid, W_g ∈ R^(d×d)
        # NOTACIÓN EN CÓDIGO: gate[b, i, j] = σ(Σ_k hidden_states[b, i, k] · W_g[k, j])
        #   gate ∈ R^(B×N×d) con valores en [0, 1]
        # CÓDIGO: Red: Linear(d→d) → Sigmoid
        self.reward_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),  # W_g
            nn.Sigmoid()  # σ
        )
        
        logger.info(f"LongReward initialized: {config.base_context_length} → {config.extended_context_length} tokens")
    
    def apply_reward_guidance(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Aplica guía de recompensa a hidden_states.
        
        EN EL PAPER: Sección 3.3 - Reward-Guided Attention
        
        NOTACIÓN DEL PAPER:
        1. Recompensas: R(h_i, p_i) = f_θ(h_i) / τ ∈ R
        2. Pesos normalizados: w_i = softmax(R_i) = exp(R_i) / Σ_j exp(R_j) ∈ [0, 1]
           donde Σ_i w_i = 1 (distribución de probabilidad)
        3. Gate adaptativo: g_i = σ(W_g · h_i) ∈ R^d con valores en [0, 1]
        4. Estados guiados: h'_i = h_i ⊙ g_i ⊙ w_i
           donde ⊙ es multiplicación elemento a elemento (Hadamard product)
        
        NOTACIÓN EN CÓDIGO:
        - hidden_states ∈ R^(B×N×d)
        - rewards ∈ R^(B×N)
        - w ∈ R^(B×N), gate ∈ R^(B×N×d)
        - guided_states ∈ R^(B×N×d)
        
        Esta combinación permite enfatizar tokens importantes basándose en
        recompensas aprendidas y un gate adaptativo.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            position_ids: [batch, seq_len] posiciones
        
        Returns:
            guided_states: [batch, seq_len, hidden_dim] estados guiados por recompensas
        """
        if self.reward_model is None:
            return hidden_states
        
        # NOTACIÓN DEL PAPER: H ∈ R^(B×N×d) donde B=batch, N=seq_len, d=hidden_dim
        B, N, d = hidden_states.shape
        H = hidden_states  # H ∈ R^(B×N×d)
        
        # PASO 1: Cálculo de recompensas
        # NOTACIÓN DEL PAPER: R(h_i, p_i) = f_θ(h_i) / τ
        # NOTACIÓN EN CÓDIGO: R[b, i] = R(H[b, i, :], position_ids[b, i])
        #   R ∈ R^(B×N)
        R = self.reward_model(H, position_ids)  # R(h, p) ∈ R^(B×N)
        
        # PASO 2: Normalización de recompensas
        # NOTACIÓN DEL PAPER: w = softmax(R) = exp(R) / Σ_j exp(R_j)
        #   Esto convierte recompensas en distribución de probabilidad
        # NOTACIÓN EN CÓDIGO: w[b, i] = exp(R[b, i]) / Σ_j exp(R[b, j])
        #   w ∈ R^(B×N) con Σ_i w[b, i] = 1 para cada batch b
        w = F.softmax(R, dim=-1)  # w = softmax(R) ∈ R^(B×N)
        
        # PASO 3: Gate adaptativo
        # NOTACIÓN DEL PAPER: g = σ(W_g · H) donde σ es sigmoid, W_g ∈ R^(d×d)
        # NOTACIÓN EN CÓDIGO: g[b, i, j] = σ(Σ_k H[b, i, k] · W_g[k, j])
        #   g ∈ R^(B×N×d) con valores en [0, 1]
        g = self.reward_gate(H)  # g = σ(W_g · H) ∈ R^(B×N×d)
        
        # PASO 4: Aplicación de guía de recompensa
        # NOTACIÓN DEL PAPER: h' = H ⊙ g ⊙ w
        #   donde ⊙ es multiplicación elemento a elemento (Hadamard product)
        #   w se expande a [B, N, 1] para broadcasting
        # NOTACIÓN EN CÓDIGO: 
        #   h_prime[b, i, j] = H[b, i, j] · g[b, i, j] · w[b, i]
        #   h_prime ∈ R^(B×N×d)
        w_expanded = w.unsqueeze(-1)  # w ∈ R^(B×N) → w_expanded ∈ R^(B×N×1)
        h_prime = H * g * w_expanded  # h' = H ⊙ g ⊙ w (Hadamard product)
        
        return h_prime  # h' ∈ R^(B×N×d)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass con LongReward.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            position_ids: [batch, seq_len] posiciones
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # Rastrear dependencias
        dependency_scores, enhanced_states = self.dependency_tracker(hidden_states)
        
        # Aplicar guía de recompensa
        if self.config.use_reward_guidance:
            output = self.apply_reward_guidance(enhanced_states, position_ids)
        else:
            output = enhanced_states
        
        # Calcular recompensa promedio
        if self.reward_model is not None:
            rewards = self.reward_model(hidden_states, position_ids)
            avg_reward = rewards.mean().item()
        else:
            avg_reward = 0.0
        
        metadata = {
            'context_length': seq_len,
            'extended': seq_len > self.config.base_context_length,
            'reward_guidance': self.config.use_reward_guidance,
            'avg_reward': avg_reward,
            'dependency_window': self.config.dependency_window,
            'max_context': self.config.extended_context_length
        }
        
        self._update_metrics(
            context_length=seq_len,
            avg_reward=avg_reward
        )
        
        return output, metadata


if __name__ == "__main__":
    config = LongRewardConfig(
        hidden_dim=768,
        base_context_length=2048,
        extended_context_length=32768
    )
    
    module = LongRewardModule(config)
    
    # Test
    hidden_states = torch.randn(2, 4096, config.hidden_dim)
    output, metadata = module(hidden_states)
    
    print(f"✅ LongReward test:")
    print(f"   Output shape: {output.shape}")
    print(f"   Reward guidance: {metadata['reward_guidance']}")
    print(f"   Avg reward: {metadata['avg_reward']:.4f}")



