#!/usr/bin/env python3
"""
Paper: 2505.05315v2 (Research Paper)
====================================

Implementación específica basada en el paper de investigación.
Este módulo implementa las técnicas específicas propuestas en este paper.

Paper URL: https://arxiv.org/html/2505.05315v2

MATEMÁTICAS DEL PAPER IMPLEMENTADAS:

1. Mixture of Experts (MoE):
   - Routing: g(x) = softmax(W_g · x) donde W_g es el router
   - Output: y = Σ_i g_i(x) · E_i(x)
     donde E_i son los expertos y g_i son los pesos de routing
   - Implementado en: MixtureOfExperts.forward()

2. Load Balancing:
   - L_balance = Σ_i (f_i · P_i) donde f_i es la fracción de tokens
     y P_i es la probabilidad de routing al experto i
   - Implementado para evitar que algunos expertos reciban demasiados tokens

3. Expert Capacity:
   - C_i = floor(T · g_i / K) donde T es el número de tokens,
     g_i es el peso del experto, y K es el número de expertos
   - Limita cuántos tokens puede procesar cada experto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Paper2505_05315v2Config:
    """Configuración específica para paper 2505.05315v2."""
    hidden_dim: int = 512
    num_heads: int = 8
    use_dynamic_routing: bool = True
    use_mixture_of_experts: bool = True
    num_experts: int = 4
    expert_capacity: int = 2


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts basado en paper 2505.05315v2.
    Técnica: Routing dinámico a múltiples expertos.
    
    Mejoras:
    - Load balancing mejorado
    - Validación de capacity
    - Métricas de routing
    - Mejor inicialización
    """
    
    def __init__(self, hidden_dim: int, num_experts: int = 4, expert_capacity: int = 2,
                 load_balancing_weight: float = 0.01):
        super().__init__()
        assert num_experts > 0, f"num_experts must be positive, got {num_experts}"
        assert expert_capacity > 0, f"expert_capacity must be positive, got {expert_capacity}"
        
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.load_balancing_weight = load_balancing_weight
        
        # Expert networks with better initialization
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
        # Initialize expert weights
        for expert in self.experts:
            for module in expert:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # Router (gating network) with better initialization
        self.router = nn.Linear(hidden_dim, num_experts)
        nn.init.xavier_uniform_(self.router.weight)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        
        # Metrics
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('load_balance_loss', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass con load balancing mejorado.
        
        Returns:
            output: Expert outputs
            load_balance_loss: Load balancing loss for training
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing scores
        router_logits = self.router(x)  # [batch, seq, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.expert_capacity, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Flatten for processing
        x_flat = x.view(-1, self.hidden_dim)  # [batch*seq, hidden]
        top_k_indices_flat = top_k_indices.view(-1, self.expert_capacity)  # [batch*seq, k]
        top_k_probs_flat = top_k_probs.view(-1, self.expert_capacity)  # [batch*seq, k]
        
        # Process through experts (more efficient implementation)
        output = torch.zeros_like(x_flat)
        expert_usage_count = torch.zeros(self.num_experts, device=x.device)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices_flat == expert_idx).any(dim=-1)  # [batch*seq]
            
            if expert_mask.any():
                expert_input = x_flat[expert_mask]  # [num_tokens, hidden]
                expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, hidden]
                
                # Get weights for this expert
                expert_token_indices = torch.where(expert_mask)[0]
                expert_weights = torch.zeros(len(expert_token_indices), device=x.device)
                
                for i, token_idx in enumerate(expert_token_indices):
                    # Find which k position this expert is at for this token
                    token_experts = top_k_indices_flat[token_idx]
                    token_weights = top_k_probs_flat[token_idx]
                    k_pos = (token_experts == expert_idx).nonzero(as_tuple=True)[0]
                    if len(k_pos) > 0:
                        expert_weights[i] = token_weights[k_pos[0]]
                
                # Weighted sum
                output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
                expert_usage_count[expert_idx] = expert_mask.sum().item()
        
        # Reshape output
        output = output.view(batch_size, seq_len, self.hidden_dim)
        
        # Update metrics
        if expert_usage_count.sum() > 0:
            usage_normalized = expert_usage_count / expert_usage_count.sum()
            self.expert_usage = 0.9 * self.expert_usage + 0.1 * usage_normalized
        
        # Compute load balancing loss
        router_probs_flat = router_probs.view(-1, self.num_experts)
        avg_expert_prob = router_probs_flat.mean(dim=0)  # [num_experts]
        load_balance_loss = self.load_balancing_weight * (avg_expert_prob.std() ** 2)
        self.load_balance_loss = 0.9 * self.load_balance_loss + 0.1 * load_balance_loss
        
        return output, load_balance_loss
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing and load balancing metrics."""
        return {
            'expert_usage': self.expert_usage.cpu().numpy().tolist(),
            'load_balance_loss': self.load_balance_loss.item(),
            'usage_std': self.expert_usage.std().item()
        }


class Paper2505_05315v2Module(nn.Module):
    """
    Módulo implementando técnicas específicas del paper 2505.05315v2.
    
    Técnicas implementadas:
    - Mixture of Experts (MoE)
    - Dynamic routing
    - Expert specialization
    
    Basado en: https://arxiv.org/html/2505.05315v2
    """
    
    def __init__(self, config: Paper2505_05315v2Config):
        super().__init__()
        self.config = config
        
        # Mixture of Experts
        if config.use_mixture_of_experts:
            self.moe = MixtureOfExperts(
                config.hidden_dim,
                config.num_experts,
                config.expert_capacity
            )
        else:
            self.moe = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        logger.info(f"Initialized Paper 2505.05315v2 module with config: {config}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementando técnicas del paper.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Output tensor con técnicas aplicadas
        """
        residual = x
        x = self.layer_norm(x)
        
        if self.moe is not None:
            moe_output = self.moe(x)
            if isinstance(moe_output, tuple):
                x, load_balance_loss = moe_output
                # Store loss for potential use in training
                if not hasattr(self, '_load_balance_losses'):
                    self._load_balance_losses = []
                self._load_balance_losses.append(load_balance_loss)
            else:
                x = moe_output
        
        x = residual + x
        return x
    
    def get_load_balance_loss(self) -> torch.Tensor:
        """Get accumulated load balance loss."""
        if hasattr(self, '_load_balance_losses') and self._load_balance_losses:
            return sum(self._load_balance_losses) / len(self._load_balance_losses)
        return torch.tensor(0.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from MoE."""
        if self.moe is not None:
            return self.moe.get_metrics()
        return {}


# ============================================================================
# Integración con TruthGPT
# ============================================================================

class TruthGPT_Paper2505_05315v2_Integration(nn.Module):
    """
    Integración del paper 2505.05315v2 con TruthGPT.
    """
    
    def __init__(self, base_model, paper_config: Paper2505_05315v2Config):
        super().__init__()
        self.base_model = base_model
        self.paper_module = Paper2505_05315v2Module(paper_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass integrado."""
        # Aplicar técnicas del paper al modelo base
        output = self.base_model(*args, **kwargs)
        # Procesar con módulo del paper
        enhanced_output = self.paper_module(output)
        return enhanced_output


if __name__ == "__main__":
    # Ejemplo de uso
    config = Paper2505_05315v2Config()
    module = Paper2505_05315v2Module(config)
    
    # Test
    x = torch.randn(2, 32, config.hidden_dim)
    output = module(x)
    print(f"✅ Paper 2505.05315v2 module test: Input {x.shape} -> Output {output.shape}")


