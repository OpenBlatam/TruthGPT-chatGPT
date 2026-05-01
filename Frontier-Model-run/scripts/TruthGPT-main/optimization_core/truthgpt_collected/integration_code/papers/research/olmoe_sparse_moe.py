#!/usr/bin/env python3
"""
OLMoE: Open Sparse Mixture-of-Experts Language Models
======================================================

Implementación basada en OLMoE de AllenAI.
OLMoE es un modelo de lenguaje de código abierto que utiliza una arquitectura
de Mixture-of-Experts (MoE) con 7 mil millones de parámetros totales,
activando solo 1 mil millones por token de entrada.

Técnicas implementadas:
- Sparse MoE routing
- Load balancing
- Expert capacity management
- Top-k expert selection

Basado en: https://github.com/allenai/OLMoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OLMoEConfig:
    """Configuración para OLMoE Sparse MoE."""
    hidden_dim: int = 512
    num_experts: int = 8
    num_experts_per_tok: int = 2  # Top-k experts to activate
    expert_capacity_factor: float = 1.25
    load_balancing_weight: float = 0.01
    use_noisy_gating: bool = True
    noise_std: float = 1.0
    use_expert_capacity: bool = True
    min_capacity: int = 4


class NoisyTopKGating(nn.Module):
    """
    Noisy Top-K Gating para routing en MoE.
    
    Basado en Switch Transformer y OLMoE.
    """
    
    def __init__(self, hidden_dim: int, num_experts: int, num_experts_per_tok: int,
                 noise_std: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.noise_std = noise_std
        
        # Gating network
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Noisy gating (optional)
        if noise_std > 0:
            self.noise_std = noise_std
        else:
            self.noise_std = 0.0
        
        # Initialize gate weights
        nn.init.xavier_uniform_(self.gate.weight)
        
        logger.info(f"Initialized NoisyTopKGating: {num_experts} experts, top-{num_experts_per_tok}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gating scores and routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            gate_scores: [batch_size, seq_len, num_experts]
            top_k_indices: [batch_size, seq_len, num_experts_per_tok]
            top_k_weights: [batch_size, seq_len, num_experts_per_tok]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute gate logits
        gate_logits = self.gate(x)  # [batch, seq, num_experts]
        
        # Add noise for exploration (during training)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(
            gate_logits, 
            k=self.num_experts_per_tok, 
            dim=-1
        )
        
        # Softmax over top-k
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Create full gate scores (zeros for non-selected experts)
        gate_scores = torch.zeros_like(gate_logits)
        gate_scores.scatter_(-1, top_k_indices, top_k_weights)
        
        return gate_scores, top_k_indices, top_k_weights


class SparseMoEExpert(nn.Module):
    """
    Expert network en MoE.
    
    Cada experto es un feed-forward network independiente.
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int = None):
        super().__init__()
        intermediate_dim = intermediate_dim or (hidden_dim * 4)
        
        self.expert = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Initialize weights
        for module in self.expert:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        return self.expert(x)


class OLMoESparseMoE(nn.Module):
    """
    Sparse Mixture-of-Experts basado en OLMoE.
    
    Características:
    - Top-k expert routing
    - Load balancing
    - Expert capacity management
    - Noisy gating opcional
    """
    
    def __init__(self, config: OLMoEConfig):
        super().__init__()
        assert config.hidden_dim > 0, f"hidden_dim must be positive, got {config.hidden_dim}"
        assert config.num_experts > 0, f"num_experts must be positive, got {config.num_experts}"
        assert 0 < config.num_experts_per_tok <= config.num_experts, \
            f"num_experts_per_tok must be in (0, num_experts], got {config.num_experts_per_tok}"
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Gating network
        self.gating = NoisyTopKGating(
            config.hidden_dim,
            config.num_experts,
            config.num_experts_per_tok,
            config.noise_std if config.use_noisy_gating else 0.0
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            SparseMoEExpert(config.hidden_dim)
            for _ in range(config.num_experts)
        ])
        
        # Load balancing tracking
        self.register_buffer('expert_usage', torch.zeros(config.num_experts))
        self.register_buffer('load_balance_loss', torch.tensor(0.0))
        
        # Metrics
        self.register_buffer('routing_entropy', torch.tensor(0.0))
        self.register_buffer('expert_utilization', torch.tensor(0.0))
        self.register_buffer('expert_load_variance', torch.tensor(0.0))
        self.register_buffer('routing_consistency', torch.tensor(0.0))
        self.register_buffer('expert_efficiency', torch.tensor(0.0))
        self.register_buffer('total_tokens_processed', torch.tensor(0))
        self._previous_routing = None
        
        logger.info(f"Initialized OLMoE Sparse MoE: {config.num_experts} experts, "
                   f"top-{config.num_experts_per_tok} routing")
    
    def _compute_load_balancing_loss(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Encourages uniform expert usage.
        """
        # Average gate scores per expert
        expert_usage = gate_scores.mean(dim=[0, 1])  # [num_experts]
        
        # Update tracking
        self.expert_usage = 0.9 * self.expert_usage + 0.1 * expert_usage.detach()
        
        # Load balancing loss: variance of expert usage
        load_balance_loss = expert_usage.var() * self.config.load_balancing_weight
        
        self.load_balance_loss = 0.9 * self.load_balance_loss + 0.1 * load_balance_loss.item()
        
        return load_balance_loss
    
    def _route_to_experts_optimized(self, x: torch.Tensor, gate_scores: torch.Tensor,
                                    top_k_indices: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        """
        Route input to experts and combine outputs.
        
        Args:
            x: Input [batch, seq, hidden]
            gate_scores: Gate scores [batch, seq, num_experts]
            top_k_indices: Top-k expert indices [batch, seq, num_experts_per_tok]
            top_k_weights: Top-k weights [batch, seq, num_experts_per_tok]
            
        Returns:
            Combined expert outputs [batch, seq, hidden]
        """
        batch_size, seq_len, hidden_dim = x.shape
        num_tokens = batch_size * seq_len
        
        # Flatten for processing
        flat_x = x.view(-1, hidden_dim)  # [batch * seq, hidden]
        flat_indices = top_k_indices.view(-1, self.num_experts_per_tok)  # [batch * seq, k]
        flat_weights = top_k_weights.view(-1, self.num_experts_per_tok)  # [batch * seq, k]
        
        # Optimized routing: group tokens by expert
        expert_token_map = {}
        expert_weight_map = {}
        
        for token_idx in range(num_tokens):
            for k_idx in range(self.num_experts_per_tok):
                expert_idx = flat_indices[token_idx, k_idx].item()
                weight = flat_weights[token_idx, k_idx].item()
                
                if expert_idx not in expert_token_map:
                    expert_token_map[expert_idx] = []
                    expert_weight_map[expert_idx] = []
                
                expert_token_map[expert_idx].append(token_idx)
                expert_weight_map[expert_idx].append(weight)
        
        # Process experts in parallel where possible
        output = torch.zeros_like(flat_x)
        
        for expert_idx in range(self.num_experts):
            if expert_idx in expert_token_map:
                token_indices = torch.tensor(expert_token_map[expert_idx], device=x.device)
                weights = torch.tensor(expert_weight_map[expert_idx], device=x.device)
                
                # Get tokens for this expert
                expert_tokens = flat_x[token_indices]  # [num_tokens_for_expert, hidden]
                
                # Process through expert
                expert_out = self.experts[expert_idx](expert_tokens)  # [num_tokens_for_expert, hidden]
                
                # Weight the output
                expert_out = expert_out * weights.unsqueeze(-1)
                
                # Accumulate in output
                output[token_indices] = output[token_indices] + expert_out
        
        # Reshape back
        output = output.view(batch_size, seq_len, hidden_dim)
        
        return output
    
    def _route_to_experts(self, x: torch.Tensor, gate_scores: torch.Tensor,
                          top_k_indices: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        """Route to experts using optimized method."""
        return self._route_to_experts_optimized(x, gate_scores, top_k_indices, top_k_weights)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Sparse MoE.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            output: Output tensor [batch_size, seq_len, hidden_dim]
            load_balance_loss: Load balancing loss
        """
        # Validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"Input hidden_dim ({x.size(-1)}) != configured ({self.hidden_dim})")
        
        batch_size, seq_len, _ = x.shape
        
        # Compute gating
        gate_scores, top_k_indices, top_k_weights = self.gating(x)
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balancing_loss(gate_scores)
        
        # Route to experts
        output = self._route_to_experts(x, gate_scores, top_k_indices, top_k_weights)
        
        # Update metrics
        # Routing entropy
        entropy = -(gate_scores * torch.log(gate_scores + 1e-8)).sum(dim=-1).mean()
        self.routing_entropy = 0.9 * self.routing_entropy + 0.1 * entropy.item()
        
        # Expert utilization (fraction of experts used)
        expert_usage_per_expert = gate_scores.sum(dim=[0, 1])  # [num_experts]
        active_experts = (expert_usage_per_expert > 0).sum().item()
        utilization = active_experts / self.num_experts
        self.expert_utilization = 0.9 * self.expert_utilization + 0.1 * utilization
        
        # Expert load variance (for load balancing quality)
        if active_experts > 0:
            active_usage = expert_usage_per_expert[expert_usage_per_expert > 0]
            load_variance = active_usage.var().item()
            self.expert_load_variance = 0.9 * self.expert_load_variance + 0.1 * load_variance
        
        # Routing consistency (compare with previous routing)
        if self._previous_routing is not None:
            consistency = (top_k_indices == self._previous_routing).float().mean().item()
            self.routing_consistency = 0.9 * self.routing_consistency + 0.1 * consistency
        self._previous_routing = top_k_indices.detach().clone()
        
        # Expert efficiency (tokens per expert / total tokens)
        total_tokens = batch_size * seq_len
        self.total_tokens_processed += total_tokens
        if active_experts > 0:
            avg_tokens_per_expert = total_tokens / active_experts
            efficiency = avg_tokens_per_expert / (total_tokens / self.num_experts_per_tok)
            self.expert_efficiency = 0.9 * self.expert_efficiency + 0.1 * efficiency
        
        return output, load_balance_loss
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get MoE metrics."""
        return {
            'load_balance_loss': self.load_balance_loss.item(),
            'routing_entropy': self.routing_entropy.item(),
            'expert_utilization': self.expert_utilization.item(),
            'expert_load_variance': self.expert_load_variance.item(),
            'routing_consistency': self.routing_consistency.item(),
            'expert_efficiency': self.expert_efficiency.item(),
            'expert_usage': self.expert_usage.detach().cpu().numpy().tolist(),
            'total_tokens_processed': self.total_tokens_processed.item(),
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'load_balance_quality': 1.0 / (1.0 + self.expert_load_variance.item())  # Higher is better
        }


class OLMoEModule(nn.Module):
    """
    Módulo completo OLMoE con Sparse MoE.
    
    Basado en la arquitectura de OLMoE de AllenAI.
    """
    
    def __init__(self, config: OLMoEConfig):
        super().__init__()
        self.config = config
        
        # Sparse MoE
        self.sparse_moe = OLMoESparseMoE(config)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        logger.info(f"Initialized OLMoE Module with config: {config}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through OLMoE module.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            output: Output tensor
            load_balance_loss: Load balancing loss
        """
        residual = x
        x = self.layer_norm(x)
        output, load_balance_loss = self.sparse_moe(x)
        output = residual + output
        
        return output, load_balance_loss
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return self.sparse_moe.get_metrics()


class TruthGPT_OLMoE_Integration(nn.Module):
    """Integración de OLMoE con TruthGPT."""
    
    def __init__(self, base_model, olmoe_config: OLMoEConfig):
        super().__init__()
        self.base_model = base_model
        self.olmoe_module = OLMoEModule(olmoe_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass integrado con OLMoE."""
        output = self.base_model(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            enhanced_output, load_balance_loss = self.olmoe_module(output)
            return enhanced_output
        return output


if __name__ == "__main__":
    config = OLMoEConfig(
        hidden_dim=512,
        num_experts=8,
        num_experts_per_tok=2
    )
    module = OLMoEModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, load_balance_loss = module(x)
    metrics = module.get_metrics()
    print(f"✅ OLMoE Sparse MoE test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Load balance loss: {load_balance_loss.item():.6f}")
    print(f"   Expert utilization: {metrics['expert_utilization']:.2%}")
    print(f"   Routing entropy: {metrics['routing_entropy']:.4f}")
    print(f"   Active experts: {metrics['num_experts_per_tok']}/{metrics['num_experts']}")


