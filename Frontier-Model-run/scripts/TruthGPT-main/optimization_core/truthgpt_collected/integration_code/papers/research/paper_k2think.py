#!/usr/bin/env python3
"""
K2-Think: A Parameter-Efficient Reasoning System
=================================================

No es un paper de conferencia (tech report), pero es relevante. Usa Qwen2.5-32B y
razonamiento eficiente con múltiples rollouts para mejorar su desempeño en AIME-2024
entre otros.

Muy útil para sistemas donde no se puede sacrificar muchos parámetros.

Técnica principal: Parameter-efficient reasoning with multiple rollouts.
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
class K2ThinkConfig:
    """Configuración para K2-Think."""
    hidden_dim: int = 512
    num_rollouts: int = 5
    use_parameter_efficient: bool = True
    adapter_dim: int = 64
    use_rollout_aggregation: bool = True
    aggregation_method: str = "weighted"  # weighted, average, best
    use_confidence_weighting: bool = True


class ParameterEfficientAdapter(nn.Module):
    """
    Adaptador eficiente en parámetros para razonamiento.
    """
    
    def __init__(self, config: K2ThinkConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.adapter_dim = config.adapter_dim
        
        # Adapter (bottleneck architecture)
        self.down_proj = nn.Linear(config.hidden_dim, config.adapter_dim)
        self.up_proj = nn.Linear(config.adapter_dim, config.hidden_dim)
        
        # Activation
        self.activation = nn.GELU()
        
        # Initialize
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        
        logger.info(f"Initialized ParameterEfficientAdapter: adapter_dim={config.adapter_dim}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply parameter-efficient adapter.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            adapted_states: [batch, seq, hidden_dim]
        """
        # Adapter forward
        adapted = self.down_proj(hidden_states)  # [batch, seq, adapter_dim]
        adapted = self.activation(adapted)
        adapted = self.up_proj(adapted)  # [batch, seq, hidden_dim]
        
        return adapted


class RolloutReasoner(nn.Module):
    """
    Genera múltiples rollouts de razonamiento.
    """
    
    def __init__(self, config: K2ThinkConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Reasoning network (lightweight)
        self.reasoner = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Confidence estimator
        if config.use_confidence_weighting:
            self.confidence_estimator = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.confidence_estimator = None
        
        # Initialize
        for module in self.reasoner:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info(f"Initialized RolloutReasoner: num_rollouts={config.num_rollouts}")
    
    def forward(self, hidden_states: torch.Tensor, num_rollouts: int = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Generate multiple reasoning rollouts.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            num_rollouts: Number of rollouts to generate
            
        Returns:
            rollouts: List of [batch, seq, hidden_dim] tensors
            confidences: [batch, num_rollouts]
        """
        num_rollouts = num_rollouts or self.config.num_rollouts
        batch_size, seq_len, _ = hidden_states.shape
        
        rollouts = []
        confidences = []
        
        for i in range(num_rollouts):
            # Add small noise for diversity
            noise = torch.randn_like(hidden_states) * 0.01
            noisy_states = hidden_states + noise
            
            # Reason
            reasoned = self.reasoner(noisy_states)  # [batch, seq, hidden_dim]
            rollouts.append(reasoned)
            
            # Estimate confidence
            if self.confidence_estimator is not None:
                last_token = reasoned[:, -1, :]  # [batch, hidden_dim]
                confidence = self.confidence_estimator(last_token).squeeze(-1)  # [batch]
                confidences.append(confidence)
            else:
                confidences.append(torch.ones(batch_size, device=hidden_states.device))
        
        confidence_tensor = torch.stack(confidences, dim=1)  # [batch, num_rollouts]
        
        return rollouts, confidence_tensor


class RolloutAggregator(nn.Module):
    """
    Agrega múltiples rollouts en una solución final.
    """
    
    def __init__(self, config: K2ThinkConfig):
        super().__init__()
        self.config = config
        self.aggregation_method = config.aggregation_method
        
        logger.info(f"Initialized RolloutAggregator: method={config.aggregation_method}")
    
    def forward(self, rollouts: List[torch.Tensor], confidences: torch.Tensor) -> torch.Tensor:
        """
        Aggregate rollouts.
        
        Args:
            rollouts: List of [batch, seq, hidden_dim] tensors
            confidences: [batch, num_rollouts]
            
        Returns:
            aggregated: [batch, seq, hidden_dim]
        """
        if len(rollouts) == 0:
            return torch.zeros_like(rollouts[0]) if rollouts else None
        
        stacked = torch.stack(rollouts, dim=0)  # [num_rollouts, batch, seq, hidden_dim]
        
        if self.aggregation_method == "weighted":
            # Weight by confidence
            confidence_weights = F.softmax(confidences, dim=1)  # [batch, num_rollouts]
            confidence_weights = confidence_weights.unsqueeze(-1).unsqueeze(-1)  # [batch, num_rollouts, 1, 1]
            aggregated = (stacked * confidence_weights).sum(dim=0)  # [batch, seq, hidden_dim]
        
        elif self.aggregation_method == "best":
            # Use best rollout (highest confidence)
            best_indices = confidences.argmax(dim=1)  # [batch]
            aggregated = stacked[best_indices, torch.arange(stacked.size(1)), :, :]  # [batch, seq, hidden_dim]
        
        else:  # average
            # Simple average
            aggregated = stacked.mean(dim=0)  # [batch, seq, hidden_dim]
        
        return aggregated


class K2ThinkModule(nn.Module):
    """
    Módulo K2-Think completo.
    """
    
    def __init__(self, config: K2ThinkConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Components
        if config.use_parameter_efficient:
            self.adapter = ParameterEfficientAdapter(config)
        else:
            self.adapter = None
        
        self.reasoner = RolloutReasoner(config)
        
        if config.use_rollout_aggregation:
            self.aggregator = RolloutAggregator(config)
        else:
            self.aggregator = None
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Metrics
        self.register_buffer('avg_confidence', torch.tensor(0.5))
        self.register_buffer('rollout_diversity', torch.tensor(0.0))
        self.register_buffer('parameter_efficiency', torch.tensor(1.0))
        
        logger.info(f"Initialized K2ThinkModule: num_rollouts={config.num_rollouts}")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: parameter-efficient reasoning with rollouts.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with reasoning info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply adapter if enabled
        if self.adapter is not None:
            adapted = self.adapter(hidden_states)
            reasoning_input = adapted
        else:
            reasoning_input = hidden_states
        
        # Generate rollouts
        rollouts, confidences = self.reasoner(reasoning_input, self.config.num_rollouts)
        
        # Aggregate rollouts
        if self.aggregator is not None:
            aggregated = self.aggregator(rollouts, confidences)
        else:
            aggregated = rollouts[0]  # Use first rollout
        
        # Project output
        output = self.output_projection(aggregated)
        
        # Combine with original
        output = hidden_states + 0.3 * output
        
        # Update metrics
        avg_confidence = confidences.mean().item()
        self.avg_confidence = 0.9 * self.avg_confidence + 0.1 * avg_confidence
        
        # Rollout diversity (variance across rollouts)
        if len(rollouts) > 1:
            stacked = torch.stack(rollouts, dim=0)  # [num_rollouts, batch, seq, hidden_dim]
            variance = stacked.var(dim=0).mean().item()
            self.rollout_diversity = 0.9 * self.rollout_diversity + 0.1 * variance
        
        # Parameter efficiency (ratio of adapter params to full params)
        if self.adapter is not None:
            adapter_params = sum(p.numel() for p in self.adapter.parameters())
            full_params = self.hidden_dim * self.hidden_dim * 2  # Approximate
            efficiency = adapter_params / (full_params + 1e-8)
            self.parameter_efficiency = 0.9 * self.parameter_efficiency + 0.1 * efficiency
        
        metadata = {
            'num_rollouts': self.config.num_rollouts,
            'avg_confidence': avg_confidence,
            'rollout_diversity': self.rollout_diversity.item()
        }
        
        return output, metadata
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for efficiency analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters()) if self.adapter else 0
        
        return {
            'total_params': total_params,
            'adapter_params': adapter_params,
            'efficiency_ratio': adapter_params / (total_params + 1e-8)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            'avg_confidence': self.avg_confidence.item(),
            'rollout_diversity': self.rollout_diversity.item(),
            'parameter_efficiency': self.parameter_efficiency.item(),
            'num_rollouts': self.config.num_rollouts,
            'parameter_count': self.get_parameter_count()
        }


if __name__ == "__main__":
    config = K2ThinkConfig(
        hidden_dim=512,
        num_rollouts=5,
        use_parameter_efficient=True
    )
    module = K2ThinkModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ K2-Think test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Rollouts: {metadata['num_rollouts']}")
    print(f"   Avg confidence: {metadata['avg_confidence']:.4f}")
    print(f"   Parameter efficiency: {metrics['parameter_efficiency']:.4f}")



