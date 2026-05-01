#!/usr/bin/env python3
"""
Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies - 2025
=============================================================================================

Mejora razonamiento task-specific (+10-15% en benchmarks de multiturn).
Elimina prompts frágiles.

Técnicas principales:
- Framework MoR (Mixture of Reasonings)
- Estrategias adaptativas
- Templates diversos
- Entrenamiento en múltiples estrategias
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
class MixtureOfReasoningsConfig:
    """Configuración para Mixture of Reasonings (MoR)."""
    hidden_dim: int = 512
    num_strategies: int = 5
    strategy_dim: int = 256
    use_adaptive_selection: bool = True
    temperature: float = 1.0


class ReasoningStrategy(nn.Module):
    """Estrategia individual de razonamiento."""
    
    def __init__(self, config: MixtureOfReasoningsConfig, strategy_id: int):
        super().__init__()
        self.config = config
        self.strategy_id = strategy_id
        
        # Strategy-specific processor
        self.processor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.strategy_dim),
            nn.GELU(),
            nn.Linear(config.strategy_dim, config.hidden_dim)
        )
        
        # Strategy quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq, hidden_dim]
        Returns:
            processed: [batch, seq, hidden_dim]
            quality_score: [batch]
        """
        processed = self.processor(hidden_states)
        last_token = processed[:, -1, :]  # [batch, hidden_dim]
        quality = self.quality_scorer(last_token).squeeze(-1)  # [batch]
        return processed, quality


class MixtureOfReasoningsModule(nn.Module):
    """
    Módulo Mixture of Reasonings con estrategias adaptativas.
    """
    
    def __init__(self, config: MixtureOfReasoningsConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Multiple reasoning strategies
        self.strategies = nn.ModuleList([
            ReasoningStrategy(config, i) for i in range(config.num_strategies)
        ])
        
        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.num_strategies),
            nn.Softmax(dim=-1)
        )
        
        # Strategy mixer
        self.strategy_mixer = nn.Sequential(
            nn.Linear(config.hidden_dim * config.num_strategies, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Metrics
        self.register_buffer('strategy_usage', torch.zeros(config.num_strategies))
        self.register_buffer('reasoning_quality', torch.tensor(0.5))
        self.register_buffer('adaptive_selection_rate', torch.tensor(0.0))
        
        logger.info(f"Initialized MixtureOfReasoningsModule with {config.num_strategies} strategies")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: adaptive mixture of reasoning strategies.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with strategy info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get strategy selection weights
        last_token = hidden_states[:, -1, :]  # [batch, hidden_dim]
        strategy_weights = self.strategy_selector(last_token)  # [batch, num_strategies]
        
        # Apply all strategies
        strategy_outputs = []
        strategy_qualities = []
        
        for strategy in self.strategies:
            processed, quality = strategy(hidden_states)
            strategy_outputs.append(processed)
            strategy_qualities.append(quality)
        
        # Stack outputs
        stacked = torch.stack(strategy_outputs, dim=1)  # [batch, num_strategies, seq, hidden_dim]
        
        # Weighted combination
        weights = strategy_weights.unsqueeze(-1).unsqueeze(-1)  # [batch, num_strategies, 1, 1]
        weighted = (stacked * weights).sum(dim=1)  # [batch, seq, hidden_dim]
        
        # Alternative: mix all strategies
        stacked_flat = stacked.permute(0, 2, 1, 3).contiguous()  # [batch, seq, num_strategies, hidden_dim]
        stacked_flat = stacked_flat.view(batch_size, seq_len, -1)  # [batch, seq, num_strategies * hidden_dim]
        mixed = self.strategy_mixer(stacked_flat)  # [batch, seq, hidden_dim]
        
        # Combine both approaches
        enhanced_states = hidden_states + 0.3 * weighted + 0.2 * mixed
        
        # Update metrics
        selected_strategy = strategy_weights.argmax(dim=-1)[0].item()
        self.strategy_usage[selected_strategy] += 1
        
        avg_quality = torch.stack(strategy_qualities).mean().item()
        self.reasoning_quality = 0.9 * self.reasoning_quality + 0.1 * avg_quality
        
        metadata = {
            'selected_strategy': selected_strategy,
            'strategy_weights': strategy_weights[0].cpu().tolist(),
            'avg_quality': avg_quality,
            'num_strategies': self.config.num_strategies
        }
        
        return enhanced_states, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        usage = self.strategy_usage.cpu().numpy()
        total = usage.sum()
        if total > 0:
            usage_normalized = (usage / total).tolist()
        else:
            usage_normalized = [0.0] * len(usage)
        
        return {
            'strategy_usage': usage_normalized,
            'reasoning_quality': self.reasoning_quality.item(),
            'adaptive_selection_rate': self.adaptive_selection_rate.item(),
            'num_strategies': self.config.num_strategies
        }


if __name__ == "__main__":
    config = MixtureOfReasoningsConfig(
        hidden_dim=512,
        num_strategies=5,
        use_adaptive_selection=True
    )
    module = MixtureOfReasoningsModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ Mixture of Reasonings test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Selected strategy: {metadata['selected_strategy']}")
    print(f"   Avg quality: {metadata['avg_quality']:.4f}")



