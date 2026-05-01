#!/usr/bin/env python3
"""
SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training - 2025
=============================================================================================

RL mejora OOD y reconocimiento visual (+8-12% en tareas de razonamiento textual/visual vs. SFT).

Técnicas principales:
- Estudio empírico comparando SFT y RL
- Generalización OOD
- Reconocimiento visual mejorado
- Post-training con RL
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
class SFTRLGeneralizationConfig:
    """Configuración para SFT vs RL Generalization."""
    hidden_dim: int = 512
    use_rl_training: bool = True
    use_ood_detection: bool = True
    reward_scale: float = 1.0
    generalization_weight: float = 0.5


class OODDetector(nn.Module):
    """Detector de datos fuera de distribución (OOD)."""
    
    def __init__(self, config: SFTRLGeneralizationConfig):
        super().__init__()
        self.config = config
        
        self.detector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info("Initialized OODDetector")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden_dim]
        Returns:
            ood_scores: [batch] - probability of being OOD
        """
        last_token = hidden_states[:, -1, :]
        ood_scores = self.detector(last_token).squeeze(-1)
        return ood_scores


class RLPolicy(nn.Module):
    """Policy network para RL training."""
    
    def __init__(self, config: SFTRLGeneralizationConfig):
        super().__init__()
        self.config = config
        
        self.policy = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Value function
        self.value_function = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        logger.info("Initialized RLPolicy")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq, hidden_dim]
        Returns:
            policy_output: [batch, seq, hidden_dim]
            value: [batch]
        """
        policy_output = self.policy(hidden_states)
        last_token = hidden_states[:, -1, :]
        value = self.value_function(last_token).squeeze(-1)
        return policy_output, value


class SFTRLGeneralizationModule(nn.Module):
    """
    Módulo comparando SFT y RL para generalización.
    """
    
    def __init__(self, config: SFTRLGeneralizationConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Components
        if config.use_rl_training:
            self.rl_policy = RLPolicy(config)
        else:
            self.rl_policy = None
        
        if config.use_ood_detection:
            self.ood_detector = OODDetector(config)
        else:
            self.ood_detector = None
        
        # SFT-style processor (for comparison)
        self.sft_processor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Metrics
        self.register_buffer('generalization_score', torch.tensor(0.5))
        self.register_buffer('ood_detection_rate', torch.tensor(0.0))
        self.register_buffer('rl_advantage', torch.tensor(0.0))
        
        logger.info("Initialized SFTRLGeneralizationModule")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: RL-based generalization with OOD detection.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with generalization info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # OOD detection
        if self.ood_detector:
            ood_scores = self.ood_detector(hidden_states)  # [batch]
            ood_mask = (ood_scores > 0.5).float()
        else:
            ood_scores = torch.zeros(batch_size, device=hidden_states.device)
            ood_mask = torch.zeros(batch_size, device=hidden_states.device)
        
        # RL policy (better for generalization)
        if self.rl_policy:
            rl_output, rl_value = self.rl_policy(hidden_states)
        else:
            rl_output = hidden_states
            rl_value = torch.zeros(batch_size, device=hidden_states.device)
        
        # SFT processor (for comparison)
        sft_output = self.sft_processor(hidden_states)
        
        # Combine: use RL more for OOD cases
        ood_mask_expanded = ood_mask.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        rl_weight = 0.7 * ood_mask_expanded + 0.3 * (1 - ood_mask_expanded)
        sft_weight = 1.0 - rl_weight
        
        enhanced_states = hidden_states + rl_weight * 0.3 * rl_output + sft_weight * 0.2 * sft_output
        
        # Update metrics
        self.generalization_score = 0.9 * self.generalization_score + 0.1 * (1 - ood_scores.mean())
        self.ood_detection_rate = 0.9 * self.ood_detection_rate + 0.1 * ood_mask.mean()
        self.rl_advantage = 0.9 * self.rl_advantage + 0.1 * rl_value.mean()
        
        metadata = {
            'ood_detection_rate': ood_mask.mean().item(),
            'rl_value': rl_value.mean().item(),
            'generalization_score': (1 - ood_scores.mean()).item(),
            'use_rl': self.config.use_rl_training
        }
        
        return enhanced_states, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            'generalization_score': self.generalization_score.item(),
            'ood_detection_rate': self.ood_detection_rate.item(),
            'rl_advantage': self.rl_advantage.item(),
            'use_rl_training': self.config.use_rl_training
        }


if __name__ == "__main__":
    config = SFTRLGeneralizationConfig(
        hidden_dim=512,
        use_rl_training=True,
        use_ood_detection=True
    )
    module = SFTRLGeneralizationModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ SFT vs RL Generalization test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   OOD detection rate: {metadata['ood_detection_rate']:.4f}")
    print(f"   RL advantage: {metadata['rl_value']:.4f}")



