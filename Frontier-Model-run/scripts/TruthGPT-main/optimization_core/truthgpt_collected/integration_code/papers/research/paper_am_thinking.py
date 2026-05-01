#!/usr/bin/env python3
"""
AM-Thinking-v1: Advancing the Frontier of Reasoning at 32B Scale
================================================================

Ji, Tian, Zhao, Wang, Chen, Peng, Zhao, Li. May 2025. arXiv

Modelo denso de 32B que alcanza muy buenas cifras en AIME (2024 y 2025) y LiveCodeBench,
compitiendo con modelos mucho más grandes con una pipeline de SFT + RL.

Técnica principal: Dense 32B model with SFT + RL pipeline for reasoning.
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
class AMThinkingConfig:
    """Configuración para AM-Thinking-v1."""
    hidden_dim: int = 512
    num_layers: int = 24  # For 32B scale
    num_heads: int = 16
    intermediate_size: int = 2048
    use_sft_training: bool = True
    use_rl_training: bool = True
    rl_kl_penalty: float = 0.1
    rl_reward_scale: float = 1.0
    use_reasoning_heads: bool = True
    reasoning_head_dim: int = 256
    reasoning_dim: int = 256  # Alias for reasoning_head_dim


class ReasoningHead(nn.Module):
    """
    Cabeza de razonamiento especializada.
    """
    
    def __init__(self, config: AMThinkingConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.reasoning_dim = getattr(config, 'reasoning_dim', config.reasoning_head_dim)
        
        # Reasoning network
        self.reasoning_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.reasoning_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.reasoning_dim * 2, config.reasoning_dim)
        )
        
        # Reasoning quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(config.reasoning_dim, config.reasoning_dim // 2),
            nn.GELU(),
            nn.Linear(config.reasoning_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in self.reasoning_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized ReasoningHead")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply reasoning.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            reasoned_states: [batch, seq, reasoning_dim]
            quality_scores: [batch, seq]
        """
        # Apply reasoning
        reasoned = self.reasoning_net(hidden_states)  # [batch, seq, reasoning_dim]
        
        # Score quality
        quality = self.quality_scorer(reasoned).squeeze(-1)  # [batch, seq]
        
        return reasoned, quality


class TransformerBlock(nn.Module):
    """
    Bloque Transformer para AM-Thinking.
    """
    
    def __init__(self, config: AMThinkingConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        logger.info("Initialized TransformerBlock")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attended, _ = self.attention(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + attended
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states


class AMThinkingModule(nn.Module):
    """
    Módulo AM-Thinking-v1 completo.
    """
    
    def __init__(self, config: AMThinkingConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        
        # Reasoning heads
        if config.use_reasoning_heads:
            self.reasoning_head = ReasoningHead(config)
        else:
            self.reasoning_head = None
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Metrics
        self.register_buffer('reasoning_quality', torch.tensor(0.5))
        self.register_buffer('sft_loss', torch.tensor(0.0))
        self.register_buffer('rl_reward', torch.tensor(0.0))
        
        logger.info(f"Initialized AMThinkingModule: {config.num_layers} layers")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: AM-Thinking reasoning.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with reasoning info
        """
        # Apply transformer layers
        x = hidden_states
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Apply reasoning head if enabled
        reasoning_quality = None
        if self.reasoning_head is not None:
            reasoned, quality_scores = self.reasoning_head(x)
            reasoning_quality = quality_scores.mean().item()
            self.reasoning_quality = 0.9 * self.reasoning_quality + 0.1 * reasoning_quality
            
            # Project reasoning back to hidden_dim
            reasoned_proj = nn.Linear(self.config.reasoning_head_dim, self.hidden_dim).to(x.device)
            reasoned_hidden = reasoned_proj(reasoned)
            x = x + 0.2 * reasoned_hidden
        
        # Output projection
        output = self.output_projection(x)
        
        # Combine with original
        output = hidden_states + 0.3 * output
        
        metadata = {
            'reasoning_quality': reasoning_quality,
            'num_layers': self.config.num_layers
        }
        
        return output, metadata
    
    def compute_sft_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised fine-tuning loss.
        
        Args:
            logits: [batch, seq, vocab_size]
            labels: [batch, seq]
            
        Returns:
            loss: Scalar loss
        """
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        self.sft_loss = 0.9 * self.sft_loss + 0.1 * loss.item()
        return loss
    
    def compute_rl_loss(self, logits: torch.Tensor, rewards: torch.Tensor, 
                        ref_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute RL loss (PPO-style).
        
        Args:
            logits: [batch, seq, vocab_size]
            rewards: [batch] - rewards for each sequence
            ref_logits: Optional reference logits for KL penalty
            
        Returns:
            loss: Scalar loss
        """
        # Get action probabilities
        probs = F.softmax(logits, dim=-1)  # [batch, seq, vocab_size]
        
        # Sample actions (for simplicity, use most likely token)
        actions = probs.argmax(dim=-1)  # [batch, seq]
        
        # Compute policy loss
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [batch, seq]
        
        # Average over sequence
        avg_log_probs = selected_log_probs.mean(dim=1)  # [batch]
        
        # Policy loss
        policy_loss = -(avg_log_probs * rewards).mean()
        
        # KL penalty
        kl_loss = torch.tensor(0.0, device=logits.device)
        if ref_logits is not None:
            ref_probs = F.softmax(ref_logits, dim=-1)
            kl = (probs * (torch.log(probs + 1e-8) - torch.log(ref_probs + 1e-8))).sum(dim=-1).mean()
            kl_loss = self.config.rl_kl_penalty * kl
        
        # Total loss
        total_loss = policy_loss + kl_loss
        
        self.rl_reward = 0.9 * self.rl_reward + 0.1 * rewards.mean().item()
        
        return total_loss
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            'reasoning_quality': self.reasoning_quality.item(),
            'sft_loss': self.sft_loss.item(),
            'rl_reward': self.rl_reward.item(),
            'num_layers': self.config.num_layers
        }


if __name__ == "__main__":
    config = AMThinkingConfig(
        hidden_dim=512,
        num_layers=24,
        use_reasoning_heads=True
    )
    module = AMThinkingModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ AM-Thinking test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Reasoning quality: {metadata['reasoning_quality']:.4f}")
    print(f"   Num layers: {metadata['num_layers']}")


