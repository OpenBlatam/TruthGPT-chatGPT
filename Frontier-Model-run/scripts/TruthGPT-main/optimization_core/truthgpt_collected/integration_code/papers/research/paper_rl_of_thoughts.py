#!/usr/bin/env python3
"""
RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning
===================================================================================

Hao, Li, Yuan, Li. May 2025. arXiv

Entrenan un "navegador" ligero con RL para elegir dinámicamente entre diferentes
bloques de razonamiento (cadena, árbol, etc) según la tarea. Logran mejoras de
hasta +13.4% en benchmarks como AIME, MATH y GPQA.

Técnica principal: Lightweight RL navigator for dynamic reasoning block selection.

ArXiv ID: 2501.12948
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
class RLOfThoughtsConfig:
    """Configuración para RL of Thoughts."""
    hidden_dim: int = 512
    num_reasoning_blocks: int = 4  # chain, tree, graph, hybrid
    navigator_dim: int = 128
    use_value_function: bool = True
    use_advantage_estimation: bool = True
    exploration_rate: float = 0.1
    discount_factor: float = 0.99
    temperature: float = 1.0


class ReasoningBlock(nn.Module):
    """
    Bloque de razonamiento individual.
    """
    
    def __init__(self, config: RLOfThoughtsConfig, block_type: str):
        super().__init__()
        self.config = config
        self.block_type = block_type
        self.hidden_dim = config.hidden_dim
        
        # Block-specific processing
        self.processor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Initialize
        for module in self.processor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info(f"Initialized ReasoningBlock: {block_type}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process with reasoning block."""
        return self.processor(hidden_states)


class RLNavigator(nn.Module):
    """
    Navegador RL ligero para seleccionar bloques de razonamiento.
    """
    
    def __init__(self, config: RLOfThoughtsConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.navigator_dim = config.navigator_dim
        self.num_blocks = config.num_reasoning_blocks
        
        # Policy network (selects reasoning block)
        self.policy_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.navigator_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.navigator_dim, config.num_reasoning_blocks),
            nn.Softmax(dim=-1)
        )
        
        # Value network (estimates value of state)
        if config.use_value_function:
            self.value_net = nn.Sequential(
                nn.Linear(config.hidden_dim, config.navigator_dim),
                nn.GELU(),
                nn.Linear(config.navigator_dim, 1)
            )
        else:
            self.value_net = None
        
        # Initialize
        for module in self.policy_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        if self.value_net is not None:
            for module in self.value_net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        
        logger.info(f"Initialized RLNavigator: {config.num_reasoning_blocks} blocks")
    
    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Navigate: select reasoning block.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            training: Whether in training mode
            
        Returns:
            action_probs: [batch, num_blocks]
            selected_action: [batch]
            value_estimate: [batch] or None
        """
        # Use last token for navigation
        last_token = hidden_states[:, -1, :]  # [batch, hidden_dim]
        
        # Get action probabilities
        action_probs = self.policy_net(last_token)  # [batch, num_blocks]
        
        # Select action
        if training:
            # Sample with exploration
            if torch.rand(1).item() < self.config.exploration_rate:
                selected_action = torch.randint(0, self.num_blocks, (last_token.size(0),), device=last_token.device)
            else:
                selected_action = torch.multinomial(action_probs, 1).squeeze(-1)
        else:
            # Greedy selection
            selected_action = action_probs.argmax(dim=-1)
        
        # Get value estimate
        value_estimate = None
        if self.value_net is not None:
            value_estimate = self.value_net(last_token).squeeze(-1)  # [batch]
        
        return action_probs, selected_action, value_estimate


class RLOfThoughtsModule(nn.Module):
    """
    Módulo RL of Thoughts completo.
    """
    
    def __init__(self, config: RLOfThoughtsConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # RL Navigator
        self.navigator = RLNavigator(config)
        
        # Reasoning blocks
        block_types = ["chain", "tree", "graph", "hybrid"]
        self.reasoning_blocks = nn.ModuleList([
            ReasoningBlock(config, block_types[i % len(block_types)])
            for i in range(config.num_reasoning_blocks)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Metrics
        self.register_buffer('block_usage', torch.zeros(config.num_reasoning_blocks))
        self.register_buffer('avg_value', torch.tensor(0.0))
        self.register_buffer('navigation_confidence', torch.tensor(0.5))
        
        logger.info(f"Initialized RLOfThoughtsModule: {config.num_reasoning_blocks} blocks")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: RL-guided reasoning.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with RL info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Navigate: select reasoning block
        action_probs, selected_action, value_estimate = self.navigator(hidden_states, training=self.training)
        
        # Apply selected reasoning blocks
        reasoned_outputs = []
        for i in range(batch_size):
            block_idx = selected_action[i].item()
            self.block_usage[block_idx] += 1
            
            # Get reasoning block
            reasoning_block = self.reasoning_blocks[block_idx]
            
            # Apply to this sample
            sample_states = hidden_states[i:i+1, :, :]  # [1, seq, hidden_dim]
            reasoned = reasoning_block(sample_states)  # [1, seq, hidden_dim]
            reasoned_outputs.append(reasoned)
        
        # Stack results
        reasoned_output = torch.cat(reasoned_outputs, dim=0)  # [batch, seq, hidden_dim]
        
        # Project output
        output = self.output_projection(reasoned_output)
        
        # Combine with original
        output = hidden_states + 0.3 * output
        
        # Update metrics
        if value_estimate is not None:
            self.avg_value = 0.9 * self.avg_value + 0.1 * value_estimate.mean().item()
        
        # Navigation confidence (entropy of action probs)
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean().item()
        confidence = 1.0 - action_entropy / math.log(self.config.num_reasoning_blocks)
        self.navigation_confidence = 0.9 * self.navigation_confidence + 0.1 * confidence
        
        metadata = {
            'selected_actions': selected_action.cpu().numpy().tolist(),
            'action_probs': action_probs.mean(dim=0).cpu().numpy().tolist(),
            'value_estimate': value_estimate.mean().item() if value_estimate is not None else None,
            'navigation_confidence': confidence
        }
        
        return output, metadata
    
    def compute_rl_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                       rewards: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute RL loss for training.
        
        Args:
            states: [batch, seq, hidden_dim]
            actions: [batch] - selected actions
            rewards: [batch] - rewards
            next_states: Optional [batch, seq, hidden_dim]
            
        Returns:
            Dict with loss components
        """
        # Get action probabilities and values
        action_probs, _, value_estimate = self.navigator(states, training=True)
        
        # Policy loss (REINFORCE)
        log_probs = torch.log(action_probs + 1e-8)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Advantage estimation
        if self.config.use_advantage_estimation and value_estimate is not None:
            advantages = rewards - value_estimate.detach()
        else:
            advantages = rewards
        
        # Policy loss
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Value loss
        value_loss = torch.tensor(0.0, device=states.device)
        if self.value_net is not None and value_estimate is not None:
            value_loss = F.mse_loss(value_estimate, rewards)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        total_usage = self.block_usage.sum().item()
        block_usage_pct = (self.block_usage / (total_usage + 1e-8)).cpu().numpy().tolist()
        
        return {
            'block_usage': block_usage_pct,
            'avg_value': self.avg_value.item(),
            'navigation_confidence': self.navigation_confidence.item(),
            'num_blocks': self.config.num_reasoning_blocks
        }


if __name__ == "__main__":
    config = RLOfThoughtsConfig(
        hidden_dim=512,
        num_reasoning_blocks=4,
        use_value_function=True
    )
    module = RLOfThoughtsModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ RL of Thoughts test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Selected actions: {metadata['selected_actions']}")
    print(f"   Navigation confidence: {metadata['navigation_confidence']:.4f}")



