#!/usr/bin/env python3
"""
SOLAR: Scalable Optimization of Large-scale Architecture for Reasoning
=======================================================================

Li, Luo, Bolimera, Ahmed, Srinivasan, Gokhale, Savvides. Mar 2025. arXiv

Optimiza dinámicamente la estructura de razonamiento (chain, tree, graph) y usa
aprendizaje curricular para mejorar tareas como MATH y GSM8K, con ganancias notables
en precisión y eficiencia.

Técnica principal: Dynamic structure optimization + curriculum learning for reasoning.
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
class SOLARConfig:
    """Configuración para SOLAR."""
    hidden_dim: int = 512
    num_reasoning_layers: int = 3
    structure_types: List[str] = None  # ["chain", "tree", "graph"]
    use_curriculum_learning: bool = True
    curriculum_schedule: str = "linear"  # linear, exponential, adaptive
    use_dynamic_structure: bool = True
    structure_selector_dim: int = 128
    temperature: float = 1.0

    def __post_init__(self):
        if self.structure_types is None:
            self.structure_types = ["chain", "tree", "graph"]


class StructureSelector(nn.Module):
    """
    Selector dinámico de estructura de razonamiento.
    """
    
    def __init__(self, config: SOLARConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.structure_types = config.structure_types
        
        # Structure selector network
        self.selector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.structure_selector_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.structure_selector_dim, len(config.structure_types)),
            nn.Softmax(dim=-1)
        )
        
        # Initialize
        for module in self.selector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info(f"Initialized StructureSelector: {config.structure_types}")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Select optimal structure.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            structure_probs: [batch, num_structures]
            selected_structure: str
        """
        # Use last token for selection
        last_token = hidden_states[:, -1, :]  # [batch, hidden_dim]
        
        # Get structure probabilities
        structure_probs = self.selector(last_token)  # [batch, num_structures]
        
        # Select structure (greedy or sample)
        if self.training:
            # Sample during training
            structure_idx = torch.multinomial(structure_probs, 1).squeeze(-1)  # [batch]
        else:
            # Greedy during inference
            structure_idx = structure_probs.argmax(dim=-1)  # [batch]
        
        # Get most common structure
        selected_idx = structure_idx.mode().values.item()
        selected_structure = self.structure_types[selected_idx]
        
        return structure_probs, selected_structure


class CurriculumScheduler:
    """
    Scheduler de aprendizaje curricular.
    """
    
    def __init__(self, config: SOLARConfig, total_steps: int = 10000):
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.schedule = config.curriculum_schedule
    
    def get_difficulty(self) -> float:
        """
        Get current difficulty level (0.0 = easy, 1.0 = hard).
        """
        if not self.config.use_curriculum_learning:
            return 1.0
        
        progress = min(self.current_step / self.total_steps, 1.0)
        
        if self.schedule == "linear":
            difficulty = progress
        elif self.schedule == "exponential":
            difficulty = 1.0 - math.exp(-3.0 * progress)
        else:  # adaptive
            # Start easy, ramp up
            difficulty = min(progress * 1.5, 1.0)
        
        return difficulty
    
    def step(self):
        """Update scheduler step."""
        self.current_step += 1


class ReasoningLayer(nn.Module):
    """
    Capa de razonamiento con estructura optimizable.
    """
    
    def __init__(self, config: SOLARConfig, structure_type: str = "chain"):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.structure_type = structure_type
        
        # Reasoning network
        self.reasoning_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Structure-specific components
        if structure_type == "graph":
            self.graph_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
        
        # Initialize
        for module in self.reasoning_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info(f"Initialized ReasoningLayer: {structure_type}")
    
    def _chain_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Chain reasoning: sequential."""
        # Process sequentially
        output = hidden_states[:, 0:1, :]
        for i in range(1, hidden_states.size(1)):
            current = hidden_states[:, i:i+1, :]
            combined = torch.cat([output[:, -1:, :], current], dim=1)
            processed = self.reasoning_net(combined)
            output = torch.cat([output, processed[:, -1:, :]], dim=1)
        return output
    
    def _tree_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Tree reasoning: hierarchical."""
        # Hierarchical processing
        current = hidden_states
        while current.size(1) > 1:
            # Process pairs
            num_pairs = current.size(1) // 2
            pairs = []
            for i in range(num_pairs):
                pair = current[:, [i*2, i*2+1], :]
                processed = self.reasoning_net(pair.mean(dim=1, keepdim=True))
                pairs.append(processed.squeeze(1))
            
            if current.size(1) % 2 == 1:
                pairs.append(current[:, -1, :])
            
            current = torch.stack(pairs, dim=1) if len(pairs) > 1 else pairs[0].unsqueeze(1)
        
        # Expand back to original length
        return current.expand(-1, hidden_states.size(1), -1)
    
    def _graph_reasoning(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Graph reasoning: fully connected."""
        # Graph attention
        attended, _ = self.graph_attention(hidden_states, hidden_states, hidden_states)
        # Process with reasoning net
        output = self.reasoning_net(attended)
        return output
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply reasoning based on structure type."""
        if self.structure_type == "chain":
            return self._chain_reasoning(hidden_states)
        elif self.structure_type == "tree":
            return self._tree_reasoning(hidden_states)
        else:  # graph
            return self._graph_reasoning(hidden_states)


class SOLARModule(nn.Module):
    """
    Módulo SOLAR completo.
    """
    
    def __init__(self, config: SOLARConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Structure selector
        if config.use_dynamic_structure:
            self.structure_selector = StructureSelector(config)
        else:
            self.structure_selector = None
        
        # Reasoning layers (one per structure type)
        self.reasoning_layers = nn.ModuleDict({
            structure: ReasoningLayer(config, structure)
            for structure in config.structure_types
        })
        
        # Curriculum scheduler
        if config.use_curriculum_learning:
            self.curriculum_scheduler = CurriculumScheduler(config)
        else:
            self.curriculum_scheduler = None
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Metrics
        self.register_buffer('structure_usage', torch.zeros(len(config.structure_types)))
        self.register_buffer('curriculum_difficulty', torch.tensor(0.0))
        self.register_buffer('reasoning_improvement', torch.tensor(0.0))
        
        logger.info(f"Initialized SOLARModule: {config.structure_types}")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: SOLAR reasoning.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with reasoning info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Select structure
        if self.structure_selector is not None:
            structure_probs, selected_structure = self.structure_selector(hidden_states)
            structure_idx = self.config.structure_types.index(selected_structure)
            self.structure_usage[structure_idx] += 1
        else:
            selected_structure = self.config.structure_types[0]
            structure_probs = None
        
        # Apply reasoning with selected structure
        reasoning_layer = self.reasoning_layers[selected_structure]
        reasoned_output = reasoning_layer(hidden_states)
        
        # Apply curriculum learning (adjust difficulty)
        if self.curriculum_scheduler is not None:
            difficulty = self.curriculum_scheduler.get_difficulty()
            self.curriculum_difficulty = 0.9 * self.curriculum_difficulty + 0.1 * difficulty
            
            # Adjust reasoning based on difficulty
            difficulty_weight = difficulty
            reasoned_output = hidden_states + difficulty_weight * (reasoned_output - hidden_states)
        
        # Project output
        output = self.output_projection(reasoned_output)
        
        # Combine with original
        output = hidden_states + 0.3 * output
        
        metadata = {
            'selected_structure': selected_structure,
            'structure_probs': structure_probs.mean(dim=0).cpu().numpy().tolist() if structure_probs is not None else None,
            'curriculum_difficulty': self.curriculum_difficulty.item() if self.curriculum_scheduler else None
        }
        
        return output, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        total_usage = self.structure_usage.sum().item()
        structure_usage_pct = (self.structure_usage / (total_usage + 1e-8)).cpu().numpy().tolist()
        
        return {
            'structure_usage': dict(zip(self.config.structure_types, structure_usage_pct)),
            'curriculum_difficulty': self.curriculum_difficulty.item() if self.curriculum_scheduler else None,
            'reasoning_improvement': self.reasoning_improvement.item(),
            'selected_structures': self.config.structure_types
        }
    
    def update_curriculum(self):
        """Update curriculum scheduler."""
        if self.curriculum_scheduler is not None:
            self.curriculum_scheduler.step()


if __name__ == "__main__":
    config = SOLARConfig(
        hidden_dim=512,
        use_dynamic_structure=True,
        use_curriculum_learning=True
    )
    module = SOLARModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ SOLAR test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Selected structure: {metadata['selected_structure']}")
    print(f"   Curriculum difficulty: {metadata['curriculum_difficulty']:.4f}")



