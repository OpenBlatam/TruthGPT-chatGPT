#!/usr/bin/env python3
"""
Recursive Decomposition of Logical Thoughts: Framework for Superior Reasoning and Knowledge Propagation
======================================================================================================

Qasim, Zhang, Alsahfi, Butt. Ene 2025. arXiv

Decomponen recursivamente tareas complejas y usan un módulo para propagar "pensamientos buenos"
(knowledge propagation), lo que mejora benchmarks como GSM8K, SVAMP, Gaokao Math, etc.

Técnica principal: Recursive decomposition + knowledge propagation.
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
class RDoLTConfig:
    """Configuración para RDoLT."""
    hidden_dim: int = 512
    max_decomposition_depth: int = 5
    decomposition_dim: int = 256
    use_knowledge_propagation: bool = True
    knowledge_dim: int = 128
    propagation_strength: float = 0.5
    use_recursive_refinement: bool = True


class RecursiveDecomposer(nn.Module):
    """
    Descompone recursivamente tareas complejas.
    """
    
    def __init__(self, config: RDoLTConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.decomposition_dim = config.decomposition_dim
        self.max_depth = config.max_decomposition_depth
        
        # Decomposition network
        self.decomposer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.decomposition_dim * 2)  # For 2 subproblems
        )
        
        # Complexity estimator (decides if further decomposition needed)
        self.complexity_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in self.decomposer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info(f"Initialized RecursiveDecomposer: max_depth={config.max_decomposition_depth}")
    
    def forward(self, hidden_states: torch.Tensor, depth: int = 0) -> Tuple[List[torch.Tensor], bool]:
        """
        Recursively decompose.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            depth: Current decomposition depth
            
        Returns:
            subproblems: List of decomposed states
            should_continue: Whether to continue decomposing
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Check if we should continue decomposing
        if depth >= self.max_depth:
            return [hidden_states], False
        
        # Estimate complexity
        last_token = hidden_states[:, -1, :]  # [batch, hidden_dim]
        complexity = self.complexity_estimator(last_token).squeeze(-1)  # [batch]
        avg_complexity = complexity.mean().item()
        
        # If complexity is low, don't decompose further
        if avg_complexity < 0.3:
            return [hidden_states], False
        
        # Decompose into subproblems
        decomposition_features = self.decomposer(last_token)  # [batch, decomposition_dim * 2]
        
        # Split into two subproblems
        subproblem1_features = decomposition_features[:, :self.decomposition_dim]
        subproblem2_features = decomposition_features[:, self.decomposition_dim:]
        
        # Expand to match sequence length
        subproblem1 = subproblem1_features.unsqueeze(1).expand(-1, seq_len, -1)
        subproblem2 = subproblem2_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Project back to hidden_dim
        subproblem1_proj = nn.Linear(self.decomposition_dim, self.hidden_dim).to(hidden_states.device)
        subproblem2_proj = nn.Linear(self.decomposition_dim, self.hidden_dim).to(hidden_states.device)
        
        subproblem1 = subproblem1_proj(subproblem1)
        subproblem2 = subproblem2_proj(subproblem2)
        
        # Recursively decompose if needed
        subproblems = []
        should_continue = avg_complexity > 0.5
        
        if should_continue and depth < self.max_depth - 1:
            # Recursively decompose subproblems
            sub1_list, _ = self.forward(subproblem1, depth + 1)
            sub2_list, _ = self.forward(subproblem2, depth + 1)
            subproblems.extend(sub1_list)
            subproblems.extend(sub2_list)
        else:
            subproblems = [subproblem1, subproblem2]
        
        return subproblems, should_continue


class KnowledgePropagator(nn.Module):
    """
    Propaga "pensamientos buenos" entre subproblemas.
    """
    
    def __init__(self, config: RDoLTConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.knowledge_dim = config.knowledge_dim
        
        # Knowledge extraction
        self.knowledge_extractor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.knowledge_dim),
            nn.GELU(),
            nn.Linear(config.knowledge_dim, config.knowledge_dim)
        )
        
        # Knowledge propagation
        self.propagation_net = nn.Sequential(
            nn.Linear(config.hidden_dim + config.knowledge_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Knowledge quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(config.knowledge_dim, config.knowledge_dim // 2),
            nn.GELU(),
            nn.Linear(config.knowledge_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in [self.knowledge_extractor, self.propagation_net, self.quality_scorer]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        
        logger.info("Initialized KnowledgePropagator")
    
    def forward(self, subproblems: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Propagate knowledge between subproblems.
        
        Args:
            subproblems: List of [batch, seq, hidden_dim] tensors
            
        Returns:
            enhanced_subproblems: List of enhanced subproblems
        """
        if len(subproblems) <= 1:
            return subproblems
        
        # Extract knowledge from each subproblem
        knowledge_list = []
        quality_scores = []
        
        for subproblem in subproblems:
            last_token = subproblem[:, -1, :]  # [batch, hidden_dim]
            knowledge = self.knowledge_extractor(last_token)  # [batch, knowledge_dim]
            knowledge_list.append(knowledge)
            
            # Score quality
            quality = self.quality_scorer(knowledge).squeeze(-1)  # [batch]
            quality_scores.append(quality)
        
        # Aggregate knowledge (weighted by quality)
        knowledge_tensor = torch.stack(knowledge_list, dim=1)  # [batch, num_subproblems, knowledge_dim]
        quality_tensor = torch.stack(quality_scores, dim=1)  # [batch, num_subproblems]
        quality_weights = F.softmax(quality_tensor, dim=1).unsqueeze(-1)  # [batch, num_subproblems, 1]
        
        aggregated_knowledge = (knowledge_tensor * quality_weights).sum(dim=1)  # [batch, knowledge_dim]
        
        # Propagate to each subproblem
        enhanced_subproblems = []
        for i, subproblem in enumerate(subproblems):
            # Expand aggregated knowledge
            batch_size, seq_len, _ = subproblem.shape
            knowledge_expanded = aggregated_knowledge.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, knowledge_dim]
            
            # Combine with subproblem
            combined = torch.cat([subproblem, knowledge_expanded], dim=-1)  # [batch, seq, hidden_dim + knowledge_dim]
            
            # Propagate
            enhanced = self.propagation_net(combined)  # [batch, seq, hidden_dim]
            
            # Combine with original
            enhanced_subproblem = subproblem + self.config.propagation_strength * enhanced
            enhanced_subproblems.append(enhanced_subproblem)
        
        return enhanced_subproblems


class RDoLTModule(nn.Module):
    """
    Módulo RDoLT completo.
    """
    
    def __init__(self, config: RDoLTConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Components
        self.decomposer = RecursiveDecomposer(config)
        if config.use_knowledge_propagation:
            self.knowledge_propagator = KnowledgePropagator(config)
        else:
            self.knowledge_propagator = None
        
        # Recomposer (combine subproblems back)
        self.recomposer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Metrics
        self.register_buffer('avg_decomposition_depth', torch.tensor(0.0))
        self.register_buffer('avg_num_subproblems', torch.tensor(0.0))
        self.register_buffer('knowledge_quality', torch.tensor(0.5))
        
        logger.info(f"Initialized RDoLTModule: max_depth={config.max_decomposition_depth}")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: recursive decomposition + knowledge propagation.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with decomposition info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Recursively decompose
        subproblems, should_continue = self.decomposer(hidden_states, depth=0)
        
        # Propagate knowledge
        if self.knowledge_propagator is not None:
            enhanced_subproblems = self.knowledge_propagator(subproblems)
        else:
            enhanced_subproblems = subproblems
        
        # Recompose subproblems
        if len(enhanced_subproblems) > 1:
            # Combine subproblems
            stacked = torch.stack(enhanced_subproblems, dim=0)  # [num_subproblems, batch, seq, hidden_dim]
            combined = stacked.mean(dim=0)  # [batch, seq, hidden_dim]
        else:
            combined = enhanced_subproblems[0]
        
        # Recompose
        recomposed = self.recomposer(combined)
        
        # Combine with original
        output = hidden_states + 0.3 * recomposed
        
        # Update metrics
        decomposition_depth = len(subproblems)  # Approximate depth
        self.avg_decomposition_depth = 0.9 * self.avg_decomposition_depth + 0.1 * decomposition_depth
        self.avg_num_subproblems = 0.9 * self.avg_num_subproblems + 0.1 * len(subproblems)
        
        metadata = {
            'num_subproblems': len(subproblems),
            'decomposition_depth': decomposition_depth,
            'should_continue': should_continue
        }
        
        return output, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            'avg_decomposition_depth': self.avg_decomposition_depth.item(),
            'avg_num_subproblems': self.avg_num_subproblems.item(),
            'knowledge_quality': self.knowledge_quality.item(),
            'max_depth': self.config.max_decomposition_depth
        }


if __name__ == "__main__":
    config = RDoLTConfig(
        hidden_dim=512,
        max_decomposition_depth=5,
        use_knowledge_propagation=True
    )
    module = RDoLTModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ RDoLT test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Subproblems: {metadata['num_subproblems']}")
    print(f"   Decomposition depth: {metadata['decomposition_depth']}")



