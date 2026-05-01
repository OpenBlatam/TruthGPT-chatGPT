#!/usr/bin/env python3
"""
Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures
===================================================================================================

Pandey, Ghukasyan, Goktas, Radha. Feb 2025. arXiv

Introduce un método de inferencia dinámico (Grafo de pensamientos) para descomponer
preguntas complejas en subproblemas. Mejora el desempeño en razonamiento científico,
matemático y multi-hop.

Técnica principal: Dynamic inference with adaptive graph structures that can switch
between chain, tree, and graph reasoning patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveGoTConfig:
    """Configuración para Adaptive Graph of Thoughts."""
    hidden_dim: int = 512
    max_subproblems: int = 10
    reasoning_structure: str = "adaptive"  # adaptive, chain, tree, graph
    use_dynamic_decomposition: bool = True
    subproblem_dim: int = 256
    graph_attention_heads: int = 4
    use_knowledge_propagation: bool = True
    temperature: float = 1.0


class SubproblemDecomposer(nn.Module):
    """
    Descompone preguntas complejas en subproblemas.
    """
    
    def __init__(self, config: AdaptiveGoTConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.subproblem_dim = config.subproblem_dim
        
        # Decomposition network
        self.decomposer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.subproblem_dim)
        )
        
        # Subproblem importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(config.subproblem_dim, config.subproblem_dim // 2),
            nn.GELU(),
            nn.Linear(config.subproblem_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in self.decomposer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        logger.info("Initialized SubproblemDecomposer")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Decompose into subproblems.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            subproblems: List of [batch, subproblem_dim] tensors
            importance_scores: [batch, num_subproblems]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use last token for decomposition
        last_token = hidden_states[:, -1, :]  # [batch, hidden_dim]
        
        # Generate subproblems
        subproblem_features = self.decomposer(last_token)  # [batch, subproblem_dim]
        
        # Split into multiple subproblems
        num_subproblems = min(self.config.max_subproblems, seq_len)
        subproblems = []
        importance_scores = []
        
        for i in range(num_subproblems):
            # Extract subproblem representation
            start_idx = (i * self.config.subproblem_dim) % subproblem_features.size(-1)
            end_idx = min(start_idx + self.config.subproblem_dim, subproblem_features.size(-1))
            
            if end_idx > start_idx:
                subproblem = subproblem_features[:, start_idx:end_idx]
                # Pad if necessary
                if subproblem.size(-1) < self.config.subproblem_dim:
                    padding = torch.zeros(batch_size, self.config.subproblem_dim - subproblem.size(-1),
                                         device=subproblem.device)
                    subproblem = torch.cat([subproblem, padding], dim=-1)
            else:
                subproblem = torch.zeros(batch_size, self.config.subproblem_dim, device=subproblem_features.device)
            
            subproblems.append(subproblem)
            
            # Score importance
            importance = self.importance_scorer(subproblem).squeeze(-1)  # [batch]
            importance_scores.append(importance)
        
        importance_tensor = torch.stack(importance_scores, dim=1)  # [batch, num_subproblems]
        
        return subproblems, importance_tensor


class GraphReasoningStructure(nn.Module):
    """
    Estructura de razonamiento tipo grafo que puede adaptarse dinámicamente.
    """
    
    def __init__(self, config: AdaptiveGoTConfig):
        super().__init__()
        self.config = config
        self.subproblem_dim = config.subproblem_dim
        self.num_heads = config.graph_attention_heads
        
        # Graph attention for connecting subproblems
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=config.subproblem_dim,
            num_heads=config.graph_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Structure selector (chain, tree, graph)
        self.structure_selector = nn.Sequential(
            nn.Linear(config.subproblem_dim, config.subproblem_dim // 2),
            nn.GELU(),
            nn.Linear(config.subproblem_dim // 2, 3),  # chain, tree, graph
            nn.Softmax(dim=-1)
        )
        
        # Knowledge propagation
        if config.use_knowledge_propagation:
            self.knowledge_propagation = nn.Sequential(
                nn.Linear(config.subproblem_dim, config.subproblem_dim),
                nn.GELU(),
                nn.Linear(config.subproblem_dim, config.subproblem_dim)
            )
        
        logger.info(f"Initialized GraphReasoningStructure: {config.reasoning_structure}")
    
    def _chain_reasoning(self, subproblems: List[torch.Tensor]) -> torch.Tensor:
        """Chain reasoning: sequential processing."""
        # Stack subproblems
        stacked = torch.stack(subproblems, dim=1)  # [batch, num_subproblems, subproblem_dim]
        
        # Sequential processing
        result = stacked[:, 0, :]
        for i in range(1, stacked.size(1)):
            result = result + stacked[:, i, :]  # Accumulate
        
        return result
    
    def _tree_reasoning(self, subproblems: List[torch.Tensor]) -> torch.Tensor:
        """Tree reasoning: hierarchical processing."""
        stacked = torch.stack(subproblems, dim=1)  # [batch, num_subproblems, subproblem_dim]
        
        # Hierarchical aggregation (binary tree)
        current = stacked
        while current.size(1) > 1:
            # Pair up and aggregate
            num_pairs = current.size(1) // 2
            pairs = []
            for i in range(num_pairs):
                pair = current[:, [i*2, i*2+1], :].mean(dim=1)  # [batch, subproblem_dim]
                pairs.append(pair)
            
            # Add remaining if odd
            if current.size(1) % 2 == 1:
                pairs.append(current[:, -1, :])
            
            current = torch.stack(pairs, dim=1)  # [batch, new_num, subproblem_dim]
        
        return current[:, 0, :]
    
    def _graph_reasoning(self, subproblems: List[torch.Tensor]) -> torch.Tensor:
        """Graph reasoning: fully connected with attention."""
        stacked = torch.stack(subproblems, dim=1)  # [batch, num_subproblems, subproblem_dim]
        
        # Graph attention
        attended, _ = self.graph_attention(stacked, stacked, stacked)  # [batch, num_subproblems, subproblem_dim]
        
        # Aggregate
        result = attended.mean(dim=1)  # [batch, subproblem_dim]
        
        return result
    
    def forward(self, subproblems: List[torch.Tensor], importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply reasoning structure.
        
        Args:
            subproblems: List of [batch, subproblem_dim] tensors
            importance_scores: [batch, num_subproblems]
            
        Returns:
            reasoned_output: [batch, subproblem_dim]
        """
        if len(subproblems) == 0:
            return torch.zeros(subproblems[0].size(0), self.subproblem_dim, device=subproblems[0].device)
        
        # Weight subproblems by importance
        stacked = torch.stack(subproblems, dim=1)  # [batch, num_subproblems, subproblem_dim]
        importance_expanded = importance_scores.unsqueeze(-1)  # [batch, num_subproblems, 1]
        weighted = stacked * importance_expanded  # [batch, num_subproblems, subproblem_dim]
        
        # Select structure
        if self.config.reasoning_structure == "adaptive":
            # Use first subproblem to select structure
            structure_logits = self.structure_selector(weighted[:, 0, :])  # [batch, 3]
            structure_probs = structure_logits
            
            # Weighted combination of structures
            chain_result = self._chain_reasoning(subproblems)
            tree_result = self._tree_reasoning(subproblems)
            graph_result = self._graph_reasoning(subproblems)
            
            # Combine based on structure probabilities
            result = (structure_probs[:, 0:1] * chain_result.unsqueeze(1) +
                     structure_probs[:, 1:2] * tree_result.unsqueeze(1) +
                     structure_probs[:, 2:3] * graph_result.unsqueeze(1)).squeeze(1)
        elif self.config.reasoning_structure == "chain":
            result = self._chain_reasoning(subproblems)
        elif self.config.reasoning_structure == "tree":
            result = self._tree_reasoning(subproblems)
        else:  # graph
            result = self._graph_reasoning(subproblems)
        
        # Knowledge propagation
        if self.config.use_knowledge_propagation:
            result = result + self.knowledge_propagation(result)
        
        return result


class AdaptiveGoTModule(nn.Module):
    """
    Módulo Adaptive Graph of Thoughts completo.
    """
    
    def __init__(self, config: AdaptiveGoTConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.subproblem_dim = config.subproblem_dim
        
        # Components
        self.decomposer = SubproblemDecomposer(config)
        self.graph_structure = GraphReasoningStructure(config)
        
        # Projection back to hidden_dim
        self.output_projection = nn.Sequential(
            nn.Linear(config.subproblem_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Metrics
        self.register_buffer('avg_num_subproblems', torch.tensor(0.0))
        self.register_buffer('structure_usage', torch.zeros(3))  # chain, tree, graph
        self.register_buffer('reasoning_quality', torch.tensor(0.5))
        self.register_buffer('decomposition_efficiency', torch.tensor(0.5))
        
        logger.info(f"Initialized AdaptiveGoTModule: structure={config.reasoning_structure}")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: adaptive graph of thoughts reasoning.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            enhanced_states: [batch, seq, hidden_dim]
            metadata: Dict with reasoning info
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Decompose into subproblems
        subproblems, importance_scores = self.decomposer(hidden_states)
        
        # Apply graph reasoning
        reasoned_output = self.graph_structure(subproblems, importance_scores)  # [batch, subproblem_dim]
        
        # Project back to hidden_dim
        enhanced_features = self.output_projection(reasoned_output)  # [batch, hidden_dim]
        
        # Combine with original hidden states
        # Expand to match sequence length
        enhanced_expanded = enhanced_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, hidden_dim]
        
        # Weighted combination
        combination_weight = 0.3  # How much to use enhanced features
        output = hidden_states + combination_weight * enhanced_expanded
        
        # Update metrics
        num_subproblems = len(subproblems)
        self.avg_num_subproblems = 0.9 * self.avg_num_subproblems + 0.1 * num_subproblems
        
        # Compute reasoning quality (based on importance scores)
        reasoning_quality = importance_scores.mean().item()
        self.reasoning_quality = 0.9 * self.reasoning_quality + 0.1 * reasoning_quality
        
        metadata = {
            'num_subproblems': num_subproblems,
            'avg_importance': importance_scores.mean().item(),
            'reasoning_structure': self.config.reasoning_structure,
            'reasoning_quality': reasoning_quality
        }
        
        return output, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            'avg_num_subproblems': self.avg_num_subproblems.item(),
            'structure_usage': self.structure_usage.cpu().numpy().tolist(),
            'reasoning_quality': self.reasoning_quality.item(),
            'decomposition_efficiency': self.decomposition_efficiency.item(),
            'reasoning_structure': self.config.reasoning_structure
        }


if __name__ == "__main__":
    config = AdaptiveGoTConfig(
        hidden_dim=512,
        reasoning_structure="adaptive",
        use_dynamic_decomposition=True
    )
    module = AdaptiveGoTModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ AdaptiveGoT test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Subproblems: {metadata['num_subproblems']}")
    print(f"   Reasoning quality: {metadata['reasoning_quality']:.4f}")



