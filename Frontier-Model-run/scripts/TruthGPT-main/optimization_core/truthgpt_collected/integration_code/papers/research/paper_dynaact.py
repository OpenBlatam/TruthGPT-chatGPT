#!/usr/bin/env python3
"""
DynaAct: Large Language Model Reasoning with Dynamic Action Spaces
===================================================================

Propuesta para mejorar el razonamiento secuencial, creando un "espacio de acciones"
compacto dinámico que mejora el desempeño sin penalizar mucho la latencia.

Técnica principal: Dynamic action space construction for sequential reasoning.

Basado en: arXiv paper (November 2025)
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
class DynaActConfig:
    """Configuración para DynaAct Dynamic Action Spaces."""
    hidden_dim: int = 512
    max_action_space_size: int = 100
    min_action_space_size: int = 10
    action_embedding_dim: int = 256
    use_adaptive_pruning: bool = True
    pruning_threshold: float = 0.1
    use_action_attention: bool = True
    num_attention_heads: int = 8


class DynamicActionSpace(nn.Module):
    """
    Dynamic Action Space para razonamiento secuencial.
    
    Técnica: Construye un espacio de acciones compacto y dinámico
    que se adapta al contexto actual.
    """
    
    def __init__(self, config: DynaActConfig):
        super().__init__()
        assert config.hidden_dim > 0, f"hidden_dim must be positive, got {config.hidden_dim}"
        assert config.max_action_space_size > config.min_action_space_size, \
            f"max_action_space_size must be > min_action_space_size"
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.action_embedding_dim = config.action_embedding_dim
        
        # Action embeddings (learnable)
        self.action_embeddings = nn.Parameter(
            torch.randn(config.max_action_space_size, config.action_embedding_dim) * 0.02
        )
        
        # Context-to-action mapping
        self.context_projection = nn.Linear(config.hidden_dim, config.action_embedding_dim)
        self.action_scorer = nn.Linear(config.action_embedding_dim, 1)
        
        # Initialize
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.xavier_uniform_(self.action_scorer.weight)
        if self.context_projection.bias is not None:
            nn.init.zeros_(self.context_projection.bias)
        if self.action_scorer.bias is not None:
            nn.init.zeros_(self.action_scorer.bias)
        
        # Action attention (optional)
        if config.use_action_attention:
            self.action_attention = nn.MultiheadAttention(
                config.action_embedding_dim,
                config.num_attention_heads,
                batch_first=True
            )
        else:
            self.action_attention = None
        
        # Metrics
        self.register_buffer('avg_action_space_size', torch.tensor(config.min_action_space_size))
        self.register_buffer('action_usage', torch.zeros(config.max_action_space_size))
        self.register_buffer('action_selection_entropy', torch.tensor(0.0))
        self.register_buffer('pruning_efficiency', torch.tensor(1.0))
        self.register_buffer('action_coverage', torch.tensor(0.0))
        
        logger.info(f"Initialized DynamicActionSpace: max_size={config.max_action_space_size}, "
                   f"min_size={config.min_action_space_size}")
    
    def _compute_action_scores(self, context: torch.Tensor) -> torch.Tensor:
        """
        Compute relevance scores for each action given context.
        
        Args:
            context: [batch, hidden_dim] or [batch, seq, hidden_dim]
            
        Returns:
            scores: [batch, max_action_space_size]
        """
        if context.dim() == 3:
            # Use last token
            context = context[:, -1, :]
        
        # Project context to action space
        context_proj = self.context_projection(context)  # [batch, action_embedding_dim]
        
        # Compute similarity scores
        # [batch, action_embedding_dim] x [max_actions, action_embedding_dim]^T
        scores = torch.matmul(context_proj, self.action_embeddings.transpose(-2, -1))
        
        # Score each action
        action_scores = self.action_scorer(self.action_embeddings).squeeze(-1)  # [max_actions]
        scores = scores + action_scores.unsqueeze(0)  # Broadcast
        
        return scores
    
    def _prune_actions(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune actions based on scores to create dynamic action space.
        
        Args:
            scores: [batch, max_action_space_size]
            
        Returns:
            pruned_scores: [batch, dynamic_size]
            action_indices: [batch, dynamic_size]
        """
        batch_size = scores.size(0)
        
        if self.config.use_adaptive_pruning:
            # Adaptive pruning: select top-k actions per batch
            # k is determined by score distribution
            score_mean = scores.mean(dim=-1, keepdim=True)  # [batch, 1]
            score_std = scores.std(dim=-1, keepdim=True)  # [batch, 1]
            threshold = score_mean + self.config.pruning_threshold * score_std
            
            # Select actions above threshold
            mask = scores > threshold
            dynamic_sizes = mask.sum(dim=-1)  # [batch]
            
            # Ensure within bounds
            dynamic_sizes = torch.clamp(
                dynamic_sizes,
                min=self.config.min_action_space_size,
                max=self.config.max_action_space_size
            )
            
            # Select top-k for each sample
            pruned_scores_list = []
            action_indices_list = []
            
            for i in range(batch_size):
                k = dynamic_sizes[i].item()
                topk_scores, topk_indices = torch.topk(scores[i], k, dim=-1)
                pruned_scores_list.append(topk_scores)
                action_indices_list.append(topk_indices)
            
            # Pad to max size for batching
            max_k = dynamic_sizes.max().item()
            pruned_scores = torch.zeros(batch_size, max_k, device=scores.device)
            action_indices = torch.zeros(batch_size, max_k, dtype=torch.long, device=scores.device)
            
            for i in range(batch_size):
                k = len(pruned_scores_list[i])
                pruned_scores[i, :k] = pruned_scores_list[i]
                action_indices[i, :k] = action_indices_list[i]
            
            # Update metrics
            avg_size = dynamic_sizes.float().mean().item()
            self.avg_action_space_size = 0.9 * self.avg_action_space_size + 0.1 * avg_size
            
        else:
            # Fixed size: top-k
            k = self.config.min_action_space_size
            pruned_scores, action_indices = torch.topk(scores, k, dim=-1)
            self.avg_action_space_size = k
        
        return pruned_scores, action_indices
    
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: create dynamic action space.
        
        Args:
            context: [batch, seq, hidden_dim] or [batch, hidden_dim]
            
        Returns:
            action_embeddings: [batch, dynamic_size, action_embedding_dim]
            action_weights: [batch, dynamic_size]
            metadata: Dict with action space info
        """
        # Compute action scores
        scores = self._compute_action_scores(context)  # [batch, max_actions]
        
        # Prune to dynamic size
        pruned_scores, action_indices = self._prune_actions(scores)  # [batch, dynamic_size]
        
        # Get action embeddings
        batch_size, dynamic_size = action_indices.shape
        action_embeddings = self.action_embeddings[action_indices]  # [batch, dynamic_size, action_embedding_dim]
        
        # Apply attention if enabled
        if self.action_attention is not None:
            # Use context as query
            if context.dim() == 3:
                context_query = context[:, -1:, :]  # [batch, 1, hidden_dim]
                context_proj = self.context_projection(context_query)  # [batch, 1, action_embedding_dim]
            else:
                context_proj = self.context_projection(context.unsqueeze(1))  # [batch, 1, action_embedding_dim]
            
            # Attention: query=context, key=value=actions
            attended_actions, _ = self.action_attention(
                context_proj, action_embeddings, action_embeddings
            )
            action_embeddings = attended_actions + action_embeddings
        
        # Compute weights (softmax over pruned scores)
        action_weights = F.softmax(pruned_scores, dim=-1)
        
        # Update usage tracking
        for i in range(batch_size):
            for j in range(dynamic_size):
                action_idx = action_indices[i, j].item()
                if action_idx < self.config.max_action_space_size:
                    self.action_usage[action_idx] += action_weights[i, j].item()
        
        # Compute selection entropy
        entropy = -(action_weights * torch.log(action_weights + 1e-8)).sum(dim=-1).mean()
        self.action_selection_entropy = 0.9 * self.action_selection_entropy + 0.1 * entropy.item()
        
        # Pruning efficiency (how much we reduced from max)
        efficiency = dynamic_size / self.config.max_action_space_size
        self.pruning_efficiency = 0.9 * self.pruning_efficiency + 0.1 * efficiency
        
        # Action coverage (fraction of actions used at least once)
        unique_actions = len(set(action_indices.flatten().cpu().numpy().tolist()))
        coverage = unique_actions / self.config.max_action_space_size
        self.action_coverage = 0.9 * self.action_coverage + 0.1 * coverage
        
        metadata = {
            'dynamic_size': dynamic_size,
            'avg_size': self.avg_action_space_size.item(),
            'action_indices': action_indices.detach().cpu().numpy().tolist(),
            'selection_entropy': entropy.item(),
            'pruning_efficiency': efficiency,
            'action_coverage': coverage
        }
        
        return action_embeddings, action_weights, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get action space metrics."""
        return {
            'avg_action_space_size': self.avg_action_space_size.item(),
            'action_usage': self.action_usage.detach().cpu().numpy().tolist(),
            'action_selection_entropy': self.action_selection_entropy.item(),
            'pruning_efficiency': self.pruning_efficiency.item(),
            'action_coverage': self.action_coverage.item(),
            'max_action_space_size': self.config.max_action_space_size,
            'min_action_space_size': self.config.min_action_space_size,
            'space_utilization': self.action_usage.sum().item() / self.config.max_action_space_size
        }


class DynaActModule(nn.Module):
    """
    Módulo DynaAct completo para razonamiento con espacios de acciones dinámicos.
    
    Mejoras avanzadas:
    - Caching de espacios de acciones
    - Adaptive learning rate para acciones
    - Action importance scoring
    - Temporal consistency tracking
    """
    
    def __init__(self, config: DynaActConfig):
        super().__init__()
        self.config = config
        
        # Dynamic action space
        self.action_space = DynamicActionSpace(config)
        
        # Action processor (processes selected actions)
        self.action_processor = nn.Sequential(
            nn.Linear(config.action_embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Initialize
        for module in self.action_processor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Action importance scorer
        self.action_importance = nn.Linear(config.action_embedding_dim, 1)
        nn.init.xavier_uniform_(self.action_importance.weight)
        if self.action_importance.bias is not None:
            nn.init.zeros_(self.action_importance.bias)
        
        # Caching
        self._cached_action_space = None
        self._cached_context_hash = None
        
        # Metrics
        self.register_buffer('action_importance_score', torch.tensor(0.0))
        self.register_buffer('cache_hit_rate', torch.tensor(0.0))
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Initialized DynaActModule with config: {config}")
    
    def forward(self, context: torch.Tensor, use_cache: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: reasoning with dynamic action space.
        
        Mejoras:
        - Caching de espacios de acciones
        - Action importance scoring
        - Temporal consistency
        
        Args:
            context: [batch, seq, hidden_dim]
            use_cache: Si usar cache de espacios de acciones
            
        Returns:
            enhanced_context: [batch, seq, hidden_dim]
            metadata: Dict with action space info
        """
        # Check cache
        context_hash = None
        if use_cache and context.dim() == 3:
            # Simple hash based on context norm
            context_hash = context.norm().item()
            if self._cached_context_hash is not None and abs(context_hash - self._cached_context_hash) < 0.01:
                # Use cached action space
                action_embeddings, action_weights, metadata = self._cached_action_space
                self._cache_hits += 1
            else:
                # Compute new action space
                action_embeddings, action_weights, metadata = self.action_space(context)
                self._cached_action_space = (action_embeddings, action_weights, metadata)
                self._cached_context_hash = context_hash
                self._cache_misses += 1
        else:
            # Compute new action space
            action_embeddings, action_weights, metadata = self.action_space(context)
            self._cache_misses += 1
        
        # Update cache hit rate
        total = self._cache_hits + self._cache_misses
        if total > 0:
            self.cache_hit_rate = torch.tensor(self._cache_hits / total)
        
        # Compute action importance
        importance_scores = self.action_importance(action_embeddings).squeeze(-1)  # [batch, dynamic_size]
        importance_weights = F.softmax(importance_scores, dim=-1)
        avg_importance = (importance_weights * importance_scores).sum(dim=-1).mean().item()
        self.action_importance_score = 0.9 * self.action_importance_score + 0.1 * avg_importance
        
        # Combine importance with action weights
        combined_weights = (action_weights + importance_weights) / 2.0
        combined_weights = F.softmax(combined_weights, dim=-1)
        
        # Weighted combination of actions
        weighted_actions = (action_embeddings * combined_weights.unsqueeze(-1)).sum(dim=1)  # [batch, action_embedding_dim]
        
        # Process actions
        action_features = self.action_processor(weighted_actions)  # [batch, hidden_dim]
        
        # Combine with context
        if context.dim() == 3:
            # Add to last token
            enhanced_context = context.clone()
            enhanced_context[:, -1, :] = enhanced_context[:, -1, :] + action_features
        else:
            enhanced_context = context + action_features
        
        # Update metadata
        metadata['action_importance'] = avg_importance
        metadata['cache_hit'] = (self._cache_hits > self._cache_misses) if total > 0 else False
        
        return enhanced_context, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        metrics = self.action_space.get_metrics()
        metrics.update({
            'action_importance_score': self.action_importance_score.item(),
            'cache_hit_rate': self.cache_hit_rate.item(),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses
        })
        return metrics
    
    def clear_cache(self):
        """Clear action space cache."""
        self._cached_action_space = None
        self._cached_context_hash = None
        self._cache_hits = 0
        self._cache_misses = 0


class TruthGPT_DynaAct_Integration(nn.Module):
    """Integración de DynaAct con TruthGPT."""
    
    def __init__(self, base_model, dynaact_config: DynaActConfig):
        super().__init__()
        self.base_model = base_model
        self.dynaact_module = DynaActModule(dynaact_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass integrado con DynaAct."""
        output = self.base_model(*args, **kwargs)
        if isinstance(output, torch.Tensor) and output.dim() >= 2:
            enhanced_output, metadata = self.dynaact_module(output)
            return enhanced_output
        return output


if __name__ == "__main__":
    config = DynaActConfig(
        hidden_dim=512,
        max_action_space_size=50,
        min_action_space_size=10
    )
    module = DynaActModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    output, metadata = module(x)
    metrics = module.get_metrics()
    print(f"✅ DynaAct test:")
    print(f"   Input {x.shape} -> Output {output.shape}")
    print(f"   Dynamic action space size: {metadata['dynamic_size']}")
    print(f"   Avg action space size: {metrics['avg_action_space_size']:.2f}")


