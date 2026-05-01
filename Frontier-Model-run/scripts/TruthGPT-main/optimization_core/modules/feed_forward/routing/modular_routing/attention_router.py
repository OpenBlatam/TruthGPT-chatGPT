"""
Attention-Based Router Module
Router that uses attention mechanisms to determine expert assignments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .base_router import BaseRouter, RouterConfig, RoutingResult, RoutingStrategy

@dataclass
class AttentionRouterConfig(RouterConfig):
    """Configuration for attention-based router."""
    attention_heads: int = 8
    attention_dropout: float = 0.1
    temperature: float = 1.0
    expert_embedding_dim: int = 64
    use_relative_position: bool = True
    max_relative_position: int = 128
    attention_type: str = "multi_head"  # multi_head, sparse, local
    sparsity_threshold: float = 0.1
    local_window_size: int = 64

class AttentionRouter(BaseRouter):
    """
    Attention-based router that uses multi-head attention to route tokens to experts.
    """
    
    def __init__(self, config: AttentionRouterConfig):
        super().__init__(config)
        self.config = config
        self.attention_layer = None
        self.expert_embeddings = None
        self.position_embeddings = None
        self.gate_network = None
        
    def initialize(self) -> None:
        """Initialize the attention router."""
        # Create attention layer
        if self.config.attention_type == "multi_head":
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.config.hidden_size,
                num_heads=self.config.attention_heads,
                dropout=self.config.attention_dropout,
                batch_first=True
            )
        elif self.config.attention_type == "sparse":
            self.attention_layer = SparseAttention(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.attention_heads,
                sparsity_threshold=self.config.sparsity_threshold
            )
        elif self.config.attention_type == "local":
            self.attention_layer = LocalAttention(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.attention_heads,
                window_size=self.config.local_window_size
            )
        
        # Create expert embeddings
        self.expert_embeddings = nn.Parameter(
            torch.randn(self.config.num_experts, self.config.expert_embedding_dim)
        )
        
        # Create position embeddings if needed
        if self.config.use_relative_position:
            self.position_embeddings = nn.Parameter(
                torch.randn(2 * self.config.max_relative_position + 1, self.config.hidden_size)
            )
        
        # Create gate network
        self.gate_network = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.config.attention_dropout),
            nn.Linear(self.config.hidden_size // 2, self.config.num_experts)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        self._initialized = True
        self.logger.info(f"Attention router initialized with {self.config.attention_type} attention")
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize expert embeddings
        nn.init.normal_(self.expert_embeddings, mean=0, std=0.02)
        
        if self.position_embeddings is not None:
            nn.init.normal_(self.position_embeddings, mean=0, std=0.02)
    
    def route_tokens(
        self, 
        input_tokens: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """Route tokens using attention mechanism."""
        start_time = time.time()
        
        # Validate input
        self.validate_input(input_tokens)
        
        # Check cache
        cache_key = self.get_cache_key(input_tokens, context)
        if cache_key:
            cached_result = self.get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        batch_size, seq_len, hidden_size = input_tokens.shape
        
        # Compute attention scores
        attention_scores = self._compute_attention_scores(input_tokens, attention_mask)
        
        # Compute expert scores
        expert_scores = self._compute_expert_scores(input_tokens, attention_scores)
        
        # Apply temperature scaling
        expert_scores = expert_scores / self.config.temperature
        
        # Compute routing probabilities
        routing_probs = F.softmax(expert_scores, dim=-1)
        
        # Select experts
        expert_indices, expert_weights = self._select_experts(routing_probs)
        
        # Compute routing confidence
        confidence = self._compute_confidence(routing_probs, expert_indices)
        
        # Create routing result
        result = RoutingResult(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            routing_confidence=confidence,
            routing_time=time.time() - start_time,
            strategy_used=self.config.strategy.value,
            metadata={
                'attention_scores': attention_scores.detach().cpu().numpy(),
                'expert_scores': expert_scores.detach().cpu().numpy(),
                'routing_probs': routing_probs.detach().cpu().numpy()
            }
        )
        
        # Cache result
        if cache_key:
            self.cache_result(cache_key, result)
        
        # Record metrics and log
        self.record_metrics(result)
        self.log_routing(result, input_tokens.shape)
        
        return result
    
    def _compute_attention_scores(
        self, 
        input_tokens: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention scores."""
        if self.config.attention_type == "multi_head":
            # Standard multi-head attention
            attn_output, attn_weights = self.attention_layer(
                input_tokens, input_tokens, input_tokens,
                key_padding_mask=attention_mask
            )
            return attn_weights.mean(dim=1)  # Average across heads
        
        elif self.config.attention_type == "sparse":
            # Sparse attention
            return self.attention_layer(input_tokens, attention_mask)
        
        elif self.config.attention_type == "local":
            # Local attention
            return self.attention_layer(input_tokens, attention_mask)
        
        else:
            raise ValueError(f"Unknown attention type: {self.config.attention_type}")
    
    def _compute_expert_scores(
        self, 
        input_tokens: torch.Tensor, 
        attention_scores: torch.Tensor
    ) -> torch.Tensor:
        """Compute expert scores using gate network."""
        # Apply gate network
        expert_logits = self.gate_network(input_tokens)
        
        # Weight by attention scores
        if attention_scores is not None:
            # Broadcast attention scores to match expert logits
            attention_weights = attention_scores.unsqueeze(-1).expand_as(expert_logits)
            expert_logits = expert_logits * attention_weights
        
        return expert_logits
    
    def _select_experts(
        self, 
        routing_probs: torch.Tensor
    ) -> Tuple[List[int], List[float]]:
        """Select experts based on routing probabilities."""
        batch_size, seq_len, num_experts = routing_probs.shape
        
        # Get top-k experts for each token
        k = min(self.config.max_tokens_per_expert, num_experts)
        top_k_probs, top_k_indices = torch.topk(routing_probs, k, dim=-1)
        
        # Ensure minimum tokens per expert
        if self.config.min_tokens_per_expert > 0:
            # Add minimum tokens to least used experts
            expert_counts = torch.zeros(batch_size, num_experts)
            for i in range(batch_size):
                for j in range(seq_len):
                    for expert_idx in top_k_indices[i, j]:
                        expert_counts[i, expert_idx] += 1
            
            # Find experts with insufficient tokens
            min_threshold = self.config.min_tokens_per_expert
            for i in range(batch_size):
                for expert_idx in range(num_experts):
                    if expert_counts[i, expert_idx] < min_threshold:
                        # Add this expert to some tokens
                        token_indices = torch.randperm(seq_len)[:min_threshold]
                        for token_idx in token_indices:
                            if expert_idx not in top_k_indices[i, token_idx]:
                                # Replace least important expert
                                min_prob_idx = torch.argmin(top_k_probs[i, token_idx])
                                top_k_indices[i, token_idx, min_prob_idx] = expert_idx
                                top_k_probs[i, token_idx, min_prob_idx] = routing_probs[i, token_idx, expert_idx]
        
        # Convert to lists
        expert_indices = []
        expert_weights = []
        
        for i in range(batch_size):
            batch_indices = []
            batch_weights = []
            
            for j in range(seq_len):
                token_indices = top_k_indices[i, j].tolist()
                token_weights = top_k_probs[i, j].tolist()
                
                batch_indices.extend(token_indices)
                batch_weights.extend(token_weights)
            
            expert_indices.append(batch_indices)
            expert_weights.append(batch_weights)
        
        return expert_indices, expert_weights
    
    def _compute_confidence(
        self, 
        routing_probs: torch.Tensor, 
        expert_indices: List[List[int]]
    ) -> float:
        """Compute routing confidence."""
        # Compute entropy of routing probabilities
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1)
        max_entropy = math.log(self.config.num_experts)
        normalized_entropy = entropy / max_entropy
        
        # Confidence is inverse of normalized entropy
        confidence = 1.0 - normalized_entropy.mean().item()
        
        return max(0.0, min(1.0, confidence))
    
    def get_router_info(self) -> Dict[str, Any]:
        """Get router information and statistics."""
        return {
            'router_type': 'attention_based',
            'strategy': self.config.strategy.value,
            'attention_type': self.config.attention_type,
            'num_experts': self.config.num_experts,
            'hidden_size': self.config.hidden_size,
            'attention_heads': self.config.attention_heads,
            'expert_embedding_dim': self.config.expert_embedding_dim,
            'temperature': self.config.temperature,
            'cache_enabled': self.config.enable_caching,
            'cache_size': len(self.cache) if self.cache else 0,
            'metrics': self.get_metrics()
        }

class SparseAttention(nn.Module):
    """Sparse attention implementation."""
    
    def __init__(self, hidden_size: int, num_heads: int, sparsity_threshold: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.sparsity_threshold = sparsity_threshold
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute Q, K, V
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparsity mask
        if self.sparsity_threshold > 0:
            sparse_mask = scores > self.sparsity_threshold
            scores = scores.masked_fill(~sparse_mask, float('-inf'))
        
        # Apply attention mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.view(batch_size, seq_len, hidden_size)
        
        return self.out_linear(out)

class LocalAttention(nn.Module):
    """Local attention implementation."""
    
    def __init__(self, hidden_size: int, num_heads: int, window_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute Q, K, V
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Create local attention mask
        local_mask = self._create_local_mask(seq_len, x.device)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~local_mask.unsqueeze(1), float('-inf'))
        
        # Apply attention mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.view(batch_size, seq_len, hidden_size)
        
        return self.out_linear(out)
    
    def _create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create local attention mask."""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = True
        
        return mask





