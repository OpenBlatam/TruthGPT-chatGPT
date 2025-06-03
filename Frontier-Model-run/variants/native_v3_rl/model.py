"""
Native DeepSeek-V3 with Reinforcement Learning
Based on DeepSeek-V3 architecture with integrated RL capabilities.

Key Features:
- Multi-Head Latent Attention (MLA) with LoRA-style compression
- Mixture of Experts with routed and shared experts
- Advanced RoPE with YARN scaling
- RMSNorm normalization
- Integrated PPO-based reinforcement learning
- Multi-objective reward optimization
- No API dependencies - fully native implementation
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


@dataclass
class NativeV3RLConfig:
    """Configuration for Native DeepSeek-V3 with RL."""
    
    # Model architecture
    vocab_size: int = 102400
    max_position_embeddings: int = 16384
    hidden_size: int = 2048  # dim in DeepSeek-V3
    intermediate_size: int = 10944  # inter_dim
    moe_intermediate_size: int = 1408  # moe_inter_dim
    num_hidden_layers: int = 27  # n_layers
    num_dense_layers: int = 1  # n_dense_layers
    num_attention_heads: int = 16  # n_heads
    
    # MoE configuration
    num_routed_experts: int = 64  # n_routed_experts
    num_shared_experts: int = 2  # n_shared_experts
    num_activated_experts: int = 6  # n_activated_experts
    num_expert_groups: int = 1  # n_expert_groups
    num_limited_groups: int = 1  # n_limited_groups
    score_func: str = "softmax"  # "softmax" or "sigmoid"
    route_scale: float = 1.0
    
    # MLA (Multi-Head Latent Attention) configuration
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128  # query-key without positional encoding
    qk_rope_head_dim: int = 64   # query-key with rotary encoding
    v_head_dim: int = 128
    
    # RoPE configuration
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 0.707
    original_seq_len: int = 4096
    
    # RL configuration
    use_reinforcement_learning: bool = True
    reward_types: List[str] = None
    use_multi_objective: bool = True
    use_ppo: bool = True
    use_value_heads: bool = True
    use_curiosity: bool = True
    use_experience_replay: bool = True
    
    # PPO hyperparameters
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Curiosity parameters
    curiosity_coef: float = 0.1
    curiosity_feature_dim: int = 512
    
    # Experience replay
    replay_buffer_size: int = 10000
    replay_batch_size: int = 32
    
    # Training parameters
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    
    def __post_init__(self):
        if self.reward_types is None:
            self.reward_types = ["accuracy", "fluency", "helpfulness", "safety"]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YARN scaling."""
    
    def __init__(self, config: NativeV3RLConfig):
        super().__init__()
        self.config = config
        self.rope_theta = config.rope_theta
        self.rope_factor = config.rope_factor
        self.beta_fast = config.beta_fast
        self.beta_slow = config.beta_slow
        self.mscale = config.mscale
        self.original_seq_len = config.original_seq_len
        
        # Precompute frequency tensor
        dim = config.qk_rope_head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        # YARN scaling for extended sequences
        if seq_len > self.original_seq_len:
            scale_factor = seq_len / self.original_seq_len
            inv_freq = self._yarn_scaling(self.inv_freq, scale_factor)
        else:
            inv_freq = self.inv_freq
            
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale
        
        return cos, sin
    
    def _yarn_scaling(self, inv_freq, scale_factor):
        """Apply YARN scaling to frequency tensor."""
        # Simplified YARN scaling implementation
        low_freq_factor = 1.0
        high_freq_factor = scale_factor
        
        # Interpolate between low and high frequency factors
        freq_factor = torch.where(
            inv_freq < 1.0 / self.beta_fast,
            low_freq_factor,
            torch.where(
                inv_freq > 1.0 / self.beta_slow,
                high_freq_factor,
                low_freq_factor + (high_freq_factor - low_freq_factor) * 
                (torch.log(inv_freq * self.beta_fast) / torch.log(self.beta_slow / self.beta_fast))
            )
        )
        
        return inv_freq * freq_factor


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) from DeepSeek-V3."""
    
    def __init__(self, config: NativeV3RLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # MLA dimensions
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        # Query projection
        if self.q_lora_rank == 0:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
                bias=False
            )
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
                bias=False
            )
        
        # Key-Value projection with LoRA-style compression
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False
        )
        
        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(config)
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Query projection
        if self.q_lora_rank == 0:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_proj(hidden_states))
        
        # Key-Value projection
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(compressed_kv)
        
        # Reshape and split
        q = q.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # Expand k_pe for multi-head
        k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        
        # Apply rotary embedding
        cos, sin = self.rotary_emb(hidden_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        
        # Concatenate nope and pe parts
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Handle past key values for caching
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, present_key_value


class MoEGate(nn.Module):
    """Mixture of Experts Gating mechanism."""
    
    def __init__(self, config: NativeV3RLConfig):
        super().__init__()
        self.config = config
        self.num_routed_experts = config.num_routed_experts
        self.num_activated_experts = config.num_activated_experts
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        
        self.weight = nn.Parameter(torch.empty(config.hidden_size, config.num_routed_experts))
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute routing scores
        routing_logits = torch.matmul(hidden_states, self.weight) * self.route_scale
        
        if self.score_func == "softmax":
            routing_weights = F.softmax(routing_logits, dim=-1)
        elif self.score_func == "sigmoid":
            routing_weights = torch.sigmoid(routing_logits)
        else:
            raise ValueError(f"Unknown score function: {self.score_func}")
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_activated_experts, dim=-1
        )
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts


class MoEExpert(nn.Module):
    """Single expert in MoE layer."""
    
    def __init__(self, config: NativeV3RLConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MoELayer(nn.Module):
    """Mixture of Experts layer with both routed and shared experts."""
    
    def __init__(self, config: NativeV3RLConfig):
        super().__init__()
        self.config = config
        self.num_routed_experts = config.num_routed_experts
        self.num_shared_experts = config.num_shared_experts
        
        # Routed experts
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList([
            MoEExpert(config) for _ in range(config.num_routed_experts)
        ])
        
        # Shared experts
        if config.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                MoEExpert(config) for _ in range(config.num_shared_experts)
            ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Routed experts
        routing_weights, selected_experts = self.gate(hidden_states)
        
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_states[expert_mask]
                expert_output = expert(expert_input)
                
                # Weight by routing scores
                expert_weights = routing_weights[expert_mask]
                expert_weights = expert_weights[:, selected_experts[expert_mask] == i]
                expert_output = expert_output * expert_weights.sum(dim=-1, keepdim=True)
                
                final_hidden_states[expert_mask] += expert_output
        
        # Shared experts
        if hasattr(self, 'shared_experts'):
            shared_output = torch.zeros_like(hidden_states)
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(hidden_states)
            final_hidden_states += shared_output / len(self.shared_experts)
        
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        
        # Auxiliary losses for load balancing
        aux_losses = {
            'load_balancing_loss': self._compute_load_balancing_loss(routing_weights, selected_experts)
        }
        
        return final_hidden_states, aux_losses
    
    def _compute_load_balancing_loss(self, routing_weights: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage."""
        # Count how many tokens are assigned to each expert
        expert_counts = torch.zeros(self.num_routed_experts, device=routing_weights.device)
        for i in range(self.num_routed_experts):
            expert_counts[i] = (selected_experts == i).float().sum()
        
        # Normalize by total number of tokens
        expert_probs = expert_counts / expert_counts.sum()
        
        # Compute load balancing loss (encourage uniform distribution)
        uniform_prob = 1.0 / self.num_routed_experts
        load_balancing_loss = F.mse_loss(expert_probs, torch.full_like(expert_probs, uniform_prob))
        
        return load_balancing_loss


class CuriosityModule(nn.Module):
    """Curiosity-driven exploration module for RL."""
    
    def __init__(self, config: NativeV3RLConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.curiosity_feature_dim
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Forward model (predicts next state features)
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + config.vocab_size, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Inverse model (predicts action from state transition)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, config.vocab_size)
        )
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode states
        state_features = self.feature_encoder(states)
        next_state_features = self.feature_encoder(next_states)
        
        # Forward model prediction
        action_onehot = F.one_hot(actions, num_classes=self.config.vocab_size).float()
        predicted_next_features = self.forward_model(
            torch.cat([state_features, action_onehot], dim=-1)
        )
        
        # Inverse model prediction
        predicted_actions = self.inverse_model(
            torch.cat([state_features, next_state_features], dim=-1)
        )
        
        # Compute intrinsic reward (prediction error)
        intrinsic_reward = F.mse_loss(
            predicted_next_features, next_state_features.detach(), reduction='none'
        ).mean(dim=-1)
        
        return {
            'intrinsic_reward': intrinsic_reward,
            'forward_loss': F.mse_loss(predicted_next_features, next_state_features.detach()),
            'inverse_loss': F.cross_entropy(predicted_actions, actions)
        }


class ValueHead(nn.Module):
    """Value head for RL training."""
    
    def __init__(self, config: NativeV3RLConfig, reward_type: str):
        super().__init__()
        self.reward_type = reward_type
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden_states).squeeze(-1)


class NativeV3RLDecoderLayer(nn.Module):
    """Single decoder layer with MLA and MoE."""
    
    def __init__(self, config: NativeV3RLConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Multi-Head Latent Attention
        self.self_attn = MultiHeadLatentAttention(config, layer_idx)
        
        # MoE or dense layer
        if layer_idx >= config.num_dense_layers:
            self.mlp = MoELayer(config)
            self.is_moe = True
        else:
            # Dense layer for first few layers
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            )
            self.is_moe = False
        
        # Layer normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.is_moe:
            hidden_states, aux_losses = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            aux_losses = {}
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        if self.is_moe:
            outputs += (aux_losses,)
        
        return outputs


class NativeV3RLModel(nn.Module):
    """Native DeepSeek-V3 model with RL capabilities."""
    
    def __init__(self, config: NativeV3RLConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NativeV3RLDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # RL components
        if config.use_reinforcement_learning:
            # Value heads for each reward type
            if config.use_value_heads:
                self.value_heads = nn.ModuleDict({
                    reward_type: ValueHead(config, reward_type)
                    for reward_type in config.reward_types
                })
            
            # Curiosity module
            if config.use_curiosity:
                self.curiosity_module = CuriosityModule(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rewards: Optional[Dict[str, torch.Tensor]] = None,
        compute_values: Optional[bool] = False,
    ) -> Union[Tuple, Dict[str, Any]]:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        
        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.layers))
        else:
            past_length = past_key_values[0][0].shape[2]
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_length
            )
        
        hidden_states = inputs_embeds
        
        if output_hidden_states:
            all_hidden_states = ()
        
        if output_attentions:
            all_self_attns = ()
        
        next_decoder_cache = () if use_cache else None
        all_aux_losses = []
        
        # Decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # Collect auxiliary losses from MoE layers
            if len(layer_outputs) > (3 if output_attentions and use_cache else 2 if output_attentions or use_cache else 1):
                aux_losses = layer_outputs[-1]
                if aux_losses:
                    all_aux_losses.append(aux_losses)
        
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        # Compute values if requested
        values = {}
        if compute_values and hasattr(self, 'value_heads'):
            for reward_type, value_head in self.value_heads.items():
                values[reward_type] = value_head(hidden_states)
        
        # Compute curiosity if enabled
        curiosity_info = {}
        if hasattr(self, 'curiosity_module') and rewards is not None:
            # For curiosity, we need state transitions
            # This is a simplified implementation
            if input_ids is not None and input_ids.shape[1] > 1:
                states = hidden_states[:, :-1]
                next_states = hidden_states[:, 1:]
                actions = input_ids[:, 1:]
                
                curiosity_info = self.curiosity_module(states, actions, next_states)
        
        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_aux_losses,
                values,
                curiosity_info
            ] if v is not None)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states if output_hidden_states else None,
            'attentions': all_self_attns if output_attentions else None,
            'aux_losses': all_aux_losses,
            'values': values,
            'curiosity_info': curiosity_info
        }
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Prepare causal attention mask."""
        batch_size, seq_length = input_shape
        seq_length_with_past = seq_length + past_key_values_length
        
        # Create causal mask
        causal_mask = torch.full(
            (seq_length, seq_length_with_past),
            torch.finfo(inputs_embeds.dtype).min,
            device=inputs_embeds.device
        )
        
        if seq_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=past_key_values_length + 1)
        
        causal_mask = causal_mask[None, None, :, :].expand(
            batch_size, 1, seq_length, seq_length_with_past
        )
        
        if attention_mask is not None:
            expanded_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, seq_length_with_past
            ).to(causal_mask.dtype)
            causal_mask = causal_mask.masked_fill(expanded_mask == 0, torch.finfo(inputs_embeds.dtype).min)
        
        return causal_mask


class NativeV3RLForCausalLM(nn.Module):
    """Native DeepSeek-V3 with RL for causal language modeling."""
    
    def __init__(self, config: NativeV3RLConfig):
        super().__init__()
        self.config = config
        self.model = NativeV3RLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # RL components
        if config.use_reinforcement_learning:
            self.ppo_clip_ratio = config.ppo_clip_ratio
            self.value_loss_coef = config.value_loss_coef
            self.entropy_coef = config.entropy_coef
            self.curiosity_coef = config.curiosity_coef
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rewards: Optional[Dict[str, torch.Tensor]] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        compute_values: Optional[bool] = False,
    ) -> Union[Tuple, Dict[str, Any]]:
        
        return_dict = return_dict if return_dict is not None else True
        
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rewards=rewards,
            compute_values=compute_values,
        )
        
        if return_dict:
            hidden_states = outputs['last_hidden_state']
            aux_losses = outputs.get('aux_losses', [])
            values = outputs.get('values', {})
            curiosity_info = outputs.get('curiosity_info', {})
        else:
            hidden_states = outputs[0]
            aux_losses = outputs[4] if len(outputs) > 4 else []
            values = outputs[5] if len(outputs) > 5 else {}
            curiosity_info = outputs[6] if len(outputs) > 6 else {}
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # Add auxiliary losses from MoE layers
            if aux_losses:
                aux_loss = sum(
                    aux_loss_dict.get('load_balancing_loss', 0.0)
                    for aux_loss_dict in aux_losses
                ) / len(aux_losses)
                loss = loss + 0.01 * aux_loss  # Small coefficient for auxiliary loss
        
        # RL-specific computations
        rl_info = {}
        if self.config.use_reinforcement_learning and rewards is not None:
            # Compute policy loss (PPO)
            if old_log_probs is not None:
                log_probs = F.log_softmax(logits, dim=-1)
                if labels is not None:
                    # Get log probabilities for the actual tokens
                    current_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                    
                    # PPO clipped loss
                    ratio = torch.exp(current_log_probs - old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio)
                    
                    # Compute advantages (simplified - in practice, use GAE)
                    advantages = {}
                    for reward_type, reward_values in rewards.items():
                        if reward_type in values:
                            advantages[reward_type] = reward_values - values[reward_type].detach()
                    
                    # Multi-objective policy loss
                    policy_losses = []
                    for reward_type, advantage in advantages.items():
                        policy_loss1 = ratio * advantage
                        policy_loss2 = clipped_ratio * advantage
                        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                        policy_losses.append(policy_loss)
                    
                    rl_info['policy_loss'] = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
            
            # Value loss
            if values:
                value_losses = []
                for reward_type, reward_values in rewards.items():
                    if reward_type in values:
                        value_loss = F.mse_loss(values[reward_type], reward_values)
                        value_losses.append(value_loss)
                
                rl_info['value_loss'] = sum(value_losses) / len(value_losses) if value_losses else 0.0
                
                if loss is not None:
                    loss = loss + self.value_loss_coef * rl_info['value_loss']
            
            # Entropy loss for exploration
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            rl_info['entropy'] = entropy
            
            if loss is not None:
                loss = loss - self.entropy_coef * entropy
            
            # Curiosity loss
            if curiosity_info:
                curiosity_loss = (
                    curiosity_info.get('forward_loss', 0.0) +
                    curiosity_info.get('inverse_loss', 0.0)
                )
                rl_info['curiosity_loss'] = curiosity_loss
                
                if loss is not None:
                    loss = loss + self.curiosity_coef * curiosity_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        result = {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.get('past_key_values'),
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions'),
            'aux_losses': aux_losses,
            'values': values,
            'rl_info': rl_info,
            'curiosity_info': curiosity_info
        }
        
        return result
    
    def generate_with_rl(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate text with RL-aware sampling."""
        
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        log_probs = []
        values_history = {reward_type: [] for reward_type in self.config.reward_types}
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated_ids[:, -1:] if past_key_values is not None else generated_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    compute_values=True,
                    return_dict=True
                )
                
                logits = outputs['logits'][:, -1, :] / temperature
                past_key_values = outputs['past_key_values']
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Store log probabilities
                log_prob = F.log_softmax(logits, dim=-1).gather(-1, next_token)
                log_probs.append(log_prob.squeeze(-1))
                
                # Store values
                if 'values' in outputs and outputs['values']:
                    for reward_type, value in outputs['values'].items():
                        values_history[reward_type].append(value[:, -1])
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == self.config.vocab_size - 1:  # Assuming last token is EOS
                    break
        
        return {
            'generated_ids': generated_ids,
            'log_probs': torch.stack(log_probs, dim=1) if log_probs else None,
            'values_history': {k: torch.stack(v, dim=1) if v else None for k, v in values_history.items()}
        }


# Export the main classes
__all__ = [
    'NativeV3RLConfig',
    'NativeV3RLForCausalLM',
    'NativeV3RLModel'
]