#!/usr/bin/env python3
"""
Mixture of Experts (MoE) Transformer Implementation
Sparse expert routing for efficient scaling without using DeepSeek.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from dataclasses import dataclass


@dataclass
class MoETransformerConfig:
    """Configuration for MoE Transformer model."""
    vocab_size: int = 50257
    max_position_embeddings: int = 8192
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 16384
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # MoE specific parameters
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    use_expert_choice: bool = False
    expert_dropout: float = 0.1
    
    # Load balancing
    load_balancing_loss_coef: float = 0.01
    use_auxiliary_loss: bool = True
    
    # Expert routing
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001
    
    # Hierarchical experts
    use_hierarchical_experts: bool = False
    num_expert_groups: int = 2
    
    # Expert specialization
    use_expert_specialization: bool = True
    specialization_loss_coef: float = 0.001
    
    # Advanced features
    use_rotary_embeddings: bool = True
    use_rms_norm: bool = True
    use_pre_norm: bool = True


class TopKRouter(nn.Module):
    """Top-K router for expert selection."""
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        # Router network
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.dropout = nn.Dropout(config.expert_dropout)
        
        # Expert capacity
        self.expert_capacity_factor = config.expert_capacity_factor
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            expert_weights: [batch_size, seq_len, num_experts_per_token]
            expert_indices: [batch_size, seq_len, num_experts_per_token]
            router_logits: [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, 
            self.num_experts_per_token, 
            dim=-1
        )
        
        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return expert_weights, expert_indices, router_logits


class ExpertChoiceRouter(nn.Module):
    """Expert-choice router where experts choose tokens."""
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.expert_capacity_factor = config.expert_capacity_factor
        
        # Router network
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert-choice routing."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        total_tokens = batch_size * seq_len
        
        # Compute router logits
        router_logits = self.router(hidden_states.view(-1, hidden_size))  # [total_tokens, num_experts]
        
        # Expert capacity
        expert_capacity = int(self.expert_capacity_factor * total_tokens / self.num_experts)
        
        # Each expert chooses top tokens
        expert_weights = torch.zeros(total_tokens, self.num_experts, device=hidden_states.device)
        expert_indices = torch.zeros(total_tokens, self.num_experts, dtype=torch.long, device=hidden_states.device)
        
        for expert_idx in range(self.num_experts):
            expert_scores = router_logits[:, expert_idx]
            top_tokens = torch.topk(expert_scores, min(expert_capacity, total_tokens), dim=0)
            
            expert_weights[top_tokens.indices, expert_idx] = F.softmax(top_tokens.values, dim=0)
            expert_indices[top_tokens.indices, expert_idx] = expert_idx
        
        # Reshape back
        expert_weights = expert_weights.view(batch_size, seq_len, self.num_experts)
        expert_indices = expert_indices.view(batch_size, seq_len, self.num_experts)
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)
        
        return expert_weights, expert_indices, router_logits


class Expert(nn.Module):
    """Individual expert network."""
    
    def __init__(self, config: MoETransformerConfig, expert_id: int = 0):
        super().__init__()
        self.config = config
        self.expert_id = expert_id
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Expert-specific parameters
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Expert specialization tracking
        if config.use_expert_specialization:
            self.register_buffer('usage_count', torch.zeros(1))
            self.register_buffer('specialization_score', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        intermediate = self.dropout(intermediate)
        output = self.down_proj(intermediate)
        
        # Update usage statistics
        if self.config.use_expert_specialization and self.training:
            self.usage_count += x.shape[0] * x.shape[1]  # batch_size * seq_len
        
        return output


class HierarchicalExpertLayer(nn.Module):
    """Hierarchical expert layer with grouped experts."""
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.experts_per_group = self.num_experts // self.num_expert_groups
        
        # Group routers
        self.group_router = nn.Linear(config.hidden_size, self.num_expert_groups, bias=False)
        
        # Expert groups
        self.expert_groups = nn.ModuleList([
            nn.ModuleList([
                Expert(config, group_id * self.experts_per_group + expert_id)
                for expert_id in range(self.experts_per_group)
            ])
            for group_id in range(self.num_expert_groups)
        ])
        
        # Expert routers for each group
        self.expert_routers = nn.ModuleList([
            nn.Linear(config.hidden_size, self.experts_per_group, bias=False)
            for _ in range(self.num_expert_groups)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Hierarchical expert routing."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Group selection
        group_logits = self.group_router(hidden_states)
        group_probs = F.softmax(group_logits, dim=-1)
        selected_group = torch.argmax(group_probs, dim=-1)  # [batch_size, seq_len]
        
        # Expert selection within groups
        output = torch.zeros_like(hidden_states)
        aux_losses = {}
        
        for group_id in range(self.num_expert_groups):
            # Mask for tokens assigned to this group
            group_mask = (selected_group == group_id)
            if not group_mask.any():
                continue
            
            # Get tokens for this group
            group_tokens = hidden_states[group_mask]  # [num_group_tokens, hidden_size]
            
            # Expert routing within group
            expert_logits = self.expert_routers[group_id](group_tokens)
            expert_probs = F.softmax(expert_logits, dim=-1)
            
            # Select top experts
            expert_weights, expert_indices = torch.topk(
                expert_probs, 
                self.config.num_experts_per_token, 
                dim=-1
            )
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
            
            # Apply experts
            group_output = torch.zeros_like(group_tokens)
            for expert_idx in range(self.experts_per_group):
                expert_mask = (expert_indices == expert_idx).any(dim=-1)
                if not expert_mask.any():
                    continue
                
                expert_tokens = group_tokens[expert_mask]
                expert_output = self.expert_groups[group_id][expert_idx](expert_tokens)
                
                # Weight by expert selection probability
                weights = expert_weights[expert_mask]
                expert_weight = weights[expert_indices[expert_mask] == expert_idx].unsqueeze(-1)
                group_output[expert_mask] += expert_weight * expert_output
            
            # Assign back to output
            output[group_mask] = group_output
        
        return output, aux_losses


class MoELayer(nn.Module):
    """Mixture of Experts layer."""
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        # Router
        if config.use_expert_choice:
            self.router = ExpertChoiceRouter(config)
        else:
            self.router = TopKRouter(config)
        
        # Experts
        if config.use_hierarchical_experts:
            self.expert_layer = HierarchicalExpertLayer(config)
        else:
            self.experts = nn.ModuleList([
                Expert(config, expert_id) for expert_id in range(config.num_experts)
            ])
        
        # Load balancing
        self.load_balancing_loss_coef = config.load_balancing_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef

    def _compute_auxiliary_losses(self, router_logits: torch.Tensor, expert_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for load balancing."""
        aux_losses = {}
        
        # Load balancing loss
        if self.load_balancing_loss_coef > 0:
            # Compute expert usage
            router_probs = F.softmax(router_logits, dim=-1)
            expert_usage = torch.mean(router_probs, dim=[0, 1])  # [num_experts]
            
            # Ideal uniform distribution
            uniform_usage = torch.ones_like(expert_usage) / self.num_experts
            
            # Load balancing loss (encourage uniform usage)
            load_balance_loss = F.mse_loss(expert_usage, uniform_usage)
            aux_losses['load_balance_loss'] = self.load_balancing_loss_coef * load_balance_loss
        
        # Router z-loss (encourage low router logit magnitudes)
        if self.router_z_loss_coef > 0:
            z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
            aux_losses['router_z_loss'] = self.router_z_loss_coef * z_loss
        
        return aux_losses

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through MoE layer."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if self.config.use_hierarchical_experts:
            return self.expert_layer(hidden_states)
        
        # Route tokens to experts
        expert_weights, expert_indices, router_logits = self.router(hidden_states)
        
        # Compute auxiliary losses
        aux_losses = self._compute_auxiliary_losses(router_logits, expert_indices)
        
        # Apply experts
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx)
            if not expert_mask.any():
                continue
            
            # Get tokens and weights for this expert
            token_indices = torch.where(expert_mask)
            expert_tokens = hidden_states[token_indices[0], token_indices[1]]  # [num_expert_tokens, hidden_size]
            token_weights = expert_weights[expert_mask].unsqueeze(-1)  # [num_expert_tokens, 1]
            
            # Apply expert
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Weight and accumulate output
            weighted_output = token_weights * expert_output
            output[token_indices[0], token_indices[1]] += weighted_output
        
        return output, aux_losses


class MoETransformerLayer(nn.Module):
    """Transformer layer with MoE."""
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Self attention (reuse from native transformer)
        from ..native_transformer.model import AdaptiveAttention
        self.self_attn = AdaptiveAttention(config)
        
        # MoE layer instead of regular MLP
        self.moe = MoELayer(config)
        
        # Normalization layers
        if config.use_rms_norm:
            from ..native_transformer.model import RMSNorm
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        residual = hidden_states
        
        # Pre-norm
        if self.config.use_pre_norm:
            hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]
        
        # Residual connection
        hidden_states = residual + attn_output
        
        if not self.config.use_pre_norm:
            hidden_states = self.input_layernorm(hidden_states)
        
        # MoE
        residual = hidden_states
        if self.config.use_pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        moe_output, aux_losses = self.moe(hidden_states)
        
        # Residual connection
        hidden_states = residual + moe_output
        
        if not self.config.use_pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        outputs = (hidden_states, aux_losses)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[-1],)
        
        return outputs


class MoETransformerModel(nn.Module):
    """MoE Transformer Model."""
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.vocab_size - 1
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Position embeddings (if not using RoPE)
        if not config.use_rotary_embeddings:
            self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([MoETransformerLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Final norm
        if config.use_rms_norm:
            from ..native_transformer.model import RMSNorm
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        
        # Input processing (similar to native transformer)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Add position embeddings if not using RoPE
        if not self.config.use_rotary_embeddings:
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            position_embeddings = self.embed_positions(position_ids)
            hidden_states = inputs_embeds + position_embeddings
        else:
            hidden_states = inputs_embeds
        
        # Transformer layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_aux_losses = []
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            aux_losses = layer_outputs[1]
            all_aux_losses.append(aux_losses)
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[2] if len(layer_outputs) > 2 else None,)
        
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "aux_losses": all_aux_losses,
        }


class MoETransformerForCausalLM(nn.Module):
    """MoE Transformer Model for Causal Language Modeling."""
    
    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.model = MoETransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        
        # Decoder outputs
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
        )
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
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
            
            # Add auxiliary losses
            aux_losses = outputs.get("aux_losses", [])
            for layer_aux_losses in aux_losses:
                for aux_loss_name, aux_loss_value in layer_aux_losses.items():
                    loss += aux_loss_value
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "aux_losses": outputs.get("aux_losses", []),
        }