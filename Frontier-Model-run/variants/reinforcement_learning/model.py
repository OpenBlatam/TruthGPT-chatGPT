#!/usr/bin/env python3
"""
Reinforcement Learning Transformer Implementation
Advanced RL training with multiple reward signals without using DeepSeek.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class RLTransformerConfig:
    """Configuration for RL Transformer model."""
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
    
    # RL specific parameters
    rl_algorithm: str = "ppo"  # "ppo", "a2c", "sac", "dqn"
    value_head_hidden_size: int = 1024
    num_value_heads: int = 1
    
    # PPO parameters
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 4
    ppo_batch_size: int = 64
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Multi-objective RL
    use_multi_objective: bool = True
    reward_types: List[str] = None  # ['accuracy', 'fluency', 'safety', 'creativity']
    reward_weights: List[float] = None
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 5
    curriculum_threshold: float = 0.8
    
    # Self-play
    use_self_play: bool = False
    self_play_opponents: int = 3
    
    # Experience replay
    use_experience_replay: bool = True
    replay_buffer_size: int = 10000
    replay_batch_size: int = 32
    
    # Reward shaping
    use_reward_shaping: bool = True
    intrinsic_motivation: bool = True
    curiosity_coef: float = 0.1
    
    # Advanced features
    use_rotary_embeddings: bool = True
    use_rms_norm: bool = True
    use_pre_norm: bool = True


class ValueHead(nn.Module):
    """Value function head for RL."""
    
    def __init__(self, config: RLTransformerConfig, reward_type: str = "default"):
        super().__init__()
        self.config = config
        self.reward_type = reward_type
        self.hidden_size = config.hidden_size
        self.value_hidden_size = config.value_head_hidden_size
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.value_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.value_hidden_size, self.value_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.value_hidden_size // 2, 1)
        )
        
        # Advantage estimation
        self.advantage_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.value_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.value_hidden_size, config.vocab_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute value and advantage estimates.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            values: [batch_size, seq_len, 1]
            advantages: [batch_size, seq_len, vocab_size]
        """
        # Value estimation
        values = self.value_net(hidden_states)
        
        # Advantage estimation (for dueling architecture)
        advantages = self.advantage_net(hidden_states)
        
        return values, advantages


class CuriosityModule(nn.Module):
    """Intrinsic curiosity module for exploration."""
    
    def __init__(self, config: RLTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Forward model (predicts next state given current state and action)
        self.forward_model = nn.Sequential(
            nn.Linear(self.hidden_size + config.vocab_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Inverse model (predicts action given current and next state)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, config.vocab_size)
        )
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2)
        )

    def forward(
        self, 
        current_states: torch.Tensor, 
        actions: torch.Tensor, 
        next_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute curiosity-driven intrinsic reward.
        
        Args:
            current_states: [batch_size, seq_len, hidden_size]
            actions: [batch_size, seq_len, vocab_size] (one-hot)
            next_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            intrinsic_reward: [batch_size, seq_len]
            forward_loss: scalar
            inverse_loss: scalar
        """
        # Encode features
        current_features = self.feature_encoder(current_states)
        next_features = self.feature_encoder(next_states)
        
        # Forward model prediction
        forward_input = torch.cat([current_features, actions], dim=-1)
        predicted_next_features = self.forward_model(forward_input)
        
        # Forward model loss (prediction error as intrinsic reward)
        forward_loss = F.mse_loss(predicted_next_features, next_features, reduction='none')
        intrinsic_reward = forward_loss.mean(dim=-1)  # [batch_size, seq_len]
        
        # Inverse model prediction
        inverse_input = torch.cat([current_features, next_features], dim=-1)
        predicted_actions = self.inverse_model(inverse_input)
        
        # Inverse model loss
        inverse_loss = F.cross_entropy(
            predicted_actions.view(-1, predicted_actions.size(-1)),
            actions.argmax(dim=-1).view(-1),
            reduction='mean'
        )
        
        return intrinsic_reward, forward_loss.mean(), inverse_loss


class ExperienceReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Dict[str, torch.Tensor]):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Stack experiences
        stacked_batch = {}
        for key in batch[0].keys():
            stacked_batch[key] = torch.stack([exp[key] for exp in batch])
        
        return stacked_batch
    
    def __len__(self):
        return len(self.buffer)


class MultiObjectiveRewardAggregator(nn.Module):
    """Aggregates multiple reward signals."""
    
    def __init__(self, config: RLTransformerConfig):
        super().__init__()
        self.config = config
        self.reward_types = config.reward_types or ['default']
        self.num_rewards = len(self.reward_types)
        
        # Learnable reward weights
        if config.reward_weights:
            self.reward_weights = nn.Parameter(torch.tensor(config.reward_weights))
        else:
            self.reward_weights = nn.Parameter(torch.ones(self.num_rewards) / self.num_rewards)
        
        # Reward normalization
        self.reward_normalizers = nn.ModuleDict({
            reward_type: nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh()
            ) for reward_type in self.reward_types
        })

    def forward(self, rewards: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple reward signals.
        
        Args:
            rewards: Dict mapping reward type to tensor [batch_size, seq_len]
            
        Returns:
            aggregated_reward: [batch_size, seq_len]
        """
        normalized_rewards = []
        
        for i, reward_type in enumerate(self.reward_types):
            if reward_type in rewards:
                reward = rewards[reward_type].unsqueeze(-1)  # [batch_size, seq_len, 1]
                normalized_reward = self.reward_normalizers[reward_type](reward).squeeze(-1)
                normalized_rewards.append(self.reward_weights[i] * normalized_reward)
        
        if normalized_rewards:
            return torch.stack(normalized_rewards).sum(dim=0)
        else:
            return torch.zeros_like(list(rewards.values())[0])


class RLTransformerLayer(nn.Module):
    """Transformer layer with RL capabilities."""
    
    def __init__(self, config: RLTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Self attention
        from ..native_transformer.model import AdaptiveAttention
        self.self_attn = AdaptiveAttention(config)
        
        # MLP
        from ..native_transformer.model import NativeMLP
        self.mlp = NativeMLP(config)
        
        # Value heads for different reward types
        if config.use_multi_objective:
            self.value_heads = nn.ModuleDict({
                reward_type: ValueHead(config, reward_type)
                for reward_type in (config.reward_types or ['default'])
            })
        else:
            self.value_head = ValueHead(config)
        
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
        compute_values: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
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
        
        # MLP
        residual = hidden_states
        if self.config.use_pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        mlp_output = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + mlp_output
        
        if not self.config.use_pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Compute values if requested
        rl_info = {}
        if compute_values:
            if self.config.use_multi_objective:
                values = {}
                advantages = {}
                for reward_type, value_head in self.value_heads.items():
                    v, a = value_head(hidden_states)
                    values[reward_type] = v
                    advantages[reward_type] = a
                rl_info['values'] = values
                rl_info['advantages'] = advantages
            else:
                values, advantages = self.value_head(hidden_states)
                rl_info['values'] = values
                rl_info['advantages'] = advantages
        
        outputs = (hidden_states, rl_info)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if use_cache:
            outputs += (attn_outputs[-1],)
        
        return outputs


class RLTransformerModel(nn.Module):
    """RL Transformer Model."""
    
    def __init__(self, config: RLTransformerConfig):
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
        self.layers = nn.ModuleList([
            RLTransformerLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        if config.use_rms_norm:
            from ..native_transformer.model import RMSNorm
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # RL components
        if config.use_multi_objective:
            self.reward_aggregator = MultiObjectiveRewardAggregator(config)
        
        if config.intrinsic_motivation:
            self.curiosity_module = CuriosityModule(config)
        
        # Experience replay
        if config.use_experience_replay:
            self.replay_buffer = ExperienceReplayBuffer(config.replay_buffer_size)
        
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
        compute_values: bool = False,
    ) -> Dict[str, Any]:
        
        # Input processing
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
        all_rl_info = []
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
                compute_values=compute_values,
            )
            
            hidden_states = layer_outputs[0]
            rl_info = layer_outputs[1]
            all_rl_info.append(rl_info)
            
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
            "rl_info": all_rl_info,
        }

    def compute_intrinsic_reward(
        self, 
        current_states: torch.Tensor, 
        actions: torch.Tensor, 
        next_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute intrinsic curiosity reward."""
        if self.config.intrinsic_motivation:
            return self.curiosity_module(current_states, actions, next_states)
        else:
            batch_size, seq_len = current_states.shape[:2]
            return (
                torch.zeros(batch_size, seq_len, device=current_states.device),
                torch.tensor(0.0, device=current_states.device),
                torch.tensor(0.0, device=current_states.device)
            )

    def aggregate_rewards(self, rewards: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate multiple reward signals."""
        if self.config.use_multi_objective:
            return self.reward_aggregator(rewards)
        else:
            return list(rewards.values())[0]

    def store_experience(self, experience: Dict[str, torch.Tensor]):
        """Store experience in replay buffer."""
        if self.config.use_experience_replay:
            self.replay_buffer.push(experience)

    def sample_experience(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample experience from replay buffer."""
        if self.config.use_experience_replay:
            return self.replay_buffer.sample(batch_size)
        else:
            return {}


class RLTransformerForCausalLM(nn.Module):
    """RL Transformer Model for Causal Language Modeling with RL training."""
    
    def __init__(self, config: RLTransformerConfig):
        super().__init__()
        self.config = config
        self.model = RLTransformerModel(config)
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
        rewards: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_values: bool = False,
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
            compute_values=compute_values,
        )
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        # Compute policy loss
        policy_loss = None
        value_loss = None
        
        if labels is not None:
            # Standard language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            policy_loss = loss_fct(shift_logits, shift_labels)
        
        # Compute value loss if values are computed
        if compute_values and rewards is not None:
            rl_info = outputs.get("rl_info", [])
            if rl_info and len(rl_info) > 0:
                # Use values from the last layer
                last_layer_info = rl_info[-1]
                if 'values' in last_layer_info:
                    values = last_layer_info['values']
                    
                    if self.config.use_multi_objective:
                        # Aggregate rewards
                        aggregated_rewards = self.model.aggregate_rewards(rewards)
                        
                        # Compute value loss for each reward type
                        total_value_loss = 0
                        for reward_type, value in values.items():
                            if reward_type in rewards:
                                target_values = rewards[reward_type].unsqueeze(-1)
                                value_loss_component = F.mse_loss(value, target_values)
                                total_value_loss += value_loss_component
                        
                        value_loss = total_value_loss / len(values)
                    else:
                        # Single reward type
                        reward_values = list(rewards.values())[0].unsqueeze(-1)
                        value_loss = F.mse_loss(values, reward_values)
        
        # Combine losses
        total_loss = None
        if policy_loss is not None:
            total_loss = policy_loss
            if value_loss is not None:
                total_loss += self.config.value_loss_coef * value_loss
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "rl_info": outputs.get("rl_info", []),
        }

    def generate_with_rl(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        reward_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Generate text with RL-based sampling."""
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated_ids = input_ids.clone()
        past_key_values = None
        
        # Generation loop
        for step in range(max_length):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_ids[:, -1:] if past_key_values is not None else generated_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    compute_values=True,
                )
                
                logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check for end of sequence
                if (next_token == self.config.vocab_size - 1).all():  # Assuming EOS token
                    break
        
        return {
            "generated_ids": generated_ids,
            "generated_text": generated_ids,  # Would need tokenizer to decode
        }