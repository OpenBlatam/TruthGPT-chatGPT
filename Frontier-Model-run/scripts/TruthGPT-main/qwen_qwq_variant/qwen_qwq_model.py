"""
Qwen QwQ Model Implementation - Billion Parameter Scale
Advanced reasoning model with MoE, Flash Attention, and GRPO integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import math
import warnings

@dataclass
class QwenQwQConfig:
    """Configuration for Qwen QwQ billion-parameter model."""
    vocab_size: int = 151936
    hidden_size: int = 8192  # Billion-parameter scale
    intermediate_size: int = 49152
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    max_position_embeddings: int = 131072  # Extended context
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 70
    tie_word_embeddings: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    reasoning_depth: int = 8
    reasoning_width: int = 16
    use_chain_of_thought: bool = True
    use_step_by_step: bool = True
    
    use_moe: bool = True
    num_experts: int = 128  # Increased for billion-parameter scale
    num_experts_per_tok: int = 8
    shared_expert_intermediate_size: int = 2816
    n_routed_experts: int = 128
    n_activated_experts: int = 8
    n_shared_experts: int = 4
    moe_inter_dim: int = 49152
    
    use_flash_attention: bool = True
    flash_attention_version: str = "v2"
    use_triton_kernels: bool = True
    
    enable_quantization: bool = True
    quantization_bits: int = 8
    enable_gradient_checkpointing: bool = True
    enable_compilation: bool = True
    use_fused_kernels: bool = True
    use_memory_efficient_attention: bool = True
    
    use_grpo: bool = True
    grpo_beta: float = 0.1
    grpo_gamma: float = 0.99
    grpo_eps: float = 1e-8

class QwQRMSNorm(nn.Module):
    """RMS Normalization optimized for QwQ."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class QwQRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for QwQ with extended context support."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 131072, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QwQMLP(nn.Module):
    """MLP layer for QwQ with optimizations."""
    
    def __init__(self, config: QwenQwQConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class QwQMoEGate(nn.Module):
    """Advanced MoE gating mechanism for QwQ."""
    
    def __init__(self, config: QwenQwQConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        self.weight = nn.Parameter(torch.empty((config.num_experts, config.hidden_size)))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = F.linear(hidden_states, self.weight, None)
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        aux_loss = self.compute_aux_loss(router_logits, selected_experts)
        
        return routing_weights, selected_experts, aux_loss
        
    def compute_aux_loss(self, router_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss for load balancing."""
        num_tokens = router_logits.shape[0]
        
        expert_counts = torch.zeros(self.num_experts, device=router_logits.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (selected_experts == expert_idx).sum()
        
        expert_usage = expert_counts / num_tokens
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        aux_loss = F.mse_loss(expert_usage, target_usage)
        
        return aux_loss

class QwQExpert(nn.Module):
    """Individual expert in MoE."""
    
    def __init__(self, config: QwenQwQConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_inter_dim
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class QwQMoE(nn.Module):
    """Mixture of Experts layer for QwQ."""
    
    def __init__(self, config: QwenQwQConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        self.gate = QwQMoEGate(config)
        self.experts = nn.ModuleList([QwQExpert(config) for _ in range(config.num_experts)])
        
        self.shared_experts = nn.ModuleList([
            QwQExpert(config) for _ in range(config.n_shared_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        routing_weights, selected_experts, aux_loss = self.gate(hidden_states)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx)
            
            if expert_mask.any():
                expert_indices = torch.where(expert_mask)
                batch_indices, expert_positions = expert_indices
                
                if len(batch_indices) > 0:
                    expert_input = hidden_states[batch_indices]
                    expert_output = expert_layer(expert_input)
                    
                    weights = routing_weights[batch_indices, expert_positions].unsqueeze(-1)
                    final_hidden_states[batch_indices] += expert_output * weights
        
        shared_output = torch.zeros_like(final_hidden_states)
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(hidden_states)
        
        final_hidden_states += shared_output / len(self.shared_experts)
        
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), aux_loss

class QwQFlashAttention(nn.Module):
    """Flash Attention implementation for QwQ."""
    
    def __init__(self, config: QwenQwQConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = QwQRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads if n_kv_heads < n_heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class QwQReasoningLayer(nn.Module):
    """Specialized reasoning layer for QwQ."""
    
    def __init__(self, config: QwenQwQConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.cot_processor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        self.step_processor = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(config.reasoning_depth)
        ])
        
        self.reasoning_norm = QwQRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.use_chain_of_thought:
            cot_output = self.cot_processor(hidden_states)
            hidden_states = hidden_states + cot_output
        
        if self.config.use_step_by_step:
            for step_layer in self.step_processor:
                step_output = step_layer(hidden_states)
                hidden_states = hidden_states + step_output
                hidden_states = self.reasoning_norm(hidden_states)
        
        return hidden_states

class QwQDecoderLayer(nn.Module):
    """QwQ Transformer decoder layer."""
    
    def __init__(self, config: QwenQwQConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.self_attn = QwQFlashAttention(config)
        
        if layer_idx >= config.num_hidden_layers // 2:
            self.mlp = QwQMoE(config)
            self.use_moe = True
        else:
            self.mlp = QwQMLP(config)
            self.use_moe = False
            
        self.input_layernorm = QwQRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwQRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if layer_idx % (config.num_hidden_layers // config.reasoning_depth) == 0:
            self.reasoning_layer = QwQReasoningLayer(config)
            self.use_reasoning = True
        else:
            self.use_reasoning = False

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
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
            
        hidden_states = residual + hidden_states
        
        if self.use_reasoning:
            hidden_states = self.reasoning_layer(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        if self.use_moe:
            outputs += (aux_loss,)

        return outputs

class QwenQwQModel(nn.Module):
    """QwQ Model with billion-parameter scale."""
    
    def __init__(self, config: QwenQwQConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.vocab_size - 1
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            QwQDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = QwQRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

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
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, 'use_cache') else False

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

        if attention_mask is not None and hasattr(F, "scaled_dot_product_attention"):
            attention_mask = attention_mask.bool()
        elif attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                warnings.warn(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
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
                
            if len(layer_outputs) > (3 if output_attentions and use_cache else 2 if output_attentions or use_cache else 1):
                aux_loss_idx = -1
                if isinstance(layer_outputs[aux_loss_idx], torch.Tensor):
                    total_aux_loss += layer_outputs[aux_loss_idx]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        class ModelOutput:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            aux_loss=total_aux_loss
        )

    def _gradient_checkpointing_func(self, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(*args, **kwargs)

class QwenQwQForCausalLM(nn.Module):
    """QwQ Model for Causal Language Modeling."""
    
    def __init__(self, config: QwenQwQConfig):
        super().__init__()
        self.config = config
        self.model = QwenQwQModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, 'use_return_dict') else True

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

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        aux_loss = getattr(outputs, 'aux_loss', torch.tensor(0.0, device=logits.device))
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            if aux_loss.numel() > 0:
                loss = loss + 0.01 * aux_loss  # Small weight for aux loss

        class CausalLMOutput:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_loss=aux_loss
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

def create_qwen_qwq_model(config: Dict[str, Any]) -> QwenQwQForCausalLM:
    """Factory function to create QwQ model."""
    qwq_config = QwenQwQConfig(**config)
    model = QwenQwQForCausalLM(qwq_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"QwQ Model created successfully!")
    print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
    print(f"Model size: ~{total_params * 4 / 1e9:.2f} GB (FP32)")
    
    return model
