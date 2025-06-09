"""
Qwen model implementation with advanced optimizations for TruthGPT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math

@dataclass
class QwenConfig:
    """Configuration for Qwen model."""
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    tie_word_embeddings: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    use_flash_attention: bool = True
    use_moe: bool = True
    num_experts: int = 64
    num_experts_per_tok: int = 8
    shared_expert_intermediate_size: int = 1408
    
    enable_quantization: bool = True
    quantization_bits: int = 8
    enable_gradient_checkpointing: bool = True
    enable_compilation: bool = True

class QwenRMSNorm(nn.Module):
    """RMS Normalization for Qwen."""
    
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

class QwenRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Qwen."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
            
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

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

class QwenMLP(nn.Module):
    """MLP module for Qwen."""
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        down = self.down_proj(F.silu(gate) * up)
        return down

class QwenMoEGate(nn.Module):
    """MoE gating mechanism for Qwen."""
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.num_experts
        
        self.scoring_func = "softmax"
        self.alpha = 1.0
        self.seq_aux = True
        self.norm_topk_prob = True
        
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.dim() == 2:
            bsz, seq_len = 1, hidden_states.shape[0]
            h = hidden_states.shape[1]
            hidden_states = hidden_states.unsqueeze(0)  # Add batch dimension
        else:
            bsz, seq_len, h = hidden_states.shape
        
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            scores = logits
            
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
            
        topk_weight = topk_weight.view(bsz, seq_len, -1)
        topk_idx = topk_idx.view(bsz, seq_len, -1)
        
        aux_loss = None
        if self.training and self.seq_aux:
            aux_loss = self.compute_aux_loss(logits, self.n_routed_experts, self.top_k)
            
        return topk_idx, topk_weight, aux_loss
    
    def compute_aux_loss(self, logits: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        
        counts = torch.zeros(n_experts, device=logits.device)
        for i in range(n_experts):
            counts[i] = (torch.argmax(probs, dim=-1) == i).float().sum()
            
        route_prob = torch.mean(probs, dim=0)
        route_frac = counts / counts.sum()
        
        aux_loss = torch.sum(route_prob * route_frac) * n_experts
        return aux_loss

class QwenMoE(nn.Module):
    """Mixture of Experts for Qwen."""
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        
        self.gate = QwenMoEGate(config)
        self.experts = nn.ModuleList([QwenMLP(config) for _ in range(config.num_experts)])
        
        if hasattr(config, 'shared_expert_intermediate_size'):
            self.shared_expert = QwenMLP(config)
            self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            shared_expert_output = shared_expert_output * shared_expert_gate
        else:
            shared_expert_output = 0
            
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits, top_k_weights, aux_loss = self.gate(hidden_states)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        for expert_idx in range(self.config.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_output = expert_layer(hidden_states)
            final_hidden_states += expert_output / self.config.num_experts
            
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states + shared_expert_output
            
        return final_hidden_states, aux_loss

class QwenAttention(nn.Module):
    """Multi-head attention for Qwen."""
    
    def __init__(self, config: QwenConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = QwenRotaryEmbedding(
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
        
        cos = cos[:kv_seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin[:kv_seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        
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
                is_causal=self.is_causal and attention_mask is None and q_len > 1,
            )
            attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
                
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            
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

class QwenDecoderLayer(nn.Module):
    """Qwen decoder layer."""
    
    def __init__(self, config: QwenConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = QwenAttention(config, layer_idx)
        
        if config.use_moe:
            self.mlp = QwenMoE(config)
        else:
            self.mlp = QwenMLP(config)
            
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
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
        
        if isinstance(self.mlp, QwenMoE):
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            aux_loss = None
            
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
            
        if use_cache:
            outputs += (present_key_value,)
            
        if aux_loss is not None:
            outputs += (aux_loss,)
            
        return outputs

class QwenModel(nn.Module):
    """Qwen model implementation."""
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.vocab_size - 1
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if config.enable_gradient_checkpointing:
            self.gradient_checkpointing = True
        else:
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
    ) -> Dict[str, torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions if hasattr(self.config, 'output_attentions') else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states if hasattr(self.config, 'output_hidden_states') else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, 'use_cache') else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, 'use_return_dict') else True
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
            
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False
                
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        if attention_mask is not None and hasattr(F, "scaled_dot_product_attention") and not output_attentions:
            attention_mask = None
        elif attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
            
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            
        hidden_states = inputs_embeds
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False
                
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        aux_loss = 0.0
        
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
                
            if len(layer_outputs) > 3:
                aux_loss += layer_outputs[-1]
                
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        next_cache = next_decoder_cache if use_cache else None
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "aux_loss": aux_loss,
        }
    
    def _gradient_checkpointing_func(self, func, *args, **kwargs):
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

class QwenForCausalLM(nn.Module):
    """Qwen model for causal language modeling."""
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.model = QwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            
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
    ) -> Dict[str, torch.Tensor]:
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
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        aux_loss = outputs.get("aux_loss", 0.0)
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            if aux_loss is not None and aux_loss > 0:
                loss = loss + 0.01 * aux_loss
                
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "aux_loss": aux_loss,
        }
    
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

def create_qwen_model(config: Dict[str, Any]) -> QwenForCausalLM:
    """Create Qwen model from configuration."""
    qwen_config = QwenConfig(
        vocab_size=config.get('vocab_size', 151936),
        hidden_size=config.get('hidden_size', 4096),
        intermediate_size=config.get('intermediate_size', 22016),
        num_hidden_layers=config.get('num_hidden_layers', 32),
        num_attention_heads=config.get('num_attention_heads', 32),
        num_key_value_heads=config.get('num_key_value_heads', 32),
        max_position_embeddings=config.get('max_position_embeddings', 32768),
        rope_theta=config.get('rope_theta', 1000000.0),
        rms_norm_eps=config.get('rms_norm_eps', 1e-6),
        use_sliding_window=config.get('use_sliding_window', False),
        sliding_window=config.get('sliding_window', 4096),
        max_window_layers=config.get('max_window_layers', 28),
        tie_word_embeddings=config.get('tie_word_embeddings', False),
        dropout=config.get('dropout', 0.0),
        attention_dropout=config.get('attention_dropout', 0.0),
        use_flash_attention=config.get('use_flash_attention', True),
        use_moe=config.get('use_moe', True),
        num_experts=config.get('num_experts', 64),
        num_experts_per_tok=config.get('num_experts_per_tok', 8),
        shared_expert_intermediate_size=config.get('shared_expert_intermediate_size', 1408),
        enable_quantization=config.get('enable_quantization', True),
        quantization_bits=config.get('quantization_bits', 8),
        enable_gradient_checkpointing=config.get('enable_gradient_checkpointing', True),
        enable_compilation=config.get('enable_compilation', True)
    )
    
    return QwenForCausalLM(qwen_config)
