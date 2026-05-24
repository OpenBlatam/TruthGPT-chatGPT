"""
DeepSeek-R1-Qwen3 Frontier Model Implementation

A native implementation combining DeepSeek-R1's advanced reasoning capabilities
with Qwen3's efficient architecture, enhanced with frontier reasoning features.

Key Features:
- Qwen3 architecture with 8B parameters
- DeepSeek-R1 reasoning enhancements
- Chain-of-thought optimization
- Multi-step reasoning capabilities
- Advanced thinking mechanisms
- YARN-scaled RoPE for long sequences
"""

import math
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class DeepSeekR1Qwen3Config(PretrainedConfig):
    """Configuration class for DeepSeek-R1-Qwen3 model."""
    
    model_type = "deepseek_r1_qwen3"
    
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        max_window_layers: int = 36,
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = None,
        
        # DeepSeek-R1 reasoning enhancements
        reasoning_depth: int = 5,
        thinking_tokens: int = 23000,
        chain_of_thought_layers: List[int] = None,
        reasoning_temperature: float = 0.6,
        reasoning_top_p: float = 0.95,
        use_thinking_head: bool = True,
        thinking_head_size: int = 1024,
        
        # Advanced reasoning features
        use_step_by_step: bool = True,
        use_verification: bool = True,
        use_reflection: bool = True,
        max_reasoning_steps: int = 10,
        reasoning_confidence_threshold: float = 0.8,
        
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_window_layers = max_window_layers
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        
        # DeepSeek-R1 reasoning parameters
        self.reasoning_depth = reasoning_depth
        self.thinking_tokens = thinking_tokens
        self.chain_of_thought_layers = chain_of_thought_layers or [12, 18, 24, 30]
        self.reasoning_temperature = reasoning_temperature
        self.reasoning_top_p = reasoning_top_p
        self.use_thinking_head = use_thinking_head
        self.thinking_head_size = thinking_head_size
        
        # Advanced reasoning features
        self.use_step_by_step = use_step_by_step
        self.use_verification = use_verification
        self.use_reflection = use_reflection
        self.max_reasoning_steps = max_reasoning_steps
        self.reasoning_confidence_threshold = reasoning_confidence_threshold
        
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YARN scaling."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000, device=None, scaling_factor: float = 1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # YARN scaling
        if scaling_factor != 1.0:
            self._yarn_scaling()

    def _yarn_scaling(self):
        """Apply YARN scaling to the frequency."""
        scale = self.scaling_factor
        dim = self.dim
        
        # YARN scaling formula
        alpha = 1.0
        beta = 32.0
        
        # Apply scaling
        self.inv_freq = self.inv_freq / scale

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
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


class Qwen3Attention(nn.Module):
    """Multi-head attention with reasoning enhancements."""
    
    def __init__(self, config: DeepSeekR1Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
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
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Reasoning enhancement: thinking attention for CoT layers
        self.is_thinking_layer = layer_idx in config.chain_of_thought_layers if layer_idx is not None else False
        if self.is_thinking_layer:
            self.thinking_proj = nn.Linear(self.hidden_size, config.thinking_head_size)
            self.thinking_gate = nn.Linear(self.hidden_size, 1)
        
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["rope_type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "yarn":
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Reasoning enhancement: apply thinking mechanism for CoT layers
        if self.is_thinking_layer:
            thinking_features = self.thinking_proj(hidden_states)
            thinking_gate = torch.sigmoid(self.thinking_gate(hidden_states))
            attn_output = attn_output + thinking_gate * thinking_features

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3MLP(nn.Module):
    """MLP with reasoning enhancements."""
    
    def __init__(self, config: DeepSeekR1Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ReasoningModule(nn.Module):
    """Advanced reasoning module for step-by-step thinking."""
    
    def __init__(self, config: DeepSeekR1Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Step-by-step reasoning components
        self.step_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.step_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.step_gate = nn.Linear(config.hidden_size, 1)
        
        # Verification mechanism
        self.verification_head = nn.Linear(config.hidden_size, 2)  # correct/incorrect
        
        # Reflection mechanism
        self.reflection_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.reflection_gate = nn.Linear(config.hidden_size, 1)
        
        # Confidence estimation
        self.confidence_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        step_idx: int = 0,
        previous_steps: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Step-by-step reasoning
        step_features = self.step_encoder(hidden_states)
        step_gate = torch.sigmoid(self.step_gate(hidden_states))
        step_output = step_gate * step_features + (1 - step_gate) * hidden_states
        
        # Verification
        verification_logits = self.verification_head(step_output)
        verification_probs = F.softmax(verification_logits, dim=-1)
        
        # Reflection on previous steps
        reflection_output = step_output
        if previous_steps and self.config.use_reflection:
            # Aggregate previous steps
            prev_context = torch.stack(previous_steps, dim=1).mean(dim=1)  # [batch, hidden]
            prev_context = prev_context.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden]
            
            reflection_features = self.reflection_proj(prev_context)
            reflection_gate = torch.sigmoid(self.reflection_gate(hidden_states))
            reflection_output = step_output + reflection_gate * reflection_features
        
        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_head(reflection_output))
        
        return {
            "reasoning_output": reflection_output,
            "verification_probs": verification_probs,
            "confidence": confidence,
            "step_features": step_features,
        }


class Qwen3DecoderLayer(nn.Module):
    """Transformer decoder layer with reasoning enhancements."""
    
    def __init__(self, config: DeepSeekR1Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Add reasoning module for specific layers
        self.use_reasoning = layer_idx in config.chain_of_thought_layers
        if self.use_reasoning:
            self.reasoning_module = ReasoningModule(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        reasoning_state: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Apply reasoning module if this is a reasoning layer
        reasoning_outputs = None
        if self.use_reasoning and reasoning_state is not None:
            reasoning_outputs = self.reasoning_module(
                hidden_states,
                step_idx=reasoning_state.get("step_idx", 0),
                previous_steps=reasoning_state.get("previous_steps", []),
            )
            hidden_states = reasoning_outputs["reasoning_output"]

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        if reasoning_outputs:
            outputs += (reasoning_outputs,)

        return outputs


class Qwen3PreTrainedModel(PreTrainedModel):
    """Base class for Qwen3 models."""
    config_class = DeepSeekR1Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen3Model(Qwen3PreTrainedModel):
    """The bare Qwen3 Model outputting raw hidden-states."""
    
    def __init__(self, config: DeepSeekR1Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()

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
        reasoning_mode: bool = False,
        **kwargs,
    ) -> Union[Tuple, Dict]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # Initialize reasoning state for reasoning mode
        reasoning_state = None
        if reasoning_mode:
            reasoning_state = {
                "step_idx": 0,
                "previous_steps": [],
                "reasoning_outputs": [],
            }

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    use_cache,
                    reasoning_state,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    reasoning_state=reasoning_state,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
            # Update reasoning state
            if reasoning_mode and len(layer_outputs) > 3:
                reasoning_outputs = layer_outputs[-1]
                reasoning_state["previous_steps"].append(reasoning_outputs["step_features"])
                reasoning_state["reasoning_outputs"].append(reasoning_outputs)
                reasoning_state["step_idx"] += 1

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        outputs = {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }
        
        if reasoning_mode and reasoning_state:
            outputs["reasoning_outputs"] = reasoning_state["reasoning_outputs"]
            
        return outputs


class DeepSeekR1Qwen3ForCausalLM(Qwen3PreTrainedModel):
    """DeepSeek-R1-Qwen3 Model with a language modeling head and reasoning capabilities."""
    
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepSeekR1Qwen3Config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Thinking head for reasoning mode
        if config.use_thinking_head:
            self.thinking_head = nn.Linear(config.hidden_size, config.thinking_head_size)
            self.thinking_to_vocab = nn.Linear(config.thinking_head_size, config.vocab_size)
        
        self.post_init()

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
        reasoning_mode: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            reasoning_mode=reasoning_mode,
        )

        hidden_states = outputs["last_hidden_state"] if return_dict else outputs[0]
        
        # Use thinking head in reasoning mode
        if reasoning_mode and self.config.use_thinking_head:
            thinking_features = self.thinking_head(hidden_states)
            logits = self.thinking_to_vocab(thinking_features)
        else:
            logits = self.lm_head(hidden_states)
        
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get("past_key_values"),
            hidden_states=outputs.get("hidden_states"),
            attentions=outputs.get("attentions"),
        )
        
        # Add reasoning outputs if available
        if "reasoning_outputs" in outputs:
            result.reasoning_outputs = outputs["reasoning_outputs"]
            
        return result

    def generate_with_reasoning(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        max_reasoning_steps: int = None,
        temperature: float = None,
        top_p: float = None,
        confidence_threshold: float = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Generate text with step-by-step reasoning."""
        max_reasoning_steps = max_reasoning_steps or self.config.max_reasoning_steps
        temperature = temperature or self.config.reasoning_temperature
        top_p = top_p or self.config.reasoning_top_p
        confidence_threshold = confidence_threshold or self.config.reasoning_confidence_threshold
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize reasoning state
        reasoning_steps = []
        current_ids = input_ids
        
        for step in range(max_reasoning_steps):
            # Forward pass with reasoning mode
            outputs = self.forward(
                input_ids=current_ids,
                reasoning_mode=True,
                use_cache=False,
                return_dict=True,
            )
            
            # Get reasoning outputs
            if hasattr(outputs, 'reasoning_outputs') and outputs.reasoning_outputs:
                latest_reasoning = outputs.reasoning_outputs[-1]
                confidence = latest_reasoning["confidence"].mean().item()
                
                # Check if confidence is high enough
                if confidence >= confidence_threshold:
                    break
                    
                reasoning_steps.append({
                    "step": step,
                    "confidence": confidence,
                    "verification_probs": latest_reasoning["verification_probs"],
                })
            
            # Generate next tokens
            logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # Stop if we reach max length
            if current_ids.shape[-1] >= max_length:
                break
        
        # Final generation without reasoning mode
        final_outputs = self.generate(
            current_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            **kwargs,
        )
        
        return {
            "generated_ids": final_outputs,
            "reasoning_steps": reasoning_steps,
            "num_reasoning_steps": len(reasoning_steps),
        }

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

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


# Helper functions for attention mask preparation
def _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """Prepare 4D causal attention mask."""
    batch_size, seq_length = input_shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device

    # Create causal mask
    causal_mask = torch.full((seq_length, seq_length), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask_cond = torch.arange(causal_mask.size(-1), device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
    causal_mask = causal_mask.to(dtype)

    if past_key_values_length > 0:
        causal_mask = torch.cat([torch.zeros(seq_length, past_key_values_length, dtype=dtype, device=device), causal_mask], dim=-1)

    expanded_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length + past_key_values_length)
    if attention_mask is not None:
        expanded_mask = expanded_mask + attention_mask[:, None, None, :].to(dtype)

    return expanded_mask


def _prepare_4d_causal_attention_mask_for_sdpa(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """Prepare 4D causal attention mask for SDPA."""
    batch_size, seq_length = input_shape

    if attention_mask is not None and len(attention_mask.shape) == 2:
        expanded_mask = attention_mask[:, None, None, :]
        return expanded_mask

    return None


# Cache implementation for compatibility
class Cache:
    """Base class for cache implementations."""
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        raise NotImplementedError("Subclasses must implement get_seq_length")
    
    def get_max_length(self) -> Optional[int]:
        raise NotImplementedError("Subclasses must implement get_max_length")
    
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)


class DynamicCache(Cache):
    """Dynamic cache implementation."""
    
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.seen_tokens = 0

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self.key_cache)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self.key_cache)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], ...]:
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                cache.update(key_states, value_states, layer_idx)
        return cache