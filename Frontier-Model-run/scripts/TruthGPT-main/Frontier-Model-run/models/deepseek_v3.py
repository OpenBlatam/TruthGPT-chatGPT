import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import torch
import torch.nn.functional as F

def act_quant(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Fallback implementation for activation quantization."""
    if x.dtype == torch.float8_e4m3fn:
        return x
    scale = x.abs().max() / 127.0
    quantized = torch.round(x / scale).clamp(-128, 127)
    return quantized * scale

def weight_dequant(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Fallback implementation for weight dequantization."""
    if weight.dtype == torch.float8_e4m3fn:
        return weight.to(torch.float16) * weight_scale
    return weight

def fp8_gemm(x: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Fallback implementation for FP8 GEMM operation."""
    x_fp16 = x.to(torch.float16) if x.dtype == torch.float8_e4m3fn else x
    weight_fp16 = weight.to(torch.float16) if weight.dtype == torch.float8_e4m3fn else weight
    
    if scale is not None:
        x_fp16 = x_fp16 * scale
    if weight_scale is not None:
        weight_fp16 = weight_fp16 * weight_scale
    
    return F.linear(x_fp16, weight_fp16)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 30
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    moe_intermediate_size: int = 1407
    shared_intermediate_size: int = 1024
    
    use_fp8: bool = False
    
    original_seq_len: int = 4096
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        
        if weight is not None and weight.element_size() == 1:
            return fp8_gemm(x, None, weight, None)
        else:
            return F.linear(x, weight, bias)

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, gather_output: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            return output
        return output

class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, input_is_parallel: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        
        self.weight = Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = Parameter(torch.empty((vocab_size, dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_yarn: bool = False, 
                        original_seq_len: int = 4096, rope_factor: float = 40, 
                        beta_fast: int = 32, beta_slow: int = 1, mscale: float = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    if use_yarn and end > original_seq_len:
        scale_factor = rope_factor
        low_freq_factor = 1.0
        high_freq_factor = 1.0
        
        freqs = freqs / scale_factor
        
        wavelength = 2 * math.pi / freqs
        freqs = torch.where(
            wavelength > beta_slow,
            freqs * low_freq_factor,
            torch.where(
                wavelength < beta_fast,
                freqs * high_freq_factor,
                freqs
            )
        )
    
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim
        
        self.q_a_proj = Linear(args.dim, args.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank)
        self.q_b_proj = Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        
        self.kv_a_proj_with_mqa = Linear(args.dim, args.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank)
        self.kv_b_proj = Linear(args.kv_lora_rank, self.n_kv_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        
        self.o_proj = Linear(self.n_heads * self.v_head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        
        q = self.q_a_proj(x)
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q).view(bsz, seqlen, self.n_heads, self.head_dim)
        
        kv = self.kv_a_proj_with_mqa(x)
        kv_a, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)
        
        k_nope, v = kv.split([self.n_kv_heads * self.qk_nope_head_dim, self.n_kv_heads * self.v_head_dim], dim=-1)
        k_nope = k_nope.view(bsz, seqlen, self.n_kv_heads, self.qk_nope_head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.v_head_dim)
        
        q_rope = q[..., :self.qk_rope_head_dim]
        q_nope = q[..., self.qk_rope_head_dim:]
        k_rope = k_rope.view(bsz, seqlen, 1, self.qk_rope_head_dim).expand(-1, -1, self.n_kv_heads, -1)
        
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis)
        
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)
        
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        
        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)

class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.n_activated_experts
        self.n_routed_experts = args.n_routed_experts
        self.gate = Linear(args.dim, args.n_routed_experts, bias=False)

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, selected_experts = torch.topk(gate_logits, self.top_k)
        weights = F.softmax(weights, dim=-1)
        return weights, selected_experts

class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_routed_experts = args.n_routed_experts
        self.n_shared_experts = args.n_shared_experts
        self.n_activated_experts = args.n_activated_experts
        
        if self.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                nn.Sequential(
                    Linear(args.dim, args.shared_intermediate_size, bias=False),
                    nn.SiLU(),
                    Linear(args.shared_intermediate_size, args.dim, bias=False)
                ) for _ in range(self.n_shared_experts)
            ])
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                Linear(args.dim, args.moe_intermediate_size, bias=False),
                nn.SiLU(),
                Linear(args.moe_intermediate_size, args.dim, bias=False)
            ) for _ in range(self.n_routed_experts)
        ])
        
        self.gate = MoEGate(args)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        shared_output = torch.zeros_like(x_flat)
        if self.n_shared_experts > 0:
            for expert in self.shared_experts:
                shared_output += expert(x_flat)
            shared_output /= self.n_shared_experts
        
        weights, selected_experts = self.gate(x_flat)
        routed_output = torch.zeros_like(x_flat)
        
        for i in range(self.n_activated_experts):
            expert_weights = weights[:, i:i+1]
            expert_indices = selected_experts[:, i]
            
            for expert_idx in range(self.n_routed_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    routed_output[mask] += expert_weights[mask] * expert_output
        
        total_output = shared_output + routed_output
        return total_output.view(batch_size, seq_len, hidden_dim)

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = MLA(args)
        self.feed_forward = MoE(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.qk_rope_head_dim,
            params.max_seq_len * 2,
            params.rope_theta,
            use_yarn=True,
            original_seq_len=params.original_seq_len,
            rope_factor=params.rope_factor,
            beta_fast=params.beta_fast,
            beta_slow=params.beta_slow,
            mscale=params.mscale
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

def create_deepseek_v3_model(config: Dict[str, Any]) -> Transformer:
    """Create a DeepSeek-V3 model from configuration with optimization_core integration."""
    model_args = ModelArgs(
        dim=config.get('hidden_size', 4096),
        n_layers=config.get('num_hidden_layers', 30),
        n_heads=config.get('num_attention_heads', 32),
        n_kv_heads=config.get('num_key_value_heads'),
        vocab_size=config.get('vocab_size', 102400),
        norm_eps=config.get('layer_norm_eps', 1e-5),
        rope_theta=config.get('rope_theta', 10000.0),
        max_seq_len=config.get('max_position_embeddings', 2048),
        
        q_lora_rank=config.get('q_lora_rank', 1536),
        kv_lora_rank=config.get('kv_lora_rank', 512),
        qk_rope_head_dim=config.get('qk_rope_head_dim', 64),
        v_head_dim=config.get('v_head_dim', 128),
        qk_nope_head_dim=config.get('qk_nope_head_dim', 128),
        
        n_routed_experts=config.get('n_routed_experts', 64),
        n_shared_experts=config.get('n_shared_experts', 2),
        n_activated_experts=config.get('n_activated_experts', 6),
        moe_intermediate_size=config.get('moe_intermediate_size', 1407),
        shared_intermediate_size=config.get('shared_intermediate_size', 1024),
        
        use_fp8=config.get('use_fp8', False),
        
        original_seq_len=config.get('original_seq_len', 4096),
        rope_factor=config.get('rope_factor', 40),
        beta_fast=config.get('beta_fast', 32),
        beta_slow=config.get('beta_slow', 1),
        mscale=config.get('mscale', 1.0)
    )
    
    return Transformer(model_args)
