import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

@dataclass
class LlamaArgs:
    """Configuration for Llama-3.1-405B model."""
    dim: int = 16384  # 405B model dimension
    n_layers: int = 126  # 405B layers
    n_heads: int = 128  # 405B attention heads
    n_kv_heads: int = 8  # GQA with 8 KV heads
    vocab_size: int = 128256  # Llama 3.1 vocab size
    multiple_of: int = 1024
    ffn_dim_multiplier: float = 1.3
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0  # Extended for long context
    max_seq_len: int = 131072  # 128K context length
    use_scaled_rope: bool = True
    rope_scaling_factor: float = 8.0
    
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_quantization: bool = False
    quantization_bits: int = 8

class LlamaRMSNorm(nn.Module):
    """RMS Normalization for Llama."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LlamaLinear(nn.Module):
    """Optimized linear layer for Llama with optional quantization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, 
                 use_quantization: bool = False, quantization_bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        
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

    def forward(self, x):
        if self.use_quantization and self.training:
            weight = self.quantize_weight(self.weight)
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)
    
    def quantize_weight(self, weight):
        """Simple quantization for demonstration."""
        if self.quantization_bits == 8:
            scale = weight.abs().max() / 127.0
            quantized = torch.round(weight / scale).clamp(-128, 127)
            return quantized * scale
        return weight

def precompute_freqs_cis_llama(dim: int, end: int, theta: float = 500000.0, 
                               scaling_factor: float = 8.0, use_scaled: bool = True):
    """Precompute the frequency tensor for complex exponentials (cis) with RoPE scaling."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    if use_scaled:
        freqs = freqs / scaling_factor
    
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb_llama(xq, xk, freqs_cis):
    """Apply rotary embeddings to queries and keys."""
    print(f"DEBUG: xq.shape = {xq.shape}, xk.shape = {xk.shape}, freqs_cis.shape = {freqs_cis.shape}")
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    print(f"DEBUG: xq_.shape = {xq_.shape}, xk_.shape = {xk_.shape}")
    
    # Ensure freqs_cis matches the sequence length dimension
    seq_len = xq_.shape[1]
    if freqs_cis.shape[0] != seq_len:
        freqs_cis = freqs_cis[:seq_len]
    
    if len(freqs_cis.shape) == 2:
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # Add batch and head dims
    
    print(f"DEBUG: After adjustment, freqs_cis.shape = {freqs_cis.shape}")
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class LlamaAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) for Llama-3.1-405B."""
    
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.use_flash_attention = args.use_flash_attention

        self.wq = LlamaLinear(args.dim, args.n_heads * self.head_dim, bias=False,
                             use_quantization=args.use_quantization, 
                             quantization_bits=args.quantization_bits)
        self.wk = LlamaLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False,
                             use_quantization=args.use_quantization,
                             quantization_bits=args.quantization_bits)
        self.wv = LlamaLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False,
                             use_quantization=args.use_quantization,
                             quantization_bits=args.quantization_bits)
        self.wo = LlamaLinear(args.n_heads * self.head_dim, args.dim, bias=False,
                             use_quantization=args.use_quantization,
                             quantization_bits=args.quantization_bits)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if freqs_cis.shape[0] < seqlen:
            freqs_cis = freqs_cis[:seqlen]
        elif freqs_cis.shape[0] > seqlen:
            freqs_cis = freqs_cis[:seqlen]
        
        xq, xk = apply_rotary_emb_llama(xq, xk, freqs_cis=freqs_cis)

        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.use_flash_attention:
            try:
                import flash_attn
                output = flash_attn.flash_attn_func(xq, xk, xv, causal=True)
            except ImportError:
                output = self._standard_attention(xq, xk, xv, mask)
        else:
            output = self._standard_attention(xq, xk, xv, mask)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def _standard_attention(self, xq, xk, xv, mask):
        """Standard scaled dot-product attention."""
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        return output

class LlamaFeedForward(nn.Module):
    """SwiGLU Feed Forward Network for Llama."""
    
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, 
                 ffn_dim_multiplier: Optional[float], use_quantization: bool = False,
                 quantization_bits: int = 8):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = LlamaLinear(dim, hidden_dim, bias=False, 
                             use_quantization=use_quantization,
                             quantization_bits=quantization_bits)
        self.w2 = LlamaLinear(hidden_dim, dim, bias=False,
                             use_quantization=use_quantization,
                             quantization_bits=quantization_bits)
        self.w3 = LlamaLinear(dim, hidden_dim, bias=False,
                             use_quantization=use_quantization,
                             quantization_bits=quantization_bits)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LlamaTransformerBlock(nn.Module):
    """Transformer block for Llama with optional gradient checkpointing."""
    
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = LlamaAttention(args)
        self.feed_forward = LlamaFeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            use_quantization=args.use_quantization,
            quantization_bits=args.quantization_bits,
        )
        self.layer_id = layer_id
        self.attention_norm = LlamaRMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = LlamaRMSNorm(args.dim, eps=args.norm_eps)
        self.use_gradient_checkpointing = args.use_gradient_checkpointing

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, start_pos, freqs_cis, mask, use_reentrant=False
            )
        else:
            return self._forward_impl(x, start_pos, freqs_cis, mask)
    
    def _forward_impl(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class LlamaTransformer(nn.Module):
    """Llama-3.1-405B Transformer model with all optimizations."""
    
    def __init__(self, params: LlamaArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(LlamaTransformerBlock(layer_id, params))
        self.norm = LlamaRMSNorm(params.dim, eps=params.norm_eps)
        self.output = LlamaLinear(params.dim, params.vocab_size, bias=False,
                                 use_quantization=params.use_quantization,
                                 quantization_bits=params.quantization_bits)

        self.freqs_cis = precompute_freqs_cis_llama(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.rope_scaling_factor,
            params.use_scaled_rope
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

def create_llama_3_1_405b_model(config: Optional[Dict[str, Any]] = None) -> LlamaTransformer:
    """Factory function to create Llama-3.1-405B model with optimizations."""
    
    default_config = {
        'dim': 16384,
        'n_layers': 126,
        'n_heads': 128,
        'n_kv_heads': 8,
        'vocab_size': 128256,
        'multiple_of': 1024,
        'ffn_dim_multiplier': 1.3,
        'norm_eps': 1e-5,
        'rope_theta': 500000.0,
        'max_seq_len': 131072,
        'use_scaled_rope': True,
        'rope_scaling_factor': 8.0,
        'use_flash_attention': True,
        'use_gradient_checkpointing': True,
        'use_quantization': False,
        'quantization_bits': 8,
    }
    
    if config:
        default_config.update(config)
    
    args = LlamaArgs(**default_config)
    model = LlamaTransformer(args)
    
    try:
        from enhanced_model_optimizer import create_universal_optimizer
        optimizer = create_universal_optimizer({
            'enable_fp16': True,
            'enable_gradient_checkpointing': True,
            'use_advanced_normalization': True,
            'use_enhanced_mlp': True,
            'use_mcts_optimization': True
        })
        model = optimizer.optimize_model(model, "Llama-3.1-405B")
        print(f"✅ Applied optimization_core to Llama-3.1-405B model")
    except ImportError:
        pass
    
    print(f"✅ Created Llama-3.1-405B model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🔧 Optimizations: Flash Attention={args.use_flash_attention}, "
          f"Gradient Checkpointing={args.use_gradient_checkpointing}, "
          f"Quantization={args.use_quantization}")
    
    return model
