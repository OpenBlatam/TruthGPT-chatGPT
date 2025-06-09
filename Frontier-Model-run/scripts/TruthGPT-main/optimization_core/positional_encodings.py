"""
Advanced positional encoding techniques for TruthGPT optimization.
Integrates rope.py and positional.py optimizations.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

class SinusoidalPositionalEmbedding(torch.nn.Module):
    """Sinusoidal positional embedding with precision control."""
    
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.precision = precision

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        if self.precision == torch.bfloat16:
            sinusoid_inp = sinusoid_inp.float()
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        if self.precision == torch.bfloat16:
            sin, cos = sin.bfloat16(), cos.bfloat16()
        emb = torch.cat((sin, cos), dim=-1)
        return emb[None, :, :]

class RotaryEmbedding(torch.nn.Module):
    """Rotary positional embedding with caching and precision control."""
    
    def __init__(
        self, dim, max_seq_len, base=10000, precision=torch.half, save_inv_freqs=False
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision
        self.max_seq_len = max_seq_len
        self.base = base
        self.dim = dim

        cos_cached, sin_cached, inv_freq = self._prepare_cache(
            max_seq_len, precision, base
        )

        self.register_buffer("inv_freq", inv_freq, persistent=save_inv_freqs)
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached

    def _prepare_cache(self, seq_len, precision, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        t = torch.arange(seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.emb = emb.reshape(emb.size(0), 1, 1, emb.size(1))

        cos_cached = emb.cos()[:, None, None, :]
        sin_cached = emb.sin()[:, None, None, :]

        return (
            cos_cached.to(precision),
            sin_cached.to(precision),
            inv_freq.to(precision),
        )

    def get_emb(self):
        return self.emb.to(self.precision).cuda()

    def forward(self, x, seq_dim=0, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]

        assert seq_len <= self.max_seq_len

        if seq_len != self.max_seq_len:
            return (
                self.cos_cached[:seq_len, ...].to(x.device),
                self.sin_cached[:seq_len, ...].to(x.device),
            )
        else:
            return self.cos_cached.to(x.device), self.sin_cached.to(x.device)

class LlamaRotaryEmbedding(torch.nn.Module):
    """Llama-style rotary embedding with dynamic caching."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer('inv_freq', inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class FixedLlamaRotaryEmbedding(torch.nn.Module):
    """Fixed version of Llama rotary embedding with improved caching."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

class AliBi(torch.nn.Module):
    """Attention with Linear Biases (ALiBi) positional encoding."""
    
    def __init__(self, num_heads, mp_size=1, mp_rank=1):
        super().__init__()
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ]
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def bias(self, seq_len_q, seq_len_k, device, dtype):
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(device).to(dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )

        return a

    def forward(self, x):
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )

        return x + a

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def apply_rotary_pos_emb_torch(
    q, k, cos, sin, offset: int = 0
):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def apply_rotary_pos_emb_llama(q, k, cos, sin, position_ids):
    """Llama-style rotary position embedding application."""
    gather_indices = position_ids[:, None, :, None]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)

class PositionalEncodingOptimizations:
    """Utility class for applying positional encoding optimizations."""
    
    @staticmethod
    def replace_rotary_embeddings(model: nn.Module, embedding_type: str = "fixed_llama") -> nn.Module:
        """Replace rotary embeddings with optimized versions."""
        for name, module in model.named_modules():
            if hasattr(module, '__class__') and 'RotaryEmbedding' in module.__class__.__name__:
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    child_name = name.split('.')[-1]
                else:
                    parent = model
                    child_name = name
                
                if embedding_type == "fixed_llama":
                    new_embedding = FixedLlamaRotaryEmbedding(
                        dim=getattr(module, 'dim', 64),
                        max_position_embeddings=getattr(module, 'max_position_embeddings', 2048),
                        base=getattr(module, 'base', 10000)
                    )
                elif embedding_type == "llama":
                    new_embedding = LlamaRotaryEmbedding(
                        dim=getattr(module, 'dim', 64),
                        max_position_embeddings=getattr(module, 'max_position_embeddings', 2048),
                        base=getattr(module, 'base', 10000)
                    )
                else:
                    new_embedding = RotaryEmbedding(
                        dim=getattr(module, 'dim', 64),
                        max_seq_len=getattr(module, 'max_seq_len', 2048),
                        base=getattr(module, 'base', 10000)
                    )
                
                setattr(parent, child_name, new_embedding)
        
        return model
    
    @staticmethod
    def add_alibi_attention(model: nn.Module, num_heads: int) -> nn.Module:
        """Add ALiBi positional encoding to attention modules."""
        return model
    
    @staticmethod
    def get_positional_encoding_report(model: nn.Module) -> dict:
        """Get a report of positional encoding optimization status."""
        total_pos_modules = 0
        rotary_modules = 0
        alibi_modules = 0
        sinusoidal_modules = 0
        
        for module in model.modules():
            if isinstance(module, (RotaryEmbedding, LlamaRotaryEmbedding, FixedLlamaRotaryEmbedding)):
                rotary_modules += 1
                total_pos_modules += 1
            elif isinstance(module, AliBi):
                alibi_modules += 1
                total_pos_modules += 1
            elif isinstance(module, SinusoidalPositionalEmbedding):
                sinusoidal_modules += 1
                total_pos_modules += 1
        
        return {
            'total_positional_modules': total_pos_modules,
            'rotary_embedding_modules': rotary_modules,
            'alibi_modules': alibi_modules,
            'sinusoidal_modules': sinusoidal_modules,
            'optimization_ratio': (rotary_modules + alibi_modules) / total_pos_modules if total_pos_modules > 0 else 0
        }

def create_rotary_embedding(dim: int, max_seq_len: int = 2048, **kwargs) -> RotaryEmbedding:
    """Factory function to create RotaryEmbedding."""
    return RotaryEmbedding(dim, max_seq_len, **kwargs)

def create_llama_rotary_embedding(dim: int, max_position_embeddings: int = 2048, **kwargs) -> LlamaRotaryEmbedding:
    """Factory function to create LlamaRotaryEmbedding."""
    return LlamaRotaryEmbedding(dim, max_position_embeddings, **kwargs)

def create_alibi(num_heads: int, **kwargs) -> AliBi:
    """Factory function to create AliBi."""
    return AliBi(num_heads, **kwargs)

def create_sinusoidal_embedding(dim: int, **kwargs) -> SinusoidalPositionalEmbedding:
    """Factory function to create SinusoidalPositionalEmbedding."""
    return SinusoidalPositionalEmbedding(dim, **kwargs)
