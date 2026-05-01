from typing import Optional
import torch
import torch.nn.functional as F

from optimization_core.factories.registry import Registry

ATTENTION_BACKENDS = Registry()


def sdpa_attention(q, k, v, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = True):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


@ATTENTION_BACKENDS.register("sdpa")
def build_sdpa():
    return sdpa_attention


@ATTENTION_BACKENDS.register("flash")
def build_flash():
    # Fallback to SDPA; real flash-attn integration can be injected here
    return sdpa_attention


@ATTENTION_BACKENDS.register("triton")
def build_triton():
    # Placeholder that falls back to SDPA
    return sdpa_attention






