from typing import Optional
import torch
from optimization_core.factories.registry import Registry
from optimization_core.modules.attention.ultra_efficient_kv_cache import PagedKVCache

KV_CACHE = Registry()


@KV_CACHE.register("none")
def build_none(*args, **kwargs):
    return None


@KV_CACHE.register("paged")
def build_paged(num_heads: int, head_dim: int, max_tokens: int, block_size: int = 128, dtype: Optional[torch.dtype] = None):
    dtype = dtype or torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return PagedKVCache(num_heads=num_heads, head_dim=head_dim, max_tokens=max_tokens, block_size=block_size, dtype=dtype)






