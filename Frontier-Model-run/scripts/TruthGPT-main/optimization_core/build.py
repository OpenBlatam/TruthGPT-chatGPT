from typing import Any, Dict
import torch

from optimization_core.factories.attention import ATTENTION_BACKENDS
from optimization_core.factories.kv_cache import KV_CACHE
from optimization_core.factories.memory import MEMORY_MANAGERS
from optimization_core.factories.datasets import DATASETS
from optimization_core.factories.collate import COLLATE


def build_components(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {})
    attn_cfg = model_cfg.get("attention", {})
    kv_cfg = model_cfg.get("kv_cache", {})
    mem_cfg = model_cfg.get("memory", {})

    attn_name = attn_cfg.get("backend", "sdpa")
    attn_fn = ATTENTION_BACKENDS.build(attn_name)

    kv_type = kv_cfg.get("type", "none")
    kv_block = int(kv_cfg.get("block_size", 128))
    # kv cache builder returns a call to create with params later
    kv_builder = lambda h, d, max_tokens, dtype=None: KV_CACHE.build(
        kv_type, num_heads=h, head_dim=d, max_tokens=max_tokens, block_size=kv_block, dtype=dtype
    )

    mem_policy = mem_cfg.get("policy", "adaptive")
    memory_manager = MEMORY_MANAGERS.build(mem_policy)

    return {
        "attention": attn_fn,
        "kv_cache_builder": kv_builder,
        "memory_manager": memory_manager,
        "datasets": DATASETS,
        "collate": COLLATE,
    }



