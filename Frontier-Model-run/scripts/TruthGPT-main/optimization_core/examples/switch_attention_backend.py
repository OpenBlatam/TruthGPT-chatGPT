import yaml
import torch
from optimization_core.build import build_components
from modules.memory.advanced_memory_manager import create_advanced_memory_manager


def main():
    cfg = yaml.safe_load(open("optimization_core/configs/llm_default.yaml", "r", encoding="utf-8"))
    comps = build_components(cfg)
    attn = comps["attention"]
    mm = comps["memory_manager"]

    h, d, ctx = 16, 64, 4096
    dtype = mm.select_dtype_adaptive()
    suggested = mm.suggest_kv_block_size(h, d, ctx, memory_fraction=0.1)
    kv = comps["kv_cache_builder"](h, d, ctx, dtype)

    print("Attention backend:", attn.__name__)
    print("KV block size (suggested):", suggested)
    print("KV length initial:", kv.length if kv is not None else None)


if __name__ == "__main__":
    main()






