import time
from typing import Dict

import torch
import torch.nn.functional as F


def _time_op(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - start) / iters


def benchmark_attention(h: int, t: int, d: int, dtype: torch.dtype) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(1, h, t, d, device=device, dtype=dtype)
    k = torch.randn(1, h, t, d, device=device, dtype=dtype)
    v = torch.randn(1, h, t, d, device=device, dtype=dtype)

    def run_sdpa():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def run_naive():
        attn = torch.matmul(q, k.transpose(-1, -2)) * (d ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    results = {}
    if hasattr(F, "scaled_dot_product_attention"):
        results["sdpa"] = _time_op(run_sdpa)
    results["naive"] = _time_op(run_naive)
    return results


def choose_best_backend(h: int, t: int, d: int, dtype: torch.dtype) -> str:
    times = benchmark_attention(h, t, d, dtype)
    best = min(times, key=times.get)
    return best  # "sdpa" or "naive"






