import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_model(model_name: str, use_tf32: bool, use_compile: bool, dtype: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and use_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(dtype, torch.float32)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass
    model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def measure_tps(model, tokenizer, device, prompt: str, max_new_tokens: int = 128, warmup: int = 1, iters: int = 3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # warmup
    for _ in range(warmup):
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start
    gen_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
    tps = gen_tokens / elapsed if elapsed > 0 else 0.0
    return tps, elapsed, gen_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="The theory of transformers is")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    configs = [
        {"tf32": False, "compile": False},
        {"tf32": True, "compile": False},
        {"tf32": True, "compile": True},
    ]

    results = []
    for cfg in configs:
        model, tok, device = setup_model(args.model, cfg["tf32"], cfg["compile"], args.dtype)
        tps, sec, toks = measure_tps(model, tok, device, args.prompt, args.max_new_tokens)
        results.append((cfg, tps, sec, toks))
        print(f"tf32={cfg['tf32']} compile={cfg['compile']} -> tokens/s={tps:.1f} time={sec:.2f}s tokens={toks}")

    best = max(results, key=lambda x: x[1])
    cfg = best[0]
    print(f"Best: tf32={cfg['tf32']} compile={cfg['compile']} with {best[1]:.1f} tokens/s")


if __name__ == "__main__":
    main()






