#!/usr/bin/env python3
"""
Validate training configuration YAML file.
Checks for required fields, valid values, and consistency.
"""

import argparse
import sys
import yaml
from pathlib import Path


def validate_config(config_path: str) -> tuple[bool, list[str]]:
    """Validate configuration file. Returns (is_valid, errors)."""
    errors = []
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        return False, [f"Config file not found: {config_path}"]
    except yaml.YAMLError as e:
        return False, [f"YAML parse error: {e}"]
    
    # Required top-level sections
    required = ["model", "training", "data"]
    for section in required:
        if section not in cfg:
            errors.append(f"Missing required section: {section}")
    
    # Validate model section
    if "model" in cfg:
        model = cfg["model"]
        if "name_or_path" not in model:
            errors.append("model.name_or_path is required")
        if "attention" in model:
            attn = model["attention"]
            if "backend" in attn and attn["backend"] not in ["sdpa", "flash", "triton"]:
                errors.append(f"Invalid attention.backend: {attn['backend']} (must be sdpa|flash|triton)")
        if "kv_cache" in model:
            kv = model["kv_cache"]
            if "type" in kv and kv["type"] not in ["none", "paged"]:
                errors.append(f"Invalid kv_cache.type: {kv['type']} (must be none|paged)")
    
    # Validate training section
    if "training" in cfg:
        train = cfg["training"]
        if "mixed_precision" in train and train["mixed_precision"] not in ["none", "fp16", "bf16"]:
            errors.append(f"Invalid training.mixed_precision: {train['mixed_precision']} (must be none|fp16|bf16)")
        if "scheduler" in train and train["scheduler"] not in ["cosine", "linear", "constant"]:
            errors.append(f"Invalid training.scheduler: {train['scheduler']} (must be cosine|linear|constant)")
        if "callbacks" in train:
            valid_callbacks = ["print", "wandb", "tensorboard"]
            for cb in train["callbacks"]:
                if cb not in valid_callbacks:
                    errors.append(f"Unknown callback: {cb} (valid: {valid_callbacks})")
    
    # Validate optimizer section
    if "optimizer" in cfg:
        opt = cfg["optimizer"]
        if "type" in opt and opt["type"] not in ["adamw", "lion", "adafactor"]:
            errors.append(f"Invalid optimizer.type: {opt['type']} (must be adamw|lion|adafactor)")
    
    # Validate data section
    if "data" in cfg:
        data = cfg["data"]
        if "source" in data and data["source"] not in ["hf", "jsonl", "webdataset"]:
            errors.append(f"Invalid data.source: {data['source']} (must be hf|jsonl|webdataset)")
        if "collate" in data and data["collate"] not in ["lm", "cv"]:
            errors.append(f"Invalid data.collate: {data['collate']} (must be lm|cv)")
    
    # Validate eval section
    if "eval" in cfg:
        eval_cfg = cfg["eval"]
        if "select_best_by" in eval_cfg and eval_cfg["select_best_by"] not in ["loss", "ppl"]:
            errors.append(f"Invalid eval.select_best_by: {eval_cfg['select_best_by']} (must be loss|ppl)")
    
    # Validate numeric ranges
    if "training" in cfg:
        train = cfg["training"]
        if "learning_rate" in train:
            lr = train["learning_rate"]
            if not (1e-6 <= lr <= 1.0):
                errors.append(f"training.learning_rate {lr} should be between 1e-6 and 1.0")
        if "warmup_ratio" in train:
            wr = train["warmup_ratio"]
            if not (0.0 <= wr <= 1.0):
                errors.append(f"training.warmup_ratio {wr} should be between 0.0 and 1.0")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate training configuration YAML")
    parser.add_argument(
        "config",
        type=str,
        default="configs/llm_default.yaml",
        nargs="?",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    is_valid, errors = validate_config(str(config_path))
    
    if is_valid:
        print(f"✅ Configuration file '{config_path}' is valid!")
        sys.exit(0)
    else:
        print(f"❌ Configuration file '{config_path}' has {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()



