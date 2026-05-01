"""
Complete workflow example demonstrating all modular features.
Shows how to switch components via YAML without code changes.
"""

import yaml
from pathlib import Path
from optimization_core.scripts.legacy.build_trainer import build_trainer
from trainers.trainer import TrainerConfig


def create_config_variant(name: str, **overrides) -> dict:
    """Create a config variant with overrides."""
    base_path = Path("configs/llm_default.yaml")
    if not base_path.exists():
        print(f"⚠️  Config file not found: {base_path}")
        return {}
    
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Apply overrides
    for key, value in overrides.items():
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    cfg["run_name"] = name
    cfg["output_dir"] = f"runs/{name}"
    
    return cfg


def demo_workflows():
    """Demonstrate different workflow configurations."""
    
    print("🎯 TruthGPT Optimization Core - Complete Workflow Demo\n")
    
    # 1. Basic training
    print("1️⃣  Basic Training Configuration")
    print("-" * 50)
    cfg1 = create_config_variant(
        "basic",
        **{"training.epochs": 1, "training.train_batch_size": 4}
    )
    print(f"✅ Model: {cfg1['model']['name_or_path']}")
    print(f"✅ Optimizer: {cfg1['optimizer']['type']}")
    print(f"✅ Callbacks: {cfg1['training']['callbacks']}")
    print()
    
    # 2. LoRA fine-tuning
    print("2️⃣  LoRA Fine-tuning Configuration")
    print("-" * 50)
    cfg2 = create_config_variant(
        "lora",
        **{
            "model.lora.enabled": True,
            "training.mixed_precision": "bf16",
            "training.epochs": 2,
        }
    )
    print(f"✅ Model: {cfg2['model']['name_or_path']}")
    print(f"✅ LoRA enabled: {cfg2['model']['lora']['enabled']}")
    print(f"✅ LoRA rank: {cfg2['model']['lora']['r']}")
    print()
    
    # 3. Performance optimized
    print("3️⃣  Performance Optimized Configuration")
    print("-" * 50)
    cfg3 = create_config_variant(
        "perf",
        **{
            "training.allow_tf32": True,
            "training.torch_compile": True,
            "training.compile_mode": "reduce-overhead",
            "data.bucket_by_length": True,
            "training.fused_adamw": True,
        }
    )
    print(f"✅ TF32: {cfg3['training']['allow_tf32']}")
    print(f"✅ torch.compile: {cfg3['training']['torch_compile']}")
    print(f"✅ Length bucketing: {cfg3['data']['bucket_by_length']}")
    print()
    
    # 4. With W&B logging
    print("4️⃣  W&B Logging Configuration")
    print("-" * 50)
    cfg4 = create_config_variant(
        "wandb",
        **{
            "training.callbacks": ["print", "wandb"],
            "logging.project": "truthgpt-experiments",
            "logging.run_name": "exp-001",
        }
    )
    print(f"✅ Callbacks: {cfg4['training']['callbacks']}")
    print(f"✅ W&B project: {cfg4['logging']['project']}")
    print()
    
    # 5. Streaming dataset
    print("5️⃣  Streaming Dataset Configuration")
    print("-" * 50)
    cfg5 = create_config_variant(
        "streaming",
        **{
            "data.streaming": True,
            "data.collate": "lm",
        }
    )
    print(f"✅ Data source: {cfg5['data']['source']}")
    print(f"✅ Streaming: {cfg5['data']['streaming']}")
    print(f"✅ Collate: {cfg5['data']['collate']}")
    print()
    
    # 6. EMA + Resume
    print("6️⃣  EMA + Resume Configuration")
    print("-" * 50)
    cfg6 = create_config_variant(
        "ema_resume",
        **{
            "ema.enabled": True,
            "ema.decay": 0.9999,
            "resume.enabled": True,
            "checkpoint.interval_steps": 500,
        }
    )
    print(f"✅ EMA enabled: {cfg6['ema']['enabled']}")
    print(f"✅ EMA decay: {cfg6['ema']['decay']}")
    print(f"✅ Resume enabled: {cfg6['resume']['enabled']}")
    print()
    
    print("=" * 50)
    print("✅ All configurations created successfully!")
    print("\nTo use any configuration:")
    print("  python train_llm.py --config <custom_config.yaml>")
    print("\nOr modify configs/llm_default.yaml directly.")


if __name__ == "__main__":
    demo_workflows()



