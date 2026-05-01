"""
Export configuration from a checkpoint or training run.
Useful for reproducing experiments.
"""

import argparse
import json
import yaml
from pathlib import Path


def export_from_checkpoint(checkpoint_path: str, output: str = None) -> bool:
    """Export config.json from HF checkpoint to YAML."""
    ckpt_path = Path(checkpoint_path)
    
    if ckpt_path.is_dir():
        config_file = ckpt_path / "config.json"
    else:
        # If it's a .pt file, try parent directory
        parent = ckpt_path.parent
        config_file = parent / "config.json"
        if not config_file.exists():
            config_file = parent / "training_args.bin"
            print("⚠️  Training args not directly readable, checking for config in parent...")
            parent_parent = parent.parent
            config_file = parent_parent / "config.json"
    
    if not config_file.exists():
        print(f"❌ Config file not found in {checkpoint_path}")
        return False
    
    # Load JSON config
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"configs/exported_{ckpt_path.stem}.yaml")
    
    # Convert to YAML-friendly format
    yaml_config = {
        "model": {
            "name_or_path": config.get("_name_or_path", config.get("model_type", "unknown")),
        },
        "training": {
            # Map common training args
        },
    }
    
    # Write YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Exported config to: {output_path}")
    print(f"   Note: This is a basic export. Review and complete the config manually.")
    return True


def export_from_run(run_dir: str, output: str = None) -> bool:
    """Export config from a training run directory."""
    run_path = Path(run_dir)
    
    # Look for best checkpoint
    best_ckpt = run_path / "best.pt"
    if best_ckpt.exists() or (run_path / "best.pt").is_dir():
        return export_from_checkpoint(str(best_ckpt), output)
    
    # Look for last checkpoint
    last_ckpt = run_path / "last.pt"
    if last_ckpt.exists() or (run_path / "last.pt").is_dir():
        return export_from_checkpoint(str(last_ckpt), output)
    
    # Look for any config.json in run directory
    config_file = run_path / "config.json"
    if config_file.exists():
        return export_from_checkpoint(str(config_file), output)
    
    print(f"❌ No checkpoint or config found in {run_dir}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Export configuration from checkpoint or run")
    parser.add_argument(
        "source",
        type=str,
        help="Checkpoint path or run directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output YAML file path",
    )
    args = parser.parse_args()
    
    source_path = Path(args.source)
    if source_path.is_file() or (source_path.is_dir() and (source_path / "config.json").exists()):
        export_from_checkpoint(args.source, args.output)
    elif source_path.is_dir():
        export_from_run(args.source, args.output)
    else:
        print(f"❌ Source not found: {args.source}")


if __name__ == "__main__":
    main()



