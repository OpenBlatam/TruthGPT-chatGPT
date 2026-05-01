#!/usr/bin/env python3
"""
Initialize a new training project with a custom configuration.
Creates a new config file and output directory.
"""

import argparse
import yaml
from pathlib import Path
import shutil


def load_base_config(preset: str = None) -> dict:
    """Load base config, optionally from preset."""
    if preset:
        preset_path = Path(f"configs/presets/{preset}.yaml")
        if preset_path.exists():
            with open(preset_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            print(f"⚠️  Preset '{preset}' not found, using default")
    
    default_path = Path("configs/llm_default.yaml")
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def create_project_config(project_name: str, preset: str = None, model: str = None) -> dict:
    """Create a new project configuration."""
    cfg = load_base_config(preset)
    
    # Override project-specific settings
    cfg["run_name"] = project_name
    cfg["output_dir"] = f"runs/{project_name}"
    
    if model:
        cfg["model"]["name_or_path"] = model
    
    if "logging" in cfg:
        cfg["logging"]["run_name"] = project_name
    
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Initialize a new training project")
    parser.add_argument("project_name", type=str, help="Name of the project")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["lora_fast", "performance_max", "debug", None],
        default=None,
        help="Use a preset configuration",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name_or_path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output config file path (default: configs/{project_name}.yaml)",
    )
    args = parser.parse_args()
    
    # Create config
    cfg = create_project_config(args.project_name, args.preset, args.model)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"configs/{args.project_name}.yaml")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write config
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"✅ Created project config: {output_path}")
    print(f"   Run name: {cfg['run_name']}")
    print(f"   Output dir: {cfg['output_dir']}")
    if "model" in cfg:
        print(f"   Model: {cfg['model']['name_or_path']}")
    if args.preset:
        print(f"   Preset: {args.preset}")
    print(f"\n🚀 Start training with:")
    print(f"   python train_llm.py --config {output_path}")


if __name__ == "__main__":
    main()



