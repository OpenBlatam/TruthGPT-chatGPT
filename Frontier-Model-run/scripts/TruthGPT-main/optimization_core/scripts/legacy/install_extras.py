#!/usr/bin/env python3
"""
Install optional dependencies for optimization_core.
Allows selective installation of features.
"""

import argparse
import subprocess
import sys


OPTIONAL_DEPS = {
    "wandb": {
        "package": "wandb",
        "description": "Weights & Biases logging",
        "used_by": "W&B callback",
    },
    "tensorboard": {
        "package": "tensorboard",
        "description": "TensorBoard logging",
        "used_by": "TensorBoard callback",
    },
    "psutil": {
        "package": "psutil",
        "description": "System monitoring",
        "used_by": "monitor_training.py, health_check.py",
    },
    "accelerate": {
        "package": "accelerate",
        "description": "Accelerate library for distributed training",
        "used_by": "Advanced training features",
    },
    "peft": {
        "package": "peft",
        "description": "Parameter-Efficient Fine-Tuning (LoRA, etc.)",
        "used_by": "LoRA training",
    },
    "all": {
        "package": None,
        "description": "Install all optional dependencies",
        "used_by": "All optional features",
    },
}


def install_package(package_name: str) -> bool:
    """Install a package using pip."""
    try:
        print(f"📦 Installing {package_name}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"✅ {package_name} installed successfully")
            return True
        else:
            print(f"❌ Failed to install {package_name}")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error installing {package_name}: {e}")
        return False


def list_optional_deps():
    """List all optional dependencies."""
    print("\n📦 Optional Dependencies:\n")
    print(f"{'Name':<15} {'Package':<20} {'Description'}")
    print("=" * 70)
    
    for name, info in OPTIONAL_DEPS.items():
        if name == "all":
            continue
        package = info["package"]
        desc = info["description"]
        print(f"{name:<15} {package:<20} {desc}")
    
    print("\nUse: python install_extras.py <name>")
    print("Example: python install_extras.py wandb")


def install_selected(packages: list):
    """Install selected optional dependencies."""
    if "all" in packages:
        packages = [p for p in OPTIONAL_DEPS.keys() if p != "all"]
    
    print(f"\n🚀 Installing {len(packages)} optional dependency(ies)...\n")
    
    success_count = 0
    for pkg_name in packages:
        if pkg_name not in OPTIONAL_DEPS:
            print(f"⚠️  Unknown package: {pkg_name}")
            continue
        
        info = OPTIONAL_DEPS[pkg_name]
        if install_package(info["package"]):
            success_count += 1
        print()
    
    print(f"\n✅ Installed {success_count}/{len(packages)} package(s) successfully")
    
    if success_count < len(packages):
        print("⚠️  Some packages failed to install. Check errors above.")


def check_installed():
    """Check which optional dependencies are installed."""
    print("\n📋 Optional Dependencies Status:\n")
    
    for name, info in OPTIONAL_DEPS.items():
        if name == "all":
            continue
        
        package = info["package"]
        try:
            __import__(package)
            print(f"✅ {name:<15} ({package}) - Installed")
        except ImportError:
            print(f"❌ {name:<15} ({package}) - Not installed")


def main():
    parser = argparse.ArgumentParser(description="Install optional dependencies")
    parser.add_argument(
        "packages",
        nargs="*",
        help="Package names to install (or 'all' for everything)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all optional dependencies",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which optional dependencies are installed",
    )
    args = parser.parse_args()
    
    if args.list:
        list_optional_deps()
    elif args.check:
        check_installed()
    elif args.packages:
        install_selected(args.packages)
    else:
        # Default: list available
        list_optional_deps()
        print("\n💡 Use --check to see installed packages")
        print("💡 Use <package_name> to install")


if __name__ == "__main__":
    main()



