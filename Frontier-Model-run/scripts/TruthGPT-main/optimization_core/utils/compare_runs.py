"""
Compare multiple training runs.
Shows checkpoint info, file sizes, and basic metrics if available.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict


def get_run_info(run_dir: Path) -> Dict:
    """Get information about a training run."""
    info = {
        "name": run_dir.name,
        "path": str(run_dir),
        "checkpoints": [],
        "total_size_mb": 0,
        "has_best": False,
        "has_last": False,
    }
    
    if not run_dir.exists():
        return info
    
    # Find checkpoints
    for item in run_dir.iterdir():
        if item.is_file() and item.suffix == ".pt":
            size_mb = item.stat().st_size / (1024**2)
            info["checkpoints"].append({
                "name": item.name,
                "size_mb": size_mb,
            })
            info["total_size_mb"] += size_mb
            
            if item.name == "best.pt":
                info["has_best"] = True
            if item.name == "last.pt":
                info["has_last"] = True
        
        elif item.is_dir() and (item / "config.json").exists():
            # HF checkpoint format
            size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024**2)
            info["checkpoints"].append({
                "name": item.name,
                "size_mb": size_mb,
            })
            info["total_size_mb"] += size_mb
            
            if item.name == "best.pt":
                info["has_best"] = True
            if item.name == "last.pt":
                info["has_last"] = True
    
    return info


def compare_runs(runs_dir: str = "runs"):
    """Compare all runs in a directory."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"⚠️  Directory not found: {runs_dir}")
        return
    
    runs = []
    for item in runs_path.iterdir():
        if item.is_dir():
            info = get_run_info(item)
            if info["checkpoints"]:
                runs.append(info)
    
    if not runs:
        print(f"📭 No runs with checkpoints found in {runs_dir}")
        return
    
    print(f"\n📊 Comparing {len(runs)} training run(s)\n")
    print("=" * 80)
    print(f"{'Run Name':<30} {'Checkpoints':<15} {'Total Size':<15} {'Best/Last':<10}")
    print("=" * 80)
    
    for run in sorted(runs, key=lambda x: x["name"]):
        checkpoint_count = len(run["checkpoints"])
        total_size = f"{run['total_size_mb']:.1f} MB"
        best_last = "✓" if (run["has_best"] and run["has_last"]) else ("✓" if run["has_best"] else "?")
        
        print(f"{run['name']:<30} {checkpoint_count:<15} {total_size:<15} {best_last:<10}")
    
    print("=" * 80)
    
    # Show checkpoint details
    print("\n📦 Checkpoint Details:\n")
    for run in sorted(runs, key=lambda x: x["name"]):
        print(f"  {run['name']}:")
        for ckpt in sorted(run["checkpoints"], key=lambda x: x["name"]):
            print(f"    - {ckpt['name']}: {ckpt['size_mb']:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple training runs")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing runs (default: runs)",
    )
    args = parser.parse_args()
    
    compare_runs(args.runs_dir)


if __name__ == "__main__":
    main()



