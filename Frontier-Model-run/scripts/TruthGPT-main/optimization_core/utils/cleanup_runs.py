"""
Clean up old training runs and checkpoints.
Helps manage disk space.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime, timedelta


def get_run_size(run_dir: Path) -> float:
    """Get total size of a run directory in MB."""
    total = 0
    for f in run_dir.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024**2)


def cleanup_old_runs(runs_dir: str = "runs", days: int = 30, dry_run: bool = True):
    """Remove runs older than specified days."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"⚠️  Directory not found: {runs_dir}")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    removed = []
    total_size = 0.0
    
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Check modification time
        mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
        
        if mtime < cutoff_date:
            size_mb = get_run_size(run_dir)
            if dry_run:
                print(f"  Would remove: {run_dir.name} ({size_mb:.1f} MB, last modified: {mtime.strftime('%Y-%m-%d')})")
            else:
                shutil.rmtree(run_dir)
                print(f"  ✓ Removed: {run_dir.name} ({size_mb:.1f} MB)")
            
            removed.append(run_dir.name)
            total_size += size_mb
    
    if dry_run:
        print(f"\n📊 Dry run: Would remove {len(removed)} run(s) ({total_size:.1f} MB total)")
        print("   Use --execute to actually remove them")
    else:
        print(f"\n✅ Removed {len(removed)} run(s) ({total_size:.1f} MB total)")


def cleanup_checkpoints(runs_dir: str = "runs", keep: int = 3, dry_run: bool = True):
    """Keep only the last N checkpoints per run."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"⚠️  Directory not found: {runs_dir}")
        return
    
    total_removed = 0
    total_size = 0.0
    
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Find step_*.pt checkpoints
        step_checkpoints = sorted([
            f for f in run_dir.iterdir()
            if f.is_file() and f.name.startswith("step_") and f.suffix == ".pt"
        ], key=lambda x: int(x.stem.split("_")[1]) if "_" in x.stem else 0)
        
        if len(step_checkpoints) <= keep:
            continue
        
        # Remove old ones
        to_remove = step_checkpoints[:-keep]  # Keep last N
        
        for ckpt in to_remove:
            size_mb = ckpt.stat().st_size / (1024**2)
            if dry_run:
                print(f"  Would remove: {ckpt.name} ({size_mb:.1f} MB) from {run_dir.name}")
            else:
                ckpt.unlink()
                print(f"  ✓ Removed: {ckpt.name} ({size_mb:.1f} MB) from {run_dir.name}")
            
            total_removed += 1
            total_size += size_mb
    
    if dry_run:
        print(f"\n📊 Dry run: Would remove {total_removed} checkpoint(s) ({total_size:.1f} MB total)")
        print("   Use --execute to actually remove them")
    else:
        print(f"\n✅ Removed {total_removed} checkpoint(s) ({total_size:.1f} MB total)")


def main():
    parser = argparse.ArgumentParser(description="Clean up old training runs and checkpoints")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing runs (default: runs)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Remove runs older than N days (default: 30)",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Keep last N step checkpoints per run (default: 3)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove files (default: dry run)",
    )
    parser.add_argument(
        "--old-runs",
        action="store_true",
        help="Clean old runs",
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Clean old checkpoints",
    )
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    if args.old_runs:
        print(f"🧹 Cleaning runs older than {args.days} days...")
        cleanup_old_runs(args.runs_dir, args.days, dry_run)
    
    if args.checkpoints:
        print(f"🧹 Cleaning checkpoints (keeping last {args.keep_checkpoints})...")
        cleanup_checkpoints(args.runs_dir, args.keep_checkpoints, dry_run)
    
    if not args.old_runs and not args.checkpoints:
        # Default: clean both
        print(f"🧹 Cleaning runs older than {args.days} days...")
        cleanup_old_runs(args.runs_dir, args.days, dry_run)
        print(f"\n🧹 Cleaning checkpoints (keeping last {args.keep_checkpoints})...")
        cleanup_checkpoints(args.runs_dir, args.keep_checkpoints, dry_run)


if __name__ == "__main__":
    main()



