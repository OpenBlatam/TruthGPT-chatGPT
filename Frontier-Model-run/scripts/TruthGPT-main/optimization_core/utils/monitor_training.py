"""
Monitor training progress in real-time.
Shows loss, tokens/s, GPU usage, and memory stats.
"""

import argparse
import time
import os
from pathlib import Path
from typing import Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available. Install with: pip install psutil")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def get_gpu_stats():
    """Get GPU statistics if available."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    
    stats = {}
    for i in range(torch.cuda.device_count()):
        mem_used = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        
        stats[f"gpu_{i}"] = {
            "memory_used_gb": mem_used,
            "memory_reserved_gb": mem_reserved,
            "memory_total_gb": mem_total,
            "memory_percent": (mem_reserved / mem_total) * 100,
        }
    
    return stats


def get_system_stats():
    """Get system statistics."""
    if not _PSUTIL_AVAILABLE:
        return None
    
    stats = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    return stats


def watch_log_file(log_file: str, refresh_interval: float = 1.0):
    """Watch a log file for new training metrics."""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"⚠️  Log file not found: {log_file}")
        print("   Waiting for file to be created...")
        while not log_path.exists():
            time.sleep(0.5)
    
    print(f"📊 Monitoring: {log_file}")
    print("   Press Ctrl+C to stop\n")
    
    last_size = 0
    
    try:
        while True:
            if log_path.exists():
                current_size = log_path.stat().st_size
                
                if current_size > last_size:
                    # Read new content
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                        
                        for line in new_lines:
                            line = line.strip()
                            if any(keyword in line.lower() for keyword in ["loss", "step", "tokens", "ppl", "lr"]):
                                print(f"  {line}")
                    
                    last_size = current_size
                
                # Show system stats
                if _PSUTIL_AVAILABLE:
                    sys_stats = get_system_stats()
                    gpu_stats = get_gpu_stats()
                    
                    if sys_stats:
                        print(f"\r  CPU: {sys_stats['cpu_percent']:.1f}% | "
                              f"RAM: {sys_stats['memory_percent']:.1f}% "
                              f"({sys_stats['memory_used_gb']:.1f}/{sys_stats['memory_total_gb']:.1f} GB)", end="")
                    
                    if gpu_stats:
                        for gpu_name, gpu_stat in gpu_stats.items():
                            print(f" | GPU: {gpu_stat['memory_percent']:.1f}% "
                                  f"({gpu_stat['memory_reserved_gb']:.1f}/{gpu_stat['memory_total_gb']:.1f} GB)", end="")
                    
                    print(end="", flush=True)
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


def monitor_run_dir(run_dir: str, refresh_interval: float = 2.0):
    """Monitor a training run directory."""
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"⚠️  Run directory not found: {run_dir}")
        return
    
    print(f"📊 Monitoring run: {run_path.name}")
    print("   Press Ctrl+C to stop\n")
    
    last_checkpoints = set()
    
    try:
        while True:
            # Check for new checkpoints
            current_checkpoints = set([
                f.name for f in run_path.iterdir()
                if f.is_file() and f.suffix == ".pt"
            ])
            
            new_checkpoints = current_checkpoints - last_checkpoints
            if new_checkpoints:
                print(f"  📦 New checkpoint(s): {', '.join(new_checkpoints)}")
                last_checkpoints = current_checkpoints
            
            # Show checkpoint count
            checkpoint_count = len(current_checkpoints)
            if checkpoint_count > 0:
                total_size = sum(
                    f.stat().st_size for f in run_path.iterdir()
                    if f.is_file() and f.suffix == ".pt"
                ) / (1024**2)  # MB
                
                print(f"\r  Checkpoints: {checkpoint_count} | Total size: {total_size:.1f} MB", end="", flush=True)
            
            # Show system stats
            if _PSUTIL_AVAILABLE:
                sys_stats = get_system_stats()
                if sys_stats:
                    print(f" | CPU: {sys_stats['cpu_percent']:.1f}% | "
                          f"RAM: {sys_stats['memory_percent']:.1f}%", end="", flush=True)
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument(
        "source",
        type=str,
        nargs="?",
        default="runs",
        help="Log file path or run directory (default: runs)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--file",
        action="store_true",
        help="Monitor a log file (default: monitor run directory)",
    )
    args = parser.parse_args()
    
    if args.file:
        watch_log_file(args.source, args.interval)
    else:
        monitor_run_dir(args.source, args.interval)


if __name__ == "__main__":
    main()



