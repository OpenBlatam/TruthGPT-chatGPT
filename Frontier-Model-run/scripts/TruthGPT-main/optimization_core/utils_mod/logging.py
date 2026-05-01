import os
import sys
from datetime import datetime


def is_main_process() -> bool:
    # Basic heuristic; extend with torch.distributed.is_initialized if needed
    return os.environ.get("RANK", "0") == "0"


def log_info(message: str) -> None:
    if is_main_process():
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {message}")
        sys.stdout.flush()






