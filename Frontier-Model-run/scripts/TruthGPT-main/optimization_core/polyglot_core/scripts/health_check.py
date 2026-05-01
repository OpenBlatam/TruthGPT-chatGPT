#!/usr/bin/env python3
"""
Script to check polyglot_core health.

Usage:
    python -m optimization_core.polyglot_core.scripts.health_check
    python -m optimization_core.polyglot_core.scripts.health_check --json
"""

import sys
import argparse
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from optimization_core.polyglot_core import (
    check_health,
    print_health_status,
)


def main():
    parser = argparse.ArgumentParser(description="Check polyglot_core health")
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    health = check_health()
    
    if args.json:
        print(json.dumps(health.to_dict(), indent=2))
    else:
        print_health_status()


if __name__ == "__main__":
    main()













