#!/usr/bin/env python3
"""
Script to check polyglot_core backend availability.

Usage:
    python -m optimization_core.polyglot_core.scripts.check_backends
    python -m optimization_core.polyglot_core.scripts.check_backends --detailed
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from optimization_core.polyglot_core import (
    get_available_backends,
    print_backend_status,
    check_polyglot_availability,
    print_polyglot_status,
    get_test_compatibility_info,
    print_device_info,
)


def main():
    parser = argparse.ArgumentParser(description="Check polyglot_core backend availability")
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed information'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    if args.json:
        import json
        backends = get_available_backends()
        availability = check_polyglot_availability()
        compatibility = get_test_compatibility_info()
        
        data = {
            'backends': [
                {
                    'name': b.name,
                    'available': b.available,
                    'version': b.version,
                    'features': b.features,
                    'performance_multiplier': b.performance_multiplier
                }
                for b in backends
            ],
            'modules': availability,
            'compatibility': compatibility
        }
        
        print(json.dumps(data, indent=2))
        return
    
    print("=" * 80)
    print("Polyglot Core - Backend Check")
    print("=" * 80)
    print()
    
    # Device info
    print_device_info()
    
    # Backend status
    print_backend_status()
    
    # Module status
    print_polyglot_status()
    
    if args.detailed:
        # Compatibility info
        print("\n" + "=" * 80)
        print("Compatibility Information")
        print("=" * 80)
        
        compatibility = get_test_compatibility_info()
        print(f"Polyglot Available: {compatibility.get('polyglot_available', False)}")
        
        print("\nBackends:")
        for name, info in compatibility.get('backends', {}).items():
            status = "✓" if info.get('available', False) else "✗"
            print(f"  {status} {name}: {info.get('version', 'unknown')}")
            if info.get('features'):
                print(f"    Features: {', '.join(info['features'])}")
        
        print("\nModules:")
        for name, available in compatibility.get('modules', {}).items():
            status = "✓" if available else "✗"
            print(f"  {status} {name}")
        
        if compatibility.get('recommendations'):
            print("\nRecommendations:")
            for rec in compatibility['recommendations']:
                print(f"  • {rec}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()













