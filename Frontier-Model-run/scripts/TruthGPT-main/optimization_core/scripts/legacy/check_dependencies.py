#!/usr/bin/env python3
"""
Check that all pip dependencies referenced in BUILD files exist in requirements_lock.txt
"""

import re
from pathlib import Path

def extract_pip_deps_from_build_files():
    """Extract all @pip// dependencies from BUILD files"""
    base_dir = Path(__file__).parent
    deps = set()
    
    for build_file in base_dir.rglob("BUILD.bazel"):
        try:
            with open(build_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Find all @pip// references
                matches = re.findall(r'@pip//(\w+)', content)
                deps.update(matches)
        except Exception as e:
            print(f"Error reading {build_file}: {e}")
    
    return deps

def extract_requirements():
    """Extract package names from requirements_lock.txt"""
    base_dir = Path(__file__).parent
    req_file = base_dir / "requirements_lock.txt"
    
    packages = {}
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package==version
                    match = re.match(r'^([a-zA-Z0-9_-]+)==', line)
                    if match:
                        pkg_name = match.group(1)
                        # Handle package name variations
                        # e.g., python-dotenv -> python_dotenv
                        normalized = pkg_name.replace('-', '_')
                        packages[pkg_name] = normalized
                        packages[normalized] = normalized
    
    return packages

def main():
    """Main check function"""
    print("Checking pip dependencies...")
    
    # Get dependencies from BUILD files
    build_deps = extract_pip_deps_from_build_files()
    print(f"\nFound {len(build_deps)} unique pip dependencies in BUILD files:")
    for dep in sorted(build_deps):
        print(f"  - {dep}")
    
    # Get packages from requirements
    req_packages = extract_requirements()
    print(f"\nFound {len(req_packages)} packages in requirements_lock.txt")
    
    # Check for missing dependencies
    missing = []
    for dep in build_deps:
        # Check both original and normalized names
        if dep not in req_packages and dep.replace('_', '-') not in req_packages:
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  Missing dependencies in requirements_lock.txt:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nAdd these to requirements_lock.txt")
        return 1
    else:
        print("\n✅ All dependencies found in requirements_lock.txt")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())













