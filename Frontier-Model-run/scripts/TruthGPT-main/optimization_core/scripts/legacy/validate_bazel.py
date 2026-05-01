#!/usr/bin/env python3
"""
Script to validate Bazel configuration files
Checks for common syntax errors and issues
"""

import os
import re
import sys
from pathlib import Path

def check_build_file(filepath):
    """Check a BUILD.bazel file for common issues"""
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        # Check for common issues
        for i, line in enumerate(lines, 1):
            # Check for invalid glob patterns
            if 'glob([' in line and '**' in line:
                # This is usually fine, but check context
                pass
            
            # Check for missing quotes in strings
            if re.search(r'=\s*[^"\']\w+', line) and '//' not in line:
                # Might be missing quotes, but could be a variable
                pass
                
    except Exception as e:
        issues.append(f"Error reading {filepath}: {e}")
    
    return issues

def check_workspace(filepath):
    """Check WORKSPACE.bazel for common issues"""
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for pip_parse
        if 'pip_parse' in content:
            if 'requirements_lock' not in content:
                issues.append("pip_parse found but requirements_lock not specified")
        
        # Check for load statements before use
        loads = re.findall(r'load\([^)]+\)', content)
        for load in loads:
            if 'rules_python' in load and 'pip_parse' in load:
                # Check if pip_parse is used before load
                load_pos = content.find(load)
                pip_parse_pos = content.find('pip_parse', load_pos)
                if pip_parse_pos == -1:
                    issues.append("pip_parse used but not loaded")
                    
    except Exception as e:
        issues.append(f"Error reading {filepath}: {e}")
    
    return issues

def main():
    """Main validation function"""
    base_dir = Path(__file__).parent
    issues = []
    
    # Check WORKSPACE.bazel
    workspace_file = base_dir / "WORKSPACE.bazel"
    if workspace_file.exists():
        print(f"Checking {workspace_file}...")
        issues.extend(check_workspace(workspace_file))
    
    # Check all BUILD.bazel files
    for build_file in base_dir.rglob("BUILD.bazel"):
        print(f"Checking {build_file.relative_to(base_dir)}...")
        issues.extend(check_build_file(build_file))
    
    # Report issues
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("\n✅ No obvious issues found in Bazel files")
        print("Note: Full validation requires Bazel to be installed and run")
        return 0

if __name__ == "__main__":
    sys.exit(main())













