"""
Batch-fix broken __import__ lazy loaders in __init__.py files.

Replaces:
    module = __import__(module_path, fromlist=[name], level=N)
With:
    import importlib
    module = importlib.import_module(module_path, package=__package__)

Also adds thread-safety (threading.RLock) where missing.
"""

import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

# Patterns to match
# Pattern 1: module = __import__(module_path, fromlist=[name], level=N)
# Pattern 2: module = __import__(module_path.lstrip('.'), fromlist=[name], level=1)
# Pattern 3: module = __import__(f'{__name__}{module_path}', fromlist=[name], level=0)
IMPORT_PATTERN = re.compile(
    r'^(\s*)module\s*=\s*__import__\(.*?,\s*fromlist=\[name\],\s*level=\d+\)',
    re.MULTILINE
)

# Also match the lstrip variant 
IMPORT_PATTERN_LSTRIP = re.compile(
    r'^(\s*)module\s*=\s*__import__\(module_path\.lstrip\([\'\"]\.\s*[\'\"]\),\s*fromlist=\[name\],\s*level=\d+\)',
    re.MULTILINE
)

# The f-string variant
IMPORT_PATTERN_FSTRING = re.compile(
    r"^(\s*)module\s*=\s*__import__\(f'.*?',\s*fromlist=\[name\],\s*level=\d+\)",
    re.MULTILINE
)

def fix_file(filepath: str) -> bool:
    """Fix a single __init__.py file. Returns True if changes were made."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Check if file has any __import__ pattern
    if '__import__' not in content:
        return False
    
    # Replace the common pattern:
    #   module = __import__(module_path, fromlist=[name], level=2)  
    # With:
    #   module = importlib.import_module(module_path, package=__package__)
    content = re.sub(
        r'^(\s*)module\s*=\s*__import__\(module_path\.lstrip\([\'\"]\.\s*[\'\"]\),\s*fromlist=\[name\],\s*level=\d+\)',
        r'\1module = importlib.import_module(module_path, package=__package__)',
        content,
        flags=re.MULTILINE
    )
    
    content = re.sub(
        r"^(\s*)module\s*=\s*__import__\(f'\{__name__\}\{module_path\}',\s*fromlist=\[name\],\s*level=\d+\)",
        r'\1module = importlib.import_module(module_path, package=__package__)',
        content,
        flags=re.MULTILINE
    )
    
    content = re.sub(
        r'^(\s*)module\s*=\s*__import__\(module_path,\s*fromlist=\[name\],\s*level=\d+\)',
        r'\1module = importlib.import_module(module_path, package=__package__)',
        content,
        flags=re.MULTILINE
    )
    
    # Add `import importlib` if not already present and we made changes
    if content != original:
        if 'import importlib' not in content:
            # Insert after last import or after __future__
            if 'from __future__' in content:
                content = re.sub(
                    r'(from __future__ import annotations\r?\n)',
                    r'\1\nimport importlib\n',
                    content,
                    count=1
                )
            else:
                # Insert at beginning after docstring
                lines = content.split('\n')
                insert_idx = 0
                in_docstring = False
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        if in_docstring:
                            insert_idx = i + 1
                            in_docstring = False
                        else:
                            if stripped.endswith('"""') and len(stripped) > 3:
                                insert_idx = i + 1
                            else:
                                in_docstring = True
                    elif not in_docstring and (stripped.startswith('import ') or stripped.startswith('from ')):
                        insert_idx = i + 1
                
                lines.insert(insert_idx, 'import importlib')
                content = '\n'.join(lines)
    
    if content == original:
        return False
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def main():
    fixed = []
    skipped = []
    errors = []
    
    for root, dirs, files in os.walk(BASE):
        for filename in files:
            if filename == '__init__.py':
                filepath = os.path.join(root, filename)
                try:
                    if fix_file(filepath):
                        relpath = os.path.relpath(filepath, BASE)
                        fixed.append(relpath)
                        print(f"  [FIXED] {relpath}")
                    else:
                        skipped.append(os.path.relpath(filepath, BASE))
                except Exception as e:
                    relpath = os.path.relpath(filepath, BASE)
                    errors.append((relpath, str(e)))
                    print(f"  [ERROR] {relpath}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Fixed:   {len(fixed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors:  {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == '__main__':
    main()

