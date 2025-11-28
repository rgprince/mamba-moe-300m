#!/usr/bin/env python3
"""
Comprehensive syntax checker for all Python files
"""

import ast
import sys
from pathlib import Path


def check_file(filepath):
    """Check if a Python file has syntax errors"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error reading file: {e}"


def main():
    """Check all Python files in src/ directory"""
    src_dir = Path('/home/prince/my/mamba-moe-300m/src')
    
    print("=" * 60)
    print("Syntax Checker")
    print("=" * 60)
    
    python_files = list(src_dir.rglob('*.py'))
    errors_found = []
    
    for filepath in sorted(python_files):
        rel_path = filepath.relative_to(src_dir.parent)
        success, error = check_file(filepath)
        
        if success:
            print(f"✓ {rel_path}")
        else:
            print(f"✗ {rel_path}")
            print(f"  Error: {error}")
            errors_found.append((rel_path, error))
    
    print("\n" + "=" * 60)
    if errors_found:
        print(f"❌ Found {len(errors_found)} file(s) with syntax errors:")
        for path, error in errors_found:
            print(f"  - {path}")
        return False
    else:
        print(f"✅ All {len(python_files)} Python files have valid syntax!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
