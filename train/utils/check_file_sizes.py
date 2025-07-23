# scripts/check_file_sizes.py
import os
from pathlib import Path
import sys

def check_file_sizes() -> bool:
    """Checks file line counts against limits."""
    limits = {
        'engine': 400,
        'blocks': 300,
        'callbacks': 250,
        'utils': 200
    }
    success = True
    for dir_name, max_lines in limits.items():
        dir_path = Path('train') / dir_name
        for file_path in dir_path.rglob('*.py'):
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            if lines > max_lines:
                print(f"Error: {file_path} has {lines} lines, exceeds limit of {max_lines}")
                success = False
    return success

if __name__ == "__main__":
    ok = check_file_sizes()
    sys.exit(0 if ok else 1)