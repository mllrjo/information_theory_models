# File: create_test_directories.py
# Directory: ./

"""
Create missing test directory structure for device manager tests.
"""

import os
import pathlib

def create_test_directories():
    """Create the test directory structure if it doesn't exist."""
    
    test_dirs = [
        'tests',
        'tests/test_utils',
        'tests/test_data', 
        'tests/test_models',
        'tests/test_metrics',
        'tests/test_training'
    ]
    
    for dir_path in test_dirs:
        path = pathlib.Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_file = path / '__init__.py'
        if not init_file.exists():
            init_file.write_text('# Test package\n')
            
        print(f"Created directory: {path}")

if __name__ == "__main__":
    create_test_directories()
    print("Test directories created successfully")
