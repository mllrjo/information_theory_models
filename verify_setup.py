# File: verify_setup.py
# Directory: ./

import os
import sys
import importlib
import torch

def verify_setup():
    """Verify the development environment is properly configured."""
    
    print("=== Environment Verification ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PYTHONPATH
    print(f"PYTHONPATH includes src: {'src' in os.environ.get('PYTHONPATH', '')}")
    
    # Check environment type
    env_type = os.environ.get('ENVIRONMENT_TYPE', 'unknown')
    print(f"Environment type: {env_type}")
    
    # Check PyTorch installation and device
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Check required packages
    required_packages = [
        'numpy', 'matplotlib', 'tqdm', 'transformers', 
        'pytest', 'datasets', 'tokenizers'
    ]
    
    print("\n=== Package Verification ===")
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: NOT FOUND")
    
    # Check optional packages
    optional_packages = ['mamba_ssm', 'wandb']
    print("\n=== Optional Packages ===")
    for package in optional_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"- {package}: not installed (expected for CPU environment)")
    
    # Check directory structure
    print("\n=== Directory Structure ===")
    expected_dirs = ['src', 'tests', 'configs', 'logs', 'checkpoints']
    for dir_name in expected_dirs:
        exists = os.path.exists(dir_name)
        print(f"{'✓' if exists else '✗'} {dir_name}/: {'exists' if exists else 'missing'}")
    
    print("\n=== Environment Variables ===")
    env_vars = ['PROJECT_NAME', 'ENVIRONMENT_TYPE', 'LOG_LEVEL', 'PYTHONPATH']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var}: {value}")
    
    print("\n=== Setup Complete ===")
    print("Ready to proceed with Phase 1 module development")

if __name__ == "__main__":
    try:
        verify_setup()
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
