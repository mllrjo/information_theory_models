# File: device_manager.py
# Directory: ./src/utils/

"""
Device management utilities for CPU/GPU detection and configuration.
Handles cross-platform device detection and optimal settings.
"""

import torch
import platform
import os
from typing import Dict, Any


def detect_device() -> str:
    """
    Detect the best available device for computation.
    
    Returns:
        str: Device type ('cpu', 'cuda', 'mps') based on availability
        
    Raises:
        RuntimeError: If no compatible device is found
    """
    try:
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            # Verify CUDA device is actually accessible
            try:
                torch.cuda.device_count()
                return 'cuda'
            except RuntimeError:
                pass
        
        # Check for Apple Metal Performance Shaders (M1/M2 Macs)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Test MPS functionality
                test_tensor = torch.tensor([1.0], device='mps')
                test_tensor.cpu()  # Verify we can move back to CPU
                return 'mps'
            except RuntimeError:
                pass
        
        # Fallback to CPU (always available)
        if torch.cuda.is_available() or platform.system() in ['Darwin', 'Linux', 'Windows']:
            return 'cpu'
        
        raise RuntimeError("No compatible PyTorch device found")
        
    except Exception as e:
        raise RuntimeError(f"Device detection failed: {str(e)}")


def get_device_config(device_type: str) -> Dict[str, Any]:
    """
    Get device-specific configuration settings.
    
    Args:
        device_type: Device type from detect_device()
        
    Returns:
        Dict containing device settings:
        - 'device': torch.device object
        - 'dtype': recommended torch dtype
        - 'batch_size': recommended batch size
        - 'num_workers': recommended dataloader workers
        - 'pin_memory': whether to pin memory for transfers
        - 'compile_model': whether model compilation is supported
        
    Raises:
        ValueError: If device_type is not supported
    """
    if not isinstance(device_type, str):
        raise TypeError(f"device_type must be str, got {type(device_type)}")
    
    device_type = device_type.lower().strip()
    
    if device_type == 'cpu':
        return _get_cpu_config()
    elif device_type == 'cuda':
        return _get_cuda_config()
    elif device_type == 'mps':
        return _get_mps_config()
    else:
        raise ValueError(f"Unsupported device type: {device_type}. Supported: ['cpu', 'cuda', 'mps']")


def _get_cpu_config() -> Dict[str, Any]:
    """Get CPU-optimized configuration."""
    num_cores = os.cpu_count() or 2
    
    return {
        'device': torch.device('cpu'),
        'dtype': torch.float32,  # CPU works well with float32
        'batch_size': 32,        # Conservative for CPU
        'num_workers': min(2, num_cores // 2),  # Don't overwhelm CPU
        'pin_memory': False,     # No GPU transfers
        'compile_model': False,  # torch.compile less beneficial on CPU
        'gradient_accumulation': 4,  # Compensate for smaller batches
        'device_info': {
            'cores': num_cores,
            'platform': platform.system(),
            'architecture': platform.machine()
        }
    }


def _get_cuda_config() -> Dict[str, Any]:
    """Get CUDA GPU-optimized configuration."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but cuda config requested")
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(current_device)
    
    # Determine batch size based on GPU memory
    gpu_memory_gb = device_props.total_memory / (1024**3)
    
    if gpu_memory_gb >= 24:      # A100, RTX 4090, etc.
        batch_size = 128
        gradient_accumulation = 1
    elif gpu_memory_gb >= 12:    # RTX 3080, etc.
        batch_size = 64
        gradient_accumulation = 2
    elif gpu_memory_gb >= 8:     # RTX 3070, etc.
        batch_size = 32
        gradient_accumulation = 4
    else:                        # Lower-end GPUs
        batch_size = 16
        gradient_accumulation = 8
    
    return {
        'device': torch.device('cuda'),
        'dtype': torch.bfloat16 if device_props.major >= 8 else torch.float16,
        'batch_size': batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'compile_model': True,
        'gradient_accumulation': gradient_accumulation,
        'device_info': {
            'name': device_props.name,
            'memory_gb': gpu_memory_gb,
            'compute_capability': f"{device_props.major}.{device_props.minor}",
            'multiprocessor_count': device_props.multi_processor_count,
            'device_count': device_count
        }
    }


def _get_mps_config() -> Dict[str, Any]:
    """Get Apple Metal Performance Shaders configuration."""
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise RuntimeError("MPS not available but mps config requested")
    
    # Apple Silicon optimization
    return {
        'device': torch.device('mps'),
        'dtype': torch.float32,  # MPS has some bfloat16 limitations
        'batch_size': 64,        # M1/M2 can handle moderate batches
        'num_workers': 2,        # Conservative for unified memory
        'pin_memory': False,     # Not applicable for MPS
        'compile_model': False,  # torch.compile MPS support limited
        'gradient_accumulation': 2,
        'device_info': {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'unified_memory': True
        }
    }


def validate_device_config(config: Dict[str, Any]) -> bool:
    """
    Validate device configuration dictionary.
    
    Args:
        config: Device configuration from get_device_config()
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        TypeError: If config structure is invalid
        KeyError: If required keys are missing
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be dict, got {type(config)}")
    
    required_keys = {
        'device', 'dtype', 'batch_size', 'num_workers', 
        'pin_memory', 'compile_model', 'device_info'
    }
    
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")
    
    # Validate types
    validations = [
        (config['device'], torch.device, "device must be torch.device"),
        (config['dtype'], type, "dtype must be torch dtype"),
        (config['batch_size'], int, "batch_size must be int"),
        (config['num_workers'], int, "num_workers must be int"),
        (config['pin_memory'], bool, "pin_memory must be bool"),
        (config['compile_model'], bool, "compile_model must be bool"),
        (config['device_info'], dict, "device_info must be dict")
    ]
    
    for value, expected_type, error_msg in validations:
        if expected_type == type:  # Special case for torch dtypes
            if not isinstance(value, torch.dtype):
                raise TypeError(f"{error_msg}, got {type(value)}")
        elif not isinstance(value, expected_type):
            raise TypeError(f"{error_msg}, got {type(value)}")
    
    # Validate ranges
    if config['batch_size'] <= 0:
        raise ValueError(f"batch_size must be positive, got {config['batch_size']}")
    
    if config['num_workers'] < 0:
        raise ValueError(f"num_workers must be non-negative, got {config['num_workers']}")
    
    # Validate device accessibility
    device = config['device']
    try:
        test_tensor = torch.tensor([1.0], device=device)
        test_tensor.cpu()  # Ensure we can move data back
    except RuntimeError as e:
        raise RuntimeError(f"Device {device} is not accessible: {str(e)}")
    
    return True


def get_environment_info() -> Dict[str, str]:
    """
    Get comprehensive environment information for debugging.
    
    Returns:
        Dict containing environment details
    """
    info = {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': str(torch.cuda.is_available()),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'device_detected': detect_device()
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_device_count': str(torch.cuda.device_count()),
            'cuda_current_device': str(torch.cuda.current_device()),
            'cuda_device_name': torch.cuda.get_device_name()
        })
    
    if hasattr(torch.backends, 'mps'):
        info['mps_available'] = str(torch.backends.mps.is_available())
    
    return info
