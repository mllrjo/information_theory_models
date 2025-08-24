# File: __init__.py
# Directory: ./src/utils/

"""
Utilities package for information theory models project.
"""

from .device_manager import (
    detect_device,
    get_device_config,
    validate_device_config,
    get_environment_info
)

__all__ = [
    'detect_device',
    'get_device_config', 
    'validate_device_config',
    'get_environment_info'
]
