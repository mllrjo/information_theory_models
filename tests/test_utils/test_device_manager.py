# File: test_device_manager.py
# Directory: ./tests/test_utils/

"""
Comprehensive tests for device_manager module.
Tests device detection, configuration, and validation across different platforms.
"""

import pytest
import torch
import platform
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.device_manager import (
    detect_device,
    get_device_config, 
    validate_device_config,
    get_environment_info,
    _get_cpu_config,
    _get_cuda_config,
    _get_mps_config
)


class TestDeviceDetection:
    """Test device detection functionality."""
    
    def test_detect_device_returns_valid_type(self):
        """Test that detect_device returns a supported device type."""
        device = detect_device()
        assert device in ['cpu', 'cuda', 'mps']
        assert isinstance(device, str)
    
    def test_detect_device_cpu_fallback(self):
        """Test that CPU is always available as fallback."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            device = detect_device()
            assert device == 'cpu'
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_detect_device_cuda_success(self, mock_device_count, mock_cuda_available):
        """Test CUDA detection when available."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        
        device = detect_device()
        assert device == 'cuda'
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_detect_device_cuda_failure_fallback(self, mock_device_count, mock_cuda_available):
        """Test CUDA detection failure falls back to CPU."""
        mock_cuda_available.return_value = True
        mock_device_count.side_effect = RuntimeError("CUDA error")
        
        with patch('torch.backends.mps.is_available', return_value=False):
            device = detect_device()
            assert device == 'cpu'
    
    @pytest.mark.skipif(platform.system() != 'Darwin', reason="MPS only on macOS")
    def test_detect_device_mps_on_mac(self):
        """Test MPS detection on macOS (if available)."""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            with patch('torch.cuda.is_available', return_value=False):
                device = detect_device()
                # Could be 'mps' or 'cpu' depending on actual MPS support
                assert device in ['mps', 'cpu']
    
    def test_detect_device_error_handling(self):
        """Test error handling in device detection."""
        with patch('torch.cuda.is_available', side_effect=Exception("Unexpected error")):
            with pytest.raises(RuntimeError, match="Device detection failed"):
                detect_device()


class TestDeviceConfiguration:
    """Test device configuration generation."""
    
    def test_get_cpu_config_structure(self):
        """Test CPU configuration has correct structure."""
        config = _get_cpu_config()
        
        required_keys = {
            'device', 'dtype', 'batch_size', 'num_workers',
            'pin_memory', 'compile_model', 'gradient_accumulation', 'device_info'
        }
        assert set(config.keys()) >= required_keys
        
        assert config['device'] == torch.device('cpu')
        assert config['dtype'] == torch.float32
        assert isinstance(config['batch_size'], int) and config['batch_size'] > 0
        assert isinstance(config['num_workers'], int) and config['num_workers'] >= 0
        assert config['pin_memory'] is False
        assert isinstance(config['device_info'], dict)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.current_device')  
    @patch('torch.cuda.get_device_properties')
    def test_get_cuda_config_structure(self, mock_props, mock_current, mock_count, mock_available):
        """Test CUDA configuration generation."""
        mock_available.return_value = True
        mock_count.return_value = 1
        mock_current.return_value = 0
        
        # Mock device properties for a high-end GPU
        mock_device_props = MagicMock()
        mock_device_props.name = "Tesla A100"
        mock_device_props.total_memory = 40 * 1024**3  # 40GB
        mock_device_props.major = 8
        mock_device_props.minor = 0
        mock_device_props.multi_processor_count = 108
        mock_props.return_value = mock_device_props
        
        config = _get_cuda_config()
        
        assert config['device'] == torch.device('cuda')
        assert config['dtype'] in [torch.float16, torch.bfloat16]
        assert config['batch_size'] == 128  # High-memory GPU
        assert config['pin_memory'] is True
        assert config['compile_model'] is True
        assert 'device_info' in config
        assert config['device_info']['memory_gb'] > 30
    
    @patch('torch.backends.mps.is_available')
    def test_get_mps_config_structure(self, mock_mps_available):
        """Test MPS configuration generation."""
        mock_mps_available.return_value = True
        
        config = _get_mps_config()
        
        assert config['device'] == torch.device('mps')
        assert config['dtype'] == torch.float32
        assert config['pin_memory'] is False
        assert config['compile_model'] is False
        assert config['device_info']['unified_memory'] is True
    
    def test_get_device_config_cpu(self):
        """Test getting CPU configuration through main interface."""
        config = get_device_config('cpu')
        validate_device_config(config)  # Should not raise
        assert config['device'].type == 'cpu'
    
    @patch('torch.cuda.is_available')
    def test_get_device_config_cuda_not_available(self, mock_available):
        """Test CUDA config request when CUDA not available."""
        mock_available.return_value = False
        
        with pytest.raises(RuntimeError, match="CUDA not available"):
            get_device_config('cuda')
    
    @patch('torch.backends.mps.is_available')
    def test_get_device_config_mps_not_available(self, mock_available):
        """Test MPS config request when MPS not available."""
        mock_available.return_value = False
        
        with pytest.raises(RuntimeError, match="MPS not available"):
            get_device_config('mps')
    
    def test_get_device_config_invalid_type(self):
        """Test invalid device type handling."""
        with pytest.raises(ValueError, match="Unsupported device type"):
            get_device_config('invalid_device')
    
    def test_get_device_config_type_validation(self):
        """Test type validation for device_type parameter."""
        with pytest.raises(TypeError):
            get_device_config(123)  # Not a string
    
    def test_get_device_config_case_insensitive(self):
        """Test that device type is case-insensitive."""
        config1 = get_device_config('CPU')
        config2 = get_device_config('cpu')
        assert config1['device'] == config2['device']


class TestConfigValidation:
    """Test device configuration validation."""
    
    def test_validate_device_config_success(self):
        """Test successful validation of valid config."""
        config = get_device_config('cpu')
        result = validate_device_config(config)
        assert result is True
    
    def test_validate_device_config_missing_keys(self):
        """Test validation fails with missing required keys."""
        config = {'device': torch.device('cpu')}  # Missing many required keys
        
        with pytest.raises(KeyError, match="Missing required keys"):
            validate_device_config(config)
    
    def test_validate_device_config_wrong_types(self):
        """Test validation fails with wrong types."""
        config = {
            'device': 'cpu',  # Should be torch.device
            'dtype': torch.float32,
            'batch_size': 32,
            'num_workers': 2,
            'pin_memory': False,
            'compile_model': False,
            'device_info': {}
        }
        
        with pytest.raises(TypeError, match="device must be torch.device"):
            validate_device_config(config)
    
    def test_validate_device_config_invalid_ranges(self):
        """Test validation fails with invalid value ranges."""
        config = {
            'device': torch.device('cpu'),
            'dtype': torch.float32,
            'batch_size': -1,  # Invalid: must be positive
            'num_workers': 2,
            'pin_memory': False,
            'compile_model': False,
            'device_info': {}
        }
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_device_config(config)
    
    def test_validate_device_config_not_dict(self):
        """Test validation fails when config is not a dict."""
        with pytest.raises(TypeError, match="config must be dict"):
            validate_device_config("not_a_dict")
    
    @patch('torch.tensor')
    def test_validate_device_config_device_inaccessible(self, mock_tensor):
        """Test validation fails when device is not accessible."""
        mock_tensor.side_effect = RuntimeError("Device not accessible")
        
        config = get_device_config('cpu')
        
        with pytest.raises(RuntimeError, match="Device .* is not accessible"):
            validate_device_config(config)


class TestEnvironmentInfo:
    """Test environment information gathering."""
    
    def test_get_environment_info_structure(self):
        """Test environment info has expected structure."""
        info = get_environment_info()
        
        required_keys = {
            'platform', 'architecture', 'python_version', 'pytorch_version',
            'cuda_available', 'device_detected'
        }
        assert set(info.keys()) >= required_keys
        
        assert isinstance(info['platform'], str)
        assert isinstance(info['architecture'], str)
        assert isinstance(info['pytorch_version'], str)
        assert info['cuda_available'] in ['True', 'False']
        assert info['device_detected'] in ['cpu', 'cuda', 'mps']
    
    @patch('torch.cuda.is_available')
    def test_get_environment_info_cuda_details(self, mock_available):
        """Test environment info includes CUDA details when available."""
        mock_available.return_value = True
        
        with patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.get_device_name', return_value='RTX 3080'):
            
            info = get_environment_info()
            
            assert 'cuda_device_count' in info
            assert 'cuda_current_device' in info
            assert 'cuda_device_name' in info
            assert info['cuda_device_count'] == '2'
    
    def test_get_environment_info_no_cuda(self):
        """Test environment info without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            info = get_environment_info()
            
            assert info['cuda_available'] == 'False'
            assert info['cuda_version'] == 'N/A'


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_detect_configure_validate_workflow(self):
        """Test complete workflow: detect -> configure -> validate."""
        # Step 1: Detect device
        device_type = detect_device()
        
        # Step 2: Get configuration
        config = get_device_config(device_type)
        
        # Step 3: Validate configuration
        is_valid = validate_device_config(config)
        
        assert is_valid is True
        assert config['device'].type == device_type
    
    def test_environment_consistency(self):
        """Test that environment info is consistent with detected device."""
        device_detected = detect_device()
        env_info = get_environment_info()
        
        assert env_info['device_detected'] == device_detected
    
    @pytest.mark.parametrize("device_type", ['cpu'])
    def test_config_generation_deterministic(self, device_type):
        """Test that config generation is deterministic."""
        config1 = get_device_config(device_type)
        config2 = get_device_config(device_type)
        
        # Should be identical (excluding any random elements)
        assert config1['device'] == config2['device']
        assert config1['dtype'] == config2['dtype']
        assert config1['batch_size'] == config2['batch_size']


class TestErrorRecovery:
    """Test error recovery and edge cases."""
    
    def test_graceful_degradation(self):
        """Test that system gracefully degrades to CPU when advanced features fail."""
        # This should always work, even in adverse conditions
        device = detect_device()
        config = get_device_config(device)
        
        assert device in ['cpu', 'cuda', 'mps']
        assert validate_device_config(config)
    
    def test_memory_constraints(self):
        """Test configuration adapts to memory constraints."""
        config = get_device_config('cpu')
        
        # CPU config should have reasonable memory usage
        assert config['batch_size'] <= 64  # Conservative for CPU
        assert config['num_workers'] <= 8   # Don't overwhelm system


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
