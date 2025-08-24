# File: tests/test_utils/test_logging_config.py
"""
Comprehensive tests for enhanced Morris entropy logging configuration.
Tests all entropy measurements, derived metrics, and resumable state management.
"""

import json
import pytest
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from src.utils.logging_config import (
    setup_morris_entropy_logger,
    create_experiment_log_structure,
    log_entropy_measurements,
    log_derived_metrics,
    log_training_step,
    load_experiment_state,
    get_experiment_progress
)


class TestMorrisEntropyLogger:
    """Test suite for Morris entropy logging functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_experiment = "test_morris_experiment"
        
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_setup_morris_entropy_logger_success(self):
        """Test successful Morris entropy logger setup."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir,
            device=torch.device("cpu")
        )
        
        assert logger is not None
        assert logger.name == f"morris_entropy_{self.test_experiment}"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2  # File and console handlers
        
        # Check directory structure created
        entropy_dir = Path(self.temp_dir) / "entropy_measurements"
        experiment_dir = Path(self.temp_dir) / "experiments"
        assert entropy_dir.exists()
        assert experiment_dir.exists()
        
        # Check log file created
        log_file = entropy_dir / f"{self.test_experiment}_entropy.jsonl"
        assert log_file.exists()
        
    def test_setup_morris_entropy_logger_auto_device_detection(self):
        """Test automatic device detection in logger setup."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            logger = setup_morris_entropy_logger(
                experiment_name=self.test_experiment,
                log_dir=self.temp_dir
            )
            assert logger is not None
            
    def test_setup_morris_entropy_logger_invalid_experiment_name(self):
        """Test logger setup with invalid experiment name."""
        with pytest.raises(ValueError, match="experiment_name cannot be empty"):
            setup_morris_entropy_logger(
                experiment_name="",
                log_dir=self.temp_dir
            )
            
        with pytest.raises(ValueError, match="experiment_name cannot be empty"):
            setup_morris_entropy_logger(
                experiment_name="   ",
                log_dir=self.temp_dir
            )
            
    def test_create_experiment_log_structure_valid(self):
        """Test creation of valid experiment log structure."""
        log_structure = create_experiment_log_structure(
            experiment_name=self.test_experiment,
            device="cpu",
            model_params=170000,
            dataset_size=10000
        )
        
        assert log_structure["experiment_id"] == self.test_experiment
        assert log_structure["device"] == "cpu"
        assert log_structure["model_params"] == 170000
        assert log_structure["dataset_size"] == 10000
        assert log_structure["status"] == "initialized"
        assert "timestamp" in log_structure
        assert "morris_reproduction" in log_structure
        assert log_structure["morris_reproduction"]["target_figure"] == "Figure 1"
        
    def test_create_experiment_log_structure_invalid_inputs(self):
        """Test experiment log structure with invalid inputs."""
        with pytest.raises(ValueError, match="experiment_name cannot be empty"):
            create_experiment_log_structure("", "cpu")
            
        with pytest.raises(ValueError, match="device cannot be empty"):
            create_experiment_log_structure(self.test_experiment, "")
            
    def test_log_entropy_measurements_valid(self):
        """Test logging valid entropy measurements."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir,
            device=torch.device("cpu")
        )
        
        entropy_dict = {
            "H_X": 10.5,
            "H_X_given_theta": 8.2,
            "H_X_given_theta_bar": 9.1,
            "H_X_given_theta_theta_bar": 7.8
        }
        
        # Should not raise exception
        log_entropy_measurements(
            logger=logger,
            experiment_name=self.test_experiment,
            step=100,
            entropy_dict=entropy_dict,
            model_params=170000,
            dataset_size=10000,
            device="cpu"
        )
        
        # Check log file contains the entry
        log_file = Path(self.temp_dir) / "entropy_measurements" / f"{self.test_experiment}_entropy.jsonl"
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Should have initialization log + entropy measurement log
            assert len(lines) >= 2
            
            # Parse the entropy measurement log (last non-initialization line)
            for line in reversed(lines):
                if "entropy_measurements" in line:
                    # Extract JSON part from log line (after "INFO - ")
                    if " - INFO - " in line:
                        json_part = line.split(" - INFO - ", 1)[1].strip()
                    else:
                        json_part = line.strip()
                    
                    log_entry = json.loads(json_part)
                    assert log_entry["step"] == 100
                    assert log_entry["entropy_measurements"] == entropy_dict
                    assert "derived_metrics" in log_entry
                    assert "memorization" in log_entry["derived_metrics"]
                    assert "generalization" in log_entry["derived_metrics"]
                    break
                    
    def test_log_entropy_measurements_missing_keys(self):
        """Test entropy measurement logging with missing required keys."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        incomplete_entropy = {
            "H_X": 10.5,
            "H_X_given_theta": 8.2
            # Missing H_X_given_theta_bar and H_X_given_theta_theta_bar
        }
        
        with pytest.raises(KeyError, match="Missing required entropy measurement"):
            log_entropy_measurements(
                logger=logger,
                experiment_name=self.test_experiment,
                step=100,
                entropy_dict=incomplete_entropy,
                model_params=170000,
                dataset_size=10000,
                device="cpu"
            )
            
    def test_log_entropy_measurements_invalid_values(self):
        """Test entropy measurement logging with invalid values."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        # Test negative entropy
        invalid_entropy = {
            "H_X": -1.0,  # Negative entropy is invalid
            "H_X_given_theta": 8.2,
            "H_X_given_theta_bar": 9.1,
            "H_X_given_theta_theta_bar": 7.8
        }
        
        with pytest.raises(ValueError, match="cannot be negative"):
            log_entropy_measurements(
                logger=logger,
                experiment_name=self.test_experiment,
                step=100,
                entropy_dict=invalid_entropy,
                model_params=170000,
                dataset_size=10000,
                device="cpu"
            )
            
        # Test non-numeric entropy
        invalid_entropy_2 = {
            "H_X": "not_a_number",
            "H_X_given_theta": 8.2,
            "H_X_given_theta_bar": 9.1,
            "H_X_given_theta_theta_bar": 7.8
        }
        
        with pytest.raises(ValueError, match="must be numeric"):
            log_entropy_measurements(
                logger=logger,
                experiment_name=self.test_experiment,
                step=100,
                entropy_dict=invalid_entropy_2,
                model_params=170000,
                dataset_size=10000,
                device="cpu"
            )
            
    def test_log_derived_metrics_valid(self):
        """Test logging valid derived metrics."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        # Should not raise exception
        log_derived_metrics(
            logger=logger,
            experiment_name=self.test_experiment,
            step=200,
            memorization=1.3,
            generalization=0.4,
            training_loss=2.1,
            device="cpu"
        )
        
        # Verify log entry
        log_file = Path(self.temp_dir) / "entropy_measurements" / f"{self.test_experiment}_entropy.jsonl"
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
            # Find the derived metrics log
            for line in reversed(lines):
                if "metrics_type" in line and "derived" in line:
                    # Extract JSON part from log line (after "INFO - ")
                    if " - INFO - " in line:
                        json_part = line.split(" - INFO - ", 1)[1].strip()
                    else:
                        json_part = line.strip()
                    
                    log_entry = json.loads(json_part)
                    assert log_entry["step"] == 200
                    assert log_entry["derived_metrics"]["memorization"] == 1.3
                    assert log_entry["derived_metrics"]["generalization"] == 0.4
                    assert log_entry["derived_metrics"]["training_loss"] == 2.1
                    break
                    
    def test_log_derived_metrics_invalid_values(self):
        """Test derived metrics logging with invalid values."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        # Test non-numeric value
        with pytest.raises(ValueError, match="must be numeric"):
            log_derived_metrics(
                logger=logger,
                experiment_name=self.test_experiment,
                step=200,
                memorization="invalid",
                generalization=0.4,
                training_loss=2.1,
                device="cpu"
            )
            
        # Test infinite value
        with pytest.raises(ValueError, match="must be finite"):
            log_derived_metrics(
                logger=logger,
                experiment_name=self.test_experiment,
                step=200,
                memorization=float('inf'),
                generalization=0.4,
                training_loss=2.1,
                device="cpu"
            )
            
    def test_log_training_step_basic(self):
        """Test basic training step logging."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        log_training_step(
            logger=logger,
            experiment_name=self.test_experiment,
            step=50,
            loss=1.5,
            device="cpu"
        )
        
        # Verify log entry
        log_file = Path(self.temp_dir) / "entropy_measurements" / f"{self.test_experiment}_entropy.jsonl"
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
            for line in reversed(lines):
                if "training_step" in line:
                    # Extract JSON part from log line (after "INFO - ")
                    if " - INFO - " in line:
                        json_part = line.split(" - INFO - ", 1)[1].strip()
                    else:
                        json_part = line.strip()
                    
                    log_entry = json.loads(json_part)
                    assert log_entry["step"] == 50
                    assert log_entry["training_loss"] == 1.5
                    assert log_entry["log_type"] == "training_step"
                    break
                    
    def test_log_training_step_with_entropy(self):
        """Test training step logging with entropy measurements."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        entropy_measurements = {
            "H_X": 10.5,
            "H_X_given_theta": 8.2,
            "H_X_given_theta_bar": 9.1,
            "H_X_given_theta_theta_bar": 7.8
        }
        
        log_training_step(
            logger=logger,
            experiment_name=self.test_experiment,
            step=75,
            loss=1.2,
            entropy_measurements=entropy_measurements,
            device="cpu"
        )
        
        # Verify derived metrics were calculated
        log_file = Path(self.temp_dir) / "entropy_measurements" / f"{self.test_experiment}_entropy.jsonl"
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
            for line in reversed(lines):
                if "training_step" in line:
                    # Extract JSON part from log line (after "INFO - ")
                    if " - INFO - " in line:
                        json_part = line.split(" - INFO - ", 1)[1].strip()
                    else:
                        json_part = line.strip()
                    
                    log_entry = json.loads(json_part)
                    assert "entropy_measurements" in log_entry
                    assert "derived_metrics" in log_entry
                    assert "memorization" in log_entry["derived_metrics"]
                    assert "generalization" in log_entry["derived_metrics"]
                    break
                    
    def test_log_training_step_invalid_parameters(self):
        """Test training step logging with invalid parameters."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        # Test negative step
        with pytest.raises(ValueError, match="cannot be negative"):
            log_training_step(
                logger=logger,
                experiment_name=self.test_experiment,
                step=-1,
                loss=1.5
            )
            
        # Test infinite loss
        with pytest.raises(ValueError, match="must be finite"):
            log_training_step(
                logger=logger,
                experiment_name=self.test_experiment,
                step=50,
                loss=float('inf')
            )
            
    def test_load_experiment_state_existing(self):
        """Test loading existing experiment state."""
        # Set up logger and log some data
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        log_training_step(
            logger=logger,
            experiment_name=self.test_experiment,
            step=100,
            loss=1.5
        )
        
        # Load state
        state = load_experiment_state(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        assert state is not None
        assert state["step"] == 100
        assert state["training_loss"] == 1.5
        
    def test_load_experiment_state_nonexistent(self):
        """Test loading state for non-existent experiment."""
        state = load_experiment_state(
            experiment_name="nonexistent_experiment",
            log_dir=self.temp_dir
        )
        
        assert state is None
        
    def test_get_experiment_progress_not_started(self):
        """Test progress for experiment that hasn't started."""
        progress = get_experiment_progress(
            experiment_name="not_started_experiment",
            log_dir=self.temp_dir
        )
        
        assert progress["status"] == "not_started"
        assert progress["latest_step"] == 0
        assert progress["total_measurements"] == 0
        
    def test_get_experiment_progress_in_progress(self):
        """Test progress for experiment that's in progress."""
        # Set up experiment
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        # Log entropy measurements
        entropy_dict = {
            "H_X": 10.5,
            "H_X_given_theta": 8.2,
            "H_X_given_theta_bar": 9.1,
            "H_X_given_theta_theta_bar": 7.8
        }
        
        log_entropy_measurements(
            logger=logger,
            experiment_name=self.test_experiment,
            step=150,
            entropy_dict=entropy_dict,
            model_params=170000,
            dataset_size=10000,
            device="cpu"
        )
        
        # Get progress
        progress = get_experiment_progress(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        assert progress["status"] == "in_progress"
        assert progress["latest_step"] == 150
        assert progress["total_measurements"] == 1
        assert "latest_entropy" in progress
        assert "latest_metrics" in progress
        
    def test_entropy_calculation_edge_cases(self):
        """Test entropy calculations with edge case values."""
        logger = setup_morris_entropy_logger(
            experiment_name=self.test_experiment,
            log_dir=self.temp_dir
        )
        
        # Test with zero entropy values
        entropy_dict = {
            "H_X": 0.0,
            "H_X_given_theta": 0.0,
            "H_X_given_theta_bar": 0.0,
            "H_X_given_theta_theta_bar": 0.0
        }
        
        # Should not raise exception
        log_entropy_measurements(
            logger=logger,
            experiment_name=self.test_experiment,
            step=1,
            entropy_dict=entropy_dict,
            model_params=170000,
            dataset_size=10000,
            device="cpu"
        )
        
    def test_device_compatibility(self):
        """Test logging works with different device types."""
        for device_str in ["cpu", "cuda", "mps"]:
            experiment_name = f"test_device_{device_str}"
            
            logger = setup_morris_entropy_logger(
                experiment_name=experiment_name,
                log_dir=self.temp_dir
            )
            
            log_training_step(
                logger=logger,
                experiment_name=experiment_name,
                step=1,
                loss=1.0,
                device=device_str
            )
            
            # Verify device is correctly logged
            log_file = Path(self.temp_dir) / "entropy_measurements" / f"{experiment_name}_entropy.jsonl"
            assert log_file.exists()


if __name__ == "__main__":
    pytest.main([__file__])
