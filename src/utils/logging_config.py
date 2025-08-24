# File: src/utils/logging_config.py
"""
Enhanced logging configuration for Morris reproduction with entropy tracking.
Provides structured logging for H(X), H(X|θ), H(X|θ̄), H(X|θ,θ̄) measurements
and derived memorization/generalization metrics.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch


def setup_morris_entropy_logger(
    experiment_name: str,
    log_dir: str = "logs",
    device: Optional[torch.device] = None
) -> logging.Logger:
    """
    Set up structured logger for Morris entropy measurements.
    
    Args:
        experiment_name: Unique identifier for Morris experiment
        log_dir: Directory for log files
        device: Torch device (auto-detected if None)
        
    Returns:
        Configured logger instance
        
    Raises:
        OSError: If log directory cannot be created
        ValueError: If experiment_name is empty
    """
    if not experiment_name.strip():
        raise ValueError("experiment_name cannot be empty")
        
    try:
        # Create log directory structure
        log_path = Path(log_dir)
        entropy_log_path = log_path / "entropy_measurements"
        experiment_log_path = log_path / "experiments"
        
        for path in [entropy_log_path, experiment_log_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Auto-detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        
        # Configure logger
        logger_name = f"morris_entropy_{experiment_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler for entropy measurements
        entropy_file = entropy_log_path / f"{experiment_name}_entropy.jsonl"
        file_handler = logging.FileHandler(entropy_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log experiment initialization
        init_log = create_experiment_log_structure(
            experiment_name=experiment_name,
            device=str(device)
        )
        
        logger.info(f"Morris entropy logger initialized: {json.dumps(init_log)}")
        
        return logger
        
    except Exception as e:
        raise OSError(f"Failed to setup Morris entropy logger: {str(e)}")


def create_experiment_log_structure(
    experiment_name: str,
    device: str,
    model_params: Optional[int] = None,
    dataset_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create structured log entry for Morris experiment initialization.
    
    Args:
        experiment_name: Unique experiment identifier
        device: Device string (cpu/cuda/mps)
        model_params: Number of model parameters
        dataset_size: Size of training dataset
        
    Returns:
        Structured log dictionary
        
    Raises:
        ValueError: If required parameters are invalid
    """
    if not experiment_name.strip():
        raise ValueError("experiment_name cannot be empty")
    if not device.strip():
        raise ValueError("device cannot be empty")
        
    return {
        "experiment_id": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "model_params": model_params,
        "dataset_size": dataset_size,
        "status": "initialized",
        "morris_reproduction": {
            "target_figure": "Figure 1",
            "model_type": "GPT-2",
            "target_params": 170000,
            "vocab_size": 2048,
            "sequence_length": 64
        }
    }


def log_entropy_measurements(
    logger: logging.Logger,
    experiment_name: str,
    step: int,
    entropy_dict: Dict[str, float],
    model_params: int,
    dataset_size: int,
    device: str
) -> None:
    """
    Log structured entropy measurements for Morris reproduction.
    
    Args:
        logger: Configured logger instance
        experiment_name: Experiment identifier
        step: Training step number
        entropy_dict: Dictionary containing entropy measurements
        model_params: Number of model parameters
        dataset_size: Size of training dataset
        device: Device string
        
    Raises:
        ValueError: If entropy measurements are invalid
        KeyError: If required entropy keys are missing
    """
    required_keys = ["H_X", "H_X_given_theta", "H_X_given_theta_bar", "H_X_given_theta_theta_bar"]
    
    # Validate entropy measurements
    for key in required_keys:
        if key not in entropy_dict:
            raise KeyError(f"Missing required entropy measurement: {key}")
        if not isinstance(entropy_dict[key], (int, float)):
            raise ValueError(f"Entropy measurement {key} must be numeric")
        if entropy_dict[key] < 0:
            raise ValueError(f"Entropy measurement {key} cannot be negative")
    
    # Calculate derived metrics
    try:
        memorization = entropy_dict["H_X_given_theta_bar"] - entropy_dict["H_X_given_theta_theta_bar"]
        generalization = entropy_dict["H_X_given_theta"] - entropy_dict["H_X_given_theta_theta_bar"]
    except Exception as e:
        raise ValueError(f"Failed to calculate derived metrics: {str(e)}")
    
    log_entry = {
        "experiment_id": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "model_params": model_params,
        "dataset_size": dataset_size,
        "step": step,
        "entropy_measurements": entropy_dict,
        "derived_metrics": {
            "memorization": memorization,
            "generalization": generalization
        }
    }
    
    logger.info(json.dumps(log_entry))


def log_derived_metrics(
    logger: logging.Logger,
    experiment_name: str,
    step: int,
    memorization: float,
    generalization: float,
    training_loss: float,
    device: str
) -> None:
    """
    Log derived Morris metrics (memorization, generalization, loss).
    
    Args:
        logger: Configured logger instance
        experiment_name: Experiment identifier
        step: Training step number
        memorization: Memorization metric value
        generalization: Generalization metric value
        training_loss: Training loss value
        device: Device string
        
    Raises:
        ValueError: If metrics are invalid
    """
    # Validate metrics
    metrics = {
        "memorization": memorization,
        "generalization": generalization,
        "training_loss": training_loss
    }
    
    for name, value in metrics.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Metric {name} must be numeric")
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"Metric {name} must be finite")
    
    log_entry = {
        "experiment_id": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "step": step,
        "metrics_type": "derived",
        "derived_metrics": metrics
    }
    
    logger.info(json.dumps(log_entry))


def log_training_step(
    logger: logging.Logger,
    experiment_name: str,
    step: int,
    loss: float,
    entropy_measurements: Optional[Dict[str, float]] = None,
    device: str = "cpu"
) -> None:
    """
    Log training step with optional entropy measurements.
    
    Args:
        logger: Configured logger instance
        experiment_name: Experiment identifier
        step: Training step number
        loss: Training loss value
        entropy_measurements: Optional entropy measurements dict
        device: Device string
        
    Raises:
        ValueError: If parameters are invalid
    """
    if step < 0:
        raise ValueError("Training step cannot be negative")
    if not torch.isfinite(torch.tensor(loss)):
        raise ValueError("Training loss must be finite")
    
    log_entry = {
        "experiment_id": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "step": step,
        "training_loss": loss,
        "log_type": "training_step"
    }
    
    if entropy_measurements:
        log_entry["entropy_measurements"] = entropy_measurements
        
        # Calculate derived metrics if all required entropies present
        required_keys = ["H_X", "H_X_given_theta", "H_X_given_theta_bar", "H_X_given_theta_theta_bar"]
        if all(key in entropy_measurements for key in required_keys):
            try:
                memorization = (entropy_measurements["H_X_given_theta_bar"] - 
                              entropy_measurements["H_X_given_theta_theta_bar"])
                generalization = (entropy_measurements["H_X_given_theta"] - 
                                entropy_measurements["H_X_given_theta_theta_bar"])
                
                log_entry["derived_metrics"] = {
                    "memorization": memorization,
                    "generalization": generalization
                }
            except Exception as e:
                logger.warning(f"Failed to calculate derived metrics: {str(e)}")
    
    logger.info(json.dumps(log_entry))


def load_experiment_state(
    experiment_name: str,
    log_dir: str = "logs"
) -> Optional[Dict[str, Any]]:
    """
    Load the latest experiment state for resumable training.
    
    Args:
        experiment_name: Experiment identifier
        log_dir: Directory containing log files
        
    Returns:
        Latest experiment state dict or None if not found
        
    Raises:
        OSError: If log file cannot be read
        json.JSONDecodeError: If log file contains invalid JSON
    """
    try:
        log_file = Path(log_dir) / "entropy_measurements" / f"{experiment_name}_entropy.jsonl"
        
        if not log_file.exists():
            return None
        
        latest_state = None
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        # Extract JSON part from log line (after "INFO - ")
                        if " - INFO - " in line:
                            json_part = line.split(" - INFO - ", 1)[1].strip()
                        else:
                            json_part = line.strip()
                        
                        state = json.loads(json_part)
                        if "step" in state:
                            latest_state = state
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        return latest_state
        
    except Exception as e:
        raise OSError(f"Failed to load experiment state: {str(e)}")


def get_experiment_progress(
    experiment_name: str,
    log_dir: str = "logs"
) -> Dict[str, Any]:
    """
    Get experiment progress summary for Morris reproduction.
    
    Args:
        experiment_name: Experiment identifier
        log_dir: Directory containing log files
        
    Returns:
        Progress summary dictionary
        
    Raises:
        OSError: If log files cannot be accessed
    """
    try:
        latest_state = load_experiment_state(experiment_name, log_dir)
        
        if latest_state is None:
            return {
                "experiment_id": experiment_name,
                "status": "not_started",
                "latest_step": 0,
                "total_measurements": 0
            }
        
        log_file = Path(log_dir) / "entropy_measurements" / f"{experiment_name}_entropy.jsonl"
        total_measurements = 0
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        # Extract JSON part from log line (after "INFO - ")
                        if " - INFO - " in line:
                            json_part = line.split(" - INFO - ", 1)[1].strip()
                        else:
                            json_part = line.strip()
                        
                        entry = json.loads(json_part)
                        if "entropy_measurements" in entry:
                            total_measurements += 1
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        return {
            "experiment_id": experiment_name,
            "status": "in_progress",
            "latest_step": latest_state.get("step", 0),
            "latest_timestamp": latest_state.get("timestamp"),
            "device": latest_state.get("device"),
            "total_measurements": total_measurements,
            "latest_entropy": latest_state.get("entropy_measurements", {}),
            "latest_metrics": latest_state.get("derived_metrics", {})
        }
        
    except Exception as e:
        raise OSError(f"Failed to get experiment progress: {str(e)}")
