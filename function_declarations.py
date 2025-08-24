# File: function_declarations.py
# Directory: ./
# File: function_declarations.py
# Directory: ./

"""
Central registry of all function signatures with explicit types and descriptions.
This serves as the contract for all modules in the information theory models project.
"""

from typing import Dict, Any, Tuple, Optional, List
import torch
import logging

# =============================================================================
# DEVICE MANAGEMENT (src/utils/device_manager.py)
# =============================================================================

def detect_device() -> str:
    """
    Detect the best available device for computation.
    
    Returns:
        str: Device type ('cpu', 'cuda', 'mps') based on availability
        
    Raises:
        RuntimeError: If no compatible device is found
        
    Example:
        device = detect_device()  # Returns 'cpu' on Mac, 'cuda' on GPU systems
    """

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
        
    Raises:
        ValueError: If device_type is not supported
        
    Example:
        config = get_device_config('cpu')
        # {'device': torch.device('cpu'), 'dtype': torch.float32, ...}
    """

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

# =============================================================================
# LOGGING MANAGEMENT (src/utils/logging_config.py)
# =============================================================================

def setup_logging(log_dir: str, experiment_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure comprehensive logging for experiments.
    
    Args:
        log_dir: Directory to store log files
        experiment_name: Name for this experiment run
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        OSError: If log directory cannot be created
        ValueError: If log_level is invalid
        
    Example:
        logger = setup_logging("logs", "gpt2_170k_experiment")
    """

def log_experiment_state(logger: logging.Logger, state: Dict[str, Any], checkpoint_path: str) -> None:
    """
    Log current experiment state for reproducibility.
    
    Args:
        logger: Logger instance from setup_logging()
        state: Dictionary containing experiment state
        checkpoint_path: Path where state will be/was saved
        
    Raises:
        TypeError: If state is not serializable
        OSError: If checkpoint_path is invalid
        
    Example:
        state = {'epoch': 10, 'loss': 2.3, 'model_params': 170000}
        log_experiment_state(logger, state, "checkpoints/epoch_10.pt")
    """

def load_experiment_state(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load previously saved experiment state.
    
    Args:
        checkpoint_path: Path to saved checkpoint file
        
    Returns:
        Dict containing loaded state, None if file doesn't exist
        
    Raises:
        RuntimeError: If checkpoint is corrupted
        OSError: If checkpoint_path exists but is unreadable
        
    Example:
        state = load_experiment_state("checkpoints/epoch_10.pt")
        if state: print(f"Resuming from epoch {state['epoch']}")
    """

# =============================================================================
# DATASET MANAGEMENT (src/data/dataset_manager.py)
# =============================================================================

def create_toy_dataset(vocab_size: int, seq_length: int, num_samples: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create uniform random dataset for memorization experiments (Morris Figure 1).
    
    Args:
        vocab_size: Size of vocabulary (e.g., 2048 for Morris reproduction)
        seq_length: Length of each sequence (e.g., 64)
        num_samples: Number of sequences in dataset
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (data, labels) tensors:
        - data: [num_samples, seq_length] of random tokens
        - labels: [num_samples, seq_length] shifted by 1 for language modeling
        
    Raises:
        ValueError: If any parameter is <= 0
        MemoryError: If dataset too large for available memory
        
    Example:
        data, labels = create_toy_dataset(2048, 64, 1000)
        # Creates 1000 sequences of 64 tokens each from vocab of 2048
    """

def validate_dataset(data: torch.Tensor, labels: torch.Tensor) -> bool:
    """
    Validate dataset tensors for training compatibility.
    
    Args:
        data: Input token sequences [batch_size, seq_length]
        labels: Target token sequences [batch_size, seq_length]
        
    Returns:
        bool: True if dataset is valid for training
        
    Raises:
        TypeError: If inputs are not torch.Tensor
        ValueError: If tensor shapes are incompatible
        
    Example:
        is_valid = validate_dataset(data, labels)  # True if shapes match
    """

def get_dataset_stats(data: torch.Tensor, vocab_size: int) -> Dict[str, float]:
    """
    Calculate dataset statistics for analysis.
    
    Args:
        data: Token sequences [batch_size, seq_length]
        vocab_size: Size of vocabulary
        
    Returns:
        Dict containing:
        - 'empirical_entropy': H(X) in bits
        - 'total_tokens': Total number of tokens
        - 'unique_tokens': Number of unique tokens used
        - 'vocab_coverage': Fraction of vocabulary used
        
    Raises:
        ValueError: If data contains tokens >= vocab_size
        
    Example:
        stats = get_dataset_stats(data, 2048)
        # {'empirical_entropy': 10.95, 'total_tokens': 64000, ...}
    """

# =============================================================================
# ENTROPY CALCULATIONS (src/metrics/entropy_calculator.py)
# =============================================================================

def calculate_empirical_entropy(data: torch.Tensor, vocab_size: int) -> float:
    """
    Calculate empirical entropy H(X) of dataset.
    
    Args:
        data: Token sequences [batch_size, seq_length]
        vocab_size: Total vocabulary size
        
    Returns:
        float: Empirical entropy in bits
        
    Raises:
        ValueError: If data contains invalid tokens
        RuntimeError: If calculation fails due to numerical issues
        
    Example:
        h_x = calculate_empirical_entropy(data, 2048)  # ~11.0 for uniform
    """

def calculate_conditional_entropy(data: torch.Tensor, model_probs: torch.Tensor, vocab_size: int) -> float:
    """
    Calculate conditional entropy H(X|θ) given model predictions.
    
    Args:
        data: True token sequences [batch_size, seq_length]
        model_probs: Model probability distributions [batch_size, seq_length, vocab_size]
        vocab_size: Total vocabulary size
        
    Returns:
        float: Conditional entropy H(X|θ) in bits
        
    Raises:
        ValueError: If shapes don't match or probs don't sum to 1
        RuntimeError: If numerical instability (log of 0)
        
    Example:
        h_x_given_theta = calculate_conditional_entropy(data, model_outputs, 2048)
    """

def calculate_unintended_memorization(h_x: float, h_x_given_theta_hat: float, h_x_given_theta_and_theta_hat: float) -> float:
    """
    Calculate unintended memorization following Morris et al. definition.
    
    Args:
        h_x: Empirical entropy H(X)
        h_x_given_theta_hat: Conditional entropy H(X|θ̂) from trained model
        h_x_given_theta_and_theta_hat: Joint conditional entropy H(X|θ,θ̂) from reference model
        
    Returns:
        float: Unintended memorization in bits
        
    Raises:
        ValueError: If any entropy value is negative
        
    Example:
        memorization = calculate_unintended_memorization(11.0, 5.2, 4.8)  # 0.4 bits
    """

def validate_probability_distribution(probs: torch.Tensor, tolerance: float = 1e-6) -> bool:
    """
    Validate that tensor represents valid probability distributions.
    
    Args:
        probs: Probability tensor [..., vocab_size]
        tolerance: Numerical tolerance for sum-to-1 check
        
    Returns:
        bool: True if valid probability distribution
        
    Raises:
        ValueError: If probabilities are negative or > 1
        
    Example:
        is_valid = validate_probability_distribution(model_probs, 1e-6)
    """

# =============================================================================
# ERROR HANDLING (src/utils/error_handlers.py)
# =============================================================================

def validate_function_inputs(func_name: str, **kwargs) -> None:
    """
    Validate function inputs with type checking and value ranges.
    
    Args:
        func_name: Name of calling function for error context
        **kwargs: Named arguments to validate
        
    Raises:
        TypeError: If argument types don't match expectations
        ValueError: If argument values are out of valid range
        
    Example:
        validate_function_inputs('create_toy_dataset', 
                               vocab_size=2048, seq_length=64, num_samples=1000)
    """

def handle_computation_error(error: Exception, context: str) -> None:
    """
    Handle computation errors with informative logging and recovery suggestions.
    
    Args:
        error: The original exception
        context: Description of what operation failed
        
    Raises:
        RuntimeError: Always, with enhanced error message
        
    Example:
        try:
            result = some_computation()
        except Exception as e:
            handle_computation_error(e, "entropy calculation")
    """

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Morris et al. reproduction parameters
MORRIS_SMALLEST_MODEL_PARAMS: int = 170_000  # 170K parameter model
MORRIS_VOCAB_SIZE: int = 2048
MORRIS_SEQ_LENGTH: int = 64
MORRIS_BITS_PER_PARAMETER: float = 3.6
MORRIS_DATASET_SIZES: List[int] = [100, 316, 1000, 3162, 10000, 31623, 100000]  # Log scale

# Supported model architectures
SUPPORTED_ARCHITECTURES: List[str] = ['gpt2', 'mamba']

# Device-specific settings
CPU_BATCH_SIZE: int = 32
GPU_BATCH_SIZE: int = 128
CPU_NUM_WORKERS: int = 2
GPU_NUM_WORKERS: int = 4
