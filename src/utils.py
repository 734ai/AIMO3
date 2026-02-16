"""
utils.py - Utility Functions and Helpers

Provides logging configuration, evaluation metrics, helper functions,
and other utilities for the AIMO3 pipeline.
"""

import os
import logging
import time
from typing import Dict, List, Any, Callable
from functools import wraps
from datetime import datetime


# ============================================================
# Logging Configuration
# ============================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("AIMO3")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is None:
        log_file = os.path.join(
            log_dir,
            f"aimo3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# Decorators and Timing
# ============================================================

def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = logging.getLogger("AIMO3")
        logger.debug(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        
    Returns:
        Wrapped function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("AIMO3")
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            
            logger.error(f"All {max_attempts} attempts failed")
            raise last_error
        
        return wrapper
    return decorator


# ============================================================
# Evaluation Metrics
# ============================================================

class Evaluator:
    """
    Evaluation metrics for predictions.
    """
    
    @staticmethod
    def accuracy(predictions: List[int], ground_truth: List[int]) -> float:
        """
        Calculate accuracy of predictions.
        
        Args:
            predictions: List of predicted answers
            ground_truth: List of ground truth answers
            
        Returns:
            Accuracy as fraction (0-1)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("predictions and ground_truth must have same length")
        
        if len(predictions) == 0:
            return 0.0
        
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)
    
    
    @staticmethod
    def penalized_accuracy(
        predictions: List[int],
        ground_truth_public: List[int],
        ground_truth_private: List[int],
        public_weight: float = 0.5
    ) -> float:
        """
        Calculate Kaggle-style penalized accuracy.
        
        1 point if both public and private correct
        0.5 points if one is correct
        0 points if both wrong
        
        Args:
            predictions: List of predictions
            ground_truth_public: List of public test ground truth
            ground_truth_private: List of private test ground truth
            public_weight: Weight for public accuracy
            
        Returns:
            Penalized accuracy score
        """
        if len(predictions) != len(ground_truth_public) or \
           len(predictions) != len(ground_truth_private):
            raise ValueError("All lists must have same length")
        
        if len(predictions) == 0:
            return 0.0
        
        score = 0
        for pred, public, private in zip(
            predictions, ground_truth_public, ground_truth_private
        ):
            if pred == public and pred == private:
                score += 1.0
            elif pred == public or pred == private:
                score += 0.5
        
        return score / len(predictions)
    
    
    @staticmethod
    def report(
        predictions: List[int],
        ground_truth: List[int],
        problem_ids: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed evaluation report.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth answers
            problem_ids: Optional list of problem IDs
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("predictions and ground_truth must have same length")
        
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        total = len(predictions)
        
        report = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "incorrect": total - correct,
            "incorrect_rate": (total - correct) / total if total > 0 else 0,
        }
        
        # Detailed results if problem IDs provided
        if problem_ids:
            report["details"] = []
            for pid, pred, true in zip(problem_ids, predictions, ground_truth):
                report["details"].append({
                    "problem_id": pid,
                    "prediction": pred,
                    "ground_truth": true,
                    "correct": pred == true
                })
        
        return report


# ============================================================
# File and Path Utilities
# ============================================================

def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Directory path
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_file_size(path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        path: File path
        
    Returns:
        Human-readable size string
    """
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def list_files(directory: str, pattern: str = "*") -> List[str]:
    """
    List files in directory matching pattern.
    
    Args:
        directory: Directory path
        pattern: File pattern (e.g., "*.txt")
        
    Returns:
        List of matching file paths
    """
    import glob
    return glob.glob(os.path.join(directory, pattern))


# ============================================================
# Configuration and Constants
# ============================================================

class Config:
    """
    Configuration constants for AIMO3 pipeline.
    """
    
    # Answer constraints
    ANSWER_MIN = 0
    ANSWER_MAX = 99999
    
    # Model settings
    DEFAULT_MODEL = "Open-Orca/orca_mini_3b"
    DEFAULT_DEVICE = "auto"
    
    # LLM parameters
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    
    # Kaggle constraints
    KAGGLE_CPU_TIMEOUT = 9 * 3600  # 9 hours
    KAGGLE_GPU_TIMEOUT = 5 * 3600  # 5 hours
    
    # Directories
    DEFAULT_OUTPUT_DIR = "outputs"
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_DATA_DIR = "datasets"


# ============================================================
# String Utilities
# ============================================================

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_number(number: float, decimals: int = 2) -> str:
    """
    Format number with specific decimal places.
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{number:.{decimals}f}"


# ============================================================
# Data Processing Utilities
# ============================================================

def flatten_list(nested_list: List[List]) -> List:
    """
    Flatten nested list.
    
    Args:
        nested_list: Nested list
        
    Returns:
        Flattened list
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


if __name__ == "__main__":
    # Example usage
    logger = setup_logging(log_level="INFO")
    logger.info("Utilities module loaded")
    
    # Test evaluator
    predictions = [1, 2, 3, 4, 5]
    ground_truth = [1, 2, 3, 5, 6]
    
    eval_report = Evaluator.report(predictions, ground_truth)
    print("Evaluation Report:", eval_report)
