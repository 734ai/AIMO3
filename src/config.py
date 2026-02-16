"""
config.py - Configuration and Settings for AIMO3 Pipeline

Centralized configuration management for model, training, and runtime parameters.
"""

import os
from pathlib import Path


class BaseConfig:
    """Base configuration class."""
    
    # Project structure
    PROJECT_DIR = Path(__file__).parent.parent
    SRC_DIR = PROJECT_DIR / "src"
    NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
    DATA_DIR = PROJECT_DIR / "datasets"
    OUTPUT_DIR = PROJECT_DIR / "outputs"
    LOGS_DIR = PROJECT_DIR / "logs"
    
    # Create directories if they don't exist
    for dir_path in [OUTPUT_DIR, LOGS_DIR, DATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = LOGS_DIR / f"aimo3_{os.getenv('RUN_ID', 'default')}.log"


class ModelConfig(BaseConfig):
    """Model configuration."""
    
    # Model selection
    MODEL_NAME = "Open-Orca/orca_mini_3b"
    ALTERNATIVE_MODELS = [
        "mistralai/Mistral-7B",
        "meta-llama/Llama-2-7b-hf",
    ]
    
    # Device and precision
    DEVICE = "auto"  # or "cuda", "cpu"
    TORCH_DTYPE = "float16"  # or "float32"
    
    # Model loading
    LOAD_IN_8BIT = False
    LOAD_IN_4BIT = False
    DEVICE_MAP = "auto"
    
    # LoRA configuration (for fine-tuning)
    USE_LORA = False
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]


class ReasoningConfig(BaseConfig):
    """LLM reasoning configuration."""
    
    # Generation parameters
    MAX_TOKENS = 512
    MIN_TOKENS = 10
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    
    # Prompting
    USE_CHAIN_OF_THOUGHT = True
    NUM_REASONING_STEPS = 5
    
    # Sampling
    DO_SAMPLE = True
    REPETITION_PENALTY = 1.0
    
    # Beam search (alternative to sampling)
    NUM_BEAMS = 1  # Set > 1 for beam search


class ComputationConfig(BaseConfig):
    """Computation configuration."""
    
    # SymPy settings
    SIMPLIFY_EXPRESSIONS = True
    SYMPIFY_TRANSFORMATIONS = ('lambda', 'convert_xor', 'convert_associativity')
    
    # Answer constraints
    ANSWER_MIN = 0
    ANSWER_MAX = 99999
    
    # Verification
    VERIFY_WITH_SYMPY = True
    VERIFICATION_TIMEOUT = 5  # seconds


class KaggleConfig(BaseConfig):
    """Kaggle-specific configuration."""
    
    # Submission settings
    SUBMISSION_NAME = "AIMO3_Submission"
    SUBMISSION_FORMAT = "csv"  # problem_id, predicted_answer
    
    # Time limits (Kaggle constraints)
    CPU_TIMEOUT = 9 * 3600  # 9 hours
    GPU_TIMEOUT = 5 * 3600  # 5 hours
    
    # Data paths
    KAGGLE_DATA_PATH = "/kaggle/input"
    KAGGLE_OUTPUT_PATH = "/kaggle/output"
    
    # Problem format
    AIMO_MIN_PROBLEMS = 100
    AIMO_MAX_PROBLEMS = 200


class DataConfig(BaseConfig):
    """Data configuration."""
    
    # Dataset paths
    AIMO1_PATH = DATA_DIR / "aimo1_public.csv"
    AIMO2_PATH = DATA_DIR / "aimo2_public.csv"
    AIMO3_PUBLIC_PATH = DATA_DIR / "aimo3_public.csv"
    AIMO3_PRIVATE_PATH = DATA_DIR / "aimo3_private.csv"
    
    # Training data
    SYNTHETIC_PROBLEMS_PATH = DATA_DIR / "synthetic"
    
    # Column names
    PROBLEM_ID_COLUMN = "problem_id"
    PROBLEM_TEXT_COLUMN = "problem"
    LATEX_COLUMN = "latex_problem"
    ANSWER_COLUMN = "answer"
    
    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15


class TrainingConfig(BaseConfig):
    """Fine-tuning configuration (if needed)."""
    
    # Hyperparameters
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 16
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    
    # Save and eval
    SAVE_STRATEGY = "epoch"
    EVAL_STRATEGY = "epoch"
    LOGGING_STEPS = 10
    
    # Optimizer
    OPTIMIZER = "adamw_torch"
    MAX_GRAD_NORM = 1.0


class PipelineConfig(BaseConfig):
    """Pipeline execution configuration."""
    
    # Batch processing
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    
    # Retry logic
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # Logging
    LOG_EVERY_N_PROBLEMS = 10
    SAVE_INTERMEDIATE_RESULTS = True
    
    # Caching
    CACHE_EMBEDDINGS = False
    CACHE_DIR = PROJECT_DIR / ".cache"


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    
    # Debug mode
    DEBUG = False
    
    # Testing
    TEST_MODE = False
    TEST_PROBLEMS_COUNT = 5
    
    # Profiling
    PROFILE_CODE = False
    PROFILE_DIR = LOGS_DIR / "profiles"


# Configuration selection based on environment
def get_config(environment: str = None) -> BaseConfig:
    """
    Get configuration based on environment.
    
    Args:
        environment: Environment name ('development', 'test', 'production')
        
    Returns:
        Configuration object
    """
    if environment is None:
        environment = os.getenv("AIMO_ENV", "development")
    
    configs = {
        "development": DevelopmentConfig,
        "test": DevelopmentConfig,  # Test uses development config
        "production": BaseConfig
    }
    
    return configs.get(environment, BaseConfig)


# Convenience functions
class ConfigManager:
    """Unified configuration manager."""
    
    model = ModelConfig()
    reasoning = ReasoningConfig()
    computation = ComputationConfig()
    kaggle = KaggleConfig()
    data = DataConfig()
    training = TrainingConfig()
    pipeline = PipelineConfig()
    development = DevelopmentConfig()
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert all configs to dictionary."""
        return {
            "model": cls._config_to_dict(cls.model),
            "reasoning": cls._config_to_dict(cls.reasoning),
            "computation": cls._config_to_dict(cls.computation),
            "kaggle": cls._config_to_dict(cls.kaggle),
            "data": cls._config_to_dict(cls.data),
            "training": cls._config_to_dict(cls.training),
            "pipeline": cls._config_to_dict(cls.pipeline),
            "development": cls._config_to_dict(cls.development),
        }
    
    @staticmethod
    def _config_to_dict(config) -> dict:
        """Convert config object to dictionary."""
        return {
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


if __name__ == "__main__":
    # Display configuration
    import json
    config_dict = ConfigManager.to_dict()
    print(json.dumps(config_dict, indent=2, default=str))
