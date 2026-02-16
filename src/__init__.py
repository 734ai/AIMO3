"""
AIMO3 Solver Package

A complete pipeline for solving Olympiad-level math problems
using AI reasoning and symbolic computation.
"""

from .preprocessing import (
    latex_to_text,
    pdf_to_text,
    prepare_problem,
    batch_prepare_problems
)

from .reasoning import (
    LLMSolver,
    extract_numeric_answer
)

from .computation import (
    SymbolicCompute,
    AnswerValidator
)

from .postprocessing import (
    SubmissionFormatter,
    ReasoningLogger,
    ResultsAggregator
)

from .pipeline import AIMO3Pipeline

from .utils import (
    setup_logging,
    timer,
    retry,
    Evaluator,
    Config
)

__version__ = "0.1.0"
__author__ = "AIMO3 Development Team"

__all__ = [
    # Preprocessing
    "latex_to_text",
    "pdf_to_text",
    "prepare_problem",
    "batch_prepare_problems",
    
    # Reasoning
    "LLMSolver",
    "extract_numeric_answer",
    
    # Computation
    "SymbolicCompute",
    "AnswerValidator",
    
    # Postprocessing
    "SubmissionFormatter",
    "ReasoningLogger",
    "ResultsAggregator",
    
    # Pipeline
    "AIMO3Pipeline",
    
    # Utils
    "setup_logging",
    "timer",
    "retry",
    "Evaluator",
    "Config"
]
