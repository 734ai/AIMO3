# Project Structure: AIMO3 Solver

This document outlines the folder and file structure for the AIMO3 Solver project.

AIMO3-Solver/
│
├── README.md # Project overview, goals, and usage
├── TODO.md # Task list and development roadmap
├── project-information.md # Detailed project description
├── agent-instruction.md # Instructions for AI agent development
├── requirements.md # Python dependencies and libraries
├── project-structure.md # This file
│
├── notebooks/ # Kaggle submission-ready notebooks
│ ├── aimo3_submission.ipynb # Starter notebook for solving problems
│ └── templates/ # Optional notebook templates
│
├── datasets/ # Training, validation, and test data
│ ├── aimo1_public.csv
│ ├── aimo2_public.csv
│ └── synthetic/ # Generated synthetic problems for fine-tuning
│
├── src/ # Source code for pipeline
│ ├── preprocessing.py # LaTeX parser, tokenizer
│ ├── reasoning.py # LLM reasoning engine
│ ├── computation.py # SymPy/SageMath solver
│ ├── postprocessing.py # Answer formatting and validation
│ └── utils.py # Logging, evaluation, helper functions
│
├── outputs/ # Generated answers, Kaggle submission files
│ └── submission.csv
│
└── logs/ # Logs from training, evaluation, or Kaggle runs
└── reasoning_steps.log



**Notes:**

- `notebooks/` is where all Kaggle notebook submissions will be stored.
- `datasets/` contains both **real AIMO data** and **synthetic training data** for fine-tuning.
- `src/` contains all modularized Python code for preprocessing, reasoning, computation, and postprocessing.
- `outputs/` will hold submission files ready for Kaggle evaluation.
- `logs/` captures intermediate reasoning steps, useful for debugging and writeups.
