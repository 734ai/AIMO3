# Project Information: AIMO3 Solver

## Overview

This project aims to **solve Olympiad-level math problems** using AI for the **AI Mathematical Olympiad – Progress Prize 3 (AIMO3)**. Problems span:

- Algebra
- Combinatorics
- Geometry
- Number Theory

## Goals

1. Build an **AI reasoning pipeline** capable of solving problems with high accuracy.
2. Ensure **open-source reproducibility** for submissions.
3. Maximize **private and public leaderboard performance** on Kaggle.
4. Optionally contribute to **Math Corpus Prize** or **Writeup Prize**.

## Key Components

- **LLM Reasoning**: Open-source models to parse and reason about LaTeX problems.
- **Symbolic Computation**: Python libraries (SymPy/SageMath) to compute precise numeric answers.
- **Prompting & Fine-Tuning**: Enhance model performance using Olympiad problem datasets.
- **Submission Integration**: Pipeline outputs compatible with Kaggle evaluation API.

## Expected Output

- Single integer per problem (0–99,999)
- Adheres to Kaggle notebook runtime and submission rules
- Reproducible and documented pipeline

## Success Metrics

- Accuracy on public test set
- Penalized accuracy on private test set
- Ability to solve “AI-hard” problems reliably
