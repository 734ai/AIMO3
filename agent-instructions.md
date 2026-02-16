# Agent Instruction: AI Developer for AIMO3 Solver

## Objective

You are an AI agent tasked with **developing the full pipeline** for solving Olympiad-level math problems for the AIMO3 competition. Your pipeline must:

1. Parse LaTeX problems
2. Reason step-by-step using chain-of-thought
3. Compute numeric answers accurately (0–99,999)
4. Generate Kaggle submission files using the provided API

## Constraints

- All code must be **open-source compliant**
- Notebook runtime limits: CPU ≤ 9h, GPU ≤ 5h
- No internet access during Kaggle runs
- Answers must conform to competition format

## Workflow for Agent

1. **Data Processing**
   - Convert LaTeX problems into symbolic representations
   - Tokenize expressions for LLM input

2. **Reasoning Engine**
   - Use open-source LLMs (LLaMA 3, Mistral, WizardMath)
   - Apply chain-of-thought prompting to generate solution steps

3. **Computation Module**
   - Evaluate symbolic steps using SymPy/SageMath
   - Apply modular arithmetic and integer checks

4. **Output Generation**
   - Return integer answers for each problem
   - Format output according to Kaggle API requirements

5. **Testing & Evaluation**
   - Validate pipeline on public test problems
   - Ensure reproducibility for all runs

## Optional Enhancements

- Ensemble multiple LLMs for improved accuracy
- Implement reasoning verification checks
- Log reasoning steps for debugging and writeup preparation
