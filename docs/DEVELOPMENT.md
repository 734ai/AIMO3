# AIMO3 Solver Development Guide

## Quick Start

### 1. Clone and Install

```bash
cd /path/to/aimo3-solver
pip install -r requirements.txt
pip install -e .
```

### 2. Set Up Environment

```bash
cp .env.template .env
# Edit .env with your configuration
```

### 3. Run a Single Problem

```python
from src.pipeline import AIMO3Pipeline

pipeline = AIMO3Pipeline()
result = pipeline.solve_single_problem(
    "Compute 2 + 3 × 5",
    problem_id="Example1"
)
print(f"Answer: {result['final_answer']}")
```

### 4. Batch Processing

```python
problems = [
    "Compute 2 + 3 × 5",
    "Solve 2x + 5 = 13",
]

batch_result = pipeline.solve_batch(
    problems,
    problem_ids=["P1", "P2"],
    save_results=True
)
```

## Module Overview

### preprocessing.py
- `latex_to_text()` - Convert LaTeX to plain text
- `pdf_to_text()` - Extract text from PDFs
- `prepare_problem()` - Universal problem formatter

### reasoning.py
- `LLMSolver` - Chain-of-thought reasoning with LLMs
- `extract_numeric_answer()` - Extract final answer from reasoning

### computation.py
- `SymbolicCompute` - SymPy-based symbolic computation
- `AnswerValidator` - Validate answer format and range

### postprocessing.py
- `SubmissionFormatter` - Save Kaggle-format submissions
- `ReasoningLogger` - Log detailed reasoning steps
- `ResultsAggregator` - Aggregate and analyze results

### pipeline.py
- `AIMO3Pipeline` - Complete end-to-end pipeline

### utils.py
- `setup_logging()` - Configure logging
- `Evaluator` - Compute accuracy metrics
- `Config` - Configuration constants

## Workflow

### Phase 1: Environment Setup ✅
- [x] Create directory structure
- [x] Install dependencies
- [x] Configure logging

### Phase 2: Data Preparation
- [ ] Download AIMO1/AIMO2 datasets
- [ ] Parse LaTeX problems
- [ ] Create validation set
- [ ] Generate synthetic data (optional)

### Phase 3: Model Development
- [ ] Load open-source LLM
- [ ] Implement chain-of-thought prompting
- [ ] Fine-tune on Olympiad problems
- [ ] Add ensemble logic

### Phase 4: Computation Pipeline
- [ ] Integrate SymPy/SageMath
- [ ] Handle modular arithmetic
- [ ] Implement post-processing

### Phase 5: Kaggle Integration
- [ ] Test with Kaggle API
- [ ] Validate output format
- [ ] Optimize for time limits

### Phase 6: Optimization
- [ ] Profile code execution
- [ ] Reduce token usage
- [ ] Ensure reproducibility

### Phase 7: Auxiliary Prizes
- [ ] Prepare writeup
- [ ] Document methodology
- [ ] Submit datasets

## Key Files Structure

```
aimo3-solver/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Input parsing
│   ├── reasoning.py          # LLM reasoning
│   ├── computation.py        # Symbolic computation
│   ├── postprocessing.py     # Output formatting
│   ├── pipeline.py           # Main pipeline
│   ├── utils.py              # Helper functions
│   └── config.py             # Configuration
├── notebooks/
│   └── aimo3_submission.ipynb # Kaggle submission notebook
├── datasets/                  # Data files
├── outputs/                   # Generated submissions
├── logs/                      # Execution logs
├── requirements.txt
├── setup.py
├── .env.template
└── README.md
```

## Configuration

Edit `src/config.py` to customize:
- Model selection (MODEL_NAME)
- LLM parameters (MAX_TOKENS, TEMPERATURE)
- Answer constraints (ANSWER_MIN, ANSWER_MAX)
- Kaggle settings (CPU/GPU timeouts)
- Data paths

## Testing

```python
# Test preprocessing
from src.preprocessing import latex_to_text
result = latex_to_text(r"$2 + 3 \times 5$")

# Test reasoning
from src.reasoning import LLMSolver
solver = LLMSolver()
result = solver.solve("Compute 2 + 3 × 5")

# Test computation
from src.computation import SymbolicCompute
compute = SymbolicCompute()
result = compute.evaluate_expression("2 + 3 * 5")

# Test complete pipeline
from src.pipeline import AIMO3Pipeline
pipeline = AIMO3Pipeline()
result = pipeline.solve_single_problem("Compute 2 + 3 × 5")
```

## Debugging

Enable debug logging:

```python
from src.utils import setup_logging
logger = setup_logging(log_level="DEBUG")
```

Check logs in `logs/` directory with detailed execution traces.

## Performance Optimization

1. **Reduce Token Usage**
   - Shorter prompts
   - Guided generation with constraints

2. **Cache Results**
   - Cache embeddings
   - Reuse model outputs

3. **Batch Processing**
   - Process multiple problems in parallel
   - Optimize GPU memory usage

4. **Inference Optimization**
   - Use quantization (int8/int4)
   - Use Flash Attention
   - Reduce model size with distillation

## Next Steps

1. Download AIMO1/AIMO2 datasets
2. Fine-tune LLM on Olympiad problems
3. Implement ensemble methods
4. Optimize for Kaggle runtime
5. Submit to competition

## Resources

- [AIMO3 Kaggle Competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [HuggingFace Models](https://huggingface.co/models)
- [SymPy Documentation](https://docs.sympy.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Support

For issues and improvements:
1. Check existing issues
2. Review logs in `logs/` directory
3. Enable DEBUG mode for detailed traces
4. Contact the development team

