# Quick Reference Guide

## 📂 Project Structure at a Glance

```
📦 AIMO3-Solver/
├── 📁 src/                    🔧 All source code modules
│   ├── preprocessing.py       # LaTeX/PDF/text parsing
│   ├── reasoning.py           # LLM chain-of-thought
│   ├── computation.py         # SymPy symbolic math
│   ├── postprocessing.py      # Output formatting
│   ├── pipeline.py            # Main orchestrator
│   ├── utils.py               # Helper functions
│   ├── config.py              # Configuration
│   └── __init__.py            # Package setup
├── 📁 notebooks/
│   └── aimo3_submission.ipynb  # 🎯 Kaggle submission notebook
├── 📁 datasets/               # 📊 Input data (download here)
├── 📁 outputs/                # 📤 Generated submissions
├── 📁 logs/                   # 📝 Execution logs
├── 📄 requirements.txt         # Dependencies
├── 📄 setup.py                # Installation
├── 📄 .env.template           # Environment config
├── 📄 DEVELOPMENT.md          # Developer guide
├── 📄 PROJECT_SUMMARY.md      # 📋 This project overview
└── 📄 TODO.md                 # Development roadmap
```

---

## 🚀 Quick Start

### 1. Setup (5 minutes)
```bash
cd /path/to/aimo3-solver

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.template .env
# Edit .env with your settings

# Install package in development mode
pip install -e .
```

### 2. Test Pipeline (10 minutes)
```python
from src.pipeline import AIMO3Pipeline

pipeline = AIMO3Pipeline()
result = pipeline.solve_single_problem(
    "Compute 2 + 3 × 5",
    problem_id="Test1"
)
print(result['final_answer'])  # Should output 17
```

### 3. Run Kaggle Notebook (variable time)
- Upload notebook to Kaggle
- Or run locally: `jupyter notebook notebooks/aimo3_submission.ipynb`

---

## 📚 Core Modules Overview

| Module | Purpose | Key Functions |
|--------|---------|---|
| `preprocessing.py` | Parse input | `latex_to_text()`, `pdf_to_text()` |
| `reasoning.py` | LLM reasoning | `LLMSolver.solve()` |
| `computation.py` | Math verification | `SymbolicCompute.evaluate_expression()` |
| `postprocessing.py` | Format output | `SubmissionFormatter.save_submission_csv()` |
| `pipeline.py` | Orchestration | `AIMO3Pipeline.solve_batch()` |
| `utils.py` | Utilities | `Evaluator.accuracy()`, `setup_logging()` |
| `config.py` | Configuration | `ModelConfig`, `KaggleConfig` |

---

## ⚙️ Configuration

### Important Settings (src/config.py)

```python
# Model choice
ModelConfig.MODEL_NAME = "Open-Orca/orca_mini_3b"

# Generation params
ReasoningConfig.MAX_TOKENS = 512
ReasoningConfig.TEMPERATURE = 0.7

# Kaggle limits
KaggleConfig.CPU_TIMEOUT = 9 * 3600    # 9 hours
KaggleConfig.GPU_TIMEOUT = 5 * 3600    # 5 hours

# Answer range
ComputationConfig.ANSWER_MIN = 0
ComputationConfig.ANSWER_MAX = 99999
```

---

## 🔄 Typical Workflow

### Single Problem
```python
pipeline = AIMO3Pipeline()
result = pipeline.solve_single_problem(problem_text)
print(f"Answer: {result['final_answer']}")
```

### Batch Processing
```python
problems = ["Problem 1", "Problem 2", ...]
result = pipeline.solve_batch(
    problems,
    problem_ids=["P1", "P2", ...],
    save_results=True  # Saves to outputs/ and logs/
)
```

### From CSV
```python
result = pipeline.solve_from_csv(
    "datasets/aimo3_public.csv",
    problem_column="latex_problem",
    id_column="problem_id"
)
```

---

## 📊 Output Files

After processing, you'll find:

```
outputs/
├── submission.csv           # Kaggle submission format
logs/
├── reasoning_steps.log      # Full reasoning for each problem
├── detailed_results.json    # Complete results with reasoning
├── statistics_*.json        # Statistics and metrics
└── aimo3_*.log             # Execution logs
```

---

## 🧪 Testing Examples

### Test Preprocessing
```python
from src.preprocessing import latex_to_text
result = latex_to_text(r"$\frac{1}{2} + \frac{1}{3}$")
print(result)  # "frac 1 2 + frac 1 3"
```

### Test Reasoning
```python
from src.reasoning import LLMSolver
solver = LLMSolver("Open-Orca/orca_mini_3b")
result = solver.solve("What is 2 + 2?")
```

### Test Computation
```python
from src.computation import SymbolicCompute
compute = SymbolicCompute()
result = compute.evaluate_expression("2**10")  # 1024
```

### Test Validation
```python
from src.computation import AnswerValidator
validator = AnswerValidator()
answer = validator.validate_integer(150000)  # Clamps to 99999
```

---

## 🐛 Debugging

### Enable Debug Logging
```python
from src.utils import setup_logging
logger = setup_logging(log_level="DEBUG")
```

### Check Execution Logs
```bash
ls -lh logs/
tail -f logs/aimo3_*.log
```

### Profile Code Performance
```python
from src.utils import timer

@timer
def my_function():
    # Your code here
    pass
```

---

## 📈 Performance Tips

1. **Reduce Token Usage**
   - Shorter, focused prompts
   - `MAX_TOKENS = 256` for faster inference

2. **Batch Processing**
   - Process multiple problems together
   - Better GPU utilization

3. **Model Selection**
   - Smaller models = faster (3B parameters)
   - Larger models = better quality (7B+ parameters)

4. **Caching**
   - Cache LLM outputs
   - Reuse computed results

---

## 🎯 Competition Integration

### For Kaggle Submission
1. **Use the notebook**: `notebooks/aimo3_submission.ipynb`
2. **Upload to Kaggle** as new notebook
3. **Set it as submission**: Enable in notebook settings
4. **Monitor leaderboard**: Check score after submission

### Expected Kaggle Output
```csv
problem_id,predicted_answer
P1,42
P2,100
P3,17
...
```

---

## 🔗 Resources

- **Kaggle Competition**: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3
- **HuggingFace Models**: https://huggingface.co/models
- **SymPy Docs**: https://docs.sympy.org/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Transformers Docs**: https://huggingface.co/docs/transformers/

---

## ✅ Checklist Before First Run

- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] GPU available (optional): `nvidia-smi`
- [ ] `.env` file created and configured
- [ ] `datasets/` directory ready (will populate with data)
- [ ] `outputs/` directory exists (auto-created)
- [ ] `logs/` directory exists (auto-created)

---

## 📞 Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'src'`
**Solution**: 
```bash
pip install -e .
# or run from project root directory
```

### Issue: `CUDA out of memory`
**Solution**:
- Reduce `MAX_TOKENS` in config
- Use smaller model
- Enable 4-bit quantization

### Issue: Slow inference
**Solution**:
- Profile with @timer decorator
- Reduce sequence length
- Use faster model
- Batch processing

### Issue: No answers generated
**Solution**:
- Check `logs/` for error messages
- Enable DEBUG mode
- Verify LLM is loaded correctly

---

## 🎓 Next Steps

1. **Week 1**: Get data working, test pipeline
2. **Week 2**: Fine-tune model, optimize prompts
3. **Week 3**: Generate first submission
4. **Week 4+**: Iterate, improve, reach leaderboard top

---

## 📄 Documentation Files

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Complete developer guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Detailed project overview
- **[TODO.md](TODO.md)** - Development roadmap and checklist
- **README.md** - Project overview

---

## 💡 Pro Tips

✅ **Always use the pipeline** - Don't call modules directly if possible
✅ **Check logs** - They contain useful debugging info
✅ **Use config.py** - Don't hardcode values
✅ **Monitor progress** - Use tqdm and logging
✅ **Save results** - Always save submissions and logs
✅ **Version control** - Commit working versions

---

**Status**: Phase 1 Complete ✅ | Ready for Phase 2 🚀

*All code tested and documented. Ready to download data and begin training!*
