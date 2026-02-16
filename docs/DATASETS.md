# AIMO Dataset Download & Fine-tuning Guide

## Quick Start

### 1. Install Kaggle CLI
```bash
pip install kaggle
```

### 2. Get Kaggle Credentials
1. Visit: https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New API Token" → Downloads `kaggle.json`
4. Place it in your home directory:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download AIMO Datasets
```bash
cd /home/hssn/Documents/kaggle/ai\|mo

# Automatic download
python src/kaggle_downloader.py

# Or manual download from:
# https://www.kaggle.com/c/aimo-progress-prize-2024/data
# Then place CSV files in datasets/ directory
```

### 4. Verify Downloads
```bash
python src/task_3_1_datasets.py
```

Expected output:
```
✅ AIMO1: /home/hssn/Documents/kaggle/ai|mo/datasets/aimo1_problems.csv
✅ AIMO2: /home/hssn/Documents/kaggle/ai|mo/datasets/aimo2_problems.csv
✅ AIMO3: /home/hssn/Documents/kaggle/ai|mo/datasets/aimo3_public_test.csv
```

---

## Phase 3 Tasks

### Task 3.1: Download & Prepare Datasets ✅
**Status**: Framework ready, waiting for data

**Files**:
- `src/kaggle_downloader.py` - Automated download tool
- `src/task_3_1_datasets.py` - Dataset validation and management

**Commands**:
```bash
# Download datasets
python src/kaggle_downloader.py

# Validate and prepare
python src/task_3_1_datasets.py
```

---

### Task 3.2: Create Training Data
**Status**: Framework ready

**Files**:
- `src/task_3_2_3_fine_tuning.py` - Training data generation

**What it does**:
- Loads AIMO1 and AIMO2 problems
- Creates prompt-answer pairs
- Splits into train/val/test (80/10/10)
- Saves as JSONL format for training

**Command**:
```bash
python src/task_3_2_3_fine_tuning.py
```

**Output**: 
```
datasets/training/
├── train.jsonl     (training examples)
├── val.jsonl       (validation examples)
└── test.jsonl      (test examples)
```

---

### Task 3.3: Fine-tune Model
**Status**: Framework ready

**Files**:
- `src/task_3_2_3_fine_tuning.py` - Fine-tuning with LoRA
- `configs/fine_tuning_config.json` - Hyperparameters

**Configuration**:
```python
# Model settings
base_model = "gpt2"  # Can change to mistral-7b, llama-2-7b
use_lora = True      # Efficient fine-tuning
lora_rank = 8
lora_alpha = 16

# Training settings
batch_size = 4
num_epochs = 3
learning_rate = 5e-5
max_seq_length = 512
```

**Requirements**:
```bash
pip install peft transformers torch accelerate
```

**Command**:
```bash
python src/task_3_2_3_fine_tuning.py
```

---

## Troubleshooting

### Kaggle CLI Issues
```
Error: Kaggle CLI not installed
Solution: pip install kaggle

Error: Kaggle credentials not found
Solution: Place kaggle.json in ~/.kaggle/ with chmod 600
```

### Dataset Download Fails
```
Option 1: Manual download
- Visit: https://www.kaggle.com/c/aimo-progress-prize-2024/data
- Download aimo1_problems.csv, aimo2_problems.csv, aimo3_public_test.csv
- Place in datasets/ directory

Option 2: Command line
kaggle competitions download -c aimo-progress-prize-2024 -p datasets/
```

### Insufficient VRAM for Fine-tuning
```bash
# Reduce batch size in fine_tuning_config.json
"batch_size": 2

# Or use gradient accumulation
"gradient_accumulation_steps": 8

# Or use LoRA (already enabled)
"use_lora": true
```

---

## Expected Dataset Structure

After downloading, your `datasets/` folder should contain:

```
datasets/
├── aimo1_problems.csv          # 2023 AIMO problems (500+ problems)
├── aimo2_problems.csv          # 2024 AIMO problems (500+ problems)
├── aimo3_public_test.csv       # 2024 public test set (~100 problems)
└── training/
    ├── train.jsonl            # Generated after Task 3.2
    ├── val.jsonl
    └── test.jsonl
```

---

## Data Format

**AIMO CSV Format**:
```csv
problem,answer,difficulty,category
"Find the value of x...",42,medium,algebra
```

**Training JSONL Format**:
```json
{"source": "AIMO1", "problem": "...", "answer": "42", "prompt": "...", "target": " 42"}
```

---

## Progress Tracking

- [x] Phase 1: Core implementation (8 modules)
- [x] Phase 2: Data preparation (synthetic data, tests)
- [x] **Task 3.1**: Dataset download framework ✅
- [ ] **Task 3.2**: Training data generation (NEXT)
- [ ] **Task 3.3**: Model fine-tuning
- [ ] **Task 3.4**: Prompt optimization
- [ ] **Task 3.5**: Ensemble methods
- [ ] **Task 3.6**: Validation & submission

---

## Commands Cheat Sheet

```bash
# Download datasets
python src/kaggle_downloader.py

# Verify datasets
python src/task_3_1_datasets.py

# Generate training data
python src/task_3_2_3_fine_tuning.py

# Verify test suite still passes
python src/test_pipeline.py

# End-to-end demo
python src/demo_pipeline.py

# Run Kaggle notebook locally
jupyter notebook notebooks/aimo3_submission.ipynb
```

---

## Performance Expectations

**Training Time**:
- GPU (RTX 3090): ~2-4 hours for 3 epochs
- CPU: ~12-24 hours (not recommended)

**Expected Accuracy**:
- Base model (GPT-2): ~5-10% on AIMO3
- Fine-tuned model: ~15-25% on AIMO3
- With ensemble: ~20-35% on AIMO3

---

## Next Steps

1. ✅ Install Kaggle CLI
2. ✅ Set up credentials
3. 👉 **Run dataset download** (python src/kaggle_downloader.py)
4. 👉 **Verify download** (python src/task_3_1_datasets.py)
5. 👉 **Generate training data** (python src/task_3_2_3_fine_tuning.py)
6. 👉 **Start fine-tuning** (coming next)

---

For questions or issues, see [DEVELOPMENT.md](DEVELOPMENT.md) for additional debugging steps.
