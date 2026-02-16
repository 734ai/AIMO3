# AIMO3 Starter Notebook
# Kaggle submission-ready skeleton

# -----------------------------
# 1. Install & Import Dependencies
# -----------------------------
!pip install --quiet sympy sagemath torch transformers accelerate peft datasets tqdm

import numpy as np
import pandas as pd
import sympy as sp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json, re, os

# -----------------------------
# 2. Load Model
# -----------------------------
model_name = "Open-Orca/orca_mini_3b"  # Example open-source model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# -----------------------------
# 3. Load Dataset (Public Test Set)
# -----------------------------
data_path = "../datasets/aimo3_public.csv"
df = pd.read_csv(data_path)

# -----------------------------
# 4. Preprocessing
# -----------------------------
def latex_to_text(latex_expr):
    """
    Convert LaTeX expressions into a plain-text symbolic format
    suitable for LLM input.
    """
    # Minimal preprocessing example
    text = re.sub(r"\\\\", "", latex_expr)
    text = re.sub(r"\$", "", text)
    return text

df["problem_text"] = df["latex_problem"].apply(latex_to_text)

# -----------------------------
# 5. Reasoning / LLM Prompt
# -----------------------------
def solve_with_llm(problem_text, max_tokens=512):
    prompt = f"""
    Solve the following math problem step-by-step and provide the final integer answer only (0-99999):
    Problem: {problem_text}
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -----------------------------
# 6. Computation / Validation (Optional)
# -----------------------------
def parse_numeric_answer(raw_answer):
    """
    Extract numeric answer from LLM output
    """
    match = re.search(r"\d+", raw_answer)
    return int(match.group()) if match else 0

# -----------------------------
# 7. Generate Answers
# -----------------------------
predictions = []
for problem in tqdm(df["problem_text"]):
    llm_output = solve_with_llm(problem)
    answer = parse_numeric_answer(llm_output)
    predictions.append(answer)

df["predicted_answer"] = predictions

# -----------------------------
# 8. Prepare Kaggle Submission
# -----------------------------
submission = df[["problem_id", "predicted_answer"]]
submission.to_csv("../outputs/submission.csv", index=False)
print("Submission file ready: ../outputs/submission.csv")
