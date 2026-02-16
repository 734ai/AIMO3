# AIMO3 Solver: AI Mathematical Olympiad Progress Prize 3

This project aims to develop an **AI pipeline capable of solving international-level math problems** (AIMO3 competition) written in LaTeX. The goal is to combine **open-source LLM reasoning** with **symbolic computation** to produce accurate integer answers.

---

## Features

- LaTeX parsing and symbolic conversion
- Chain-of-thought reasoning using open-source LLMs
- SymPy / SageMath integration for symbolic and numeric computation
- Modular arithmetic and integer answer enforcement
- Submission-ready Kaggle notebook support

---

## Getting Started

1. Clone this repository.
2. Install dependencies listed in `requirements.md`.
3. Fine-tune or load an open-source LLM (e.g., LLaMA 3, Mistral, WizardMath).
4. Preprocess LaTeX problems into AI-readable format.
5. Run the pipeline to generate answers in the 0–99,999 range.
6. Submit predictions using Kaggle’s evaluation API.

---

## Competition Resources

- [AIMO3 Kaggle Competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- Reference problems PDFs for testing
- Fields Model Initiative / Tinker API (GPU & API credits)

---

## Project Structure

- `agent-instruction.md` – Instructions for AI agent to develop the pipeline
- `project-information.md` – Detailed project overview
- `todo.md` – Tasks and milestones
- `requirements.md` – Python dependencies
- `notebooks/` – Submission-ready Kaggle notebooks
- `datasets/` – Training / evaluation datasets

---

## License

All code and data follow open-source licensing where applicable. Any datasets used must adhere to competition rules.
