# Multilingual LLM Symmetry Benchmark

This project explores whether a language model produces **consistent answers** across languages.  
Given an English prompt and its French equivalent, the workflow:

1. Generates several responses in each language using a **local Llama 3.1 8B Instruct GGUF model** (via `llama-cpp-python` and Apple’s MPS backend).
2. Embeds all responses into a shared multilingual embedding space using **LaBSE**.
3. Compares the two distributions using a **Sliced Kolmogorov–Smirnov (S-KS)** distance.
4. Averages results over many random projection directions to obtain a symmetry score with uncertainty.

The goal is simply to provide a clear, reproducible way to observe multilingual consistency in practice.

A sliced K-S metric compares two sets of embeddings by projecting them onto many random directions and measuring how different their 1-D distributions are. Averaging over these projections gives a single score (plus associated uncertainty) that reflects how similar the two original high-dimensional distributions are.

---

## Workflow

1. Install dependencies and activate the environment:

   ```bash
   pipenv install
   pipenv shell
   ```

2. Register the Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name=multilingual-llm-symmetry
   ```

3. Download the quantized GGUF model (Llama 3.1 8B Instruct Q4_K_M) using the setup cell in the notebook.

4. Run the notebook to:
   - sample generations in EN/FR,
   - embed with LaBSE,
   - compute symmetry scores (mean ± CI).

Everything runs locally on a MacBook using the MPS backend.

---

## Prompts

The benchmark includes simple, neutral prompts:

- **Factual:** penicillin discovery, capital of Japan, largest planet  
- **Open-ended:** morning routine ideas, small habits, relaxing activities

These provide a clean test bed for observing cross-language variation.

---

## Repository Structure

```
models/                       # downloaded GGUF model
notebooks/
  multilingual_symmetry_sliced_ks.ipynb
Pipfile
README.md
```

You can extend the prompt set, add more languages, or test additional models by editing the notebook.
