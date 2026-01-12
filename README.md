# Multilingual LLM Symmetry Benchmark

This project explores whether a language model produces **consistent answers** across languages.  
Given an English prompt and its French equivalent, the workflow:

1. Generates several responses in each language using **Cohere's APIs** (primarily the Aya family of models).
2. Embeds all responses into a shared multilingual embedding space using **Cohere's multilingual embedding model** (embed-multilingual-v3.0).
3. Compares the two distributions using a **Sliced Kolmogorov–Smirnov (S-KS)** distance.
4. Averages results over many random projection directions to obtain a symmetry score with uncertainty.

The goal is simply to provide a clear, reproducible way to observe multilingual consistency in practice.

## Statistical Treatment
The Kolmogorov–Smirnov (K-S) metric measures how different two probability distributions are by looking at the maximum gap between their cumulative curves. It’s a simple, normalized and non-parametric way to quantify how far apart two distributions are, without making assumptions about their shape. Lower values indicate higher compatibility, higher values hint at different underlying distributions.

A sliced K-S metric compares two sets of embeddings by projecting them onto many random directions and measuring how different their 1-D distributions are. Averaging over these projections gives a single score (with an associated uncertainty) that reflects how similar the two original high-dimensional distributions are.

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

3. Create a `.env` file with your Cohere API key:

   ```bash
   COHERE_API_KEY=your_api_key_here
   ```

4. Run the notebook to:
   - sample generations in EN/FR using Cohere's APIs,
   - embed with Cohere's multilingual embedding model,
   - compute symmetry scores (mean ± CI).

---

## Prompts

The benchmark includes simple, neutral prompts:

- **Factual:** penicillin discovery, capital of Japan, largest planet  
- **Open-ended:** morning routine ideas, small habits, relaxing activities

These provide a clean test bed for observing cross-language variation.

---

## Repository Structure

```
cohere-multilingual-symmetry.ipynb  # Main notebook
stats_helpers.py                     # Statistical utilities
Pipfile                              # Dependencies
README.md
.env                                 # API key (not tracked in git)
```

You can extend the prompt set, add more languages (including out-of-distribution languages like Inuktitut), or test additional Cohere models by editing the notebook.
