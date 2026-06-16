# LLM Baseline (Prompt-based) - Cached Outputs for Reproducibility

This folder contains a lightweight **LLM baseline harness** designed for rebuttal-stage experiments:

- It supports **offline reproduction** by using **cached predictions** (JSON mappings) committed under `results/llm_baseline/cached/`.
- If you want to regenerate predictions, you can plug in your own LLM call (OpenAI/Claude/local model) and then export results into the same JSON format.

## Output format

A prediction file is a JSON object:

- key: fully-qualified class name (must match `data/processed/fusion/<system>_class_order.json` entries)
- value: an integer cluster/service id

Example:

```json
{
  "com.foo.A": 0,
  "com.foo.B": 0,
  "com.foo.C": 1
}
```

## Run (cached-only)

Use the Phase4 runner:

- `python scripts/multimodal/phase4/run_llm_baseline_cached.py`

The command above evaluates every cached prediction currently present under
`results/llm_baseline/cached/`. This artifact includes the DayTrader cached
baseline; pass a specific system name only after adding its cached partition.

This will:

1. Load cached predictions
2. Evaluate with the existing evaluator (`scripts/multimodal/phase4/evaluate_partition_f1.py`)
3. Write per-system and aggregated tables under `results/llm_baseline/`

## Prompting workflow (generate cached outputs)

A recommended prompt template is provided at:
- `scripts/llm_baseline/prompt_template.md`

Workflow:

1) Prepare the model input (classes + static deps). You can reuse:
   - class order: `data/processed/fusion/<system>_class_order.json`
   - dependency matrix: `data/processed/dependency/<system>_dependency_matrix.json` (if present)

2) Send the prompt to your chosen LLM and ask it to output the JSON mapping.

3) Save the JSON mapping to:
- `results/llm_baseline/cached/<system>_llm_prompt_partition.json`

4) Evaluate offline:
- `python scripts/multimodal/phase4/run_llm_baseline_cached.py <system>`

## Notes

- We intentionally keep this baseline minimal: *prompt -> partition*.
- The repository does **not** include API keys or network calls.
- Cached outputs allow reviewers to reproduce the comparison without external services.
