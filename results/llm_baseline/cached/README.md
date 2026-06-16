# Cached LLM baseline predictions

Place cached LLM clustering outputs here for offline, reproducible evaluation.

## Expected format

Filename:
- `<system>_llm_prompt_partition.json` (e.g. `daytrader_llm_prompt_partition.json`)

JSON schema (STRICT):
- `{ "<FQCN>": "<cluster_id>" }`
  - Keys are **fully-qualified class names**.
  - Values are any JSON scalar representing a cluster ID (string or int).

## DayTrader note (Table I)

- DayTrader uses **Total=100** classes (Table I). The LLM baseline clusters all 100.

## How to generate + evaluate (DayTrader)

1. Generate prompt:
   - `python scripts/llm_baseline/export_daytrader_llm_baseline_inputs.py --system daytrader`
   - Prompt: `results/llm_baseline/prompts/daytrader_prompt_gpt4o.txt`
2. Paste into GPT-4o and save model output as:
   - `results/llm_baseline/outputs/daytrader_llm_partition.json`
     - recommended format: `{ serviceName -> [FQCN...] }`
3. Convert to cached format:
   - `python scripts/llm_baseline/convert_llm_service_to_cached_partition.py --system daytrader`
   - Output: `results/llm_baseline/cached/daytrader_llm_prompt_partition.json`
4. Evaluate:
   - `python scripts/multimodal/phase4/run_llm_baseline_cached.py daytrader`
