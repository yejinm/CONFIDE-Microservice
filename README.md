# CONFIDE - Reproducibility Package

This repository contains the artifact for:

**CONFIDE: Multimodal Conflict-aware Microservice Extraction via Evidential Deep Learning**

## Terminology note (Semantic Refiner vs DADE)

In the paper, we refer to the semantic post-processing component as **Semantic Refiner (SR)**.
In the implementation, the same component is historically named **DADE** (e.g., in script names, variables, and some output filenames such as `*sem_dade*`).
Unless otherwise stated:

- **SR (paper)** ~= **DADE (code)**

## What is included

- Source code for the CONFIDE pipeline (multi-modal similarity, **Semantic Refiner (SR)**, EDL uncertainty modeling, and conflict-aware clustering).
- A curated set of processed inputs required to reproduce the paper's main results (Table III/IV and main figures) under:
  - `data/processed/...`

## What is NOT redistributed

- Third-party subject systems (benchmark source code) and raw traces are **not** redistributed in this reproducibility package.
  See `data/raw/README.md` for upstream download links.

## Quickstart (copy/paste commands)

> All commands below assume **Windows + PowerShell** and are executed at the repository root.

### 1) Create an environment and install dependencies

```powershell
cd CONFIDE-Microservice

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Use the provided processed inputs

The repository already includes the canonical `data/processed/...` files used by the reproduction scripts. If you generate a separate paper snapshot later, `scripts/reproduce_paper.ps1` can copy it back into these canonical locations.

### 3) Reproduce Table III (overall comparison)

```powershell
# Step 1: generate Phase3 partitions (CAC + baselines)
python scripts\multimodal\phase3\phase3_cac_evaluation.py acmeair daytrader jpetstore plants

# Step 2: run baselines + ours and generate the paper table
python scripts\multimodal\phase4\run_mono_baselines_and_ours_table.py daytrader plants acmeair jpetstore
python scripts\multimodal\phase4\make_all_systems_mono_table.py
python scripts\multimodal\phase4\generate_paper_final_table.py
```

Expected key outputs:
- `results/ablation/baseline/paper_final_table.csv`
- `results/ablation/baseline/paper_final_table.md`

### 4) Reproduce Table IV (diagnosis)

```powershell
python scripts\multimodal\phase4\generate_table_IV_regress_diagnosis.py
```

Expected outputs:
- `results/artifact_tables/table_IV_regress_diagnosis.csv`
- `results/artifact_tables/table_IV_regress_diagnosis.md`

### 5) Reproduce main figures

```powershell
# Semantic smoothing (bar)
python scripts\multimodal\phase4\plot_semantic_smoothing_bar_median_iqr.py

# Semantic PDF before/after Semantic Refiner (SR)
python scripts\multimodal\phase4\plot_semantic_pdf_dade.py
```

Expected (common) figure output folders:
- `results/paper/` (when a script is invoked with `--paper_mode`)
- `results/plots/` (default for some plotting scripts)

Generated figures are written under:
- `results/paper/`
- `results/plots/`

### One-command reproduction (recommended)

After installing dependencies, you can reproduce **Table III/IV + main figures** with a single PowerShell command:

```powershell
.\scripts\reproduce_paper.ps1 -Tag paper_v1
```

This script will use the canonical `data/processed/...` inputs already present in the repository, or copy a snapshot first if `results/paper_snapshot/<tag>/` exists.

## Optional: regenerate a snapshot

Only needed if you regenerated canonical inputs locally and want to record hashes:

```powershell
python scripts\multimodal\phase4\snapshot_paper_inputs.py --tag audit_local
```

The snapshot will be written under `results/paper_snapshot/audit_local/`.

## Notes on paths

Most scripts assume the repository root as the working directory and use relative paths such as `data/processed/...`.

## Pipeline overview (dataset construction)

This package includes an automated pipeline that constructs high-fidelity multi-modal datasets from heterogeneous sources:

- **Structural modality (static structure)**
  - Java-based extractors live under `tools/` and are built as a fat JAR via Maven (`cd tools; mvn package`).
  - `scripts/structural/` invokes these extractors to generate base artifacts such as AST, call graph, and dependency graph (written under `data/processed/{ast,callgraph,dependency}/`).

- **Semantic modality (code semantics)**
  - `scripts/semantic/` uses the Maven-built extractor JAR to extract method-level semantic signals (e.g., identifiers/comments/variables), then builds embeddings and semantic similarity matrices.

- **Multi-modal similarity matrices (Phase 1 fusion inputs)**
  - `scripts/multimodal/phase1/` (notably `build_multimodal_matrices.py`) constructs the aligned similarity matrices for each modality (semantic/structural/temporal) and writes them to `data/processed/fusion/`.

- **Temporal modality (runtime behavior)**
  - Download and run the four monolith applications, start the stack under `docker/` (via `docker/docker-compose.yml`), and execute the JMeter workloads documented in `scripts/jmeter/readme`.
  - This produces JTL logs under `results/jmeter/` and trace exports; then `scripts/temporal/` builds the temporal similarity matrix (e.g., `scripts/temporal/build_S_temp.py` writes `data/processed/temporal/<system>_S_temp.npy`).

### Claimed contribution (artifact)

We develop an automated pipeline to construct high-fidelity multi-modal datasets from heterogeneous sources. By integrating non-intrusive instrumentation with domain-aware preprocessing, our pipeline provides auditable benchmarks that address the critical scarcity of multi-modal data and support reproducible research.

## Reviewer-response add-ons (rebuttal-friendly)

### Cold-start simulation (sparse runtime traces)

To simulate scenarios where runtime data is scarce, we support **random trace dropping** when constructing the temporal matrix `S_temp`.

- Script: `scripts/temporal/build_S_temp.py`
- Options:
  - `--trace_drop_rate` (0..1): fraction of traceIds to discard
  - `--trace_drop_seed`: RNG seed for deterministic reproduction

Example:

```powershell
# keep only 20% traces (drop 80%), deterministic
python scripts\temporal\build_S_temp.py --system plants --trace_drop_rate 0.80 --trace_drop_seed 1337
```

### LLM baseline (cached outputs, offline reproducible)

To address comparisons with prompt-based LLM baselines **without depending on external APIs**, this repository includes an *evaluation harness* that reads **cached LLM partitions**.

- Script: `scripts/multimodal/phase4/run_llm_baseline_cached.py`
- Cached predictions location: `results/llm_baseline/cached/<system>_llm_prompt_partition.json`

Run all cached LLM predictions included in this repository:

```powershell
python scripts\multimodal\phase4\run_llm_baseline_cached.py
```

To evaluate specific systems, pass them explicitly. If a requested cached file is missing, the script will error and tell you which file to add.

### End-to-end cold-start sweep (recommended for rebuttal)

To report *end-to-end* robustness when runtime data is scarce, use the provided sweep runner. It will:

1) Build `S_temp` with trace dropping
2) Rebuild Phase1 fused matrix `S_final`
3) Run Phase3 clustering
4) Run Phase4 evaluation and extract **BCubedF1/MoJoSim**

```powershell
# Example: Plants, sweep drop rates (default: 0,0.2,0.4,0.6,0.8,0.9)
python scripts\temporal\run_trace_drop_sweep_end2end.py --system plants
```

Outputs:
- `results/cold_start/cold_start_trace_drop_<system>.csv`
- `results/cold_start/cold_start_trace_drop_<system>.md`
