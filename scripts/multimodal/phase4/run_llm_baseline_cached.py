"""Phase4: Evaluate cached LLM baseline partitions (offline, reproducible).

Motivation:
- Reviewers asked for a prompt-based LLM baseline.
- To keep the artifact reproducible and independent of external APIs,
  we evaluate **cached** LLM predictions stored in `results/llm_baseline/cached/`.

Inputs:
- data/processed/groundtruth/<system>_ground_truth.json
- data/processed/fusion/<system>_class_order.json
- results/llm_baseline/cached/<system>_llm_prompt_partition.json

Outputs:
- results/llm_baseline/llm_baseline_cached_<system>.csv/.md
- results/llm_baseline/llm_baseline_cached_all.csv/.md

Prediction JSON format:
- { FQCN (str) : cluster_id (int or str) }

This script only *evaluates*. If no systems are provided, it evaluates all
cached partitions currently present in the cached directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "scripts" / "multimodal" / "phase4" / "evaluate_partition_f1.py"


def _python() -> str:
    return sys.executable


def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _eval(system: str, pred: Path, *, method: str) -> Dict[str, Any]:
    gt = ROOT / "data" / "processed" / "groundtruth" / f"{system}_ground_truth.json"
    order = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
    dep = ROOT / "data" / "processed" / "dependency" / f"{system}_dependency_matrix.json"

    out_dir = ROOT / "results" / "llm_baseline" / "_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"metrics_{system}_{method}.json"

    cmd = [
        _python(),
        str(EVAL),
        "--gt",
        str(gt),
        "--pred",
        str(pred),
        "--class_order",
        str(order),
        "--out_json",
        str(out_json),
    ]
    if dep.exists():
        cmd += ["--dep", str(dep)]

    _run(cmd)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*", help="Systems to run (e.g., daytrader plants acmeair jpetstore)")
    ap.add_argument(
        "--cached_dir",
        default=str((ROOT / "results" / "llm_baseline" / "cached").as_posix()),
        help="Directory containing cached LLM partitions.",
    )
    args = ap.parse_args()

    cached_dir = Path(args.cached_dir)
    systems = [s.lower().strip() for s in (args.systems or []) if str(s).strip()]
    if not systems:
        systems = sorted(
            p.name.replace("_llm_prompt_partition.json", "")
            for p in cached_dir.glob("*_llm_prompt_partition.json")
        )
        if not systems:
            raise FileNotFoundError(
                f"No cached LLM predictions found in {cached_dir}. "
                "Add files named <system>_llm_prompt_partition.json or pass systems explicitly."
            )

    out_dir = ROOT / "results" / "llm_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for system in systems:
        pred = cached_dir / f"{system}_llm_prompt_partition.json"
        if not pred.exists():
            raise FileNotFoundError(
                f"Missing cached LLM prediction for system={system}: {pred}. "
                f"Add it under results/llm_baseline/cached/ for offline reproduction."
            )

        m = _eval(system, pred, method="LLM_Prompt_Cached")
        row = {
            "system": system,
            "method": "LLM_Prompt_Cached",
            "pred_path": str(pred.relative_to(ROOT)).replace("\\", "/"),
            "bcubed_f1": float(m.get("bcubed_f1", 0.0)),
            "mojosim": float(m.get("mojosim", 0.0)),
            "pred_k": int(m.get("pred_k", 0) or 0),
            "gt_k": int(m.get("gt_k", 0) or 0),
            "k_diff": int(m.get("pred_k", 0) or 0) - int(m.get("gt_k", 0) or 0),
        }
        all_rows.append(row)

        csv_path = out_dir / f"llm_baseline_cached_{system}.csv"
        md_path = out_dir / f"llm_baseline_cached_{system}.md"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# LLM baseline (cached) - {system}\n\n")
            f.write(f"- pred: `{row['pred_path']}`\n")
            f.write("\n")
            f.write("| Method | BCubedF1 | MoJoSim | K | GT_K | K-Diff |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            f.write(
                "| {m} | {f1} | {mj} | {k} | {gtk} | {kd:+d} |\n".format(
                    m=row["method"],
                    f1=_fmt(row["bcubed_f1"], nd=4),
                    mj=_fmt(row["mojosim"], nd=2),
                    k=int(row["pred_k"]),
                    gtk=int(row["gt_k"]),
                    kd=int(row["k_diff"]),
                )
            )

    # Aggregate
    all_csv = out_dir / "llm_baseline_cached_all.csv"
    all_md = out_dir / "llm_baseline_cached_all.md"

    with all_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    with all_md.open("w", encoding="utf-8") as f:
        f.write("# LLM baseline (cached) - all systems\n\n")
        f.write("| System | Method | BCubedF1 | MoJoSim | K | GT_K | K-Diff | pred |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for r in all_rows:
            f.write(
                "| {sys} | {m} | {f1} | {mj} | {k} | {gtk} | {kd:+d} | `{pp}` |\n".format(
                    sys=r["system"],
                    m=r["method"],
                    f1=_fmt(r["bcubed_f1"], nd=4),
                    mj=_fmt(r["mojosim"], nd=2),
                    k=int(r["pred_k"]),
                    gtk=int(r["gt_k"]),
                    kd=int(r["k_diff"]),
                    pp=r["pred_path"],
                )
            )

    print(f"[OK] wrote: {all_csv}")
    print(f"[OK] wrote: {all_md}")


if __name__ == "__main__":
    main()
