"""End-to-end cold-start sweep (trace dropping) -> final metrics.

This script automates the rebuttal experiment:
- Simulate runtime data scarcity by dropping a fraction of traces when building S_temp.
- Rebuild S_final (Phase1 fusion) using the modified S_temp.
- Run Phase3 (CAC + baselines) and Phase4 (Table III-like metrics) to record BCubedF1/MoJoSim.

Design goals:
- Minimal code intrusion: reuse existing scripts as subprocesses.
- Deterministic reproduction: seed control for trace dropping.
- Stable outputs: write CSV/MD under results/cold_start/.

Example:
  python scripts/temporal/run_trace_drop_sweep_end2end.py --system plants

Notes:
- This script assumes you already have the processed inputs (class orders, embeddings, etc.)
  as in the paper snapshot workflow.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json
import shutil


ROOT = Path(__file__).resolve().parents[2]
BUILD_S_TEMP = ROOT / "scripts" / "temporal" / "build_S_temp.py"
PHASE1 = ROOT / "scripts" / "multimodal" / "phase1" / "build_multimodal_matrices.py"
PHASE3 = ROOT / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"
PHASE4_TABLE = ROOT / "scripts" / "multimodal" / "phase4" / "run_mono_baselines_and_ours_table.py"


def _python() -> str:
    return sys.executable


def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _backup_fusion_npy(system: str, *, backup_dir: Path) -> None:
    """Backup fusion matrices that may be overwritten during the sweep.

    We back up a conservative set of .npy files (S_final + modality matrices commonly used).
    Missing files are ignored.
    """
    fusion_dir = ROOT / "data" / "processed" / "fusion"
    backup_dir.mkdir(parents=True, exist_ok=True)

    patterns = [
        f"{system}_S_final.npy",
        f"{system}_S_sem.npy",
        f"{system}_S_sem_embedding.npy",
        f"{system}_S_sem_dade_*.npy",
        f"{system}_S_struct.npy",
    ]

    for pat in patterns:
        for p in fusion_dir.glob(pat):
            if p.is_file():
                shutil.copy2(p, backup_dir / p.name)


def _restore_fusion_npy(*, backup_dir: Path) -> None:
    """Restore previously backed up fusion matrices."""
    fusion_dir = ROOT / "data" / "processed" / "fusion"
    if not backup_dir.exists():
        return
    for p in backup_dir.glob("*.npy"):
        shutil.copy2(p, fusion_dir / p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["acmeair", "daytrader", "jpetstore", "plants"])
    ap.add_argument(
        "--drop_rates",
        default="0,0.2,0.4,0.6,0.8,0.9",
        help="Comma-separated drop rates in [0,1].",
    )
    ap.add_argument("--seed", type=int, default=1337, help="Seed for trace dropping.")

    # Temporal build knobs (keep defaults consistent with paper unless overridden)
    ap.add_argument("--alpha_jtl", type=float, default=0.5)
    ap.add_argument("--beta_trace", type=float, default=0.5)
    ap.add_argument("--group_by", default="thread_iteration", choices=["thread", "thread_iteration", "sliding_window"])
    ap.add_argument("--max_events", type=int, default=80)
    ap.add_argument("--min_events", type=int, default=2)
    ap.add_argument("--max_session_seconds", type=float, default=5.0)

    # NEW: optional override seeds for different drop mechanisms (defaults align to --seed)
    ap.add_argument("--jtl_drop_seed", type=int, default=None, help="Seed for JTL session dropping. Defaults to --seed.")
    ap.add_argument("--span_drop_seed", type=int, default=None, help="Seed for span/item dropping. Defaults to --seed.")
    ap.add_argument("--trace_drop_seed", type=int, default=None, help="Seed for traceId dropping. Defaults to --seed.")

    # Phase1 fusion weights (use defaults in script unless overridden)
    ap.add_argument("--w_sem", type=float, default=0.4)
    ap.add_argument("--w_struct", type=float, default=0.4)
    ap.add_argument("--w_temp", type=float, default=0.2)
    ap.add_argument("--dade_type", type=str, default="base")

    # Phase3 parameters (use same as reproduce_paper unless overridden)
    ap.add_argument("--cap", type=float, default=None, help="Override dpep cap; if omitted Phase4 pinned defaults will apply.")
    ap.add_argument("--k_lock", action="store_true", default=True, help="Use --target_from_gt during clustering.")

    args = ap.parse_args()

    # Transaction-style backup for baseline protection (no workspace snapshot needed)
    out_root = ROOT / "results" / "cold_start"
    out_root.mkdir(parents=True, exist_ok=True)
    bak_dir = out_root / "_bak_fusion" / args.system

    _backup_fusion_npy(args.system, backup_dir=bak_dir)

    try:
        rates = []
        for s in str(args.drop_rates).split(","):
            s = s.strip()
            if not s:
                continue
            r = float(s)
            if not (0.0 <= r <= 1.0):
                raise ValueError(f"drop_rate must be in [0,1], got {r}")
            rates.append(r)
        if not rates:
            raise ValueError("No drop rates provided")

        rows: List[Dict[str, Any]] = []

        for r in rates:
            run_dir = out_root / f"run_rate_{r:.2f}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # 1) Build temporal matrix with cold-start dropping
            eff_jtl_seed = int(args.seed if args.jtl_drop_seed is None else args.jtl_drop_seed)
            eff_span_seed = int(args.seed if args.span_drop_seed is None else args.span_drop_seed)
            eff_trace_seed = int(args.seed if args.trace_drop_seed is None else args.trace_drop_seed)

            _run(
                [
                    _python(),
                    str(BUILD_S_TEMP),
                    "--system",
                    args.system,
                    "--alpha-jtl",
                    str(float(args.alpha_jtl)),
                    "--beta-trace",
                    str(float(args.beta_trace)),
                    "--group-by",
                    str(args.group_by),
                    "--max-events",
                    str(int(args.max_events)),
                    "--min-events",
                    str(int(args.min_events)),
                    "--max-session-seconds",
                    str(float(args.max_session_seconds)),
                    # Apply the same drop rate to all dynamic evidence sources (full-modality cold-start)
                    "--jtl_drop_rate",
                    str(float(r)),
                    "--jtl_drop_seed",
                    str(int(eff_jtl_seed)),
                    "--span_drop_rate",
                    str(float(r)),
                    "--span_drop_seed",
                    str(int(eff_span_seed)),
                    "--trace_drop_rate",
                    str(float(r)),
                    "--trace_drop_seed",
                    str(int(eff_trace_seed)),
                ]
            )

            # Read S_temp sidecar meta (kept/dropped counts)
            physical = args.system
            # build_S_temp uses SYSTEM_NAME_MAP: daytrader7->daytrader, plantsbywebsphere->plants
            if args.system == "daytrader":
                physical = "daytrader"
            if args.system == "plants":
                physical = "plants"
            meta_path = ROOT / "data" / "processed" / "temporal" / f"{physical}_S_temp.meta.json"
            kept = None
            dropped = None
            before = None
            try:
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    before = int(meta.get("traces_before", 0))
                    kept = int(meta.get("traces_after", 0))
                    dropped = int(meta.get("traces_dropped", 0))
            except Exception:
                pass

            # 2) Rebuild S_final (Phase1 fusion) to pick up updated S_temp
            _run(
                [
                    _python(),
                    str(PHASE1),
                    "--system",
                    args.system,
                    "--w-sem",
                    str(float(args.w_sem)),
                    "--w-struct",
                    str(float(args.w_struct)),
                    "--w-temp",
                    str(float(args.w_temp)),
                    "--dade_type",
                    str(args.dade_type),
                ]
            )

            # 3) Run Phase3 to generate partitions used by Phase4 table script
            # Keep Phase3 invocation consistent with reproduce_paper (sigmoid, alpha=15, etc.)
            cmd_p3 = [
                _python(),
                str(PHASE3),
                args.system,
                "--mode",
                "sigmoid",
                "--alpha",
                "15",
            ]
            if args.cap is not None:
                cmd_p3 += ["--dpep_cap", str(float(args.cap))]
            if bool(args.k_lock):
                cmd_p3.append("--target_from_gt")
            # IMPORTANT (cold-start): do NOT merge small clusters.
            # Merging can collapse K below target_range and cause all candidates to be rejected
            # by the TargetRangeGuard, which would hide real sensitivity to sparse evidence.
            _run(cmd_p3)

            # 4) Run Phase4 table script for this system, redirecting outputs to run_dir
            _run([_python(), str(PHASE4_TABLE), args.system, "--out_dir", str(run_dir)])

            # 5) Read the produced per-system table within run_dir
            sys_csv = run_dir / f"mono_baselines_vs_ours_{args.system}.csv"
            if not sys_csv.exists():
                raise FileNotFoundError(f"Expected metrics CSV not found: {sys_csv}")

            import pandas as pd  # type: ignore

            df = pd.read_csv(sys_csv)
            # Keep the methods we care about for rebuttal: SimpleFusion baseline + ours (with U)
            keep = df[df["method"].isin(["COGCN_SimpleFusion", "Ours_CAC_withU"])]
            if keep.empty:
                raise RuntimeError(f"Unexpected CSV schema or missing rows in {sys_csv}")

            for _, row in keep.iterrows():
                rows.append(
                    {
                        "system": args.system,
                        "drop_rate": float(r),
                        "seed": int(args.seed),
                        "traces_before": ("-" if before is None else int(before)),
                        "traces_kept": ("-" if kept is None else int(kept)),
                        "traces_dropped": ("-" if dropped is None else int(dropped)),
                        "method": str(row["method"]),
                        "bcubed_f1": float(row["bcubed_f1"]),
                        "mojosim": float(row["mojosim"]),
                        "pred_k": int(row.get("pred_k", 0) or 0),
                        "gt_k": int(row.get("gt_k", 0) or 0),
                        "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                    }
                )

            print(f"[cold-start] system={args.system} drop_rate={r:.2f} done")

        # Write summary outputs
        csv_path = out_root / f"cold_start_trace_drop_{args.system}.csv"
        md_path = out_root / f"cold_start_trace_drop_{args.system}.md"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Simple markdown table
        def _fmt(x: Any, nd: int = 4) -> str:
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return str(x)

        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# Cold-start sweep (trace drop) - {args.system}\n\n")
            f.write("This table evaluates end-to-end performance under sparse runtime traces by randomly dropping traceIds.\n\n")
            f.write("| drop_rate | seed | traces_kept | traces_dropped | method | BCubedF1 | MoJoSim | K | GT_K | run_dir |\n")
            f.write("|---:|---:|---:|---:|---|---:|---:|---:|---:|---|\n")
            for r in rows:
                f.write(
                    "| {dr:.2f} | {seed} | {kept} | {drop} | {m} | {f1} | {mj} | {k} | {gtk} | `{rd}` |\n".format(
                        dr=float(r["drop_rate"]),
                        seed=int(r["seed"]),
                        kept=str(r.get("traces_kept", "-")),
                        drop=str(r.get("traces_dropped", "-")),
                        m=str(r["method"]),
                        f1=_fmt(r["bcubed_f1"], nd=4),
                        mj=_fmt(r["mojosim"], nd=2),
                        k=int(r["pred_k"]),
                        gtk=int(r["gt_k"]),
                        rd=str(r.get("run_dir", "")),
                    )
                )

        print(f"[OK] wrote: {csv_path}")
        print(f"[OK] wrote: {md_path}")

    finally:
        # Always restore baseline fusion matrices
        _restore_fusion_npy(backup_dir=bak_dir)


if __name__ == "__main__":
    main()
