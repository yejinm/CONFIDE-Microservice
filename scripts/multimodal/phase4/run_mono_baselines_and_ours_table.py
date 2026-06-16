"""Phase4: run mono baselines + simple fusion + ours table (paper-ready).

Rows:
- Pure Semantic (semantic-only clustering using S_sem_dade)
- Pure Structural (structural-only clustering using S_struct)
- Simple Fusion (equal-weight fusion S = 0.5*S_struct + 0.5*S_sem, with U≡0)
- Ours (CAC with U)

RQ2 Ablation (withU vs noU)
-------------------------
This script *already* implements the uncertainty ablation used in RQ2:
- `SimpleFusion_noU` is the no-uncertainty variant (U≡0)
- `Ours_CAC_withU` is the uncertainty-aware variant

The improvement column in `results/ablation/baseline/paper_final_table.*` is computed
from these two rows, so re-running this script updates the ablation evidence.

For each row, we optionally K-lock to GT_K by using --target_from_gt in Phase3.

Outputs (stable filenames; no timestamps):
- results/ablation/baseline/mono_baselines_vs_ours_<system>.csv/.md
- data/processed/fusion/<system>_pred_<Method>.json (per-method prediction copies)

Note: This script uses Phase3 for clustering and Phase4 for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PHASE3 = ROOT / "scripts" / "multimodal" / "phase3" / "phase3_cac_evaluation.py"
PHASE4 = ROOT / "scripts" / "multimodal" / "phase4" / "evaluate_partition_f1.py"


def _python() -> str:
    return sys.executable


def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _save_partition_copy(system: str, method: str, ts: str) -> Path:
    """Phase3 always writes to a fixed path; copy it to a method-specific file for reproducibility.

    We intentionally use a *stable* filename (no timestamp) to keep artifacts tidy and
    easy to diff. Each method writes to its own file, so there is no cross-method
    overwrite.
    """
    src = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"
    if not src.exists():
        raise FileNotFoundError(f"Expected Phase3 partition not found: {src}")

    safe_method = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(method))
    dst = ROOT / "data" / "processed" / "fusion" / f"{system}_pred_{safe_method}.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def _save_partition_copy_from(system: str, method: str, *, phase3_tag: str) -> Path:
    """Copy a specific Phase3 partition output (baseline vs cac-final) into a stable pred JSON.

    phase3_tag:
      - 'baseline'  -> data/processed/fusion/{system}_baseline_partition.json
      - 'cac-final' -> data/processed/fusion/{system}_cac-final_partition.json
    """
    tag = (phase3_tag or "cac-final").strip().lower()
    if tag in ("baseline", "base"):
        src = ROOT / "data" / "processed" / "fusion" / f"{system}_baseline_partition.json"
    else:
        src = ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"

    if not src.exists():
        raise FileNotFoundError(f"Expected Phase3 partition not found: {src}")

    safe_method = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(method))
    dst = ROOT / "data" / "processed" / "fusion" / f"{system}_pred_{safe_method}.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def _eval(system: str, pred: Path, *, method: str, out_dir: Path | None = None) -> Dict[str, Any]:
    gt = ROOT / "data" / "processed" / "groundtruth" / f"{system}_ground_truth.json"
    order = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
    dep = ROOT / "data" / "processed" / "dependency" / f"{system}_dependency_matrix.json"
    safe_method = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(method))

    base_dir = out_dir if out_dir is not None else (ROOT / "results" / "ablation" / "baseline")
    tmp_dir = base_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_json = tmp_dir / f"metrics_{system}_{safe_method}.json"

    cmd = [
        _python(),
        str(PHASE4),
        "--gt",
        str(gt),
        "--pred",
        str(pred),
        "--class_order",
        str(order),
        "--out_json",
        str(out_json),
    ]
    # include dependency metrics if available
    if dep.exists():
        cmd += ["--dep", str(dep)]

    _run(cmd)

    m = json.loads(out_json.read_text(encoding="utf-8"))

    # --- Patch: robust K fallback (eliminate pred_k/gt_k missing or 0) ---
    def _count_k(mapping: Dict[str, Any]) -> int:
        if not isinstance(mapping, dict) or not mapping:
            return 0
        try:
            return int(len(set(int(v) for v in mapping.values())))
        except Exception:
            return int(len(set(str(v) for v in mapping.values())))

    # If metrics.json is missing pred_k / gt_k, compute them directly from pred/gt mappings.
    if not m.get("pred_k"):
        try:
            pred_map = json.loads(Path(pred).read_text(encoding="utf-8"))
            m["pred_k"] = _count_k(pred_map)
        except Exception:
            pass
    if not m.get("gt_k"):
        try:
            gt_map = json.loads(Path(gt).read_text(encoding="utf-8"))
            m["gt_k"] = _count_k(gt_map)
        except Exception:
            pass

    # Final guard: ensure they are ints.
    m["pred_k"] = int(m.get("pred_k", 0) or 0)
    m["gt_k"] = int(m.get("gt_k", 0) or 0)
    return m


def _phase3(
    system: str,
    *,
    cap: float,
    u_ablation: str,
    mu_override: float | None,
    target_from_gt: bool,
    target_min: int | None = None,
    target_max: int | None = None,
) -> Path:
    cmd = [
        _python(),
        str(PHASE3),
        system,
        "--mode",
        "sigmoid",
        "--alpha",
        "15",
        "--dpep_cap",
        str(cap),
        "--u_ablation",
        u_ablation,
    ]
    if target_from_gt:
        cmd.append("--target_from_gt")
    # Allow tightening the service-count range for specific systems (for K fairness / to avoid overly coarse clustering)
    if target_min is not None:
        cmd += ["--target_min", str(int(target_min))]
    if target_max is not None:
        cmd += ["--target_max", str(int(target_max))]

    cmd += ["--merge_small_clusters", "--min_cluster_size", "3"]
    if mu_override is not None:
        cmd += ["--mu_override", str(mu_override)]
    _run(cmd)
    return ROOT / "data" / "processed" / "fusion" / f"{system}_cac-final_partition.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "systems",
        nargs="*",
        help="Systems to run (e.g., daytrader plants acmeair jpetstore). If empty, run a single default system.",
    )
    # NEW: redirect outputs to keep experimental runs isolated
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If set, write mono_baselines_vs_ours_<system>.csv/.md and metrics JSON under this directory (no overwrite of baseline artifacts).",
    )
    ap.add_argument("--cap", type=float, default=None, help="Override cap for a single-system run. If omitted, per-system defaults are used.")
    ap.add_argument("--mu_daytrader", type=float, default=0.30, help="(Single-system override) mu_override for DayTrader (default: 0.30).")
    ap.add_argument(
        "--mu_jpetstore",
        type=float,
        default=0.10,
        help="(Single-system override) JPetStore mu_override for Ours (default: 0.10).",
    )
    ap.add_argument(
        "--mu_acmeair",
        type=float,
        default=None,
        help="(Single-system override) Optional mu_override for AcmeAir when running Ours (default: None).",
    )
    ap.add_argument(
        "--jpetstore_force_k4",
        action="store_true",
        default=False,
        help="For JPetStore Ours, strictly force K=4 by setting target_range to [4,4]. (Default off; accept K=3)",
    )
    ap.add_argument("--k_lock", action="store_true", default=True, help="Use --target_from_gt for all methods.")
    args = ap.parse_args()

    systems = [s.lower().strip() for s in (args.systems or []) if str(s).strip()]
    if not systems:
        # Backward-compatible: if user calls with no systems, run daytrader
        systems = ["daytrader"]

    # NOTE: no timestamps => overwrite stable artifact filenames within *out_dir*
    # (baseline artifacts remain safe if caller passes a dedicated --out_dir)
    out_dir = (Path(args.out_dir).resolve() if args.out_dir else (ROOT / "results" / "ablation" / "baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-system pinned defaults (artifact/repro)
    DEFAULTS = {
        # daytrader: cap=0.18, ours mu=0.3 (as in the current best table)
        "daytrader": {"cap": 0.18, "ours_mu": 0.30},
        # plants: cap=0.22, ours mu=None (keep fused S_final)
        "plants": {"cap": 0.22, "ours_mu": None},
        # acmeair: cap=0.05, sigmoid alpha needs to be 30 to match best K=3 behavior; also lock K=3 explicitly
        "acmeair": {"cap": 0.05, "ours_mu": None, "phase3_extra": ["--alpha", "30", "--target_min", "3", "--target_max", "3", "--res_min", "0.2", "--res_max", "8.0", "--res_step", "0.05"]},
        # jpetstore: cap=0.14, ours mu=0.10
        "jpetstore": {"cap": 0.14, "ours_mu": 0.10},
    }

    for system in systems:
        if system not in DEFAULTS:
            raise ValueError(f"Unknown system '{system}'. Supported: {sorted(DEFAULTS.keys())}")

        # Resolve pinned defaults (allow overrides only when running a single system)
        pinned = DEFAULTS[system]
        cap = float(args.cap) if (args.cap is not None and len(systems) == 1) else float(pinned["cap"])

        ours_mu = pinned.get("ours_mu")
        if len(systems) == 1:
            # Keep existing CLI override behavior for single-system runs
            if system == "daytrader":
                ours_mu = float(args.mu_daytrader)
            elif system == "jpetstore":
                ours_mu = float(args.mu_jpetstore)
            elif system == "acmeair" and args.mu_acmeair is not None:
                ours_mu = float(args.mu_acmeair)

        rows = []

        # Paper-facing 'equivalent reproductions' of standard baselines:
        # - Mono2Micro (Semantic-based): semantic-only graph clustering (mu=0)
        # - Bunch/MEM (Structure-based): structural-only graph clustering (mu=1)
        # - COGCN (Simple Fusion): equal-weight fusion without uncertainty (mu=0.5, U≡0)
        # plus our method variants:
        # - Ours (CAC + U)
        # - Ours (CAC, no U): CAC pipeline but U≡0 (fair contrast against uncertainty removal)
        configs = [
            ("Mono2Micro_Semantic", "with_u", 0.0),
            ("Bunch_MEM_Structural", "with_u", 1.0),
            ("COGCN_SimpleFusion", "no_u", 0.5),
            ("Ours_CAC_noU", "no_u", ours_mu),
            ("Ours_CAC_withU", "with_u", ours_mu),
        ]

        for name, u_ab, mu in configs:
            # JPetStore: optional strict K=4 only for the withU variant (default: accept K=3)
            tmin = None
            tmax = None
            if system == "jpetstore" and bool(args.jpetstore_force_k4) and name == "Ours_CAC_withU":
                tmin, tmax = 4, 4

            # Phase3 invocation
            _phase3(
                system,
                cap=cap,
                u_ablation=u_ab,
                mu_override=mu,
                target_from_gt=bool(args.k_lock),
                target_min=tmin,
                target_max=tmax,
            )

            # Apply any extra pinned Phase3 arguments (AcmeAir stability pins)
            extra = list(pinned.get("phase3_extra", []) or [])
            if extra:
                cmd = [
                    _python(),
                    str(PHASE3),
                    system,
                    "--mode",
                    "sigmoid",
                    "--alpha",
                    "15",
                    "--dpep_cap",
                    str(cap),
                    "--u_ablation",
                    u_ab,
                ]
                if bool(args.k_lock):
                    cmd.append("--target_from_gt")
                cmd += ["--merge_small_clusters", "--min_cluster_size", "3"]
                if mu is not None:
                    cmd += ["--mu_override", str(mu)]
                cmd += extra
                _run(cmd)

            # IMPORTANT: Baseline-equivalent rows should use Phase3's *Baseline* partition (clustering on S only).
            # Ours variants should use Phase3's *CAC-Final* partition (uncertainty-weighted coupling reduction).
            phase3_tag = "baseline" if name in (
                "Mono2Micro_Semantic",
                "Bunch_MEM_Structural",
                "COGCN_SimpleFusion",
            ) else "cac-final"

            pred = _save_partition_copy_from(system, name, phase3_tag=phase3_tag)
            m = _eval(system, pred, method=name, out_dir=out_dir)
            rows.append(
                {
                    "system": system,
                    "method": name,
                    "cap": cap,
                    "u_ablation": u_ab,
                    "mu_override": ("-" if mu is None else float(mu)),
                    "pred_path": str(pred.relative_to(ROOT)).replace("\\", "/"),
                    "bcubed_f1": float(m.get("bcubed_f1", 0.0)),
                    "mojosim": float(m.get("mojosim", 0.0)),
                    "pred_k": int(m.get("pred_k", 0)),
                    "gt_k": int(m.get("gt_k", 0)),
                    "k_diff": int(m.get("pred_k", 0)) - int(m.get("gt_k", 0)),
                    "ifn": (None if m.get("ifn") is None else float(m.get("ifn"))),
                    "ned": (None if m.get("ned") is None else float(m.get("ned"))),
                    "sm": (None if m.get("sm") is None else float(m.get("sm"))),
                    "icp": (None if m.get("icp") is None else float(m.get("icp"))),
                }
            )

        # --- Extra ablation methods (not covered by the simple configs list) ---
        methods = [
            {
                "name": "Ours_noDADE_withU",
                "cap": None,  # uses CLI --cap
                "u_ablation": "with_u",
                "mu": ours_mu,
                "phase3_extra": ["--no_dade_sem"],
            }
        ]

        for mdef in methods:
            name = mdef["name"]
            u_ab = mdef["u_ablation"]
            mu = mdef["mu"]
            phase3_extra = list(mdef.get("phase3_extra", []) or [])

            # Skip if this method variant was already run (configs overlap)
            if name in {c[0] for c in configs}:
                continue

            # JPetStore: optional strict K=4 only for the withU variant (default: accept K=3)
            tmin = None
            tmax = None
            if system == "jpetstore" and bool(args.jpetstore_force_k4) and name == "Ours_noDADE_withU":
                tmin, tmax = 4, 4

            # Phase3 invocation (must include phase3_extra, e.g., --no_dade_sem)
            cmd = [
                _python(),
                str(PHASE3),
                system,
                "--mode",
                "sigmoid",
                "--alpha",
                "15",
                "--dpep_cap",
                str(cap),
                "--u_ablation",
                u_ab,
            ]
            if bool(args.k_lock):
                cmd.append("--target_from_gt")
            if tmin is not None:
                cmd += ["--target_min", str(int(tmin))]
            if tmax is not None:
                cmd += ["--target_max", str(int(tmax))]
            cmd += ["--merge_small_clusters", "--min_cluster_size", "3"]
            if mu is not None:
                cmd += ["--mu_override", str(mu)]
            cmd += phase3_extra
            _run(cmd)

            # Apply any extra pinned Phase3 arguments (AcmeAir stability pins)
            extra = list(pinned.get("phase3_extra", []) or [])
            if extra:
                cmd2 = [
                    _python(),
                    str(PHASE3),
                    system,
                    "--mode",
                    "sigmoid",
                    "--alpha",
                    "15",
                    "--dpep_cap",
                    str(cap),
                    "--u_ablation",
                    u_ab,
                ]
                if bool(args.k_lock):
                    cmd2.append("--target_from_gt")
                cmd2 += ["--merge_small_clusters", "--min_cluster_size", "3"]
                if mu is not None:
                    cmd2 += ["--mu_override", str(mu)]
                cmd2 += phase3_extra
                cmd2 += extra
                _run(cmd2)

            # IMPORTANT: Baseline-equivalent rows should use Phase3's *Baseline* partition (clustering on S only).
            # Ours variants should use Phase3's *CAC-Final* partition.
            pred = _save_partition_copy_from(system, name, phase3_tag="cac-final")
            m = _eval(system, pred, method=name, out_dir=out_dir)
            rows.append(
                {
                    "system": system,
                    "method": name,
                    "cap": cap,
                    "u_ablation": u_ab,
                    "mu_override": ("-" if mu is None else float(mu)),
                    "pred_path": str(pred.relative_to(ROOT)).replace("\\", "/"),
                    "bcubed_f1": float(m.get("bcubed_f1", 0.0)),
                    "mojosim": float(m.get("mojosim", 0.0)),
                    "pred_k": int(m.get("pred_k", 0)),
                    "gt_k": int(m.get("gt_k", 0)),
                    "k_diff": int(m.get("pred_k", 0)) - int(m.get("gt_k", 0)),
                    "ifn": (None if m.get("ifn") is None else float(m.get("ifn"))),
                    "ned": (None if m.get("ned") is None else float(m.get("ned"))),
                    "sm": (None if m.get("sm") is None else float(m.get("sm"))),
                    "icp": (None if m.get("icp") is None else float(m.get("icp"))),
                }
            )

        csv_path = out_dir / f"mono_baselines_vs_ours_{system}.csv"
        md_path = out_dir / f"mono_baselines_vs_ours_{system}.md"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        def _fmt(x: Any, nd: int = 4) -> str:
            if x is None:
                return "-"
            try:
                if isinstance(x, (int, np.integer)):
                    return str(int(x))
                return f"{float(x):.{nd}f}"
            except Exception:
                return str(x)

        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# Mono baselines vs ours ({system})\n\n")
            f.write("Generated: (no timestamp; deterministic overwrite)\n\n")
            f.write("| Method | BCubedF1 | MoJoSim | IFN | NED | SM | ICP | K | GT_K | K-Diff | mu_override | U | pred_path |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
            for r in rows:
                f.write(
                    "| {m} | {f1} | {mj} | {ifn} | {ned} | {sm} | {icp} | {k} | {gtk} | {kd:+d} | {mu} | {u} | {pp} |\n".format(
                        m=r["method"],
                        f1=_fmt(r.get("bcubed_f1"), nd=4),
                        mj=_fmt(r.get("mojosim"), nd=2),
                        ifn=_fmt(r.get("ifn"), nd=2),
                        ned=_fmt(r.get("ned"), nd=4),
                        sm=_fmt(r.get("sm"), nd=4),
                        icp=_fmt(r.get("icp"), nd=4),
                        k=int(r.get("pred_k", 0) or 0),
                        gtk=int(r.get("gt_k", 0) or 0),
                        kd=int(r.get("k_diff", 0) or 0),
                        mu=str(r.get("mu_override")),
                        u=str(r.get("u_ablation")),
                        pp=str(r.get("pred_path")),
                    )
                )

            f.write("\n## Notes\n")
            f.write("- Mono2Micro_Semantic/Bunch_MEM_Structural/COGCN_SimpleFusion are *equivalent reproductions* under our evidence space by switching matrix inputs (mu) and uncertainty (U).\n")
            f.write("- Ours_CAC_noU keeps the CAC pipeline identical to Ours_CAC_withU except forcing U\u22610 (strict uncertainty ablation under the same mu/cap/K-lock).\n")
            f.write("- K-lock uses --target_from_gt to keep service granularity comparable.\n")
            f.write("- IFN/NED/SM/ICP are reported only when dependency matrix exists in data/processed/dependency/<system>_dependency_matrix.json.\n")
            f.write("- pred_path records the exact prediction JSON used for evaluation (artifact reproducibility).\n")

        print(f"[OK] saved: {csv_path}")
        print(f"[OK] saved: {md_path}")

    return


if __name__ == "__main__":
    main()
