"""Plot cold-start degradation curves from `results/cold_start/cold_start_trace_drop_<system>.csv`.

Produces a single PNG with 2 subplots:
- BCubedF1 vs drop_rate
- MoJoSim vs drop_rate

Also annotates points with predicted K (cluster count) to help explain regime shifts.

Usage:
  python scripts/temporal/plot_cold_start_curve.py --system plants

Optional:
  --csv_path results/cold_start/cold_start_trace_drop_plants.csv
  --out_path results/plots/cold_start_trace_drop_plants.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _annotate_k(ax, x, y, k_values):
    for xr, yr, kr in zip(x, y, k_values):
        if pd.isna(kr):
            continue
        ax.annotate(
            f"k={int(kr)}",
            (xr, yr),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            alpha=0.85,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument(
        "--csv_path",
        default=None,
        help="Path to cold-start sweep CSV (default: results/cold_start/cold_start_trace_drop_<system>.csv)",
    )
    ap.add_argument(
        "--out_path",
        default=None,
        help="Output PNG path (default: results/plots/cold_start_trace_drop_<system>.png)",
    )
    ap.add_argument(
        "--method",
        default="Ours_CAC_withU",
        help="Method name to plot (default: Ours_CAC_withU)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    csv_path = Path(args.csv_path) if args.csv_path else repo_root / "results" / "cold_start" / f"cold_start_trace_drop_{args.system}.csv"
    out_path = Path(args.out_path) if args.out_path else repo_root / "results" / "plots" / f"cold_start_trace_drop_{args.system}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    df_m = df[df["method"] == args.method].copy()
    if df_m.empty:
        available = sorted(df["method"].unique().tolist())
        raise SystemExit(f"No rows for method={args.method}. Available: {available}")

    # Ensure clean numeric
    for col in ["drop_rate", "bcubed_f1", "mojosim", "pred_k"]:
        df_m[col] = pd.to_numeric(df_m[col], errors="coerce")

    df_m = df_m.sort_values(["drop_rate"]).reset_index(drop=True)

    x = df_m["drop_rate"].to_numpy()
    y_f1 = df_m["bcubed_f1"].to_numpy()
    y_mojo = df_m["mojosim"].to_numpy()
    k = df_m["pred_k"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)

    # --- BCubedF1
    ax = axes[0]
    ax.plot(x, y_f1, marker="o", linewidth=2)
    _annotate_k(ax, x, y_f1, k)
    ax.set_ylabel("BCubedF1")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title(f"Cold-start (dynamic evidence drop) - {args.system} - {args.method}")

    # --- MoJoSim
    ax = axes[1]
    ax.plot(x, y_mojo, marker="o", linewidth=2, color="#d95f02")
    _annotate_k(ax, x, y_mojo, k)
    ax.set_ylabel("MoJoSim")
    ax.set_xlabel("drop_rate")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Tight y-lims for readability if possible
    if pd.notna(df_m["bcubed_f1"]).all():
        pad = max(0.02, float(df_m["bcubed_f1"].max() - df_m["bcubed_f1"].min()) * 0.15)
        axes[0].set_ylim(float(df_m["bcubed_f1"].min() - pad), float(df_m["bcubed_f1"].max() + pad))
    if pd.notna(df_m["mojosim"]).all():
        pad = max(1.0, float(df_m["mojosim"].max() - df_m["mojosim"].min()) * 0.15)
        axes[1].set_ylim(float(df_m["mojosim"].min() - pad), float(df_m["mojosim"].max() + pad))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
