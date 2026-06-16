"""Plot cold-start trace-drop robustness with a safety-fuse annotation.

This script is designed for the rebuttal figure:
- X axis: trace drop rate (0 -> 0.95)
- Left Y: BCubedF1 for DayTrader/JPetStore/Plants (and AcmeAir where available)
- AcmeAir: plot available points; leave drop=0.95 missing and annotate as CAC safety fuse.

Inputs:
- results/cold_start/cold_start_trace_drop_<system>.md  (system in daytrader/jpetstore/plants)
  (AcmeAir is optional; if no file exists, only annotate the missing point.)

Outputs:
- results/plots/cold_start_trace_drop_robustness_safety_fuse.png
- results/plots/cold_start_trace_drop_robustness_safety_fuse.pdf

Usage:
  python scripts/temporal/plot_cold_start_dual_axis.py
  python scripts/temporal/plot_cold_start_dual_axis.py --out results/plots/fig_cold_start_safety_fuse.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def _parse_cold_start_md(md_path: Path, method: str = "Ours_CAC_withU") -> Tuple[List[float], List[float]]:
    """Parse a cold_start_trace_drop_<system>.md table and return (drop_rates, f1s) for a method."""

    txt = md_path.read_text(encoding="utf-8", errors="ignore")
    # table rows look like:
    # | 0.20 | 1337 | ... | Ours_CAC_withU | 0.5139 | ... |
    rows = []
    for line in txt.splitlines():
        if not line.strip().startswith("|"):
            continue
        if "drop_rate" in line and "method" in line:
            continue
        if "---" in line:
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        drop = parts[0]
        meth = parts[4]
        f1 = parts[5]
        if meth != method:
            continue
        try:
            rows.append((float(drop), float(f1)))
        except ValueError:
            continue

    rows.sort(key=lambda x: x[0])
    return [r[0] for r in rows], [r[1] for r in rows]


def _maybe_load_system(system: str) -> Optional[Tuple[List[float], List[float]]]:
    p = ROOT / "results" / "cold_start" / f"cold_start_trace_drop_{system}.md"
    if not p.exists():
        return None
    return _parse_cold_start_md(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "plots" / "cold_start_trace_drop_robustness_safety_fuse.png"),
        help="Output PNG path.",
    )
    ap.add_argument(
        "--out_pdf",
        default=str(ROOT / "results" / "plots" / "cold_start_trace_drop_robustness_safety_fuse.pdf"),
        help="Output PDF path.",
    )
    ap.add_argument(
        "--annotate_drop",
        type=float,
        default=0.95,
        help="Drop rate to annotate as safety fuse for AcmeAir.",
    )
    args = ap.parse_args()

    series: Dict[str, Tuple[List[float], List[float]]] = {}
    for sys in ["daytrader", "jpetstore", "plants"]:
        d = _maybe_load_system(sys)
        if d is None:
            raise FileNotFoundError(f"Missing: results/cold_start/cold_start_trace_drop_{sys}.md")
        series[sys] = d

    acme = _maybe_load_system("acmeair")

    # Plot
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=220)

    colors = {
        "daytrader": "#1f77b4",
        "jpetstore": "#2ca02c",
        "plants": "#ff7f0e",
        "acmeair": "#9467bd",
    }

    for sys, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, label=sys.capitalize(), color=colors[sys])

    # AcmeAir: plot if available; else only annotate fuse.
    if acme is not None:
        xs, ys = acme
        ax.plot(xs, ys, marker="s", linewidth=2.0, markersize=4.5, label="AcmeAir", color=colors["acmeair"])

    ax.set_xlabel("Trace Drop Rate")
    ax.set_ylabel("BCubed F1")
    ax.set_xlim(-0.02, 0.97)
    ax.set_ylim(0.30, 0.76)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # Safety fuse annotation at (0.95, y)
    drop_x = float(args.annotate_drop)

    # Choose a y anchor: if acme data exists, use last known point; else use mid.
    if acme is not None and len(acme[0]) > 0:
        # use the point closest to but < drop_x
        candidates = [(x, y) for x, y in zip(acme[0], acme[1]) if x < drop_x]
        if candidates:
            anchor_y = candidates[-1][1]
        else:
            anchor_y = 0.52
    else:
        anchor_y = 0.52

    # draw a light red vertical band around drop_x
    ax.axvspan(drop_x - 0.01, drop_x + 0.01, color="#ffcccc", alpha=0.55, zorder=0)

    # red point marker at the "missing" location
    ax.scatter([drop_x], [anchor_y], s=55, color="#d62728", zorder=5)

    ax.annotate(
        "Safety-fuse discovered in separate\nresolution-sweep diagnosis\n(drop=0.95, CAC-Final: K=None)",
        xy=(drop_x, anchor_y),
        xytext=(0.52, anchor_y + 0.24),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d62728", lw=1.2, alpha=0.98),
        fontsize=10.2,
        color="#7f1d1d",
        ha="left",
        va="center",
    )

    ax.legend(loc="lower left", frameon=True, framealpha=0.95)
    ax.set_title("Cold-start robustness under trace sparsity")

    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    print(f"[OK] wrote: {out_png}")
    print(f"[OK] wrote: {out_pdf}")


if __name__ == "__main__":
    main()
