"""Utility: compare two .npy matrices (NxN) and print simple diff stats.

Usage:
  python scripts/temporal/compare_matrices.py --a <pathA.npy> --b <pathB.npy>

Prints:
- shape
- off-diagonal nnz
- min/max/mean (off-diagonal)
- L1/L2 norms of delta
- fraction of entries changed (abs(delta) > eps)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _offdiag(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected NxN matrix, got {M.shape}")
    out = M.copy()
    np.fill_diagonal(out, 0.0)
    return out


def _stats(name: str, M: np.ndarray) -> None:
    O = _offdiag(M)
    nnz = int(np.count_nonzero(O))
    print(f"[{name}] shape={M.shape} offdiag_nnz={nnz} min={float(O.min()):.6g} max={float(O.max()):.6g} mean={float(O.mean()):.6g}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    A = np.load(Path(args.a))
    B = np.load(Path(args.b))

    _stats("A", A)
    _stats("B", B)

    if A.shape != B.shape:
        raise SystemExit(f"Shape mismatch: {A.shape} vs {B.shape}")

    d = _offdiag(B) - _offdiag(A)
    absd = np.abs(d)
    changed = int(np.count_nonzero(absd > float(args.eps)))
    total = int(d.size)
    l1 = float(absd.sum())
    l2 = float(np.sqrt((d * d).sum()))
    linf = float(absd.max())

    print(f"[delta] changed={changed}/{total} ({changed/total:.3%}) L1={l1:.6g} L2={l2:.6g} Linf={linf:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
