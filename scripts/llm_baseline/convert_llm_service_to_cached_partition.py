"""Convert an LLM JSON clustering to the cached-evaluation format.

Input format A (common LLM output):
{
  "ServiceA": ["fqcn1", "fqcn2"],
  "ServiceB": ["fqcn3"]
}

Output format (used by scripts/multimodal/phase4/run_llm_baseline_cached.py):
{ "fqcn1": "ServiceA", "fqcn2": "ServiceA", "fqcn3": "ServiceB" }

Also validates:
- every class in class_order appears exactly once
- no extra classes

Default paths are wired for DayTrader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def invert(ms2classes: Dict[str, List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ms, classes in ms2classes.items():
        if classes is None:
            continue
        if isinstance(classes, str):
            classes = [classes]
        for c in classes:
            if c in out:
                raise ValueError(f"Duplicate class in LLM output: {c}")
            out[c] = str(ms)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="daytrader")
    ap.add_argument(
        "--llm_json",
        default=str(ROOT / "results" / "llm_baseline" / "outputs" / "daytrader_llm_partition.json"),
        help="Path to LLM output JSON (service -> list of classes)",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "llm_baseline" / "cached" / "daytrader_llm_prompt_partition.json"),
        help="Output cached partition path (class -> cluster id)",
    )
    args = ap.parse_args()

    system = args.system
    llm_path = Path(args.llm_json)
    if not llm_path.exists():
        raise FileNotFoundError(llm_path)

    class_order = load_json(ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json")

    ms2classes = load_json(llm_path)
    cls2ms = invert(ms2classes)

    missing = [c for c in class_order if c not in cls2ms]
    extra = [c for c in cls2ms.keys() if c not in set(class_order)]
    if missing or extra:
        raise ValueError(
            f"Class set mismatch. Missing={len(missing)} Extra={len(extra)}. "
            f"First missing={missing[:5]} First extra={extra[:5]}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cls2ms, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote cached partition: {out_path} (N={len(cls2ms)})")


if __name__ == "__main__":
    main()
