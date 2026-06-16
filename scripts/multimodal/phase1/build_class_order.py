import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Keep ROOT depth consistent with build_class_index.py
ROOT = Path(__file__).resolve().parents[3]

SYSTEM_CLASS_INDEX = {
    "acmeair": "acmeair_classes.json",
    "daytrader": "daytrader_classes.json",
    "jpetstore": "jpetstore_classes.json",
    "plants": "plants_classes.json",
}


def load_system_classes(system: str) -> Dict[str, dict]:
    fname = SYSTEM_CLASS_INDEX[system]
    path = ROOT / "data" / "processed" / "fusion" / fname
    if not path.is_file():
        raise FileNotFoundError(f"{fname} not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_class_order_and_index(system: str) -> Tuple[List[str], Dict[str, int]]:
    data = load_system_classes(system)
    # Debug print: inspect index file size
    print(f"DEBUG [{system}]: Loaded index containing {len(data)} classes.")
    selected = [
        cls_name
        for cls_name, meta in data.items()
        if isinstance(meta, dict)
        and meta.get("has_semantic") is True
        and (meta.get("has_callgraph") or meta.get("has_dependency"))
    ]
    # If it still fails, print a sample meta to inspect schema
    if not selected and data:
        first_key = list(data.keys())[0]
        print(f"DEBUG [{system}]: Sample class meta: {data[first_key]}")
    if not selected:
        raise ValueError(f"[{system}] No class has both semantic and structural features.")
    class_order = sorted(selected)
    cls2idx: Dict[str, int] = {cls: i for i, cls in enumerate(class_order)}
    return class_order, cls2idx


def save_class_order(system: str, class_order: List[str]) -> None:
    out_path = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(class_order, f, indent=2, ensure_ascii=False)
    print(f"[{system}] saved class order to {out_path} (N={len(class_order)})")


def main():
    parser = argparse.ArgumentParser(description="Build class order for a given system (acmeair/daytrader/jpetstore/plants)")
    parser.add_argument("--system", choices=sorted(SYSTEM_CLASS_INDEX.keys()), required=True)
    args = parser.parse_args()

    class_order, cls2idx = build_class_order_and_index(args.system)
    save_class_order(args.system, class_order)

    print(f"First 10 classes in order for {args.system}:")
    for i, cls in enumerate(class_order[:10]):
        print(i, cls)


if __name__ == "__main__":
    main()
