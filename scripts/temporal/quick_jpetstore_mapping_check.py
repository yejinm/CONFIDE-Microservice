"""Quick sanity check: can JPetStore traces map to class_order via url.query action keys?

Usage (PowerShell):
  python scripts\temporal\quick_jpetstore_mapping_check.py

This reads:
- data/processed/traces/jpetstore.json
- data/processed/fusion/jpetstore_class_order.json
and reports whether action->controller FQCN anchors exist, and how many spans would map.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT / "data" / "processed" / "traces" / "jpetstore.json"
CLASS_ORDER_PATH = ROOT / "data" / "processed" / "fusion" / "jpetstore_class_order.json"

ACTION_TO_FQCN = {
    "viewcategory": "org.springframework.samples.jpetstore.web.spring.ViewCategoryController",
    "viewproduct": "org.springframework.samples.jpetstore.web.spring.ViewProductController",
    "viewitem": "org.springframework.samples.jpetstore.web.spring.ViewItemController",
    "search": "org.springframework.samples.jpetstore.web.spring.SearchProductsController",
    "searchproducts": "org.springframework.samples.jpetstore.web.spring.SearchProductsController",
    "signon": "org.springframework.samples.jpetstore.web.spring.SignonController",
    "additemtocart": "org.springframework.samples.jpetstore.web.spring.AddItemToCartController",
    "viewcart": "org.springframework.samples.jpetstore.web.spring.ViewCartController",
    "removeitemfromcart": "org.springframework.samples.jpetstore.web.spring.RemoveItemFromCartController",
    "updatecartquantities": "org.springframework.samples.jpetstore.web.spring.UpdateCartQuantitiesController",
}


def _attrs_to_dict(span: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in span.get("attributes", []) or []:
        key = a.get("key")
        val = a.get("value") or {}
        if not key or not isinstance(val, dict):
            continue
        if "stringValue" in val:
            out[key] = val["stringValue"]
    return out


def main() -> None:
    if not TRACE_PATH.exists():
        raise SystemExit(f"Trace file not found: {TRACE_PATH}")
    if not CLASS_ORDER_PATH.exists():
        raise SystemExit(f"Class order file not found: {CLASS_ORDER_PATH}")

    class_order = json.loads(CLASS_ORDER_PATH.read_text(encoding="utf-8"))
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    print("== Mapping presence in class_order ==")
    for action, fqcn in ACTION_TO_FQCN.items():
        print(f"  {action:20s} -> {'OK' if fqcn in class_to_idx else 'MISSING'}")

    hits = 0
    misses = 0
    seen_actions: set[str] = set()

    for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        for rs in data.get("resourceSpans", []) or []:
            for ss in rs.get("scopeSpans", []) or []:
                for sp in ss.get("spans", []) or []:
                    attrs = _attrs_to_dict(sp)
                    q = (attrs.get("url.query") or "").strip()
                    if not q:
                        continue

                    action_key = q.split("&", 1)[0].split("=", 1)[0].strip()
                    if not action_key:
                        continue

                    action_norm = action_key.replace("_", "").replace("-", "").lower()
                    fqcn = ACTION_TO_FQCN.get(action_norm)
                    if fqcn is None:
                        continue

                    seen_actions.add(action_norm)
                    if fqcn in class_to_idx:
                        hits += 1
                    else:
                        misses += 1

    print("\n== Observed actions in traces that are in mapping table ==")
    for a in sorted(seen_actions):
        print(f"  - {a}")

    print("\n== url.query-based mapping ==")
    print(f"  hits  : {hits}")
    print(f"  misses: {misses}")


if __name__ == "__main__":
    main()
