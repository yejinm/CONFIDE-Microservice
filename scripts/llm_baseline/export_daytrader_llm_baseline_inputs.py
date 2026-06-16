"""Export inputs for an LLM microservice clustering baseline on DayTrader.

What it produces (under results/llm_baseline/):
- daytrader_classes_*.json: list of classes (DayTrader uses **Total=100** classes per Table I)
- daytrader_packages.json: package summaries for context
- daytrader_dade_keywords.json: lightweight semantic keywords per class (derived from semantic extractor output)
- prompts/daytrader_prompt_gpt4o.txt: a ready-to-paste prompt asking the LLM to cluster all classes

This script is intentionally API-agnostic: you paste the generated prompt into GPT-4o/Claude/etc,
then save the model output JSON to results/llm_baseline/outputs/daytrader_llm_partition.json
and evaluate it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]

STOPWORDS = {
    # generic
    "the","a","an","and","or","to","of","in","on","for","with","as","by","from","is","are","be","this","that",
    "it","at","not","but","we","our","you","your","can","will","may","should",
    # code-ish
    "get","set","new","null","true","false","return","void","int","long","double","float","string","bool","boolean",
    "public","private","protected","static","final","class","interface","impl","bean","data","json","http","request","response",
    "servlet","filter","listener","action","trade","daytrader",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def load_class_order(system: str) -> List[str]:
    p = ROOT / "data" / "processed" / "fusion" / f"{system}_class_order.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_semantic_entries(system: str) -> List[dict]:
    p = ROOT / "data" / "processed" / "semantic" / f"{system}_semantic.json"
    return json.loads(p.read_text(encoding="utf-8"))


def split_package(cls: str) -> Tuple[str, str]:
    if "." not in cls:
        return "", cls
    pkg, simple = cls.rsplit(".", 1)
    return pkg, simple


def camel_split(s: str) -> List[str]:
    # PingServlet31AsyncRead -> Ping, Servlet, 31, Async, Read
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[0-9]+", s)
    return [p.lower() for p in parts if p]


def tokenize_text(text: str) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    return [t for t in toks if len(t) >= 3 and t not in STOPWORDS]


def build_keywords_per_class(system: str, class_order: List[str], topk: int = 12) -> Dict[str, List[str]]:
    entries = load_semantic_entries(system)
    bag: Dict[str, Counter] = defaultdict(Counter)

    for e in entries:
        cls = e.get("class")
        if not cls:
            continue
        # body tends to be noisy; keep comments + method name + variables (lighter, more "DADE-ish")
        method = e.get("method", "")
        comments = e.get("comments", "")
        variables = " ".join(e.get("variables") or [])

        toks: List[str] = []
        toks.extend(tokenize_text(comments))
        toks.extend(tokenize_text(variables))
        toks.extend(camel_split(method))

        for t in toks:
            bag[cls][t] += 1

    # also add class-name tokens (very useful signal for LLM)
    for cls in class_order:
        _pkg, simple = split_package(cls)
        for t in camel_split(simple):
            if len(t) >= 3 and t not in STOPWORDS:
                bag[cls][t] += 3

    out: Dict[str, List[str]] = {}
    for cls in class_order:
        cnt = bag.get(cls, Counter())
        out[cls] = [w for w, _ in cnt.most_common(topk)]
    return out


def build_package_summary(class_order: List[str]) -> Dict[str, dict]:
    pkg2classes: Dict[str, List[str]] = defaultdict(list)
    for cls in class_order:
        pkg, simple = split_package(cls)
        pkg2classes[pkg].append(simple)

    # keep deterministic order
    out = {}
    for pkg in sorted(pkg2classes.keys()):
        out[pkg] = {
            "count": len(pkg2classes[pkg]),
            "classes": sorted(pkg2classes[pkg]),
        }
    return out


def build_prompt(system: str, class_order: List[str], keywords: Dict[str, List[str]], few_shot: bool = True) -> str:
    # Note: few-shot uses a tiny toy example (not DayTrader) to avoid leaking GT.
    instructions = f"""You are an expert software architect.

Task: Microservice extraction via clustering.
Given a monolithic Java system, you must group classes into cohesive microservices.

System: {system}
Input: a list of 68 fully-qualified class names with their package structure, and a few lightweight semantic keywords per class.
Output: a JSON object mapping microservice names to an array of fully-qualified class names.

Hard constraints:
- Every class MUST appear in exactly one cluster.
- Do NOT drop classes.
- Do NOT create duplicate class entries.
- Use ONLY the provided class names.

Optimization goals:
- High cohesion: classes in same cluster should implement closely related business capability.
- Low coupling between clusters.
- Prefer grouping by domain functionality, not by "web vs ejb vs entity" technical layers.
- Use package structure and keywords as hints.

Output format (STRICT):
{{
  "<service_name_1>": ["com.example.A", "com.example.B"],
  "<service_name_2>": ["..."]
}}

Before producing the JSON, think step-by-step privately.
Then output ONLY the JSON and nothing else.
"""

    fewshot = ""
    if few_shot:
        fewshot = """
Few-shot toy example (format only):
Input classes:
- com.shop.web.CartController (keywords: cart, add, remove, checkout)
- com.shop.service.CartService (keywords: cart, checkout)
- com.shop.repo.CartRepository (keywords: cart, persist)
- com.shop.web.UserController (keywords: user, login, profile)
- com.shop.service.UserService (keywords: user, auth)

Valid JSON output:
{
  "Cart": [
    "com.shop.web.CartController",
    "com.shop.service.CartService",
    "com.shop.repo.CartRepository"
  ],
  "User": [
    "com.shop.web.UserController",
    "com.shop.service.UserService"
  ]
}
"""

    lines = [instructions, fewshot, f"DayTrader classes (cluster ALL {len(class_order)}):"]
    for cls in class_order:
        pkg, simple = split_package(cls)
        kw = ", ".join(keywords.get(cls, [])[:10])
        lines.append(f"- {cls}  | package: {pkg}  | keywords: {kw}")

    lines.append("\nNow output the STRICT JSON only.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="daytrader")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--no_few_shot", action="store_true")
    args = ap.parse_args()

    system = args.system

    out_root = ROOT / "results" / "llm_baseline"
    (out_root / "prompts").mkdir(parents=True, exist_ok=True)
    (out_root / "outputs").mkdir(parents=True, exist_ok=True)

    class_order = load_class_order(system)
    packages = build_package_summary(class_order)
    keywords = build_keywords_per_class(system, class_order, topk=args.topk)

    (out_root / f"{system}_classes_{len(class_order)}.json").write_text(
        json.dumps(class_order, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_root / f"{system}_packages.json").write_text(
        json.dumps(packages, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_root / f"{system}_dade_keywords.json").write_text(
        json.dumps(keywords, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    prompt = build_prompt(system, class_order, keywords, few_shot=not args.no_few_shot)
    prompt_path = out_root / "prompts" / f"{system}_prompt_gpt4o.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    print(f"[OK] Exported {len(class_order)} classes")
    print(f"[OK] Wrote prompt: {prompt_path}")
    print(
        f"Next: paste prompt into GPT-4o, save JSON to: {out_root / 'outputs' / f'{system}_llm_partition.json'}"
    )


if __name__ == "__main__":
    main()
