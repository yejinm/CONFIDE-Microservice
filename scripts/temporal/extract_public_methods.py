"""Extract public method names from a list of Java source files.

This is used to build a strict OpenTelemetry Java agent
`otel.instrumentation.methods.include` configuration that does NOT rely on `[*]`.

Usage (PowerShell):
  python .\scripts\temporal\extract_public_methods.py --base "data\\raw\\jPetStore\\src\\main\\java" --files \
      org/springframework/samples/jpetstore/domain/logic/PetStoreFacade.java \
      org/springframework/samples/jpetstore/domain/logic/PetStoreImpl.java

Output format:
  fully.qualified.ClassName[method1,method2]

Notes:
- This is heuristic: it only extracts methods declared in the given file.
- It ignores constructors and `main`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_METHOD_RE = re.compile(
    r"^\s*public\s+"  # public
    r"(?!class\b)(?!interface\b)(?!enum\b)"  # not a type decl
    r"(?:[\w<>\[\], ?]+\s+)+"  # return type + modifiers
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("  # method name
)


def _java_path_to_fqcn(java_rel_path: str) -> str:
    if java_rel_path.endswith(".java"):
        java_rel_path = java_rel_path[:-5]
    return java_rel_path.replace("/", ".").replace("\\", ".")


def extract_public_methods(java_file: Path) -> list[str]:
    methods: list[str] = []
    in_block_comment = False

    for raw in java_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw

        # crude comment stripping to avoid matching commented-out signatures
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
            continue
        if "/*" in line:
            if "*/" not in line:
                in_block_comment = True
                continue
            # one-line /* ... */
            line = re.sub(r"/\*.*?\*/", "", line)

        line = re.sub(r"//.*$", "", line)
        if not line.strip():
            continue

        m = _METHOD_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        if name == "main":
            continue
        if name not in methods:
            methods.append(name)

    return methods


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base directory that acts like a Java source root")
    ap.add_argument("--files", nargs="+", required=True, help="Java source files relative to --base")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        raise SystemExit(f"Base path does not exist: {base}")

    for rel in args.files:
        p = base / rel
        if not p.exists():
            print(f"# MISSING: {rel}")
            continue

        fqcn = _java_path_to_fqcn(rel)
        methods = extract_public_methods(p)
        if not methods:
            print(f"# WARNING: no public methods found in {rel}")
            continue

        print(f"{fqcn}[{','.join(methods)}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
