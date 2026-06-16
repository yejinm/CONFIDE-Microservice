import json
import os
import subprocess
from pathlib import Path


APPS = {
    "acmeair": "data/raw/acmeair",
    "daytrader": "data/raw/daytrader7",
    "plants": "data/raw/plantsbywebsphere",
    "jpetstore": "data/raw/jpetstore",
}

OUTPUT_DIR = Path("data/processed/semantic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


STATS_PATH = OUTPUT_DIR / "semantic_stats.json"


FAT_JAR = Path("tools/target/tools-fat.jar").resolve()

# Allow overriding java command (PowerShell-friendly):
#   $env:MM_JAVA_CMD="$env:JAVA_HOME\bin\java.exe"
# or:
#   $env:MM_JAVA_CMD='java'  (default)
JAVA_CMD = os.environ.get("MM_JAVA_CMD", "java").strip() or "java"


def _java_major_version(java_cmd: str) -> int | None:
    """Return Java major version by running `java -version` (best-effort)."""
    try:
        p = subprocess.run([java_cmd, "-version"], capture_output=True, text=True)
        s = (p.stderr or "") + "\n" + (p.stdout or "")
        # patterns:
        # - java version "1.8.0_161" -> major=8
        # - openjdk version "11.0.22" -> major=11
        m = None
        import re

        m = re.search(r'version\s+"(\d+)\.(\d+)\.', s)
        if m:
            # 1.8 => 8
            if m.group(1) == "1":
                return int(m.group(2))
            return int(m.group(1))
        m = re.search(r'version\s+"(\d+)', s)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


successful = []
failed = []
app_stats = {}


def find_source_roots(app_root: Path):
    """Auto-discover Java source roots within a project (supports multi-module layouts).

    Rules:
    - Prefer any directories matching **/src/main/java.
    - If none are found, fall back to using app_root itself (legacy/nonstandard layouts).
    """
    if not app_root.exists():
        return []

    # src/main/java
    src_roots = [p for p in app_root.rglob("src/main/java") if p.is_dir()]

    if not src_roots:
        # Fallback: use project root directly
        return [app_root]
    return src_roots


def run_semantic_extraction(app: str, src_root: str):
    if not FAT_JAR.exists():
        raise FileNotFoundError(
            f"Missing extractor JAR: {FAT_JAR}. Build it first with: cd tools; mvn package"
        )

    app_root = Path(src_root)
    source_roots = find_source_roots(app_root)

    if not source_roots:
        print(f"[WARN] {app}: no source roots found under {app_root}")
        failed.append(app)
        return

    merged_output = OUTPUT_DIR / f"{app}_semantic.json"
    temp_files = []

    print(f"\n[RUN] Running semantic extraction for {app} ...")
    print(f"   - app root: {app_root.resolve()}")
    print("   - discovered source roots:")
    for i, root in enumerate(source_roots):
        print(f"     [{i}] {root.resolve()}")

    # Per-module extraction, then simple merge (append JSON lists)
    merged_methods = []
    total_java_files = 0
    total_parsed_files = 0
    total_files_with_methods = 0
    total_parse_failed = 0

    for idx, src in enumerate(source_roots):
        module_tag = f"{app}::module{idx}"
        tmp_out = OUTPUT_DIR / f"{app}_semantic_{idx}.tmp.json"
        temp_files.append(tmp_out)

        # Preflight Java version check once (cheap, but keep log near execution)
        jv = _java_major_version(JAVA_CMD)
        if jv is not None and jv < 11:
            print(
                f"[WARN] Detected Java major={jv} for '{JAVA_CMD}'. "
                "tools-fat.jar requires Java 11+ (class file 55). "
                "Set $env:MM_JAVA_CMD to a Java 11+ java.exe and re-run."
            )

        cmd = [
            JAVA_CMD,
            "-jar",
            str(FAT_JAR),
            "semantic",
            str(src.resolve()),
            str(tmp_out.resolve()),
        ]
        print(f"   - [{module_tag}] java files root = {src.resolve()}")
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {module_tag} failed with exit code {e.returncode}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            failed.append(f"{app}::{module_tag}")
            continue

        try:
            if tmp_out.exists() and tmp_out.stat().st_size > 0:
                data = json.loads(tmp_out.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    merged_methods.extend(data)

            # Parse summary stats from stdout (simple string matching)
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if line.startswith("- Total Java files:"):
                    total_java_files += int(line.split(":")[-1].strip())
                elif line.startswith("- Parsed successfully:"):
                    total_parsed_files += int(line.split(":")[-1].strip())
                elif line.startswith("- Files with methods:"):
                    total_files_with_methods += int(line.split(":")[-1].strip())
                elif line.startswith("- Parse failed:"):
                    total_parse_failed += int(line.split(":")[-1].strip())
        except Exception as e:
            print(f"[WARN] Failed to merge semantic output for {module_tag}: {e}")

    
    merged_output.write_text(json.dumps(merged_methods, ensure_ascii=False, indent=2), encoding="utf-8")

    
    valid_methods = [m for m in merged_methods if isinstance(m, dict) and m.get("class") and m.get("method")]
    app_stats[app] = {
        "source_roots": [str(p) for p in source_roots],
        "methods_total": len(merged_methods),
        "methods_valid": len(valid_methods),
        "total_java_files": total_java_files,
        "total_parsed_files": total_parsed_files,
        "total_files_with_methods": total_files_with_methods,
        "total_parse_failed": total_parse_failed,
    }

    if len(valid_methods) == 0:
        print(f"[FAIL] {app}: merged semantic has 0 valid methods, please check directory layout and parser logs.")
        failed.append(app)
    else:
        print(f"[OK] {app} finished -> {merged_output} (valid methods: {len(valid_methods)})")
        successful.append(app)

   
    for tmp in temp_files:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    for app, src in APPS.items():
        run_semantic_extraction(app, src)

    
    print("\n=== Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    try:
        STATS_PATH.write_text(json.dumps(app_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Semantic stats saved to {STATS_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to write stats file: {e}")
