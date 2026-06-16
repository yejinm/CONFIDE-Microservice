import os
import subprocess
import json
from pathlib import Path
import logging
import sys
from collections import Counter, defaultdict

# -------------------- Path configuration --------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # workspace root
TOOLS_DIR = BASE_DIR / "tools"
RAW_DIR = BASE_DIR / "data/raw"
AST_DIR = BASE_DIR / "data/processed/ast"
CG_DIR = BASE_DIR / "data/processed/callgraph"
DEP_DIR = BASE_DIR / "data/processed/dependency"

AST_DIR.mkdir(parents=True, exist_ok=True)
CG_DIR.mkdir(parents=True, exist_ok=True)
DEP_DIR.mkdir(parents=True, exist_ok=True)


MAVEN_CMD = os.environ.get("MM_MAVEN_CMD", "mvn").strip() or "mvn"


APPS = ["acmeair", "daytrader7", "jPetStore", "plantsbywebsphere"]

# -------------------- Logging configuration --------------------
LOG_FILE = BASE_DIR / "extract_features.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("extract_features")

# -------------------- Auto-discover source roots --------------------
def find_source_roots(app_root: Path):
    src_roots = [p for p in app_root.rglob("src/main/java") if p.is_dir()]
    if not src_roots:
        return [app_root]
    return src_roots

# -------------------- JSON deduplication helper --------------------
def deduplicate_json(file_path: Path):
    if not file_path.exists():
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return  # only deduplicate list-style JSON (e.g., semantic)

        seen = set()
        unique_items = []
        for item in data:
            key = (
                item.get("class", ""),
                item.get("method_name", ""),
                " ".join(item.get("variables", []))
            )
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        if len(unique_items) < len(data):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(unique_items, f, ensure_ascii=False, indent=2)
            logger.info(f"Deduplicated {file_path.name}: {len(data)} -> {len(unique_items)}")

    except Exception as e:
        logger.warning(f"Deduplication skipped for {file_path.name}, error: {e}")

# -------------------- Multi-module AST/CallGraph/Dependency extraction --------------------
def run_ast_extractor(app_name, src_roots, output_path):
    AST_EXPORTER_JAR = TOOLS_DIR / "target" / "tools-fat.jar"
    if not AST_EXPORTER_JAR.exists():
        raise FileNotFoundError(
            f"Missing extractor JAR: {AST_EXPORTER_JAR}. Build it first with: cd tools; mvn package"
        )
    total_ast = {}
    for idx, src_dir in enumerate(src_roots):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmpf:
            tmp_json = tmpf.name
        cmd = [
            'java', '-cp', str(AST_EXPORTER_JAR),
            'extractor.ast.JavaParserASTExtractor',
            str(src_dir), tmp_json
        ]
        logger.info(f'[{app_name}] AST extraction {src_dir} ...')
        try:
            subprocess.run(cmd, check=True)
            with open(tmp_json, 'r', encoding='utf-8') as f:
                ast = json.load(f)
            for k, v in ast.items() if isinstance(ast, dict) else []:
                if k in total_ast:
                    logger.warning(f'Warning: duplicate class {k}, overwritten')
                total_ast[k] = v
        except Exception as e:
            logger.error(f'[{app_name}] AST extraction {src_dir} failed: {e}')
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(total_ast, f, indent=2, ensure_ascii=False)
    logger.info(f'[{app_name}] AST written to {output_path}, total classes: {len(total_ast)}')
    return len(total_ast)


def run_graph_extractor(app_name, extractor_type, src_roots, output_path):
    # Merge multi-module outputs for callgraph/dependency
    graph_nodes = set()
    graph_edges = set()
    for idx, src_dir in enumerate(src_roots):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmpf:
            tmp_json = tmpf.name
        cmd = [
            MAVEN_CMD, "exec:java",
            "-Dexec.mainClass=extractor.Main",
            f"-Dexec.args={extractor_type} {src_dir} {tmp_json}"
        ]
        logger.info(f'[{app_name}] {extractor_type} extraction {src_dir} ...')
        try:
            subprocess.run(cmd, cwd=str(TOOLS_DIR), shell=True, check=True)
            with open(tmp_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            nodes = set(data.get('nodes', []))
            edges = set((e['source'], e['target']) for e in data.get('edges', []))
            graph_nodes.update(nodes)
            graph_edges.update(edges)
        except Exception as e:
            logger.error(f'[{app_name}] {extractor_type} extraction {src_dir} failed: {e}')
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
    # Merge output
    merged = {
        'nodes': sorted(graph_nodes),
        'edges': [{'source': s, 'target': t} for s, t in graph_edges]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info(
        f'[{app_name}] {extractor_type} written to {output_path}, nodes: {len(graph_nodes)}, edges: {len(graph_edges)}'
    )
    return len(graph_nodes), len(graph_edges)

# -------------------- Main --------------------
def main():
    stats = {}
    for app in APPS:
        app_root = RAW_DIR / app
        src_roots = find_source_roots(app_root)
        logger.info(f'[{app}] Discovered {len(src_roots)} source root(s):')
        for d in src_roots:
            logger.info('  %s', d)
        ast_output = AST_DIR / f"{app}_ast.json"
        cg_output = CG_DIR / f"{app}_callgraph.json"
        dep_output = DEP_DIR / f"{app}_dependency.json"
        ast_count = run_ast_extractor(app, src_roots, ast_output)
        cg_nodes, cg_edges = run_graph_extractor(app, "callgraph", src_roots, cg_output)
        dep_nodes, dep_edges = run_graph_extractor(app, "dependency", src_roots, dep_output)
        stats[app] = {
            "ast_classes": ast_count,
            "callgraph_nodes": cg_nodes,
            "callgraph_edges": cg_edges,
            "dependency_nodes": dep_nodes,
            "dependency_edges": dep_edges,
            "src_roots": [str(p) for p in src_roots],
        }
    # Write stats
    stats_path = BASE_DIR / "data/processed/structural/structural_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Structural modality stats written to {stats_path}")

if __name__ == "__main__":
    main()
