import os
import json
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Data root: <repo>/data/processed
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed'))
FUSION_DIR = os.path.join(DATA_ROOT, 'fusion')
EMBEDDING_DIR = os.path.join(DATA_ROOT, 'embedding')
TEMPORAL_DIR = os.path.join(DATA_ROOT, 'temporal')
SEMANTIC_DIR = os.path.join(DATA_ROOT, 'semantic')
CALLGRAPH_DIR = os.path.join(DATA_ROOT, 'callgraph')
DEPENDENCY_DIR = os.path.join(DATA_ROOT, 'dependency')
AST_DIR = os.path.join(DATA_ROOT, 'ast')

SYSTEM_CONFIG = {
    'acmeair': {
        'class_order_file': 'acmeair_class_order.json',
        # preferred output name from scripts/semantic/extract_embeddings.py
        'class_embedding_json': 'acmeair_class_embeddings.json',
        'temporal_file': 'acmeair_S_temp.npy',
        'callgraph_file': 'acmeair_callgraph.json',
        'dependency_file': 'acmeair_dependency.json',
        'ast_file': 'acmeair_ast.json',
        'use_dade_sem': True,
    },
    'daytrader': {
        'class_order_file': 'daytrader_class_order.json',
        'class_embedding_json': 'daytrader_class_embeddings.json',
        'temporal_file': 'daytrader_S_temp.npy',
        'callgraph_file': 'daytrader7_callgraph.json',
        'dependency_file': 'daytrader7_dependency.json',
        'ast_file': 'daytrader7_ast.json',
        'use_dade_sem': True,
    },
    'jpetstore': {
        'class_order_file': 'jpetstore_class_order.json',
        'class_embedding_json': 'jpetstore_class_embeddings.json',
        'temporal_file': 'jpetstore_S_temp.npy',
        'callgraph_file': 'jpetstore_callgraph.json',
        'dependency_file': 'jpetstore_dependency.json',
        'ast_file': 'jpetstore_ast.json',
        'use_dade_sem': True,
    },
    'plants': {
        'class_order_file': 'plants_class_order.json',
        'class_embedding_json': 'plants_class_embeddings.json',
        'temporal_file': 'plants_S_temp.npy',
        'callgraph_file': 'plantsbywebsphere_callgraph.json',
        'dependency_file': 'plantsbywebsphere_dependency.json',
        'ast_file': 'plantsbywebsphere_ast.json',
        'use_dade_sem': True,
    },
}


def _pick_newest_existing(paths: List[str]) -> Optional[str]:
    existing = [p for p in paths if p and os.path.isfile(p)]
    if not existing:
        return None
    existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return existing[0]


def _resolve_semantic_inputs(system: str) -> Tuple[str, str]:
    """Resolve (class_embedding_json_path, embeddings_pt_path) with backward-compatible fallbacks.

    We currently have legacy artifacts like:
      - {system}_embeddings.pt
      - {system}_embeddings_class_embeddings.json

    and current artifacts like:
      - {system}.pt
      - {system}_class_embeddings.json

    This function picks the newest existing file among the candidates, to ensure the pipeline
    actually consumes the most recently generated semantic embeddings.
    """
    cfg = SYSTEM_CONFIG[system]

    # candidates for class-level embeddings
    class_json_candidates = [
        os.path.join(EMBEDDING_DIR, cfg.get('class_embedding_json', '')),
        os.path.join(EMBEDDING_DIR, f"{system}_class_embeddings.json"),
        os.path.join(EMBEDDING_DIR, f"{system}_embeddings_class_embeddings.json"),
    ]

    # candidates for tensor embedding file
    pt_candidates = [
        os.path.join(EMBEDDING_DIR, f"{system}.pt"),
        os.path.join(EMBEDDING_DIR, f"{system}_embeddings.pt"),
    ]

    class_json = _pick_newest_existing(class_json_candidates)
    pt_path = _pick_newest_existing(pt_candidates)

    if class_json is None:
        raise FileNotFoundError(
            f"No class embedding JSON found for {system}. Tried: {class_json_candidates}"
        )
    if pt_path is None:
        # It's OK for build_semantic_matrix (json-based) but we log so users notice.
        logging.warning("[SEM][%s] No embeddings .pt found (tried: %s)", system, pt_candidates)
        pt_path = ''

    return class_json, pt_path


def safe_normalize(M: np.ndarray) -> np.ndarray:
    M = np.array(M, dtype=np.float32)
    if M.size == 0:
        return M
    minv, maxv = M.min(), M.max()
    if maxv - minv < 1e-9:
        return np.zeros_like(M)
    return (M - minv) / (maxv - minv)


def load_class_order(system: str):
    cfg = SYSTEM_CONFIG[system]
    class_order_path = os.path.join(FUSION_DIR, cfg['class_order_file'])
    with open(class_order_path, 'r', encoding='utf-8') as f:
        class_order = json.load(f)
    return class_order


def build_semantic_matrix(system: str, class_order):
    """Unified semantic modality builder: class-level embeddings + optional DADE scaling.

    Phase-1 feature reshaping:
    - Down-weight entity/DTO edges: apply a penalty on entity-like class edges in the
      semantic similarity matrix to avoid inflated "shared data model" affinity.
    """
    cfg = SYSTEM_CONFIG[system]

    class_embedding_json_path, embeddings_pt_path = _resolve_semantic_inputs(system)
    logging.info(
        "[SEM][%s] Using class embeddings JSON: %s (mtime=%s)",
        system, class_embedding_json_path, int(os.path.getmtime(class_embedding_json_path))
    )
    if embeddings_pt_path:
        logging.info(
            "[SEM][%s] Detected embeddings PT (not required by this step): %s (mtime=%s)",
            system, embeddings_pt_path, int(os.path.getmtime(embeddings_pt_path))
        )

    if not os.path.isfile(class_embedding_json_path):
        raise FileNotFoundError(f"Class embedding JSON not found for {system}: {class_embedding_json_path}")

    with open(class_embedding_json_path, 'r', encoding='utf-8') as f:
        class_embeddings = json.load(f)

    class2embedding = {}
    for item in class_embeddings:
        cls = item.get('class')
        emb = item.get('embedding')
        if cls is None or emb is None:
            continue
        class2embedding[cls] = np.array(emb, dtype=np.float32)

    if not class2embedding:
        raise ValueError(f"No valid class embeddings found in {class_embedding_json_path} for system {system}")

    D = len(next(iter(class2embedding.values())))
    N = len(class_order)
    E = np.zeros((N, D), dtype=np.float32)
    missing = 0
    # Pre-compute global mean embedding
    global_mean_emb = np.mean(list(class2embedding.values()), axis=0)
    # Pre-build package -> embeddings list mapping
    pkg2embs = {}
    for cls, emb in class2embedding.items():
        pkg = cls.rsplit('.', 1)[0] if '.' in cls else ''
        pkg2embs.setdefault(pkg, []).append(emb)
    for i, cls in enumerate(class_order):
        if cls in class2embedding:
            E[i] = class2embedding[cls]
        else:
            missing += 1
            pkg = cls.rsplit('.', 1)[0] if '.' in cls else ''
            pkg_embs = pkg2embs.get(pkg, [])
            if pkg_embs:
                E[i] = np.mean(pkg_embs, axis=0)
            else:
                E[i] = global_mean_emb
    if missing > 0:
        logging.warning("[%s] %d classes missing embeddings, filled with package or global mean.", system, missing)

    # Persist the N×D embedding matrix needed by EDL/DADE
    emb_out_path = os.path.join(FUSION_DIR, f'{system}_S_sem_embedding.npy')
    np.save(emb_out_path, E)
    logging.info("[SEM][%s] Node embedding matrix saved: %s (shape=%s)", system, emb_out_path, E.shape)

    S_raw = cosine_similarity(E)
    S_sem_raw = (S_raw + 1.0) / 2.0

    # --- Entity/DTO/entity-like down-weighting (semantic-level) ---
    def _is_entity_like(cls_name: str) -> bool:
        name = (cls_name or "").lower()
        # Common entity/DTO naming patterns
        return any(tok in name for tok in [
            ".entity", ".entities", ".jpa", ".domain", ".model", ".dto", ".vo", ".po"
        ])

    # Penalty factors: entity-entity edges are penalized more strongly than entity-other
    entity_entity_factor = 0.6
    entity_other_factor = 0.8
    entity_mask = np.array([_is_entity_like(c) for c in class_order], dtype=bool)

    if entity_mask.any():
        # entity-entity
        ee = np.outer(entity_mask, entity_mask)
        S_sem_raw[ee] *= entity_entity_factor
        # entity-other (exclusive)
        eo = np.outer(entity_mask, ~entity_mask) | np.outer(~entity_mask, entity_mask)
        S_sem_raw[eo] *= entity_other_factor
        # diagonal keep 1
        np.fill_diagonal(S_sem_raw, 1.0)
        S_sem_raw = np.clip(S_sem_raw, 0.0, 1.0)
        logging.info(
            "[SEM][%s] Applied entity de-emphasis (ee=%.2f, eo=%.2f), entity_like=%d/%d",
            system, entity_entity_factor, entity_other_factor, int(entity_mask.sum()), len(entity_mask)
        )

    raw_sem_path = os.path.join(FUSION_DIR, f'{system}_S_sem.npy')
    np.save(raw_sem_path, S_sem_raw)
    logging.info("[SEM][%s] Raw semantic matrix saved: %s (mean=%.4f)", system, raw_sem_path, S_sem_raw.mean())

    # Optional DADE-rescaled semantic matrix
    # NOTE: Historically this repo treated `{system}_S_sem_dade.npy` as a cache and loaded it whenever it exists.
    # That can silently hide changes from new embeddings / new semantic denoising.
    # We therefore detect staleness: if class-embeddings JSON is newer than DADE, DADE is considered stale.
    dade_type = os.environ.get('MM_DADE_TYPE', 'base').strip().lower()
    dade_sem_path = os.path.join(FUSION_DIR, f"{system}_S_sem_dade_{dade_type}.npy")

    use_dade = bool(SYSTEM_CONFIG[system].get('use_dade_sem'))
    dade_exists = os.path.isfile(dade_sem_path)

    # `class_embedding_json_path` comes from `_resolve_semantic_inputs(system)` above.
    class_json_mtime = None
    dade_mtime = None
    try:
        class_json_mtime = os.path.getmtime(class_embedding_json_path)
    except Exception:
        class_json_mtime = None
    try:
        if dade_exists:
            dade_mtime = os.path.getmtime(dade_sem_path)
    except Exception:
        dade_mtime = None

    dade_is_stale = False
    if use_dade and dade_exists and class_json_mtime is not None and dade_mtime is not None:
        dade_is_stale = class_json_mtime > dade_mtime

    if use_dade and dade_exists and not dade_is_stale:
        logging.info(
            "[SEM][%s] Using DADE semantic matrix: %s (dade_mtime=%s)",
            system, dade_sem_path, int(dade_mtime) if dade_mtime is not None else 'unknown'
        )
        S_sem = np.load(dade_sem_path)
    else:
        if use_dade and dade_exists and dade_is_stale:
            logging.warning(
                "[SEM][%s] DADE semantic matrix is stale; will regenerate it. "
                "class_embeddings_json is newer (json_mtime=%s > dade_mtime=%s).",
                system,
                int(class_json_mtime) if class_json_mtime is not None else 'unknown',
                int(dade_mtime) if dade_mtime is not None else 'unknown',
            )
            try:
                # Auto-regenerate DADE so the pipeline always uses the newest semantics.
                # The generator reads `{system}_S_sem.npy` and writes `{system}_S_sem_dade.npy`.
                import importlib.util

                _dade_path = os.path.join(os.path.dirname(__file__), 'rescale_semantic_dade.py')
                spec = importlib.util.spec_from_file_location('rescale_semantic_dade', _dade_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Unable to load DADE module from: {_dade_path}")
                _dade_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_dade_mod)
                _dade_process_system = getattr(_dade_mod, 'process_system')

                target_mean_raw = os.environ.get('MM_DADE_TARGET_MEAN', '').strip()
                target_mean = float(target_mean_raw) if target_mean_raw else 0.5

                # Generate the requested DADE variant.
                # 'base' is intended to be no-topk (paper-safe).
                _dade_type = os.environ.get('MM_DADE_TYPE', 'base').strip().lower()
                _topk_ratio = 0.0 if _dade_type == 'base' else 0.0
                _dade_process_system(system, target_mean=target_mean, topk_ratio=_topk_ratio, out_dir=FUSION_DIR)

                # reload after regeneration
                if os.path.isfile(dade_sem_path):
                    S_sem = np.load(dade_sem_path)
                    logging.info(
                        "[SEM][%s] Regenerated + loaded DADE semantic matrix: %s (target_mean=%.3f)",
                        system, dade_sem_path, target_mean,
                    )
                else:
                    logging.warning("[SEM][%s] DADE regeneration did not produce file, fallback to raw S_sem.", system)
                    S_sem = S_sem_raw
            except Exception as e:
                logging.exception("[SEM][%s] Failed to regenerate DADE semantic matrix, fallback to raw S_sem. err=%s", system, e)
                S_sem = S_sem_raw
        elif use_dade and not dade_exists:
            logging.info("[SEM][%s] DADE semantic matrix not found, generating it now.", system)
            try:
                import importlib.util

                _dade_path = os.path.join(os.path.dirname(__file__), 'rescale_semantic_dade.py')
                spec = importlib.util.spec_from_file_location('rescale_semantic_dade', _dade_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Unable to load DADE module from: {_dade_path}")
                _dade_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_dade_mod)
                _dade_process_system = getattr(_dade_mod, 'process_system')

                target_mean_raw = os.environ.get('MM_DADE_TARGET_MEAN', '').strip()
                target_mean = float(target_mean_raw) if target_mean_raw else 0.5

                _dade_process_system(system, target_mean=target_mean, topk_ratio=0.2, out_dir=FUSION_DIR)

                if os.path.isfile(dade_sem_path):
                    S_sem = np.load(dade_sem_path)
                    logging.info(
                        "[SEM][%s] Generated + loaded DADE semantic matrix: %s (target_mean=%.3f)",
                        system, dade_sem_path, target_mean,
                    )
                else:
                    logging.warning("[SEM][%s] DADE generation did not produce file, fallback to raw S_sem.", system)
                    S_sem = S_sem_raw
            except Exception as e:
                logging.exception("[SEM][%s] Failed to generate DADE semantic matrix, fallback to raw S_sem. err=%s", system, e)
                S_sem = S_sem_raw
        else:
            S_sem = S_sem_raw

    return S_sem


def _intra_inter_stats_matrix(S: np.ndarray, labels: List[int]) -> Tuple[float, float, float, int, int]:
    """Compute intra/inter mean similarity given a label per node.

    labels: list[int] where -1 means unknown/skipped.
    """
    n = len(labels)
    intra = []
    inter = []
    skipped = 0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == -1 or labels[j] == -1:
                skipped += 1
                continue
            if labels[i] == labels[j]:
                intra.append(float(S[i, j]))
            else:
                inter.append(float(S[i, j]))
    intra_avg = float(np.mean(intra)) if intra else float('nan')
    inter_avg = float(np.mean(inter)) if inter else float('nan')
    ratio = intra_avg / inter_avg if inter_avg > 0 else float('inf')
    return intra_avg, inter_avg, ratio, len(intra), len(inter)


def _load_gt_labels_for_debug(system: str, class_order: List[str]) -> List[int]:
    """Load ground-truth labels aligned with class_order for debugging (Phase 1 only)."""
    cfg = SYSTEM_CONFIG[system]
    gt_path = os.path.join(DATA_ROOT, 'groundtruth', f"{system}_ground_truth.json")
    # NOTE: in this repo, class_order lives under fusion; GT under groundtruth.
    if not os.path.isfile(gt_path):
        logging.warning("[DEBUG][%s] Ground-truth not found: %s (skip decomposition stats)", system, gt_path)
        return [-1] * len(class_order)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_map = json.load(f)
    return [gt_map.get(cls, -1) for cls in class_order]


class CallGraphParser:
    """Reuse AcmeAir call-graph parsing logic and apply it consistently across systems."""

    def __init__(self, class_set, external_prefix_denylist=None):
        self.class_set = class_set
        self.cache = {}
        self.external_prefix_denylist = set(external_prefix_denylist or [])

    def parse_to_class(self, signature: str):
        if not signature:
            return None
        if signature in self.cache:
            return self.cache[signature]
        # Strip method signature suffix and normalize inner-class separators
        path_part = signature.split('(')[0].replace('$', '.')
        parts = path_part.split('.')
        # Match from the longest prefix backwards to find a class FQN in class_set
        for i in range(len(parts) - 1, 0, -1):
            potential = '.'.join(parts[:i])
            if potential in self.class_set:
                self.cache[signature] = potential
                return potential
        return None

    def get_lib_prefix(self, signature: str):
        clean_name = signature.split('(')[0]
        parts = clean_name.split('.')
        if len(parts) >= 2:
            prefix = '.'.join(parts[:2])
        else:
            prefix = 'external_lib'
        # allow filtering common framework prefixes that create cross-service noise
        if prefix in self.external_prefix_denylist:
            return None
        return prefix


def build_structural_matrix(system: str, class_order):
    """Unified structural modality builder: inheritance + dependency + calls + package structure.

    Phase-1 feature reshaping:
    - Package-distance penalty: use "package distance" (not package similarity) to avoid
      over-pulling all classes within the same package.
      Intuition: being in the same package does not necessarily mean being in the same
      service; package distance is only a weak prior, and larger distance => smaller similarity.
    """

    cfg = SYSTEM_CONFIG[system]
    N = len(class_order)
    cls2idx = {cls: i for i, cls in enumerate(class_order)}
    class_set = set(class_order)

    # ---- Callgraph: direct + external co-occurrence ----
    callgraph_path = os.path.join(CALLGRAPH_DIR, cfg['callgraph_file'])
    if not os.path.isfile(callgraph_path):
        logging.warning("[STRUCT][%s] Callgraph not found: %s, using zeros.", system, callgraph_path)
        S_call = np.zeros((N, N), dtype=np.float32)
        S_direct = np.zeros((N, N), dtype=np.float32)
        S_indirect = np.zeros((N, N), dtype=np.float32)
    else:
        with open(callgraph_path, 'r', encoding='utf-8') as f:
            cg_data = json.load(f)
        logging.info("[STRUCT][%s] Callgraph edges loaded: %d", system, len(cg_data.get('edges', [])))

        # External prefix denylist: remove ultra-common frameworks that tend to connect everything.
        # (Safe defaults; can be extended later.)
        external_deny = {
            'java.lang', 'java.util', 'java.io', 'java.time', 'java.net',
            'javax.servlet', 'jakarta.servlet', 'javax.ws', 'jakarta.ws',
            'javax.persistence', 'jakarta.persistence',
            'org.slf4j', 'org.apache', 'com.fasterxml',
            'org.springframework',
        }

        parser = CallGraphParser(class_set, external_prefix_denylist=external_deny)
        W_direct = np.zeros((N, N), dtype=np.float32)
        external_cooc = {}
        matched_direct = 0

        for edge in cg_data.get('edges', []):
            src_sig = edge.get('source')
            tgt_sig = edge.get('target')
            src_cls = parser.parse_to_class(src_sig)
            tgt_cls = parser.parse_to_class(tgt_sig)

            if src_cls is None:
                continue
            s_idx = cls2idx[src_cls]

            if tgt_cls is not None and src_cls != tgt_cls:
                W_direct[s_idx, cls2idx[tgt_cls]] += 1
                matched_direct += 1
            else:
                lib = parser.get_lib_prefix(tgt_sig or '')
                if not lib:
                    continue
                if lib not in external_cooc:
                    external_cooc[lib] = np.zeros(N, dtype=np.float32)
                external_cooc[lib][s_idx] += 1

        S_direct = safe_normalize(W_direct + W_direct.T)
        S_indirect = np.zeros((N, N), dtype=np.float32)
        if external_cooc:
            feat_matrix = np.array(list(external_cooc.values())).T  # shape: (N, num_libs)
            S_indirect = cosine_similarity(feat_matrix)

        if matched_direct > 0:
            # Default: keep a balance; indirect tends to be noisy if jars/callgraph are very complete.
            # Per-system override: daytrader is especially sensitive to shared framework calls.
            if system == 'daytrader':
                S_call = 0.70 * S_direct + 0.30 * safe_normalize(S_indirect)
            else:
                S_call = 0.50 * S_direct + 0.50 * safe_normalize(S_indirect)
        else:
            S_call = safe_normalize(S_indirect)

    # ---- Dependency graph ----
    dependency_path = os.path.join(DEPENDENCY_DIR, cfg.get('dependency_file', '')) if cfg.get('dependency_file') else ''
    S_dep = np.zeros((N, N), dtype=np.float32)
    if dependency_path and os.path.isfile(dependency_path):
        with open(dependency_path, 'r', encoding='utf-8') as f:
            dep_data = json.load(f)
        edges = dep_data.get('edges', [])
        logging.info("[STRUCT][%s] Dependency edges loaded: %d", system, len(edges))

        # --- Dependency denoising knobs (env-driven) ---
        # Defaults are set to the recommended Phase1→Phase2 baseline.
        dep_filter_entitylike = os.environ.get('MM_DEP_FILTER_ENTITYLIKE', '1').strip() in {'1', 'true', 'True'}
        dep_use_idf = os.environ.get('MM_DEP_IDF', '1').strip() in {'1', 'true', 'True'}

        # Thresholds for hub detection
        hub_indegree_cut = os.environ.get('MM_DEP_HUB_INDEGREE', '').strip()
        hub_percent_cut = os.environ.get('MM_DEP_HUB_PERCENT', '').strip()
        hub_indegree_cut = int(hub_indegree_cut) if hub_indegree_cut else None
        # default: 5% of nodes
        hub_percent_cut = float(hub_percent_cut) if hub_percent_cut else 0.05

        # Optional package rules:
        # - MM_DEP_NOISE_PKG_CONTAINS: semicolon-separated substrings, matched against target's package
        #   Example: ".dto;.entity;.model;.domain;.common;.util;.constants;.config"
        pkg_contains_raw = os.environ.get('MM_DEP_NOISE_PKG_CONTAINS', '').strip()
        noise_pkg_contains = [s.strip().lower() for s in pkg_contains_raw.split(';') if s.strip()] if pkg_contains_raw else []

        def _tgt_pkg(tgt: str) -> str:
            if not tgt or '.' not in tgt:
                return ''
            return tgt.rsplit('.', 1)[0].lower()

        def _is_entitylike_or_noise_pkg(cls_name: str) -> bool:
            n = (cls_name or '').lower()
            pkg = _tgt_pkg(cls_name)

            # user-provided package substrings have highest priority
            if noise_pkg_contains and any(sub in pkg for sub in noise_pkg_contains):
                return True

            # built-in heuristics (conservative)
            if any(tok in pkg for tok in [
                '.dto', '.vo', '.po', '.entity', '.entities', '.model', '.domain',
                '.common', '.util', '.utils', '.constant', '.constants', '.config',
            ]):
                return True

            simple = cls_name.split('.')[-1].lower() if cls_name else ''
            if simple.endswith(('dto', 'vo', 'po', 'entity', 'entities', 'model')):
                return True
            if 'exception' in simple or 'logger' in simple:
                return True
            return False

        # Build target indegree counts
        tgt_in_count = {}
        for edge in edges:
            tgt = edge.get('target')
            if tgt in class_set:
                tgt_in_count[tgt] = tgt_in_count.get(tgt, 0) + 1

        # Decide hub set
        hub_set = set()
        percent_threshold = max(1, int(np.ceil(float(hub_percent_cut) * N)))
        for tgt, deg in tgt_in_count.items():
            if hub_indegree_cut is not None:
                if deg >= hub_indegree_cut:
                    hub_set.add(tgt)
            else:
                if deg >= percent_threshold:
                    hub_set.add(tgt)

        # Convert indegree to weight: higher indegree => smaller weight
        def _idf_weight(tgt: str) -> float:
            if not dep_use_idf:
                return 1.0
            deg = float(tgt_in_count.get(tgt, 0))
            return float(np.log((N + 1.0) / (deg + 1.0)) / np.log(N + 1.0))

        # For very-common targets (almost everyone depends on), drop or heavily attenuate.
        # Default: drop when idf < 0.12 (override with MM_DEP_IDF_MIN).
        idf_min_raw = os.environ.get('MM_DEP_IDF_MIN', '').strip()
        dep_idf_min = float(idf_min_raw) if idf_min_raw else 0.12

        kept = 0
        filtered = 0
        dropped_by_idf = 0
        for edge in edges:
            src = edge.get('source')
            tgt = edge.get('target')
            if src in class_set and tgt in class_set and src != tgt:
                w = _idf_weight(tgt)

                # Drop extremely common targets regardless of package heuristics.
                if dep_use_idf and w < dep_idf_min:
                    dropped_by_idf += 1
                    continue

                # Only filter entity-like targets when they're hubs (reduces over-filtering on small projects like acmeair)
                if dep_filter_entitylike and (tgt in hub_set) and _is_entitylike_or_noise_pkg(tgt):
                    filtered += 1
                    continue

                i, j = cls2idx[src], cls2idx[tgt]
                S_dep[i, j] = max(S_dep[i, j], w)
                S_dep[j, i] = max(S_dep[j, i], w)
                kept += 1

        if dep_filter_entitylike or dep_use_idf:
            logging.info(
                "[STRUCT][%s] Dependency denoise(v3): filter_entitylike=%s(idfHubOnly) idf=%s hubSize=%d hubMode=%s idf_min=%.3f kept=%d filtered=%d drop_idf=%d",
                system,
                dep_filter_entitylike,
                dep_use_idf,
                len(hub_set),
                (f"indegree>={hub_indegree_cut}" if hub_indegree_cut is not None else f"percent>={hub_percent_cut}"),
                dep_idf_min,
                kept,
                filtered,
                dropped_by_idf,
            )
    else:
        if dependency_path:
            logging.warning("[STRUCT][%s] Dependency file not found or not configured, skipping.", system)

    # ---- AST / inheritance ----
    ast_path = os.path.join(AST_DIR, cfg.get('ast_file', '')) if cfg.get('ast_file') else ''
    S_inh = np.zeros((N, N), dtype=np.float32)
    if ast_path and os.path.isfile(ast_path):
        with open(ast_path, 'r', encoding='utf-8') as f:
            ast_data = json.load(f)
        logging.info("[STRUCT][%s] AST classes loaded: %d", system, len(ast_data))
        for cls_fqn, info in ast_data.items():
            if cls_fqn not in cls2idx:
                continue
            u = cls2idx[cls_fqn]
            parents = info.get('bases', []) + info.get('interfaces', [])
            for p_name in parents:
                if p_name in class_set:
                    v = cls2idx[p_name]
                    S_inh[u, v] = 1.0
                    S_inh[v, u] = 1.0
    else:
        if ast_path:
            logging.warning("[STRUCT][%s] AST file not found or not configured, skipping.", system)

    # ---- Package similarity (with distance penalty) ----
    S_pkg = np.zeros((N, N), dtype=np.float32)
    pkgs = ['.'.join(cls.split('.')[:-1]) for cls in class_order]

    def _pkg_distance(p1: str, p2: str) -> float:
        a = p1.split('.') if p1 else []
        b = p2.split('.') if p2 else []
        common = 0
        for x, y in zip(a, b):
            if x == y:
                common += 1
            else:
                break
        # distance: suffix length after common prefix
        return (len(a) - common) + (len(b) - common)

    for i in range(N):
        for j in range(i, N):
            if pkgs[i] == pkgs[j]:
                sim = 1.0
            else:
                dist = _pkg_distance(pkgs[i], pkgs[j])
                # Map distance to (0, 1]: larger distance => smaller similarity
                sim = 1.0 / (1.0 + dist)
            S_pkg[i, j] = S_pkg[j, i] = sim

    # ---- Weighted fusion of structural cues ----
    S_inh_n = safe_normalize(S_inh)
    S_dep_n = safe_normalize(S_dep)
    S_call_n = safe_normalize(S_call)
    S_pkg_n = safe_normalize(S_pkg)

    # Per-system structural fusion weights (defaults are conservative)
    w_inh, w_dep, w_call, w_pkg = 0.30, 0.30, 0.30, 0.10

    # Daytrader: dependency edges are still the noisiest; reduce dep weight.
    if system == 'daytrader':
        w_inh, w_dep, w_call, w_pkg = 0.30, 0.15, 0.35, 0.20

    # Plants: dep ratio tends to be <1, pkg is strong and helps separation.
    if system == 'plants':
        w_inh, w_dep, w_call, w_pkg = 0.25, 0.15, 0.35, 0.25

    # Normalize weights
    s = w_inh + w_dep + w_call + w_pkg
    w_inh, w_dep, w_call, w_pkg = w_inh / s, w_dep / s, w_call / s, w_pkg / s

    S_struct_pure = (
        w_inh * S_inh_n +
        w_dep * S_dep_n +
        w_call * S_call_n +
        w_pkg * S_pkg_n
    )
    S_struct_pure = np.clip(S_struct_pure, 0.0, 1.0)

    # Optional decomposition diagnostics (requires GT)
    if os.environ.get('MM_DEBUG_STRUCT_DECOMP', '').strip() in {'1', 'true', 'True'}:
        labels = _load_gt_labels_for_debug(system, class_order)
        for name, M in [
            ('inh', S_inh_n),
            ('dep', S_dep_n),
            ('call', S_call_n),
            ('pkg', S_pkg_n),
            ('struct', S_struct_pure),
        ]:
            intra_avg, inter_avg, ratio, n_intra, n_inter = _intra_inter_stats_matrix(M, labels)
            logging.info(
                "[DEBUG][%s][struct-decomp] %s intra=%.3f inter=%.3f ratio=%.2f pairs=%d/%d",
                system, name, intra_avg, inter_avg, ratio, n_intra, n_inter,
            )

    struct_path = os.path.join(FUSION_DIR, f"{system}_S_struct.npy")
    np.save(struct_path, S_struct_pure)
    logging.info("[STRUCT][%s] Structural matrix saved: %s (mean=%.4f)", system, struct_path, S_struct_pure.mean())

    return S_struct_pure


def load_temporal_matrix(system: str, shape_like: np.ndarray):
    """Load the temporal modality matrix; return an all-zero matrix if missing."""
    cfg = SYSTEM_CONFIG[system]
    temporal_file = cfg.get('temporal_file')
    if not temporal_file:
        logging.warning("[TEMP][%s] No temporal_file configured, using zeros.", system)
        return np.zeros_like(shape_like)

    temporal_path = os.path.join(TEMPORAL_DIR, temporal_file)
    if os.path.isfile(temporal_path):
        S_temp = np.load(temporal_path)
        if S_temp.shape != shape_like.shape:
            logging.warning(
                "[TEMP][%s] Temporal matrix shape %s mismatches structural/semantic shape %s, resizing with zeros.",
                system, S_temp.shape, shape_like.shape,
            )
            # Simple fallback: return a same-shape zero matrix to avoid crashing
            S_temp = np.zeros_like(shape_like)
        logging.info("[TEMP][%s] Using temporal matrix: %s (mean=%.4f)", system, temporal_path, S_temp.mean())
        return S_temp
    else:
        logging.warning("[TEMP][%s] Temporal matrix not found: %s, using zeros.", system, temporal_path)
        return np.zeros_like(shape_like)


def build_matrices_for_system(system: str, w_sem: float = 0.4, w_struct: float = 0.4, w_temp: float = 0.2):
    """Unified high-quality tri-modality build+fusion pipeline for all systems."""
    if system not in SYSTEM_CONFIG:
        raise ValueError(f"Unknown system: {system}")

    logging.info("[PIPELINE] Building similarity matrices for system: %s", system)

    # 1) Load class order
    class_order = load_class_order(system)
    N = len(class_order)
    logging.info("[PIPELINE][%s] Number of classes: %d", system, N)

    # 2) Semantic modality
    S_sem = build_semantic_matrix(system, class_order)

    # 3) Structural modality
    S_struct = build_structural_matrix(system, class_order)

    # 4) Temporal modality
    S_temp = load_temporal_matrix(system, S_struct)
    # --- Safety guard: force alignment ---
    if S_temp.shape != S_struct.shape:
        logging.error(f"FATAL: S_temp shape {S_temp.shape} still mismatch with {S_struct.shape}")
        N = S_struct.shape[0]
        S_temp_fixed = np.zeros((N, N), dtype=np.float32)
        min_n = min(S_temp.shape[0], N)
        S_temp_fixed[:min_n, :min_n] = S_temp[:min_n, :min_n]
        S_temp = S_temp_fixed
    # --------------------------

    # 5) Tri-modality fusion: weights (alpha/beta/gamma)
    # NOTE: avoid too-strong structural term for AcmeAir; allow per-run override.
    if system == 'acmeair':
        # safer default for acmeair: reduce structural dominance
        w_sem, w_struct, w_temp = 0.45, 0.25, 0.30

    ws = float(w_sem)
    wt = float(w_temp)
    wb = float(w_struct)
    s = ws + wb + wt
    if s <= 0:
        raise ValueError('Fusion weights must sum to a positive value')
    ws, wb, wt = ws / s, wb / s, wt / s

    logging.info("[PIPELINE][%s] Fusion weights: w_sem=%.3f w_struct=%.3f w_temp=%.3f", system, ws, wb, wt)

    S_final = ws * S_sem + wb * S_struct + wt * S_temp
    S_final = np.clip(S_final, 0.0, 1.0)

    final_path = os.path.join(FUSION_DIR, f'{system}_S_final.npy')
    np.save(final_path, S_final)
    logging.info(
        "[PIPELINE][%s] Final fused similarity matrix saved: %s (mean=%.4f)",
        system, final_path, S_final.mean(),
    )


def main():
    parser = argparse.ArgumentParser(description="Build multimodal matrices")
    parser.add_argument('--system', choices=sorted(SYSTEM_CONFIG.keys()), required=True)
    parser.add_argument('--w-sem', type=float, default=0.4)
    parser.add_argument('--w-struct', type=float, default=0.4)
    parser.add_argument('--w-temp', type=float, default=0.2)
    parser.add_argument(
        '--dade_type',
        type=str,
        default='base',
        help=(
            "Which DADE semantic matrix variant to use when use_dade_sem is enabled. "
            "Options: base (default) or rho_0p10, rho_0p08, etc. "
            "This selects fusion/{system}_S_sem_dade_<dade_type>.npy."
        ),
    )
    args = parser.parse_args()

    # Provide a CLI override for downstream semantic selection.
    # We use the existing env-based mechanism to avoid invasive refactors.
    os.environ['MM_DADE_TYPE'] = str(args.dade_type).strip().lower()

    build_matrices_for_system(args.system, w_sem=args.w_sem, w_struct=args.w_struct, w_temp=args.w_temp)


if __name__ == '__main__':
    main()