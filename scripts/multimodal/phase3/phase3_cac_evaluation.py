import numpy as np
import networkx as nx
import community as community_louvain
import json
import os
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from collections import Counter
from typing import Optional

# ---------------- Configuration ----------------
SYSTEMS = {
    'acmeair': {'range': (5, 5)},
    'daytrader': {'range': (5, 25)},
    # Narrow default range for reviewer-facing stability. Users can still override via --target_min/--target_max or --target_from_gt.
    'jpetstore': {'range': (3, 5)},
    'plants': {'range': (3, 8)}
}

# Default GT paths for Strategy A (GT>=0 universe filtering).
# Users can still override with --gt_path (single/template/list) or disable with --no_gt_filter_negative.
DEFAULT_GT_PATHS = {
    'acmeair': 'data/processed/groundtruth/acmeair_ground_truth.json',
    'daytrader': 'data/processed/groundtruth/daytrader_ground_truth.json',
    'jpetstore': 'data/processed/groundtruth/jpetstore_ground_truth.json',
    'plants': 'data/processed/groundtruth/plants_ground_truth.json',
}

def load_data(system, u_path=None, *, u_ablation: str = "with_u", mu_override: Optional[float] = None,
             no_dade_sem: bool = False):
    """Load S_final and U.

    - u_ablation='no_u' forces U≡0 (CAC w/o uncertainty)
    - mu_override re-mixes already-materialized semantic/structural matrices under fusion/.
    - no_dade_sem=True forces using raw semantic matrix S_sem.npy (instead of S_sem_dade_*),
      and rebuilds S_final = mu*S_struct + (1-mu)*S_sem with a default mu (or mu_override).

    This supports granular ablation without rerunning Phase1/2.
    """
    try:
        fusion_path = f"data/processed/fusion/{system}_S_final.npy"

        # Ablation: bypass DADE by rebuilding S_final from raw modality matrices.
        if bool(no_dade_sem):
            s_sem_path = f"data/processed/fusion/{system}_S_sem.npy"
            s_struct_path = f"data/processed/fusion/{system}_S_struct.npy"
            if not os.path.exists(s_sem_path) or not os.path.exists(s_struct_path):
                raise FileNotFoundError(
                    f"[{system}] no_dade_sem requires both {s_sem_path} and {s_struct_path} to exist. "
                    f"(Run Phase1 to materialize them.)"
                )
            S_sem = np.load(s_sem_path)
            S_struct = np.load(s_struct_path)
            if S_sem.shape != S_struct.shape:
                raise ValueError(f"[{system}] S_sem and S_struct shape mismatch: {S_sem.shape} vs {S_struct.shape}")
            # Default mu used only for DADE ablation; can be overridden by --mu_override
            mu = float(mu_override) if mu_override is not None else 0.50
            if not (0.0 <= mu <= 1.0):
                raise ValueError(f"[{system}] mu must be in [0,1], got {mu}")
            S = mu * S_struct + (1.0 - mu) * S_sem
            np.fill_diagonal(S, 1.0)
        elif mu_override is not None:
            s_sem_path = f"data/processed/fusion/{system}_S_sem_dade_base.npy"
            if not os.path.exists(s_sem_path):
                s_sem_path = f"data/processed/fusion/{system}_S_sem.npy"
            s_struct_path = f"data/processed/fusion/{system}_S_struct.npy"
            S_sem = np.load(s_sem_path)
            S_struct = np.load(s_struct_path)
            if S_sem.shape != S_struct.shape:
                raise ValueError(f"[{system}] S_sem and S_struct shape mismatch: {S_sem.shape} vs {S_struct.shape}")
            mu = float(mu_override)
            if not (0.0 <= mu <= 1.0):
                raise ValueError(f"[{system}] mu_override must be in [0,1], got {mu}")
            S = mu * S_struct + (1.0 - mu) * S_sem
            np.fill_diagonal(S, 1.0)
        else:
            S = np.load(fusion_path)

        uncertainty_path = u_path if u_path is not None else f"data/processed/edl/{system}_edl_uncertainty.npy"
        if (u_ablation or "with_u").lower().strip() == "no_u":
            U = np.zeros_like(S, dtype=float)
        else:
            U = np.load(uncertainty_path)

        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(f"[{system}] Fusion matrix is not N×N: shape={S.shape}")
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            raise ValueError(f"[{system}] Uncertainty matrix is not N×N: shape={U.shape}")
        if S.shape != U.shape:
            raise ValueError(f"[{system}] Fusion and uncertainty matrices shape mismatch: {S.shape} vs {U.shape}")

        u_min, u_max = float(U.min()), float(U.max())
        U_norm = (U - u_min) / (u_max - u_min) if u_max > u_min else U
        return S, U_norm
    except Exception as e:
        print(f"Error loading {system}: {e}")
        return None, None


def _apply_u_ablation(U: np.ndarray, *, policy: str, seed: int = 42) -> np.ndarray:
    """Return an ablated copy of U.

    policy:
      - with_u/normal: original U
      - no_u/zero:     U≡0
      - shuffle:       shuffle off-diagonal U values (sanity check)
    """
    p = (policy or "with_u").lower().strip()
    if p in ("with_u", "normal", "none", "orig", "original"):
        return U
    if p in ("no_u", "zero", "u0", "off"):
        return np.zeros_like(U)
    if p in ("shuffle", "shuffled"):
        rng = np.random.default_rng(int(seed))
        out = np.array(U, copy=True)
        iu = np.triu_indices(out.shape[0], k=1)
        vals = out[iu]
        rng.shuffle(vals)
        out[iu] = vals
        out[(iu[1], iu[0])] = vals
        np.fill_diagonal(out, 0.0)
        return out
    raise ValueError(f"Unknown U ablation policy: {policy}")


def _load_modality_matrix(system: str, kind: str) -> Optional[np.ndarray]:
    """Best-effort load modality similarity matrix (N×N).

    This enables 'mu' weight tuning *without* rerunning Phase1/2 by reconstructing
    S_final from persisted modality matrices when they exist.

    Supported kinds:
      - 'semantic': semantic similarity (e.g., tfidf/embedding)
      - 'structural': structural similarity
      - 'temporal': temporal similarity

    The repository uses different naming conventions across experiments; this
    function tries a small set of common candidates.
    """
    base = Path("data/processed")
    candidates = []
    k = (kind or "").lower().strip()
    if k == "semantic":
        candidates = [
            base / "semantic" / f"{system}_S_semantic.npy",
            base / "semantic" / f"{system}_semantic_similarity.npy",
            base / "semantic" / f"{system}_semantic.npy",
            base / "embedding" / f"{system}_S_semantic.npy",
        ]
    elif k == "structural":
        candidates = [
            base / "structural" / f"{system}_S_structural.npy",
            base / "structural" / f"{system}_structural_similarity.npy",
            base / "ast" / f"{system}_S_structural.npy",
        ]
    elif k == "temporal":
        candidates = [
            base / "temporal" / f"{system}_S_temp.npy",
            base / "temporal" / f"{system}_S_temporal.npy",
            base / "temporal" / f"{system}_temporal_similarity.npy",
        ]

    for p in candidates:
        try:
            if p.exists():
                M = np.load(str(p))
                if M.ndim == 2 and M.shape[0] == M.shape[1]:
                    return M
        except Exception:
            continue
    return None


def calculate_ifn(G, partition):
    # Compute IFN (Inter-Functionality Number): number of edges crossing communities
    cross_edges = sum(1 for u, v in G.edges() if partition[u] != partition[v])
    total_edges = G.number_of_edges()
    if total_edges > 0:
        ifn_ratio = cross_edges / total_edges
    else:
        ifn_ratio = 0.0
    return cross_edges, ifn_ratio

def _score_partition(modularity: float, ifn_ratio: float) -> float:
    """Higher is better. Slightly penalize cross-service edges."""
    return float(modularity) - 0.05 * float(ifn_ratio)


def _build_graphs(S: np.ndarray, U: np.ndarray, *, mode: str, k: float, n_power: float, edge_min_weight: float,
                 alpha: float = 15.0, beta: Optional[float] = None):
    """Build baseline (S) and CAC (S penalized by U) graphs.

    Modes:
      - exp:     W = S * exp(-k*U)
      - gate:    W = S*(1-U) (low-U),  S*(1-U)^n (high-U)
      - sigmoid: W = S * (1 - sigmoid(alpha*(U-beta)))
                where sigmoid(x)=1/(1+exp(-x)); beta defaults to median(U).
    """
    # normalize S to [0,1]
    S_min, S_max = float(S.min()), float(S.max())
    if S_max > S_min:
        S = (S - S_min) / (S_max - S_min)

    num_nodes = int(S.shape[0])

    # Baseline
    G_base = nx.Graph()
    for i in range(num_nodes):
        G_base.add_node(i)
        for j in range(i + 1, num_nodes):
            w = float(S[i, j])
            if w > edge_min_weight:
                G_base.add_edge(i, j, weight=w)

    # CAC
    G_cac = nx.Graph()
    mode = (mode or "exp").lower().strip()
    u_med = float(np.median(U))
    if beta is None:
        beta = u_med

    for i in range(num_nodes):
        G_cac.add_node(i)
        for j in range(i + 1, num_nodes):
            u = float(U[i, j])
            if mode == "exp":
                w = float(S[i, j]) * float(np.exp(-float(k) * u))
            elif mode == "gate":
                base = max(0.0, 1.0 - u)
                if u < u_med:
                    w = float(S[i, j]) * base
                else:
                    w = float(S[i, j]) * (base ** float(n_power))
            elif mode == "sigmoid":
                z = float(alpha) * (float(u) - float(beta))
                sig = 1.0 / (1.0 + np.exp(-z))
                w = float(S[i, j]) * float(1.0 - sig)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if w > edge_min_weight:
                G_cac.add_edge(i, j, weight=w)

    return G_base, G_cac


def _grid(values):
    return [v for v in values if v is not None]


def _autotune_cac(
    system: str,
    S: np.ndarray,
    U: np.ndarray,
    target_range,
    *,
    res_min: float,
    res_max: float,
    res_step: float,
    mode_candidates,
    k_candidates,
    n_power_candidates,
    edge_min_weight_candidates,
    alpha_candidates=None,
    beta_candidates=None,
    random_state: int = 42,
):
    """Search hyperparameters and return best CAC partition + params.

    Strategy:
      - For each (mode,k,n_power,edge_min_weight), build G_cac.
      - Sweep resolution.
      - Keep partitions whose service count is within target_range.
      - Score by Q - 0.05*IFN_ratio (same as existing tie-break).

    Returns:
      (best_metrics_dict, best_partition_dict, best_params_dict)
    """

    best = None  # tuple(score, metrics, partition, params)

    alpha_candidates = alpha_candidates or [15.0]
    beta_candidates = beta_candidates or [None]

    for mode in _grid(mode_candidates):
        for edge_min_weight in _grid(edge_min_weight_candidates):
            # k only relevant for exp, n_power only relevant for gate
            k_list = _grid(k_candidates) if mode == "exp" else [0.0]
            n_list = _grid(n_power_candidates) if mode == "gate" else [0.0]
            a_list = _grid(alpha_candidates) if mode == "sigmoid" else [0.0]
            b_list = _grid(beta_candidates) if mode == "sigmoid" else [None]

            for k in k_list:
                for n_power in n_list:
                    for alpha in a_list:
                        for beta in b_list:
                            _, G_cac = _build_graphs(
                                S,
                                U,
                                mode=mode,
                                k=float(k),
                                n_power=float(n_power),
                                edge_min_weight=float(edge_min_weight),
                                alpha=float(alpha) if mode == "sigmoid" else 15.0,
                                beta=beta,
                            )
                            if G_cac.number_of_edges() == 0:
                                continue

                            resolutions = np.arange(float(res_min), float(res_max), float(res_step))
                            for res in resolutions:
                                try:
                                    partition = community_louvain.best_partition(
                                        G_cac,
                                        weight="weight",
                                        resolution=float(res),
                                        random_state=int(random_state),
                                    )
                                    num_services = len(set(partition.values()))
                                    if not (target_range[0] <= num_services <= target_range[1]):
                                        continue
                                    q = float(community_louvain.modularity(partition, G_cac))
                                    cross_edges, ifn_ratio = calculate_ifn(G_cac, partition)
                                    score = _score_partition(q, ifn_ratio)
                                    metrics = {
                                        "Q": q,
                                        "IFN": int(cross_edges),
                                        "IFN_Ratio": float(ifn_ratio),
                                        "Services": int(num_services),
                                        "Resolution": float(res),
                                    }
                                    params = {
                                        "mode": mode,
                                        "k": float(k),
                                        "n_power": float(n_power),
                                        "edge_min_weight": float(edge_min_weight),
                                        "alpha": float(alpha) if mode == "sigmoid" else None,
                                        "beta": float(beta) if (mode == "sigmoid" and beta is not None) else None,
                                    }
                                    if (best is None) or (score > best[0]):
                                        best = (score, metrics, partition, params)
                                except Exception:
                                    continue

    if best is None:
        return None, None, None

    _, metrics, partition, params = best
    return metrics, partition, params


def run_cac_algorithm(
    system,
    S,
    U,
    target_range,
    non_linear_mode="exp",
    k=6,
    n_power=4,
    alpha: float = 15.0,
    beta: float = None,
    beta_policy: str = "median",
    res_min=0.1,
    res_max=8,
    res_step=0.05,
    edge_min_weight=0.01,
    verbose_graph=True,
    autotune: bool = False,
    autotune_budget: str = "default",
    k_policy: str = "auto",
    merge_small_clusters: bool = False,
    min_cluster_size: int = 3,
    run_meta=None,
    debug_resolutions: bool = False,
):
    # --- C. S matrix normalization is handled in _build_graphs() ---

    num_nodes = S.shape[0]
    best_partition = None
    best_score = -1.0
    best_num_services = 0
    best_ifn = 0
    best_type = None
    best_res = None
    results = []

    # Build baseline + CAC graphs
    mode = (non_linear_mode or "exp").lower().strip()
    if mode not in {"exp", "gate", "sigmoid"}:
        raise ValueError(f"Unknown --mode: {non_linear_mode}, expected exp|gate|sigmoid")

    used_median_u = None
    used_beta = None
    if mode == "exp" and str(k_policy).lower().strip() == "median_half":
        k, used_median_u = compute_adaptive_k(U, default_k=float(k))
        if verbose_graph:
            print(f"[KPolicy] {system}: k_policy=median_half => k={k:.6f} (median_u={used_median_u})")

    if mode == "sigmoid":
        # beta defaults to median(U) unless user explicitly sets beta or beta_policy
        if (beta is None) and (str(beta_policy).lower().strip() in {"median", "auto"}):
            used_beta = float(np.median(U))
            beta = used_beta
        else:
            used_beta = float(beta) if beta is not None else float(np.median(U))

        if isinstance(run_meta, dict):
            run_meta.setdefault("cac", {})
            run_meta["cac"].update({
                "alpha": float(alpha),
                "beta": float(beta) if beta is not None else None,
                "beta_policy": str(beta_policy),
            })

    if isinstance(run_meta, dict):
        run_meta.setdefault("cac", {})
        run_meta["cac"].update({
            "mode": mode,
            "k_policy": str(k_policy),
            "k_used": float(k),
            "median_u": float(used_median_u) if used_median_u is not None else None,
            "n_power": float(n_power),
            "edge_min_weight": float(edge_min_weight),
            "res_min": float(res_min),
            "res_max": float(res_max),
            "res_step": float(res_step),
            "autotune": bool(autotune),
            "autotune_budget": str(autotune_budget),
        })

    G_base, G_cac = _build_graphs(
        S, U, mode=mode, k=k, n_power=n_power, edge_min_weight=edge_min_weight,
        alpha=float(alpha), beta=beta,
    )

    if verbose_graph:
        denom = max(1, (num_nodes * (num_nodes - 1)) // 2)
        base_edges = G_base.number_of_edges()
        cac_edges = G_cac.number_of_edges()
        extra = ""
        if mode == "sigmoid":
            extra = f" alpha={float(alpha)} beta={float(beta) if beta is not None else 'None'}"
        print(
            f"[GraphDiag] {system} | nodes={num_nodes} | edge_min_weight={edge_min_weight} | "
            f"Baseline edges={base_edges} (density={base_edges/denom:.4f}) | "
            f"CAC edges={cac_edges} (density={cac_edges/denom:.4f}) | mode={mode} k={k} n_power={n_power}{extra}"
        )

    # --- Resolution sweep diagnostics (reviewer-facing fairness & reproducibility) ---
    # NOTE: res_* controls the resolution sweep for community_louvain.best_partition.
    try:
        rmin, rmax, rstep = float(res_min), float(res_max), float(res_step)
    except Exception:
        rmin, rmax, rstep = 0.1, 8.0, 0.05

    resolutions = np.arange(rmin, rmax, rstep)

    # IMPORTANT: restore initialization of bests and best_cac_params (a previous edit accidentally removed it)
    bests = {}
    cac_final_best_partition = None
    best_cac_params = {
        "mode": mode,
        "k": float(k),
        "n_power": float(n_power),
        "edge_min_weight": float(edge_min_weight),
        "res_min": float(rmin),
        "res_max": float(rmax),
        "res_step": float(rstep),
        "autotune": bool(autotune),
        "autotune_budget": str(autotune_budget),
    }

    for name, G in [("Baseline", G_base), ("CAC-Final", G_cac)]:
        # If CAC autotune requested, bypass fixed-graph sweep and use search.
        if name == "CAC-Final" and autotune:
            if autotune_budget == "fast":
                mode_candidates = ["gate", "exp"]
                k_candidates = [2.0, 4.0, 6.0]
                n_power_candidates = [1.0, 2.0, 4.0]
                edge_min_weight_candidates = [0.01, 0.005, 0.001]
                # slightly wider resolution search
                tune_res_min, tune_res_max, tune_res_step = 0.05, 12.0, res_step
            else:
                mode_candidates = ["gate", "exp"]
                k_candidates = [1.0, 2.0, 4.0, 6.0, 8.0]
                n_power_candidates = [1.0, 2.0, 3.0, 4.0]
                edge_min_weight_candidates = [0.02, 0.01, 0.005, 0.001]
                tune_res_min, tune_res_max, tune_res_step = 0.05, 16.0, max(0.02, float(res_step))

            tuned_metrics, tuned_partition, tuned_params = _autotune_cac(
                system,
                S,
                U,
                target_range,
                res_min=tune_res_min,
                res_max=tune_res_max,
                res_step=tune_res_step,
                mode_candidates=mode_candidates,
                k_candidates=k_candidates,
                n_power_candidates=n_power_candidates,
                edge_min_weight_candidates=edge_min_weight_candidates,
                random_state=42,
            )
            if tuned_metrics is not None:
                bests[name] = {
                    "Q": tuned_metrics["Q"],
                    "IFN": tuned_metrics["IFN"],
                    "IFN_Ratio": tuned_metrics["IFN_Ratio"],
                    "Services": tuned_metrics["Services"],
                }
                cac_final_best_partition = tuned_partition
                best_cac_params.update(tuned_params)
                best_cac_params["best_resolution"] = float(tuned_metrics["Resolution"])
                out_json = f"data/processed/fusion/{system}_{name.lower()}_partition.json"
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(tuned_partition, f, indent=2)
            else:
                bests[name] = None
            continue

        valid_partitions = []

        # Optional debug: memory-safe resolution->K logging.
        # IMPORTANT: do NOT keep a huge dict in memory (can crash VS Code / node worker).
        debug_out_json = None
        debug_out_jsonl = None
        debug_counts = None
        debug_k_min = None
        debug_k_max = None
        debug_target_hits = 0
        debug_total = 0
        if debug_resolutions:
            # Minimal sentinel (no console spam). Helps diagnose control-flow without large memory usage.
            try:
                import os
                os.makedirs("results/_tmp", exist_ok=True)
                with open("results/_tmp/_DEBUG_ENTERED.txt", "w", encoding="utf-8") as f:
                    f.write(f"entered: {system} {name}\n")
            except Exception:
                pass

            from collections import Counter
            debug_counts = Counter()
            try:
                os.makedirs("results/_tmp", exist_ok=True)
                safe_name = name.replace(" ", "_").replace("/", "-")
                debug_out_json = f"results/_tmp/{system}_{safe_name}_res_to_k.json"
                debug_out_jsonl = f"results/_tmp/{system}_{safe_name}_res_to_k.jsonl"
                # truncate previous run
                with open(debug_out_jsonl, "w", encoding="utf-8") as _f:
                    _f.write("")
            except Exception:
                debug_out_json = None
                debug_out_jsonl = None

        # Resolution sweep
        for res in resolutions:
            try:
                partition = community_louvain.best_partition(G, weight='weight', resolution=res, random_state=42)
                num_services = len(set(partition.values()))

                if debug_resolutions and debug_counts is not None:
                    k = int(num_services)
                    debug_total += 1
                    debug_counts[k] += 1
                    debug_k_min = k if debug_k_min is None else min(debug_k_min, k)
                    debug_k_max = k if debug_k_max is None else max(debug_k_max, k)
                    try:
                        if int(target_range[0]) == int(target_range[1]) and k == int(target_range[0]):
                            debug_target_hits += 1
                    except Exception:
                        pass
                    if debug_out_jsonl is not None:
                        # stream one line per resolution (keeps memory bounded)
                        try:
                            with open(debug_out_jsonl, "a", encoding="utf-8") as f:
                                f.write(json.dumps({"res": float(res), "k": k}, ensure_ascii=False) + "\n")
                        except Exception:
                            pass

                if target_range[0] <= num_services <= target_range[1]:
                    modularity = community_louvain.modularity(partition, G)
                    cross_edges, ifn_ratio = calculate_ifn(G, partition)
                    valid_partitions.append((modularity, ifn_ratio, cross_edges, num_services, partition, res))
            except Exception:
                continue

        # Emit compact debug summary JSON (safe, small) if requested
        if debug_resolutions and debug_out_json is not None and debug_counts is not None:
            try:
                missing = []
                if debug_k_min is not None and debug_k_max is not None and debug_k_max >= debug_k_min:
                    missing = [k for k in range(int(debug_k_min), int(debug_k_max) + 1) if k not in debug_counts]

                summary = {
                    "system": system,
                    "graph": name,
                    "target_range": [int(target_range[0]), int(target_range[1])],
                    "n_resolution_points": int(debug_total),
                    "k_min": int(debug_k_min) if debug_k_min is not None else None,
                    "k_max": int(debug_k_max) if debug_k_max is not None else None,
                    "missing_k_in_[min,max]": missing,
                    "target_k_hit_resolutions": int(debug_target_hits),
                    "k_counts": {str(k): int(v) for k, v in sorted(debug_counts.items(), key=lambda kv: kv[0])},
                    "mapping_jsonl": debug_out_jsonl,
                }
                with open(debug_out_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

                # Guaranteed sentinel + post-write verification (no stdout spam)
                try:
                    import os
                    os.makedirs("results/_tmp", exist_ok=True)
                    sentinel = f"results/_tmp/{system}_{safe_name}_debug_written.txt"
                    with open(sentinel, "w", encoding="utf-8") as sf:
                        sf.write(f"json={debug_out_json}\njsonl={debug_out_jsonl}\n")
                        sf.write(f"json_exists={os.path.exists(debug_out_json)}\n")
                        sf.write(f"jsonl_exists={os.path.exists(debug_out_jsonl) if debug_out_jsonl else None}\n")
                        sf.write(f"points={debug_total}\n")
                except Exception:
                    pass
            except Exception as e:
                # If we fail to write debug evidence, log it to a file (avoid console spam)
                try:
                    import os
                    os.makedirs("results/_tmp", exist_ok=True)
                    err_path = f"results/_tmp/{system}_{safe_name}_debug_write_error.txt"
                    with open(err_path, "a", encoding="utf-8") as ef:
                        ef.write(repr(e) + "\n")
                except Exception:
                    pass

        if valid_partitions:
            valid_qpos = [x for x in valid_partitions if x[0] > 0]
            if valid_qpos:
                valid_qpos.sort(key=lambda x: _score_partition(x[0], x[1]), reverse=True)
                best_mod, best_ifn_ratio, best_ifn, best_num, best_part, best_res = valid_qpos[0]
            else:
                valid_partitions.sort(key=lambda x: _score_partition(x[0], x[1]), reverse=True)
                best_mod, best_ifn_ratio, best_ifn, best_num, best_part, best_res = valid_partitions[0]

            # Cluster size stats (before merge)
            stats_before = _cluster_size_stats(best_part)
            _print_cluster_size_stats(system, name, "before_merge", stats_before)
            if isinstance(run_meta, dict):
                run_meta.setdefault("cluster_size_stats", {})
                run_meta["cluster_size_stats"].setdefault(name, {})
                run_meta["cluster_size_stats"][name]["before_merge"] = stats_before

            # --- Optional post-processing: merge tiny clusters ---
            if merge_small_clusters and (min_cluster_size is not None) and int(min_cluster_size) > 1:
                # IMPORTANT: merging can change K and violate target_range.
                # For audit-grade reproducibility, we must enforce target_range on the FINAL (post-merge) result.
                part_before_merge = best_part
                num_before_merge = len(set(part_before_merge.values()))

                merged = merge_tiny_clusters(part_before_merge, G, min_size=int(min_cluster_size))
                num_after_merge = len(set(merged.values()))

                # Cluster size stats (after merge)
                stats_after = _cluster_size_stats(merged)
                _print_cluster_size_stats(system, name, "after_merge", stats_after)
                if isinstance(run_meta, dict):
                    run_meta.setdefault("cluster_size_stats", {})
                    run_meta["cluster_size_stats"].setdefault(name, {})
                    run_meta["cluster_size_stats"][name]["after_merge"] = stats_after

                # Enforce target_range AFTER merge.
                # If violated, mark this candidate invalid (do NOT silently accept/fallback).
                if not (int(target_range[0]) <= int(num_after_merge) <= int(target_range[1])):
                    if verbose_graph:
                        print(
                            f"[TargetRangeGuard] {system} | {name}: merge changes K {num_before_merge}->{num_after_merge} "
                            f"outside target_range={target_range}; candidate rejected.",
                            flush=True,
                        )
                    # Reject this candidate by clearing best_part; the caller loop will treat as no valid partition for this resolution.
                    best_part = None
                else:
                    best_part = merged
                    best_num = int(num_after_merge)

                # If candidate rejected, skip persisting metrics for this G at this sweep best.
                if best_part is None:
                    if debug_resolutions and verbose_graph:
                        # We don't know which resolution produced the rejected best candidate after sorting;
                        # emit a hint that merge/target-range guard caused rejection.
                        print(f"[DebugRes] {system} | {name}: best candidate rejected by TargetRangeGuard after merge.", flush=True)
                    continue

            # --- Persist best info ---
            bests[name] = {
                "Q": float(best_mod),
                "IFN": int(best_ifn),
                "IFN_Ratio": float(best_ifn_ratio),
                "Services": int(best_num),
                "Resolution": float(best_res) if best_res is not None else None,
            }

            if verbose_graph:
                print(
                    f"[ResolutionBest] {system} | {name}: best_res={float(best_res):.4f} services={int(best_num)} "
                    f"Q={float(best_mod):.6f} IFN_Ratio={float(best_ifn_ratio):.6f}",
                    flush=True,
                )

            # Write best_resolution into run_meta for persistence in *_cac_best_gt_aligned.json
            if isinstance(run_meta, dict):
                run_meta.setdefault("best", {})
                run_meta["best"].setdefault(name, {})
                run_meta["best"][name].update({
                    "best_resolution": float(best_res) if best_res is not None else None,
                    "services": int(best_num),
                })

            # write partition json
            out_json = f"data/processed/fusion/{system}_{name.lower()}_partition.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(best_part, f, indent=2)

            if name == "CAC-Final":
                cac_final_best_partition = best_part
                best_cac_params["best_resolution"] = float(best_res) if best_res is not None else None

        else:
            bests[name] = None

    # --- Always persist a compact summary for tooling/reviewer reproducibility (prevents console truncation issues) ---
    try:
        summary_path = f"data/processed/fusion/{system}_cac_summary.json"
        compact = {
            "system": system,
            "target_range": [int(target_range[0]), int(target_range[1])],
            "baseline": bests.get("Baseline"),
            "cac_final": bests.get("CAC-Final"),
            "params": best_cac_params,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, indent=2, ensure_ascii=False)
        # Print an extra hint line so users can locate the final K easily
        print(f"[SummarySaved] {system}: {summary_path}", flush=True)
    except Exception:
        pass

    # Persist chosen CAC parameters for reproducibility (best-effort)
    try:
        params_path = f"data/processed/fusion/{system}_cac_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_cac_params, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # Before returning, store CAC best_resolution into run_meta (both single-run and sweep paths persist run_meta)
    if isinstance(run_meta, dict):
        run_meta.setdefault("cac", {})
        run_meta["cac"].setdefault("best_resolution", best_cac_params.get("best_resolution"))

    return bests, cac_final_best_partition

def get_dynamic_threshold(S: np.ndarray, *, percentile: float = 70.0, cap: float = 0.02, fallback: float = 0.01) -> float:
    """Density-Preserving Edge Pruning (DPEP).

    τ = min(percentile(nonzero(S), p), cap)

    - percentile: trims the long tail of weak edges (adapts to different graph densities)
    - cap: prevents sparse systems (e.g., AcmeAir) from being over-pruned into a fragmented graph
    """
    try:
        nz = S[S > 0]
        if nz.size == 0:
            return float(fallback)
        p_val = float(np.percentile(nz, float(percentile)))
        if not np.isfinite(p_val):
            return float(fallback)
        return float(min(p_val, float(cap)))
    except Exception:
        return float(fallback)

def compute_adaptive_k(U: np.ndarray, *, eps: float = 1e-12, default_k: float = 6.0):
    """Adaptively set k for exp mode so that exp(-k*u)=0.5 when u=median(U).

    Derivation: k = ln(2) / median(U)

    Returns: (k, median_u). Uses only the upper-triangular off-diagonal elements
    to avoid diagonal/self-loop effects.
    """
    try:
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            return float(default_k), None
        n = U.shape[0]
        if n < 2:
            return float(default_k), None
        tri = U[np.triu_indices(n, k=1)]
        tri = tri[np.isfinite(tri)]
        if tri.size == 0:
            return float(default_k), None
        med = float(np.median(tri))
        if not np.isfinite(med) or med <= float(eps):
            return float(default_k), float(med) if np.isfinite(med) else None
        return float(np.log(2.0) / med), float(med)
    except Exception:
        return float(default_k), None

def _partition_cluster_sizes(partition: dict) -> list[int]:
    """Return list of cluster sizes from a partition mapping (node->cluster)."""
    if not partition:
        return []
    c = Counter(int(cid) for cid in partition.values())
    return sorted(c.values(), reverse=True)


def _format_cluster_size_stats(
    sizes: list[int],
    *,
    label: str = "",
    small_threshold: int = 3,
    bins: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 1000000),
) -> str:
    """Create a compact, paper-friendly cluster-size summary string."""
    if not sizes:
        return f"[ClusterStats] {label} empty partition"

    n_clusters = len(sizes)
    n_nodes = int(sum(sizes))
    singletons = sum(1 for s in sizes if s == 1)
    small = sum(1 for s in sizes if s < int(small_threshold))

    # Histogram buckets: 1,2,3,4-5,6-10,11-20,21-50,51+
    labels = []
    counts = []
    edges = list(bins)
    # bins represent right-inclusive upper edges for convenience
    bucket_defs = [
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 3, "3"),
        (4, 5, "4-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 10**9, "51+"),
    ]

    def in_bucket(x: int, lo: int, hi: int) -> bool:
        return lo <= x <= hi

    for lo, hi, lab in bucket_defs:
        labels.append(lab)
        counts.append(sum(1 for s in sizes if in_bucket(s, lo, hi)))

    topk = ",".join(str(s) for s in sizes[:10])
    hist = " ".join(f"{lab}:{cnt}" for lab, cnt in zip(labels, counts))

    return (
        f"[ClusterStats] {label} clusters={n_clusters} nodes={n_nodes} "
        f"singleton_clusters={singletons} small(<{small_threshold})={small} "
        f"min={min(sizes)} max={max(sizes)} median={float(np.median(sizes)):.1f} "
        f"top10=[{topk}] hist={{ {hist} }}",
    )


def _cluster_size_stats(partition: dict) -> dict:
    """Compute simple cluster size statistics for reporting."""
    if not partition:
        return {
            "n_clusters": 0,
            "n_nodes": 0,
            "sizes": [],
            "min": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "max": 0,
            "n_singletons": 0,
            "top10": [],
        }

    counts = Counter(partition.values())
    sizes = sorted((int(v) for v in counts.values()))
    arr = np.asarray(sizes, dtype=float)

    def pct(p: float) -> int:
        return int(np.percentile(arr, p)) if arr.size else 0

    return {
        "n_clusters": int(len(sizes)),
        "n_nodes": int(sum(sizes)),
        "sizes": sizes,
        "min": int(arr.min()) if arr.size else 0,
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "max": int(arr.max()) if arr.size else 0,
        "n_singletons": int(sum(1 for s in sizes if s == 1)),
        "top10": sorted(sizes, reverse=True)[:10],
    }


def _print_cluster_size_stats(system: str, method: str, stage: str, stats: dict):
    if not stats or stats.get("n_clusters", 0) == 0:
        print(f"[ClusterSizes] {system} | {method} | {stage}: (empty)", flush=True)
        return

    print(
        f"[ClusterSizes] {system} | {method} | {stage} | "
        f"clusters={stats['n_clusters']} nodes={stats['n_nodes']} "
        f"min/p25/med/p75/max={stats['min']}/{stats['p25']}/{stats['median']}/{stats['p75']}/{stats['max']} "
        f"singletons={stats['n_singletons']} top10={stats['top10']}",
        flush=True,
    )

def _system_score(metrics: Optional[dict]) -> float:
    """Use the same scoring as internal tie-break: Q - 0.05*IFN_Ratio."""
    if not metrics:
        return float("-inf")
    try:
        return float(metrics.get("Q", 0.0)) - 0.05 * float(metrics.get("IFN_Ratio", 0.0))
    except Exception:
        return float("-inf")

def main():
    parser = argparse.ArgumentParser(description='Evaluate CAC Algorithm with modularity and IFN')
    parser.add_argument('systems', nargs='*', help='Systems to run (e.g., acmeair daytrader). If empty, run all.')

    # NEW: override target service count range (for forcing splits / paper comparison)
    parser.add_argument('--target_min', type=int, default=None, help='Override target min #services (default uses per-system range)')
    parser.add_argument('--target_max', type=int, default=None, help='Override target max #services (default uses per-system range)')

    # NEW: u_path parameter
    parser.add_argument('--u_path', type=str, default=None, help='Custom uncertainty matrix path(s), comma separated for multi-system')

    # Non-linear uncertainty decay hyperparameters
    parser.add_argument('--mode', type=str, default='exp', help='CAC non-linear mode: exp|gate|sigmoid')
    parser.add_argument('--k', type=float, default=6.0, help='exp mode: W=S*exp(-k*U)')
    parser.add_argument('--n_power', type=float, default=4.0, help='gate mode power for high-U edges: W=S*(1-U)^n')

    # NEW: sigmoid uncertainty decay
    parser.add_argument('--alpha', type=float, default=15.0, help='sigmoid mode slope alpha in W=S*(1-sigmoid(alpha*(U-beta)))')
    parser.add_argument('--beta', type=float, default=None, help='sigmoid mode center beta (if omitted, use --beta_policy)')
    parser.add_argument('--beta_policy', choices=['median', 'auto', 'fixed'], default='median', help='sigmoid beta policy: median/auto uses median(U); fixed requires --beta')

    # Resolution search hyperparameters
    parser.add_argument('--res_min', type=float, default=0.1)
    parser.add_argument('--res_max', type=float, default=8.0)
    parser.add_argument('--res_step', type=float, default=0.05)

    # NEW: edge threshold and diagnostic logs
    parser.add_argument('--edge_min_weight', type=float, default=0.01, help='Minimum edge weight to keep when building graphs (lower keeps more edges)')
    parser.add_argument('--no_graph_diag', action='store_true', help='Disable graph density diagnostic logs')

    # New: CAC auto-tuning
    parser.add_argument('--autotune', action='store_true', help='Auto-tune CAC hyperparameters (mode/k/n_power/edge_min_weight/resolution)')
    parser.add_argument('--autotune_budget', choices=['fast', 'default'], default='default', help='Search budget for --autotune')

    # NEW: k policy for exp mode (and logging for other modes)
    parser.add_argument(
        '--k_policy',
        choices=['fixed', 'auto', 'median_half'],
        default='fixed',
        help='Policy for selecting k in exp mode: fixed uses --k; auto uses k=ln(2)/median(U); median_half uses k=ln(2)/(0.5*median(U)).',
    )

    # === NEW: Strategy A (universe alignment to GT>=0) ===
    parser.add_argument('--gt_path', type=str, default=None, help='Ground truth path used for Strategy A filtering (GT>=0). Supports a {system} template or a comma-separated list aligned with systems.')
    # Strategy A is enabled by default (matches the paper/evaluation protocol). Disable it via --no_gt_filter_negative.
    parser.add_argument('--gt_filter_negative', action='store_true', default=True, help='(default on) Filter universe by GT>=0 before clustering (Strategy A)')
    parser.add_argument('--no_gt_filter_negative', action='store_false', dest='gt_filter_negative', help='Disable Strategy A filtering')

    # === DPEP: percentile + cap dynamic threshold (enabled by default) ===
    parser.add_argument('--dpep', action='store_true', default=True, help='(default on) Use DPEP dynamic edge pruning threshold')
    parser.add_argument('--no_dpep', action='store_false', dest='dpep', help='Disable DPEP dynamic threshold and use --edge_min_weight')
    parser.add_argument('--dpep_percentile', type=float, default=70.0, help='DPEP percentile p (recommended: 60-75)')
    parser.add_argument('--dpep_cap', type=float, default=0.02, help='DPEP cap (recommended: 0.02-0.05)')

    # === Target-range policy: around GT_K ± 1 (disabled by default; opt-in via --target_from_gt) ===
    parser.add_argument('--target_from_gt', action='store_true', default=False, help='Set target service range to [GT_K-1, GT_K+1] (requires --gt_path)')

    # Post-processing: merge tiny clusters to avoid singleton services
    parser.add_argument('--merge_small_clusters', action='store_true', default=False,
                        help='Post-process partitions by merging clusters smaller than --min_cluster_size')
    parser.add_argument('--min_cluster_size', type=int, default=3,
                        help='Minimum cluster size when --merge_small_clusters is enabled (default: 3)')

    # NEW: cluster size statistics output
    parser.add_argument('--cluster_stats', action='store_true', default=True,
                        help='(default on) Print cluster-size statistics before/after merge.')
    parser.add_argument('--no_cluster_stats', action='store_false', dest='cluster_stats',
                        help='Disable cluster-size statistics printing')
    parser.add_argument('--cluster_stats_small_threshold', type=int, default=3,
                        help='Define what counts as a small cluster in stats output (default: 3)')

    # NEW: sweep DPEP cap values and pick best GT-aligned point
    parser.add_argument(
        '--dpep_sweep',
        type=str,
        default=None,
        help='Comma-separated list of DPEP caps to sweep (e.g., "0.10,0.12,0.14"). When set, the script will run multiple times and keep the best CAC-Final by score (Q - 0.05*IFN_Ratio).',
    )

    # === NEW: U ablation (proof of uncertainty usefulness) ===
    parser.add_argument(
        "--u_ablation",
        type=str,
        default="with_u",
        choices=["with_u", "no_u", "shuffle"],
        help="U ablation: with_u (default), no_u (U≡0), shuffle (shuffle off-diagonal U; sanity check).",
    )
    parser.add_argument(
        "--u_ablation_seed",
        type=int,
        default=42,
        help="Seed used by --u_ablation=shuffle.",
    )
    parser.add_argument(
        "--mu_override",
        type=float,
        default=None,
        help="Recompute S = mu*S_struct + (1-mu)*S_sem (from fusion/*.npy) without rerunning Phase1/2.",
    )

    parser.add_argument(
        "--dump_edge_evidence",
        type=str,
        default=None,
        help="Optional: dump edge-level evidence JSON (useful for case study).",
    )
    parser.add_argument(
        "--no_dade_sem",
        action="store_true",
        help="Ablation: bypass DADE by rebuilding S_final from raw S_sem.npy + S_struct.npy (no Phase1/2 rerun).",
    )

    parser.add_argument('--debug_resolutions', action='store_true', help='Dump resolution->K mapping stats during the sweep (for diagnosing K=None).')

    args = parser.parse_args()

    print(f"[Phase3] args: systems={args.systems} mode={args.mode} dpep_cap={args.dpep_cap} merge_small_clusters={args.merge_small_clusters} min_cluster_size={args.min_cluster_size}", flush=True)
    results = []
    systems_to_run = args.systems if len(args.systems) > 0 else list(SYSTEMS.keys())

    # Parse --dpep_sweep once (comma-separated caps)
    sweep_caps_global = None
    if args.dpep_sweep:
        try:
            sweep_caps_global = [float(x.strip()) for x in str(args.dpep_sweep).split(',') if x.strip()]
        except Exception:
            sweep_caps_global = None

    # Keep original thresholds to avoid cross-system mutation
    base_edge_min_weight = float(args.edge_min_weight)

    # Parse --u_path for multiple systems
    u_paths = None
    if args.u_path:
        u_paths = [p.strip() for p in args.u_path.split(',')]
        if len(u_paths) == 1 and len(systems_to_run) > 1:
            u_paths = u_paths * len(systems_to_run)
        if len(u_paths) != len(systems_to_run):
            print("[ERROR] --u_path count does not match systems count!")
            return

    # Parse --gt_path supports:
    # - single path
    # - template path containing {system} (expanded per system)
    # - comma-separated list aligned with systems_to_run
    # - if --gt_path is omitted, fall back to DEFAULT_GT_PATHS[system]
    gt_paths = None
    if args.gt_path:
        raw = str(args.gt_path).strip()
        if '{system}' in raw and len(systems_to_run) > 1:
            gt_paths = [raw.format(system=s) for s in systems_to_run]
        else:
            parts = [p.strip() for p in raw.split(',') if p.strip()]
            if len(parts) == 1:
                gt_paths = parts * len(systems_to_run)
            elif len(parts) == len(systems_to_run):
                gt_paths = parts
            else:
                # leave as None; will error later if StrategyA enabled
                gt_paths = None
    else:
        # default mapping keeps CLI short
        gt_paths = [DEFAULT_GT_PATHS.get(s) for s in systems_to_run]

    print(f"{'System':<15} | {'Services':<10} | {'Modularity':<10} | {'Result':<10}", flush=True)
    print("-" * 55, flush=True)
    for idx, system in enumerate(systems_to_run):
        if system not in SYSTEMS:
            continue
        print(f"[Phase3] Running system={system} target_range_override=({args.target_min},{args.target_max})", flush=True)

        # Always reset per-system mutable args
        args.edge_min_weight = float(base_edge_min_weight)

        # Select u_path for this system
        u_path = u_paths[idx] if u_paths else None
        # Select gt_path for this system
        gt_path = gt_paths[idx] if gt_paths else None

        # If sweep is enabled: run multi-cap selection FIRST and skip the normal single-run path.
        if sweep_caps_global:
            print(f"[Sweep] {system}: dpep_sweep enabled | caps={sweep_caps_global}", flush=True)

            best_run = None  # dict(score, cap, bests, run_meta)

            for cap in sweep_caps_global:
                # Reload and prepare data fresh each cap
                S, U = load_data(
                    system,
                    u_path,
                    u_ablation=args.u_ablation,
                    mu_override=args.mu_override,
                    no_dade_sem=bool(args.no_dade_sem),
                )
                if S is None:
                    continue

                # Load class order
                node_file = f"data/processed/fusion/{system}_class_order.json"
                try:
                    with open(node_file, 'r', encoding='utf-8') as f:
                        node_names = json.load(f)
                except Exception as e:
                    print(f"[ERROR] Failed to load node list: {node_file}, {e}")
                    continue

                # Load GT
                gt_dict = None
                gt_k = None
                if gt_path:
                    try:
                        with open(gt_path, "r", encoding="utf-8") as f:
                            gt_dict = json.load(f)
                        gt_labels = [sid for sid in gt_dict.values() if isinstance(sid, (int, float)) and sid >= 0]
                        gt_k = len(set(gt_labels)) if gt_labels else None
                    except Exception as e:
                        print(f"[WARN] Failed to read gt_path: {gt_path}, {e}")
                        gt_dict = None
                        gt_k = None

                # Base target range
                local_target_range = SYSTEMS[system]['range']
                if args.target_min is not None or args.target_max is not None:
                    tmin = args.target_min if args.target_min is not None else local_target_range[0]
                    tmax = args.target_max if args.target_max is not None else local_target_range[1]
                    local_target_range = (int(tmin), int(tmax))

                if bool(args.target_from_gt):
                    if gt_k is None:
                        print("[ERROR] --target_from_gt requires a readable --gt_path")
                        return
                    local_target_range = (max(2, int(gt_k) - 1), int(gt_k) + 1)
                    print(f"[TargetPolicy] {system}: target_range set to [GT_K-1, GT_K+1] => {local_target_range} (GT_K={gt_k})")

                # StrategyA filter
                if bool(args.gt_filter_negative):
                    if gt_dict is None:
                        print("[ERROR] Strategy A (default on) requires a readable --gt_path (one per system)")
                        print("        Example: --gt_path data/processed/groundtruth/{system}_ground_truth.json")
                        print("        Or: --gt_path path1.json,path2.json,... (aligned with systems order)")
                        return
                    valid_indices, valid_names = [], []
                    for i, name in enumerate(node_names):
                        cls = str(name).replace('.java', '').strip()
                        if gt_dict.get(cls, -1) >= 0:
                            valid_indices.append(i)
                            valid_names.append(cls)
                    if len(valid_indices) == 0:
                        print(f"[ERROR] {system}: GT>=0 filtering produced an empty universe. Please check that GT keys match class_order.")
                        return
                    S = S[np.ix_(valid_indices, valid_indices)]
                    U = U[np.ix_(valid_indices, valid_indices)]

                # DPEP threshold for this cap
                dpep_tau = None
                edge_min_weight_final = float(args.edge_min_weight)
                if bool(args.dpep):
                    dpep_tau = get_dynamic_threshold(
                        S,
                        percentile=float(args.dpep_percentile),
                        cap=float(cap),
                        fallback=float(base_edge_min_weight),
                    )
                    edge_min_weight_final = max(float(base_edge_min_weight), float(dpep_tau))
                    print(
                        f"[GraphPolicy] {system}: sweep cap={cap:.3f} | p={args.dpep_percentile:.0f} => tau={dpep_tau:.6f} | edge_min_weight={edge_min_weight_final:.6f}",
                        flush=True,
                    )

                run_meta = {
                    "system": system,
                    "sweep": {"enabled": True, "dpep_cap": float(cap), "caps": sweep_caps_global},
                    "universe_policy": {
                        "gt_filter_negative": bool(args.gt_filter_negative),
                        "gt_path": str(gt_path),
                        "n_nodes": int(S.shape[0]),
                    },
                    "target_policy": {
                        "target_from_gt": bool(args.target_from_gt),
                        "target_range": [int(local_target_range[0]), int(local_target_range[1])],
                        "gt_k": int(gt_k) if gt_k is not None else None,
                    },
                    "dpep": {
                        "enabled": bool(args.dpep),
                        "percentile": float(args.dpep_percentile),
                        "cap": float(cap),
                        "tau": float(dpep_tau) if dpep_tau is not None else None,
                        "edge_min_weight_final": float(edge_min_weight_final),
                    },
                }

                print(f"\n=== [Phase 3] Evaluating CAC Algorithm for {system} (sweep cap={cap:.3f}) ===\n")
                bests, _ = run_cac_algorithm(
                    system,
                    S,
                    U,
                    local_target_range,
                    non_linear_mode=args.mode,
                    k=args.k,
                    n_power=args.n_power,
                    alpha=float(args.alpha),
                    beta=(float(args.beta) if args.beta is not None else None),
                    beta_policy=str(args.beta_policy),
                    res_min=args.res_min,
                    res_max=args.res_max,
                    res_step=args.res_step,
                    edge_min_weight=float(edge_min_weight_final),
                    verbose_graph=(not args.no_graph_diag),
                    autotune=bool(args.autotune),
                    autotune_budget=str(args.autotune_budget),
                    k_policy=str(args.k_policy),
                    merge_small_clusters=bool(args.merge_small_clusters),
                    min_cluster_size=int(args.min_cluster_size),
                    run_meta=run_meta,
                    debug_resolutions=bool(args.debug_resolutions),
                )

                score = _system_score(bests.get('CAC-Final'))
                print(f"[SweepScore] {system}: cap={cap:.3f} | score={score:.6f} | CAC={bests.get('CAC-Final')}", flush=True)

                if best_run is None or score > best_run["score"]:
                    best_run = {"score": float(score), "cap": float(cap), "bests": bests, "run_meta": run_meta}

            if best_run is None:
                print(f"[Sweep] {system}: no valid run found.")
                continue

            # Persist best meta snapshot
            try:
                best_meta_path = f"data/processed/fusion/{system}_cac_best_gt_aligned.json"
                with open(best_meta_path, "w", encoding="utf-8") as f:
                    json.dump(best_run["run_meta"], f, indent=2, ensure_ascii=False)
                print(f"[Sweep] {system}: best meta saved: {best_meta_path}")
            except Exception:
                pass

            best_cap = best_run["cap"]
            best_base = best_run["bests"].get('Baseline')
            best_cac = best_run["bests"].get('CAC-Final')

            print("\n" + "=" * 70)
            print(f"[SweepBest] {system}: best_cap={best_cap:.3f} | Baseline={best_base} | CAC-Final={best_cac}")
            print("=" * 70 + "\n")

            # record into results
            n_services = best_cac['Services'] if best_cac else 0
            mq = best_cac['Q'] if best_cac else 0.0
            results.append({"System": system, "Services": n_services, "Modularity": mq})
            continue

        # Select u_path for this system
        u_path = u_paths[idx] if u_paths else None
        # Select gt_path for this system
        gt_path = gt_paths[idx] if gt_paths else None

        S, U = load_data(
            system,
            u_path,
            u_ablation=args.u_ablation,
            mu_override=args.mu_override,
            no_dade_sem=bool(args.no_dade_sem),
        )
        if S is None:
            continue

        # Target service count range (can be overridden by CLI)
        target_range = SYSTEMS[system]['range']
        if args.target_min is not None or args.target_max is not None:
            tmin = args.target_min if args.target_min is not None else target_range[0]
            tmax = args.target_max if args.target_max is not None else target_range[1]
            target_range = (int(tmin), int(tmax))

        # Load node name list (index -> class name)
        node_file = f"data/processed/fusion/{system}_class_order.json"
        try:
            with open(node_file, 'r', encoding='utf-8') as f:
                node_names = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load node list: {node_file}, {e}")
            continue

        # === Read GT (for Strategy A filtering & optional target_range adaptation) ===
        gt_dict = None
        gt_k = None
        if gt_path:
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_dict = json.load(f)
                gt_labels = [sid for sid in gt_dict.values() if isinstance(sid, (int, float)) and sid >= 0]
                gt_k = len(set(gt_labels)) if gt_labels else None
            except Exception as e:
                print(f"[WARN] Failed to read gt_path: {gt_path}, {e}")
                gt_dict = None
                gt_k = None

        # --- target_range calculation: explicit target_min/max takes highest priority, then target_from_gt ---
        target_range = SYSTEMS[system]['range']
        has_explicit_target = (args.target_min is not None) or (args.target_max is not None)
        if has_explicit_target:
            tmin = args.target_min if args.target_min is not None else target_range[0]
            tmax = args.target_max if args.target_max is not None else target_range[1]
            target_range = (int(tmin), int(tmax))
            print(f"[TargetPolicy] {system}: explicit target_range override => {target_range}", flush=True)
        elif bool(args.target_from_gt):
            if gt_k is None:
                print("[ERROR] --target_from_gt requires a readable --gt_path")
                return
            target_range = (max(2, int(gt_k) - 1), int(gt_k) + 1)
            print(f"[TargetPolicy] {system}: target_range set to [GT_K-1, GT_K+1] => {target_range} (GT_K={gt_k})")

        # === Strategy A: filter universe by GT>=0 (enabled by default) ===
        if bool(args.gt_filter_negative):
            if gt_dict is None:
                print("[ERROR] Strategy A (default on) requires a readable --gt_path (one per system)")
                print("        Example: --gt_path data/processed/groundtruth/{system}_ground_truth.json")
                print("        Or: --gt_path path1.json,path2.json,... (aligned with systems order)")
                return

            valid_indices = []
            valid_names = []
            for i, name in enumerate(node_names):
                cls = str(name).replace('.java', '').strip()
                if gt_dict.get(cls, -1) >= 0:
                    valid_indices.append(i)
                    valid_names.append(cls)

            if len(valid_indices) == 0:
                print(f"[ERROR] {system}: GT>=0 filtering produced an empty universe. Please check that GT keys match class_order.")
                return

            S = S[np.ix_(valid_indices, valid_indices)]
            U = U[np.ix_(valid_indices, valid_indices)]
            node_names = valid_names
            print(f"[UniversePolicy] {system}: StrategyA enabled | kept={len(valid_indices)} / original={len(node_names)}")

        # === DPEP: percentile + cap (enabled by default) ===
        dpep_tau = None
        if bool(args.dpep):
            dpep_tau = get_dynamic_threshold(
                S,
                percentile=float(args.dpep_percentile),
                cap=float(args.dpep_cap),
                fallback=float(args.edge_min_weight),
            )
            args.edge_min_weight = max(float(args.edge_min_weight), float(dpep_tau))
            print(
                f"[GraphPolicy] {system}: DPEP enabled | p={args.dpep_percentile:.0f} cap={args.dpep_cap:.3f} "
                f"=> tau={dpep_tau:.6f} | final edge_min_weight={args.edge_min_weight:.6f}"
            )

        # --- Record run metadata (audit-friendly; fully traceable parameters) ---
        run_meta = {
            "system": system,
            "universe_policy": {
                "gt_filter_negative": bool(args.gt_filter_negative),
                "gt_path": str(gt_path),
                "n_nodes": int(S.shape[0]),
            },
            "target_policy": {
                "target_from_gt": bool(args.target_from_gt),
                "target_range": [int(target_range[0]), int(target_range[1])],
                "gt_k": int(gt_k) if gt_k is not None else None,
            },
            "dpep": {
                "enabled": bool(args.dpep),
                "percentile": float(args.dpep_percentile),
                "cap": float(args.dpep_cap),
                "tau": float(dpep_tau) if dpep_tau is not None else None,
                "edge_min_weight_final": float(args.edge_min_weight),
            },
            "fusion": {
                "mu_override": (float(args.mu_override) if args.mu_override is not None else None),
                "u_ablation": str(args.u_ablation),
            },
        }

        print(f"\n=== [Phase 3] Evaluating CAC Algorithm for {system} ===\n")
        bests, final_partition = run_cac_algorithm(
            system,
            S,
            U,
            target_range,
            non_linear_mode=args.mode,
            k=args.k,
            n_power=args.n_power,
            alpha=float(args.alpha),
            beta=(float(args.beta) if args.beta is not None else None),
            beta_policy=str(args.beta_policy),
            res_min=args.res_min,
            res_max=args.res_max,
            res_step=args.res_step,
            edge_min_weight=args.edge_min_weight,
            verbose_graph=(not args.no_graph_diag),
            autotune=bool(args.autotune),
            autotune_budget=str(args.autotune_budget),
            k_policy=str(args.k_policy),
            merge_small_clusters=bool(args.merge_small_clusters),
            min_cluster_size=int(args.min_cluster_size),
            run_meta=run_meta,
            debug_resolutions=bool(args.debug_resolutions),
        )

        # At end of run: print final best K (Services) and best_resolution (for reviewer fairness checks)
        try:
            b = (bests or {}).get("Baseline") or {}
            c = (bests or {}).get("CAC-Final") or {}
            bK = b.get("Services")
            cK = c.get("Services")
            bRes = b.get("Resolution")
            cRes = c.get("Resolution")
            if bK is not None or cK is not None:
                print(
                    f"[FinalK] {system}: Baseline K={bK} (best_res={bRes}) | CAC-Final K={cK} (best_res={cRes})",
                    flush=True,
                )
        except Exception:
            pass

        # Apply U ablation policy (B option uses: --u_ablation no_u)
        try:
            U = _apply_u_ablation(U, policy=str(args.u_ablation), seed=int(args.u_ablation_seed))
        except Exception as e:
            print(f"[WARN] U ablation failed: {e} (using normal U)")

        # Dump edge evidence for case study (use the actual S/U after StrategyA filtering & ablation)
        if args.dump_edge_evidence:
            try:
                out_path = Path(str(args.dump_edge_evidence))
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # load node_names consistently (after StrategyA, node_names is already valid_names)
                node_file = f"data/processed/fusion/{system}_class_order.json"
                try:
                    with open(node_file, 'r', encoding='utf-8') as f:
                        raw_node_names = json.load(f)
                except Exception:
                    raw_node_names = None

                # Build index->name map best-effort
                def idx_to_name(i: int) -> str:
                    if isinstance(node_names, list) and i < len(node_names):
                        return str(node_names[i]).replace('.java', '').strip()
                    if raw_node_names and i < len(raw_node_names):
                        return str(raw_node_names[i]).replace('.java', '').strip()
                    return str(i)

                # Compute W on the fly consistent with selected mode
                # NOTE: _build_graphs() normalizes S, so do same here
                S_min, S_max = float(S.min()), float(S.max())
                S_norm = (S - S_min) / (S_max - S_min) if S_max > S_min else np.array(S, copy=True)

                mode = str(args.mode).lower().strip()
                u_med = float(np.median(U))
                beta = float(args.beta) if args.beta is not None else float(np.median(U))

                iu = np.triu_indices(S_norm.shape[0], k=1)
                u_vals = U[iu]
                # take top uncertain edges
                topk = int(args.edge_evidence_topk)
                order = np.argsort(-u_vals)[: max(1, topk)]

                with out_path.open("a", encoding="utf-8") as f:
                    for t in order:
                        i = int(iu[0][t])
                        j = int(iu[1][t])
                        s_ij = float(S_norm[i, j])
                        u_ij = float(U[i, j])
                        if mode == "exp":
                            w_ij = s_ij * float(np.exp(-float(args.k) * u_ij))
                        elif mode == "gate":
                            base = max(0.0, 1.0 - u_ij)
                            w_ij = s_ij * (base if u_ij < u_med else (base ** float(args.n_power)))
                        else:  # sigmoid
                            z = float(args.alpha) * (u_ij - float(beta))
                            sig = 1.0 / (1.0 + np.exp(-z))
                            w_ij = s_ij * float(1.0 - sig)

                        rec = {
                            "system": system,
                            "u_ablation": str(args.u_ablation),
                            "mu": float(args.mu) if args.mu is not None else None,
                            "mode": mode,
                            "alpha": float(args.alpha),
                            "beta": float(beta),
                            "k": float(args.k),
                            "n_power": float(args.n_power),
                            "i": i,
                            "j": j,
                            "class_i": idx_to_name(i),
                            "class_j": idx_to_name(j),
                            "S": float(s_ij),
                            "U": float(u_ij),
                            "W": float(w_ij),
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                print(f"[CaseStudy] {system}: edge evidence appended: {out_path}", flush=True)
            except Exception as e:
                print(f"[WARN] dump_edge_evidence failed: {e}", flush=True)

def merge_tiny_clusters(partition: dict, G: nx.Graph, min_size: int = 3) -> dict:
    """Merge clusters smaller than min_size into the strongest-connected neighbor cluster.

    Lightweight post-processing to reduce singleton/tiny clusters.

    Returns a NEW partition dict.
    """
    part = dict(partition)

    def build_clusters(p: dict) -> dict[int, list[int]]:
        clusters: dict[int, list[int]] = {}
        for n, c in p.items():
            clusters.setdefault(int(c), []).append(int(n))
        return clusters

    changed = True
    while changed:
        changed = False
        clusters = build_clusters(part)
        small_cids = [cid for cid, nodes in clusters.items() if len(nodes) < int(min_size)]
        if not small_cids:
            break

        moved_any = False
        for cid in small_cids:
            for n in list(clusters.get(cid, [])):
                scores: dict[int, float] = {}
                for nbr in G.neighbors(n):
                    nbr_cid = int(part.get(nbr, cid))
                    if nbr_cid == cid:
                        continue
                    w = float(G.edges[n, nbr].get("weight", 1.0))
                    scores[nbr_cid] = scores.get(nbr_cid, 0.0) + w
                if not scores:
                    continue
                best_to = max(scores.items(), key=lambda kv: kv[1])[0]
                part[n] = best_to
                changed = True
                moved_any = True
                break
            if moved_any:
                break

        if not moved_any:
            break

    return part


if __name__ == "__main__":
    main()
