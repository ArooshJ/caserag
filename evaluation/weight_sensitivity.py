#!/usr/bin/env python3
"""
weight_sensitivity.py

Evaluate and tune the 5 case-field weights (facts/issues/statutes/precedents/holding)
offline — no DB required.

Two input modes (auto-detected by extension):
  .json  — pool_field_dump artifact (top-200 pool per query, dict-of-dicts)
  .npz   — field_similarity_matrix artifact (full corpus, 2195×3183×5 matrix)

The .npz mode is strictly better: full corpus → true Issues-only / Holding-zero numbers.
Use .json only for back-compat with pool_field_dump.py runs.

Search modes (combinable):
  Default (always) : evaluate STATIC_WEIGHTS or --weights on all records
  --grid-search    : exhaustive discrete grid, step via --grid-step
  --random-search  : Dirichlet-sampled weight vectors, count via --n-random
  --optimize       : scipy Nelder-Mead continuous optimisation
  --kfold          : wrap any/all search modes in grouped k-fold CV (unbiased)

Examples:
    # Static weights or a manual config (dict mode)
    python evaluation/weight_sensitivity.py
    python evaluation/weight_sensitivity.py --weights 0.0 1.0 0.0 0.0 0.0

    # Full-corpus matrix mode (true Issues-only, etc.)
    python evaluation/weight_sensitivity.py --matrix data/field_sim_matrix_emb3.npz
    python evaluation/weight_sensitivity.py --matrix data/field_sim_matrix_emb3.npz --weights 0 1 0 0 0
    python evaluation/weight_sensitivity.py --matrix data/field_sim_matrix_emb3.npz --weights 0.1 0.3 0.2 0.1 0.0

    # Unbiased k-fold CV with random search (matrix mode)
    python evaluation/weight_sensitivity.py --matrix data/field_sim_matrix_emb3.npz --kfold --random-search
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR  = Path(__file__).resolve().parent

for _p in (str(REPO_ROOT), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from retrieval_metrics import RetrievalMetrics  # noqa: E402

try:
    from ilpcsr_caserag_pipeline_functions import STATIC_WEIGHTS  # noqa: E402
except ImportError:
    STATIC_WEIGHTS = {"facts": 0.1, "issues": 0.3, "statutes": 0.2,
                      "precedents": 0.1, "holding": 0.3}

DATA_DIR    = REPO_ROOT / "evaluation" / "data"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"

FIELDS = ["facts", "issues", "statutes", "precedents", "holding"]


# ---------------------------------------------------------------------------
# Matrix-mode loader
# ---------------------------------------------------------------------------

def load_matrix(npz_path: Path) -> dict:
    """Load .npz matrix artifact + gold sidecar. Computes cosine at load time.
    Asserts field_order matches FIELDS (catches embedding drift)."""
    meta_path = npz_path.parent / (npz_path.stem + "_meta.json")
    gold_path = npz_path.parent / (npz_path.stem + "_gold.json")

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        field_order = meta.get("field_order", FIELDS)
        if field_order != FIELDS:
            print(f"[FAIL] field_order mismatch in meta: {field_order} != {FIELDS}")
            sys.exit(1)
        print(f"Matrix: emb_model_id={meta.get('emb_model_id')}  "
              f"'{meta.get('emb_model_name')}'  "
              f"n_q={meta.get('n_q')}  n_cases={meta.get('n_cases')}")

    print(f"Loading: {npz_path}")
    npz = np.load(npz_path)
    dot             = npz["dot"]              # (n_q, n_cases, 5) f32
    qnorm           = npz["qnorm"]            # (n_q,)
    cnorm           = npz["cnorm"]            # (n_cases, 5)
    doc_ids         = npz["doc_ids"].tolist()
    query_ids       = npz["query_ids"].tolist()
    source_case_ids = npz["source_case_ids"].tolist()

    # Cosine = dot / (||q|| * ||c_f||); guard zero-norms
    denom  = qnorm[:, None, None] * cnorm[None, :, :]  # (n_q, n_cases, 5)
    denom  = np.where(denom < 1e-12, 1.0, denom)
    cosine = (dot / denom).astype(np.float32)
    print(f"  cosine shape={cosine.shape}  mem={cosine.nbytes/1e6:.0f}MB")

    gold = {}
    if gold_path.exists():
        with open(gold_path, encoding="utf-8") as f:
            gold = json.load(f)
    else:
        print(f"  [WARN] Gold sidecar not found: {gold_path}")

    return {
        "cosine":          cosine,
        "doc_ids":         doc_ids,
        "query_ids":       query_ids,
        "source_case_ids": source_case_ids,
        "gold":            gold,
    }


# ---------------------------------------------------------------------------
# Core helpers — dict mode (pool_field_dump artifact)
# ---------------------------------------------------------------------------

def normalise(w: dict) -> dict:
    total = sum(w[f] for f in FIELDS)
    if total < 1e-12:
        return {f: 1.0 / len(FIELDS) for f in FIELDS}
    return {f: w[f] / total for f in FIELDS}


def score_and_rank(pool: list, weights: dict) -> list:
    """Rerank pool by weighted field sum; return list of case_id strings, best first."""
    scored = [(sum(weights[f] * entry[f] for f in FIELDS), entry["case_id"])
              for entry in pool]
    scored.sort(reverse=True)
    return [cid for _, cid in scored]


def compute_metrics(ranked_ids: list, gold_ids: set, metrics_obj: RetrievalMetrics) -> dict:
    grades = {g: 1 for g in gold_ids}
    full = metrics_obj.compute_all(ranked_ids, gold_ids, grades, gold_ids)
    full["MAP"] = full.get("AP", 0.0)
    full["MRR"] = full.get("RR", 0.0)
    return full


def eval_weights(records: list, weights: dict, metrics_obj: RetrievalMetrics) -> dict:
    """Evaluate a weight vector on pool_field_dump records; return macro-averaged metrics."""
    w = normalise(weights)
    totals = None
    n = 0
    for rec in records:
        gold = set(rec["ground_truth_cases"])
        if not gold:
            continue
        ranked = score_and_rank(rec["pool"], w)
        m = compute_metrics(ranked, gold, metrics_obj)
        if totals is None:
            totals = {k: 0.0 for k in m}
        for k in totals:
            totals[k] += m[k]
        n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Core helpers — matrix mode (field_similarity_matrix artifact)
# ---------------------------------------------------------------------------

def eval_weights_matrix(mat: dict, weights: dict, metrics_obj: RetrievalMetrics,
                        qi_list=None) -> dict:
    """Vectorized eval on full-corpus matrix. qi_list: query indices to include (None=all)."""
    w     = normalise(weights)
    w_arr = np.array([w[f] for f in FIELDS], dtype=np.float32)   # (5,)

    cosine    = mat["cosine"]           # (n_q, n_cases, 5)
    doc_ids   = mat["doc_ids"]
    query_ids = mat["query_ids"]
    gold      = mat["gold"]

    if qi_list is None:
        qi_list = range(len(query_ids))
    qi_arr = list(qi_list)

    scores         = cosine[qi_arr] @ w_arr          # (len(qi), n_cases)
    ranked_indices = np.argsort(-scores, axis=1)     # (len(qi), n_cases)

    totals = None
    n = 0
    for row_i, qi in enumerate(qi_arr):
        sub_id   = query_ids[qi]
        gold_ids = set(gold.get(sub_id, []))
        if not gold_ids:
            continue
        ranked = [doc_ids[ci] for ci in ranked_indices[row_i]]
        m = compute_metrics(ranked, gold_ids, metrics_obj)
        if totals is None:
            totals = {k: 0.0 for k in m}
        for k in totals:
            totals[k] += m[k]
        n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in totals.items()}


def eval_weights_full(records: list, weights: dict, metrics_obj: RetrievalMetrics) -> tuple:
    """Like eval_weights but also returns per-query results for results-file output."""
    w = normalise(weights)
    totals = None
    n = 0
    per_query = []
    for rec in records:
        gold = set(rec["ground_truth_cases"])
        ranked = score_and_rank(rec["pool"], w)
        m = compute_metrics(ranked, gold, metrics_obj) if gold else {}
        per_query.append({
            "query_id": str(rec.get("sub_query_id", rec.get("question", ""))),
            "retrieval_metrics": {"stage1": m},
        })
        if not gold:
            continue
        if totals is None:
            totals = {k: 0.0 for k in m}
        for k in totals:
            totals[k] += m[k]
        n += 1
    averages = {} if n == 0 else {k: v / n for k, v in totals.items()}
    return averages, per_query


def eval_weights_matrix_full(mat: dict, weights: dict, metrics_obj: RetrievalMetrics) -> tuple:
    """Like eval_weights_matrix but also returns per-query results for results-file output."""
    w     = normalise(weights)
    w_arr = np.array([w[f] for f in FIELDS], dtype=np.float32)

    cosine    = mat["cosine"]
    doc_ids   = mat["doc_ids"]
    query_ids = mat["query_ids"]
    gold      = mat["gold"]

    scores         = cosine @ w_arr               # (n_q, n_cases)
    ranked_indices = np.argsort(-scores, axis=1)  # (n_q, n_cases)

    totals = None
    n = 0
    per_query = []
    for qi in range(len(query_ids)):
        sub_id   = query_ids[qi]
        gold_ids = set(gold.get(sub_id, []))
        ranked   = [doc_ids[ci] for ci in ranked_indices[qi]]
        m = compute_metrics(ranked, gold_ids, metrics_obj) if gold_ids else {}
        per_query.append({"query_id": sub_id, "retrieval_metrics": {"stage1": m}})
        if not gold_ids:
            continue
        if totals is None:
            totals = {k: 0.0 for k in m}
        for k in totals:
            totals[k] += m[k]
        n += 1
    averages = {} if n == 0 else {k: v / n for k, v in totals.items()}
    return averages, per_query


def grouped_kfold_matrix(source_case_ids: list, k: int, seed: int):
    """Yield (train_qi, test_qi) per fold, grouped by source_case_id."""
    groups: dict = {}
    for qi, src_id in enumerate(source_case_ids):
        groups.setdefault(src_id, []).append(qi)
    keys = list(groups.keys())
    rng  = np.random.default_rng(seed)
    rng.shuffle(keys)
    folds = [keys[i::k] for i in range(k)]
    for i in range(k):
        test_keys = set(folds[i])
        train_qi = [qi for src, qis in groups.items() if src not in test_keys for qi in qis]
        test_qi  = [qi for src, qis in groups.items() if src     in test_keys for qi in qis]
        yield train_qi, test_qi


# ---------------------------------------------------------------------------
# Weight search methods
# ---------------------------------------------------------------------------

def grid_weights(step: float):
    """Generator: all weight vectors on the 5-field simplex at the given step size."""
    n_steps = round(1.0 / step)
    vals = [i / n_steps for i in range(n_steps + 1)]
    for combo in product(vals, repeat=len(FIELDS)):
        if abs(sum(combo) - 1.0) < 1e-9:
            yield dict(zip(FIELDS, combo))


def dirichlet_weights(n: int, seed: int):
    """Generator: n Dirichlet(1)-sampled weight vectors (uniform over simplex)."""
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(len(FIELDS)), size=n)
    for row in samples:
        yield dict(zip(FIELDS, row.tolist()))


# dict-mode search

def best_grid(records: list, step: float, metrics_obj: RetrievalMetrics) -> tuple:
    best_map, best_w = -1.0, None
    for w in grid_weights(step):
        ap = eval_weights(records, w, metrics_obj)["AP"]
        if ap > best_map:
            best_map, best_w = ap, w
    return best_w, best_map


def best_random(records: list, n: int, seed: int, metrics_obj: RetrievalMetrics) -> tuple:
    best_map, best_w = -1.0, None
    for w in dirichlet_weights(n, seed):
        ap = eval_weights(records, w, metrics_obj)["AP"]
        if ap > best_map:
            best_map, best_w = ap, w
    return best_w, best_map


def best_optimize(records: list, seed: int, metrics_obj: RetrievalMetrics) -> tuple:
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("[FAIL] scipy not installed. Run: pip install scipy")
        sys.exit(1)

    x0 = np.array([STATIC_WEIGHTS[f] for f in FIELDS], dtype=float)
    x0 /= x0.sum()

    def objective(x):
        x = np.clip(x, 0.0, None)
        s = x.sum()
        w = dict(zip(FIELDS, (x / s if s > 1e-12 else x).tolist()))
        return -eval_weights(records, w, metrics_obj)["AP"]

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6})
    x = np.clip(result.x, 0.0, None)
    x /= x.sum()
    return dict(zip(FIELDS, x.tolist())), -result.fun


# matrix-mode search (same logic; calls eval_weights_matrix with qi_list)

def best_grid_matrix(mat, qi_list, step, metrics_obj) -> tuple:
    best_map, best_w = -1.0, None
    for w in grid_weights(step):
        ap = eval_weights_matrix(mat, w, metrics_obj, qi_list)["AP"]
        if ap > best_map:
            best_map, best_w = ap, w
    return best_w, best_map


def best_random_matrix(mat, qi_list, n, seed, metrics_obj) -> tuple:
    best_map, best_w = -1.0, None
    for w in dirichlet_weights(n, seed):
        ap = eval_weights_matrix(mat, w, metrics_obj, qi_list)["AP"]
        if ap > best_map:
            best_map, best_w = ap, w
    return best_w, best_map


def best_optimize_matrix(mat, qi_list, seed, metrics_obj) -> tuple:
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("[FAIL] scipy not installed. Run: pip install scipy")
        sys.exit(1)

    x0 = np.array([STATIC_WEIGHTS[f] for f in FIELDS], dtype=float)
    x0 /= x0.sum()

    def objective(x):
        x = np.clip(x, 0.0, None)
        s = x.sum()
        w = dict(zip(FIELDS, (x / s if s > 1e-12 else x).tolist()))
        return -eval_weights_matrix(mat, w, metrics_obj, qi_list)["AP"]

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6})
    x = np.clip(result.x, 0.0, None)
    x /= x.sum()
    return dict(zip(FIELDS, x.tolist())), -result.fun


# ---------------------------------------------------------------------------
# K-fold harness — dict mode
# ---------------------------------------------------------------------------

def grouped_kfold(records: list, k: int, seed: int):
    """Yield (train_records, test_records) per fold, grouped by source_case_id."""
    groups: dict = {}
    for rec in records:
        groups.setdefault(rec["source_case_id"], []).append(rec)
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    folds = [keys[i::k] for i in range(k)]
    for i in range(k):
        test_keys = set(folds[i])
        train = [r for r in records if r["source_case_id"] not in test_keys]
        test  = [r for r in records if r["source_case_id"]     in test_keys]
        yield train, test


def run_kfold(records, k, repeats, seed,
              do_grid, grid_step, do_random, n_random, do_optimize,
              metrics_obj) -> dict:
    modes = (["grid"] if do_grid else []) + \
            (["random"] if do_random else []) + \
            (["optimize"] if do_optimize else [])
    results = {}

    for mode in modes:
        fold_maps, fold_weights_list = [], []
        for rep in range(repeats):
            rep_seed = seed + rep * 1000
            for fold_i, (train, test) in enumerate(grouped_kfold(records, k, rep_seed)):
                t0 = time.time()
                inner_seed = rep_seed + fold_i
                if mode == "grid":
                    best_w, train_map = best_grid(train, grid_step, metrics_obj)
                elif mode == "random":
                    best_w, train_map = best_random(train, n_random, inner_seed, metrics_obj)
                else:
                    best_w, train_map = best_optimize(train, inner_seed, metrics_obj)

                test_m   = eval_weights(test, best_w, metrics_obj)
                held_map = test_m["AP"]
                fold_maps.append(held_map)
                fold_weights_list.append(best_w)
                print(f"  [{mode}] rep={rep+1} fold={fold_i+1}/{k}  "
                      f"train_MAP={train_map:.4f}  held-out_MAP={held_map:.4f}  "
                      f"({time.time()-t0:.1f}s)  weights={_fmt_weights(best_w)}")

        mean_map = float(np.mean(fold_maps))
        std_map  = float(np.std(fold_maps))
        print(f"\n  [{mode}] {k}-fold x {repeats} repeats  "
              f"mean held-out MAP = {mean_map:.4f} ± {std_map:.4f}\n")
        results[mode] = {"fold_maps": fold_maps, "fold_weights": fold_weights_list,
                         "mean_map": mean_map, "std_map": std_map}
    return results


# ---------------------------------------------------------------------------
# K-fold harness — matrix mode
# ---------------------------------------------------------------------------

def run_kfold_matrix(mat, k, repeats, seed,
                     do_grid, grid_step, do_random, n_random, do_optimize,
                     metrics_obj) -> dict:
    modes = (["grid"] if do_grid else []) + \
            (["random"] if do_random else []) + \
            (["optimize"] if do_optimize else [])
    results = {}
    source_case_ids = mat["source_case_ids"]

    for mode in modes:
        fold_maps, fold_weights_list = [], []
        for rep in range(repeats):
            rep_seed = seed + rep * 1000
            for fold_i, (train_qi, test_qi) in enumerate(
                    grouped_kfold_matrix(source_case_ids, k, rep_seed)):
                t0 = time.time()
                inner_seed = rep_seed + fold_i
                if mode == "grid":
                    best_w, train_map = best_grid_matrix(mat, train_qi, grid_step, metrics_obj)
                elif mode == "random":
                    best_w, train_map = best_random_matrix(mat, train_qi, n_random,
                                                           inner_seed, metrics_obj)
                else:
                    best_w, train_map = best_optimize_matrix(mat, train_qi, inner_seed, metrics_obj)

                test_m   = eval_weights_matrix(mat, best_w, metrics_obj, test_qi)
                held_map = test_m["AP"]
                fold_maps.append(held_map)
                fold_weights_list.append(best_w)
                print(f"  [{mode}] rep={rep+1} fold={fold_i+1}/{k}  "
                      f"train_MAP={train_map:.4f}  held-out_MAP={held_map:.4f}  "
                      f"({time.time()-t0:.1f}s)  weights={_fmt_weights(best_w)}")

        mean_map = float(np.mean(fold_maps))
        std_map  = float(np.std(fold_maps))
        print(f"\n  [{mode}] {k}-fold x {repeats} repeats  "
              f"mean held-out MAP = {mean_map:.4f} ± {std_map:.4f}\n")
        results[mode] = {"fold_maps": fold_maps, "fold_weights": fold_weights_list,
                         "mean_map": mean_map, "std_map": std_map}
    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt_weights(w: dict) -> str:
    w = normalise(w)
    return "  ".join(f"{f}={w[f]:.3f}" for f in FIELDS)


def print_table(label: str, metrics: dict, weights: dict = None):
    print(f"\n{'─'*58}")
    print(f"  {label}")
    if weights:
        print(f"  weights : {_fmt_weights(weights)}")
    print(f"{'─'*58}")
    row = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"  {row}")
    print(f"{'─'*58}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate and tune field weights offline (dict mode or full-corpus matrix mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input — two modes
    ap.add_argument(
        "--dump", type=Path,
        default=DATA_DIR / "pool_field_dump_emb3_cosine.json",
        help="pool_field_dump JSON (dict mode, top-200 pool). "
             "Ignored when --matrix is given.",
    )
    ap.add_argument(
        "--matrix", type=Path, default=None,
        help="field_sim_matrix .npz (full-corpus matrix mode). "
             "When given, overrides --dump. "
             "Default search: data/field_sim_matrix_emb3.npz",
    )
    ap.add_argument(
        "--weights", type=float, nargs=5,
        metavar=("FACTS", "ISSUES", "STATUTES", "PRECEDENTS", "HOLDING"),
        default=None,
        help="Manual weight vector (normalised to sum=1). Default: STATIC_WEIGHTS.",
    )
    # Search modes
    ap.add_argument("--grid-search",   action="store_true")
    ap.add_argument("--grid-step",     type=float, default=0.1)
    ap.add_argument("--random-search", action="store_true")
    ap.add_argument("--n-random",      type=int, default=2000)
    ap.add_argument("--optimize",      action="store_true")
    # K-fold
    ap.add_argument("--kfold",   action="store_true")
    ap.add_argument("--k",       type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed",    type=int, default=42)
    # Output
    ap.add_argument("--output",   type=Path, default=None)
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10, 20, 30, 50, 100])
    args = ap.parse_args()

    any_search = args.grid_search or args.random_search or args.optimize
    if args.kfold and not any_search:
        print("[FAIL] --kfold requires at least one of --grid-search / --random-search / --optimize")
        sys.exit(1)

    matrix_mode = args.matrix is not None
    metrics_obj = RetrievalMetrics(k_values=args.k_values)
    all_results = {}

    # ── Load data ────────────────────────────────────────────────────────
    if matrix_mode:
        mat = load_matrix(args.matrix)
        n_queries = len(mat["query_ids"])
        print(f"Matrix mode: {n_queries} queries  {len(mat['doc_ids'])} corpus cases")
    else:
        if not args.dump.exists():
            print(f"[FAIL] Dump file not found: {args.dump}")
            print("       Run pool_field_dump.py first, or pass --matrix for matrix mode.")
            sys.exit(1)
        print(f"Loading: {args.dump}")
        with open(args.dump, encoding="utf-8") as f:
            dump = json.load(f)
        records = dump["records"]
        meta    = dump.get("_meta", {})
        print(f"Records      : {len(records)}")
        print(f"Embedding    : {meta.get('emb_model_name', '?')}  "
              f"distance_op={meta.get('distance_op', '?')}")
        print(f"Pool sizes   : {meta.get('pool_size_stats', {})}")

    # ── Manual / static (always runs) ────────────────────────────────────
    if args.weights is not None:
        manual_w = normalise(dict(zip(FIELDS, args.weights)))
    else:
        manual_w = normalise(dict(STATIC_WEIGHTS))

    mode_label = "matrix" if matrix_mode else f"pool (top-{meta.get('pool_size_stats',{}).get('median','?')})"
    print(f"\n── Manual / static evaluation [{mode_label}] ──")
    print(f"   weight order: facts  issues  statutes  precedents  holding")
    print(f"   using: {_fmt_weights(manual_w)}")

    if matrix_mode:
        manual_m, per_query_results = eval_weights_matrix_full(mat, manual_w, metrics_obj)
    else:
        manual_m, per_query_results = eval_weights_full(records, manual_w, metrics_obj)

    print_table("Manual weights — all queries", manual_m, manual_w)
    all_results["manual"] = {"weights": manual_w, "average_metrics": {"retrieval": {"stage1": manual_m}}}

    # ── K-fold CV ────────────────────────────────────────────────────────
    if args.kfold:
        print(f"\n── K-fold CV  (k={args.k}, repeats={args.repeats}, seed={args.seed}) ──")
        if matrix_mode:
            kfold_res = run_kfold_matrix(
                mat, args.k, args.repeats, args.seed,
                args.grid_search, args.grid_step,
                args.random_search, args.n_random,
                args.optimize, metrics_obj,
            )
        else:
            kfold_res = run_kfold(
                records, args.k, args.repeats, args.seed,
                args.grid_search, args.grid_step,
                args.random_search, args.n_random,
                args.optimize, metrics_obj,
            )
        all_results["kfold"] = kfold_res

        if len(kfold_res) > 1:
            print(f"\n{'─'*58}")
            print(f"  K-fold CV summary  (k={args.k} x {args.repeats} repeats)")
            print(f"{'─'*58}")
            for mode, res in kfold_res.items():
                print(f"  {mode:<10}  mean held-out MAP = {res['mean_map']:.4f} ± {res['std_map']:.4f}")
            print(f"  {'static':<10}  MAP = {manual_m['AP']:.4f}  (all queries, no k-fold)")
            print(f"{'─'*58}")

    elif any_search:
        print("\n[WARN] Running search on full dataset without --kfold.")
        print("       This result is biased. Do NOT report as system performance.\n")

        if args.grid_search:
            print(f"── Grid search (full data, step={args.grid_step}) ──")
            t0 = time.time()
            if matrix_mode:
                best_w, _ = best_grid_matrix(mat, None, args.grid_step, metrics_obj)
                m = eval_weights_matrix(mat, best_w, metrics_obj)
            else:
                best_w, _ = best_grid(records, args.grid_step, metrics_obj)
                m = eval_weights(records, best_w, metrics_obj)
            print_table(f"Best grid — full data [BIASED] ({time.time()-t0:.1f}s)", m, best_w)
            all_results["grid_full_data"] = {"weights": best_w, "average_metrics": {"retrieval": {"stage1": m}}}

        if args.random_search:
            print(f"── Random search (full data, n={args.n_random}) ──")
            t0 = time.time()
            if matrix_mode:
                best_w, _ = best_random_matrix(mat, None, args.n_random, args.seed, metrics_obj)
                m = eval_weights_matrix(mat, best_w, metrics_obj)
            else:
                best_w, _ = best_random(records, args.n_random, args.seed, metrics_obj)
                m = eval_weights(records, best_w, metrics_obj)
            print_table(f"Best random — full data [BIASED] ({time.time()-t0:.1f}s)", m, best_w)
            all_results["random_full_data"] = {"weights": best_w, "average_metrics": {"retrieval": {"stage1": m}}}

        if args.optimize:
            print("── Scipy optimize (full data) ──")
            t0 = time.time()
            if matrix_mode:
                best_w, _ = best_optimize_matrix(mat, None, args.seed, metrics_obj)
                m = eval_weights_matrix(mat, best_w, metrics_obj)
            else:
                best_w, _ = best_optimize(records, args.seed, metrics_obj)
                m = eval_weights(records, best_w, metrics_obj)
            print_table(f"Optimised — full data [BIASED] ({time.time()-t0:.1f}s)", m, best_w)
            all_results["optimize_full_data"] = {"weights": best_w, "average_metrics": {"retrieval": {"stage1": m}}}

    # ── Save ─────────────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_doc = {
            "average_metrics": {"retrieval": {"stage1": manual_m}},
            "per_query_results": per_query_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_doc, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
