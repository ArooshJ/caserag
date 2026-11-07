#!/usr/bin/env python
"""
significance.py - Paired statistical significance testing across result files.

WHY THIS EXISTS
    evaluate.py scores exactly ONE result file and has no notion of comparing
    systems. Significance testing is inherently a BETWEEN-systems operation over
    aligned per-query scores, so it needs its own entry point. IR venues treat
    significance testing as table stakes: a raw "MAP 0.1338 vs 0.1308" reads as a
    win when a paired test shows it is statistically noise. This script answers,
    for every CaseRAG ablation config against each baseline (and optionally
    config-vs-config), "is this difference real or noise?".

WHAT IT DOES
    1. Loads several evaluate.py output JSONs (the files in results/).
    2. Aligns them on the intersection of queries (same query scored by every
       system), keyed by (query_id, question) so file order does not matter.
    3. For each metric and each system pair, runs a PAIRED BOOTSTRAP on per-query
       differences: mean diff, 95% percentile CI, and a one-sided indicator. A
       pair/metric is SIGNIFICANT when the 95% CI excludes 0. Bootstrap is used
       (not a t-test) because IR metrics like AP are bounded, skewed, and zero-
       heavy, so normality does not hold.
    4. Optionally adds a Wilcoxon signed-rank p-value (--wilcoxon), the conventional
       named test reviewers expect, when SciPy is available.

USAGE (from repo root)
    python evaluation/significance.py
    python evaluation/significance.py --wilcoxon --all-pairs
    python evaluation/significance.py --stage stage1 --metrics AP RR Recall@100
    python evaluation/significance.py --output evaluation/results/significance.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

# SciPy is an optional dependency: only needed for --wilcoxon. Probe once so the
# script degrades gracefully (prints a note) instead of crashing when it is absent.
try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "significance.json"

DEFAULT_METRICS = ["AP", "RR", "Recall@5", "Recall@20", "Recall@100"]


def system_label(path: Path) -> str:
    """Derive a short, readable system name from a result filename.

    Why: comparison tables and the saved JSON should say "CaseRAG_k1=200_k2=100"
    or "BM25", not "results_model=1_k1=200_k2=100.json". Keeps the paper-facing
    output legible without the caller having to pass labels by hand.
    """
    name = path.stem
    low = name.lower()
    if "bm25" in low:
        return "BM25"
    if "dense" in low:
        return "Dense"
    label = name
    if label.startswith("results_"):
        label = label[len("results_"):]
    label = label.replace("model=1_", "").replace("model=1", "").strip("_")
    return f"CaseRAG_{label}" if label else name


def is_baseline(path: Path) -> bool:
    """A file is a baseline if its name names bm25 or dense; else it is a system
    under test (a CaseRAG config). Used to build the default comparison set."""
    low = path.stem.lower()
    return ("bm25" in low) or ("dense" in low)


def discover_files(results_dir: Path):
    """Default file set: the CaseRAG ablation result files plus the two baselines.
    Old/one-off result dumps (old_*, pld_*) are excluded by the glob patterns."""
    caserag = sorted(results_dir.glob("results_model=*.json"))
    baselines = [p for p in (results_dir / "bm25_results.json",
                             results_dir / "dense_results.json") if p.exists()]
    return caserag, baselines


def load_per_query(path: Path, stage: str) -> dict:
    """Read one evaluate.py result file into {(query_id, question[:60]): metric_dict}
    for the requested stage. The composite key lets the same query be matched across
    systems even when the files are in different orders."""
    data = json.load(open(path, "r", encoding="utf-8"))
    rows = data["per_query_results"] if isinstance(data, dict) else data
    out = {}
    for r in rows:
        key = (str(r.get("query_id")), (r.get("question") or "")[:60])
        out[key] = r.get("retrieval_metrics", {}).get(stage, {}) or {}
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> dict:
    """Bootstrap the mean of the paired differences a_i - b_i.

    Resamples the differences with replacement n_boot times; the 2.5/97.5 percentiles
    of the resample means form the 95% CI. `p_le0` is the fraction of resample means
    <= 0 (a one-sided indicator that A does not beat B). Significance = CI excludes 0.
    """
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_le0": float((boot_means <= 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
    }


def wilcoxon_p(a: np.ndarray, b: np.ndarray):
    """Two-sided Wilcoxon signed-rank p (median paired difference != 0).

    Two-sided on purpose: it is reported next to the two-sided bootstrap CI, so both
    columns must answer the same direction-agnostic question. Read the SIGN of the
    effect from `mean_diff`, not from this p. (A one-sided 'greater' test would be
    structurally blind to losses, printing p=1.000 wherever the system is worse.)
    Returns None if SciPy is absent or every paired difference is zero (undefined).
    """
    if not _SCIPY_AVAILABLE:
        return None
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.all(diffs == 0):
        return None
    try:
        _, p = _scipy_wilcoxon(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return None


def build_pairs(labels_paths, baselines_set, all_pairs: bool):
    """Default: each system-under-test vs each baseline. With --all-pairs, also add
    every system-vs-system unordered pair (to test the k1 effect statistically)."""
    sut = [lp for lp in labels_paths if lp[1] not in baselines_set]
    base = [lp for lp in labels_paths if lp[1] in baselines_set]
    pairs = [(a, b) for a in sut for b in base]
    if all_pairs:
        allsys = labels_paths
        for i in range(len(allsys)):
            for j in range(i + 1, len(allsys)):
                pair = (allsys[i], allsys[j])
                if pair not in pairs and (pair[1], pair[0]) not in pairs:
                    pairs.append(pair)
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Paired significance testing across evaluate.py result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="Directory of evaluate.py result JSONs (default: evaluation/results)")
    ap.add_argument("--files", nargs="+", default=None,
                    help="Explicit result files to compare (overrides auto-discovery)")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                    help=f"Per-query metrics to test (default: {' '.join(DEFAULT_METRICS)})")
    ap.add_argument("--stage", choices=["stage1", "stage2"], default="stage2",
                    help="Which retrieval stage's per-query scores to compare (default: stage2)")
    ap.add_argument("--all-pairs", action="store_true",
                    help="Also compare config-vs-config, not just systems-vs-baselines")
    ap.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples (default: 10000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible bootstrap")
    ap.add_argument("--wilcoxon", action="store_true", help="Also report Wilcoxon signed-rank p (needs SciPy)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to save the significance JSON")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    if args.files:
        paths = [Path(f) for f in args.files]
        baselines = [p for p in paths if is_baseline(p)]
    else:
        caserag, baselines = discover_files(results_dir)
        paths = caserag + baselines
    paths = [p for p in paths if p.exists()]
    if len(paths) < 2:
        print(f"[FAIL] Need >= 2 result files to compare; found {len(paths)} in {results_dir}")
        sys.exit(1)

    if args.wilcoxon and not _SCIPY_AVAILABLE:
        print("[WARN] --wilcoxon requested but SciPy is not installed; skipping Wilcoxon (bootstrap still runs).")

    # Load every system, align on the intersection of query keys.
    loaded = {}                       # label -> {key: metric_dict}
    labels_paths = []                 # [(path, label)] preserving discovery order
    for p in paths:
        lbl = system_label(p)
        loaded[lbl] = load_per_query(p, args.stage)
        labels_paths.append((p, lbl))
        print(f"  loaded {lbl:28s} <- {p.name}  ({len(loaded[lbl])} queries)")

    common = set.intersection(*[set(d.keys()) for d in loaded.values()])
    keys = sorted(common)
    print(f"\n[OK] aligned queries across all {len(loaded)} systems: {len(keys)}  (stage={args.stage})")

    baselines_set = {system_label(p) for p in baselines}
    pairs = build_pairs(labels_paths, baselines_set, args.all_pairs)
    rng = np.random.default_rng(args.seed)

    comparisons = []
    for metric in args.metrics:
        print("\n" + "=" * 78)
        print(f"METRIC: {metric}   (stage={args.stage}, n={len(keys)}, bootstrap={args.n_boot})")
        print("=" * 78)
        for (pa, la), (pb, lb) in pairs:
            # Compare over queries where BOTH systems have the metric defined.
            # A query with no ground truth yields an empty metric dict in every
            # system; dropping it (rather than imputing 0) keeps the paired test
            # honest instead of letting one undefined query void the comparison.
            paired = [(loaded[la][k].get(metric), loaded[lb][k].get(metric)) for k in keys]
            paired = [(x, y) for x, y in paired if x is not None and y is not None]
            if len(paired) < 2:
                print(f"  {la} vs {lb}: metric '{metric}' absent for both systems - skipped")
                continue
            va = np.array([p[0] for p in paired], dtype=float)
            vb = np.array([p[1] for p in paired], dtype=float)
            stats = paired_bootstrap(va, vb, args.n_boot, rng)
            stats["n"] = len(paired)
            wp = wilcoxon_p(va, vb) if args.wilcoxon else None
            tag = "SIGNIFICANT" if stats["significant"] else "n.s."
            # `significant` (CI excludes 0) is true for significant LOSSES too;
            # `favors` makes the direction explicit for anyone reading the JSON alone.
            favors = la if stats["mean_diff"] > 0 else (lb if stats["mean_diff"] < 0 else "tie")
            if wp is None:
                wtxt = ""
            else:
                wtxt = "  wilcoxon_p<0.001" if wp < 0.001 else f"  wilcoxon_p={wp:.3f}"
            print(f"  {la:26s} - {lb:14s}: "
                  f"{stats['mean_a']:.4f} vs {stats['mean_b']:.4f}  "
                  f"diff={stats['mean_diff']:+.4f}  "
                  f"95%CI[{stats['ci_low']:+.4f},{stats['ci_high']:+.4f}]  "
                  f"p(diff<=0)={stats['p_le0']:.3f}{wtxt}  -> {tag}")
            comparisons.append({
                "metric": metric, "system_a": la, "system_b": lb,
                "favors": favors, "wilcoxon_p": wp, **stats,
            })

    # --- END OF COMPARISON LOOP ---
    
    new_run_data = {
        "config": {
            "stage": args.stage, "metrics": args.metrics, "n_boot": args.n_boot,
            "seed": args.seed, "all_pairs": args.all_pairs,
            "files": [p.name for p in paths],
        },
        "aligned_queries": len(keys),
        "comparisons": comparisons,
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- SMART APPEND LOGIC ---
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            
            # If it's an older format (a single dict from your previous runs), 
            # wrap it in a list so we can start appending to it cleanly.
            if isinstance(existing_data, dict):
                existing_data = [existing_data]
                
            # Append the new evaluation run
            existing_data.append(new_run_data)
            final_output = existing_data
            
        except json.JSONDecodeError:
            # If the file exists but is empty or corrupted, start a fresh list
            final_output = [new_run_data]
    else:
        # First time running, start a new list
        final_output = [new_run_data]

    # Save the aggregated list back to the file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\n[OK] appended {len(comparisons)} comparisons -> {out_path} (Total runs stored: {len(final_output)})")


if __name__ == "__main__":
    main()
