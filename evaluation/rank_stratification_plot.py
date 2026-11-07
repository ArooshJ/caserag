"""
rank_stratification_plot.py - "Where is the Truth?" rank-stratification comparison.

Run with:
    python evaluation/rank_stratification_plot.py

WHY THIS EXISTS
    A flat MAP/MRR number hides WHERE in the ranking gold cases land. This script buckets
    every query by the rank of its FIRST relevant (gold) case in a system's final ranked
    list, and compares two systems (default: Dense vs CaseRAG, both emb3/Kanon-2, cosine,
    multi-query fused) side by side. Buckets: Rank 1 / Rank 2-5 / Rank 6-20 / Rank 21-50 /
    Rank 51-100 / Missed (no gold case anywhere in the returned list).

WHY THE OLD NOTEBOOK CELL (Qualitative_Analysis.ipynb) WAS BROKEN
    It read item["retrieved_cases"] and item["ground_truth_cases"] directly off the
    EVALUATED, already-scored result files (evaluation/results/*_fused_*.json).
    Those files' per_query_results entries only contain
        {query_id, question, retrieval_metrics: {stage1: {...}, stage2: {...}}, generation_metrics}
    -- there is no "retrieved_cases" or "ground_truth_cases" key at that level (those raw
    ranked lists live in the UNFUSED per-sub-query files under evaluation/data/, in a
    different shape, and are not what evaluate.py's output carries forward). So gt_cases was
    always an empty set, every query was skipped by `if not gt_cases: continue`, every bucket
    counter stayed at 0, and any downstream percentage computation divided 0/0.

    Fix: derive the rank of the first gold hit from the "RR" (Reciprocal Rank) field that IS
    present in these files (computed once, correctly, by retrieval_metrics.py's
    reciprocal_rank(), which scans the FULL ranked list, not just some prefix):
        RR = 1 / rank_of_first_relevant_document   =>   rank = round(1 / RR)
    This recovery is exact (rank is always a positive integer), not an approximation. RR == 0
    means the gold case never appeared anywhere in that system's returned ranking at all --
    exactly the "Missed" bucket.

INPUT FILES
    Two evaluate.py output files of identical shape:
        {"average_metrics": {...},
         "per_query_results": [{"query_id": ..., "retrieval_metrics":
             {"stage1": {...}, "stage2": {"RR": ..., "AP": ..., ...}}}, ...]}
    We read "stage2" by default -- each system's FINAL returned ranking (for single-stage
    Dense, "stage1" is empty and "stage2" holds its one ranking; for CaseRAG, "stage2" is the
    paragraph/chunk-search result collapsed to cases -- see Section 5.5 of the paper). This
    keeps the comparison to what each system actually outputs, not an intermediate stage.

OUTPUT
    - <plot-out>  grouped bar chart, baseline vs system, one group per rank bucket
    - <json-out>  bucket counts + percentages for both systems, plus every query's assigned
      rank under each system (for any follow-up analysis, e.g. a per-query movement plot)

USAGE
    # Default comparison: Dense vs CaseRAG, both emb3 (Kanon-2) cosine, multi-query fused
    python evaluation/rank_stratification_plot.py

    # Any other pair of already-evaluated fused result files, e.g. the BGE embedder:
    python evaluation/rank_stratification_plot.py \\
        --baseline evaluation/results/dense_emb1_multiquery_fused.json --baseline-label "Dense (BGE)" \\
        --system   evaluation/results/caserag_emb1_multiquery_fused_k200k100.json --system-label "CaseRAG (BGE)" \\
        --plot-out evaluation/results/rank_stratification_plot_emb1.png \\
        --json-out evaluation/results/rank_stratification_summary_emb1.json
"""

import os
import json
import argparse
import csv
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

DEFAULT_BASELINE = os.path.join(RESULTS_DIR, "dense_emb3_cosine_multiquery_fused.json")
DEFAULT_SYSTEM   = os.path.join(RESULTS_DIR, "caserag_emb3_cosine_multiquery_fused_k200k100.json")
DEFAULT_PLOT_OUT = os.path.join(FIGURES_DIR, "rank_stratification_plot.png")
DEFAULT_JSON_OUT = os.path.join(DATA_DIR, "rank_stratification_summary.json")
DEFAULT_CSV_OUT  = os.path.join(DATA_DIR, "rank_stratification.csv")

# (label, low, high) inclusive rank range. Anything outside every range (including RR == 0)
# falls into "Missed". 100 is the natural ceiling here: both compared systems return at most
# 100 candidate cases in their final (stage2) ranking (k2=100 / fused --top-n=100), so no gold
# case can ever be observed at a rank beyond 100 in this data -- "Missed" and "rank > 100" are
# the same event, not two different things being conflated.
RANK_BUCKETS = [
    ("Rank 1", 1, 1),
    ("Rank 2-5", 2, 5),
    ("Rank 6-20", 6, 20),
    ("Rank 21-50", 21, 50),
    ("Rank 51-100", 51, 100),
]
BUCKET_LABELS = [b[0] for b in RANK_BUCKETS] + ["Missed"]


def load_data(filepath: str) -> Dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def first_hit_rank(rr: Optional[float]) -> Optional[int]:
    """Recover the exact rank of the first gold hit from Reciprocal Rank (see module
    docstring). None means no gold case appeared anywhere in the returned ranking."""
    if not rr:
        return None
    return round(1.0 / rr)


def bucket_for_rank(rank: Optional[int]) -> str:
    if rank is None:
        return "Missed"
    for label, lo, hi in RANK_BUCKETS:
        if lo <= rank <= hi:
            return label
    return "Missed"  # rank > 100 -- see RANK_BUCKETS comment; not expected given the k2=100 cap


def per_query_ranks(data: Dict, stage: str) -> Dict[str, Optional[int]]:
    """query_id -> rank of first gold hit (None = missed), read from
    per_query_results[*].retrieval_metrics[stage].RR."""
    ranks = {}
    for item in data.get("per_query_results", []):
        qid = item.get("query_id")
        rr = item.get("retrieval_metrics", {}).get(stage, {}).get("RR")
        ranks[qid] = first_hit_rank(rr)
    return ranks


def bucket_counts(ranks: Dict[str, Optional[int]]) -> Dict[str, int]:
    counts = {label: 0 for label in BUCKET_LABELS}
    for rank in ranks.values():
        counts[bucket_for_rank(rank)] += 1
    return counts


def make_plot(baseline_counts: Dict[str, int], system_counts: Dict[str, int],
              baseline_label: str, system_label: str, n_queries: int, out_path: str) -> None:
    x = np.arange(len(BUCKET_LABELS))
    width = 0.35

    baseline_vals = [baseline_counts[b] for b in BUCKET_LABELS]
    system_vals = [system_counts[b] for b in BUCKET_LABELS]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label=baseline_label, color="#8c8c8c")
    bars2 = ax.bar(x + width / 2, system_vals, width, label=system_label, color="#1f6feb")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            pct = 100.0 * h / n_queries if n_queries else 0.0
            ax.annotate(f"{int(h)}\n({pct:.1f}%)", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Number of queries")
    ax.set_xlabel("Rank of first relevant (gold) case in the returned ranking")
    ax.set_title(f"Rank of first gold hit (n={n_queries} test queries)")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Plot saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Rank-stratification comparison of two evaluated systems.")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--baseline-label", default="Dense")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--system-label", default="CaseRAG")
    ap.add_argument("--stage", default="stage2", choices=["stage1", "stage2"],
                     help="which stage's ranking to compare (stage2 = each system's final output)")
    ap.add_argument("--plot-out", default=DEFAULT_PLOT_OUT)
    ap.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    ap.add_argument("--csv-out", default=DEFAULT_CSV_OUT,
                     help="tidy CSV (label, bucket, count, pct) for plotting elsewhere")
    args = ap.parse_args()

    baseline_data = load_data(args.baseline)
    system_data = load_data(args.system)

    baseline_ranks = per_query_ranks(baseline_data, args.stage)
    system_ranks = per_query_ranks(system_data, args.stage)

    common_ids = set(baseline_ranks) & set(system_ranks)
    if set(baseline_ranks) != set(system_ranks):
        print(f"[WARN] query_id sets differ: baseline={len(baseline_ranks)}, "
              f"system={len(system_ranks)}, common={len(common_ids)}. "
              "Restricting the comparison to the common (paired) set.")
    n_queries = len(common_ids)
    baseline_ranks = {q: baseline_ranks[q] for q in common_ids}
    system_ranks = {q: system_ranks[q] for q in common_ids}

    baseline_counts = bucket_counts(baseline_ranks)
    system_counts = bucket_counts(system_ranks)

    print(f"\n{'Bucket':<14} {args.baseline_label:>20} {args.system_label:>20}")
    for b in BUCKET_LABELS:
        bp = 100.0 * baseline_counts[b] / n_queries if n_queries else 0.0
        sp = 100.0 * system_counts[b] / n_queries if n_queries else 0.0
        print(f"{b:<14} {baseline_counts[b]:>6} ({bp:5.1f}%)     {system_counts[b]:>6} ({sp:5.1f}%)")

    make_plot(baseline_counts, system_counts, args.baseline_label, args.system_label,
              n_queries, args.plot_out)

    summary = {
        "n_queries": n_queries,
        "stage": args.stage,
        "baseline": {
            "file": args.baseline, "label": args.baseline_label,
            "counts": baseline_counts,
            "pct": {b: round(100.0 * baseline_counts[b] / n_queries, 2) for b in BUCKET_LABELS},
        },
        "system": {
            "file": args.system, "label": args.system_label,
            "counts": system_counts,
            "pct": {b: round(100.0 * system_counts[b] / n_queries, 2) for b in BUCKET_LABELS},
        },
        "per_query_ranks": {"baseline": baseline_ranks, "system": system_ranks},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved -> {args.json_out}")

    os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)), exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "bucket", "count", "pct", "n_queries"])
        writer.writeheader()
        for label, counts in ((args.baseline_label, baseline_counts), (args.system_label, system_counts)):
            for b in BUCKET_LABELS:
                writer.writerow({"label": label, "bucket": b, "count": counts[b],
                                  "pct": round(100.0 * counts[b] / n_queries, 2) if n_queries else 0.0,
                                  "n_queries": n_queries})
    print(f"Tidy CSV saved -> {args.csv_out}")


if __name__ == "__main__":
    main()
