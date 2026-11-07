"""
recall_at_k_curve_plot.py - Recall@K curve across the full cutoff range.

Run with:
    python evaluation/recall_at_k_curve_plot.py

WHY THIS EXISTS
    The rank-stratification bar chart (rank_stratification_plot.py) shows where the FIRST
    gold hit lands, bucketed. This script complements it with the continuous view IR readers
    expect: Recall@K for K=1..100, one line per system. Together they tell the same story two
    ways -- a discrete "how many queries land in each rank bucket" view and a continuous
    "cumulative fraction of ALL gold cases found by depth K" view. The Recall@K curve is
    specifically where the paper's precision/recall tradeoff (CaseRAG wins at low K, Dense
    wins by K=100) becomes visible as CROSSING lines rather than a single bucket count.

INPUT FILES (already evaluated; stage2 = each system's final output)
    Default: the emb3 (Kanon-2) cosine multi-query fused pair (the paper's headline configs):
        evaluation/results/dense_emb3_cosine_multiquery_fused.json
        evaluation/results/caserag_emb3_cosine_multiquery_fused_k200k100.json
    Pass --baseline/--system (and --extra NAME=FILE, repeatable) to plot other files, e.g. to
    add BM25 as a third reference line.

OUTPUT
    research_paper/ablation/figures/recall_at_k_curve.png
    research_paper/ablation/data/recall_at_k_curve.csv  (label, k, recall)

USAGE
    python evaluation/recall_at_k_curve_plot.py
    python evaluation/recall_at_k_curve_plot.py \\
        --extra "BM25=evaluation/results/bm25_ng5_multiquery_fused.json"
"""

import os
import csv
import json
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

DEFAULT_BASELINE = os.path.join(RESULTS_DIR, "dense_emb3_cosine_multiquery_fused.json")
DEFAULT_SYSTEM = os.path.join(RESULTS_DIR, "caserag_emb3_cosine_multiquery_fused_k200k100.json")

K_VALUES = [1, 3, 5, 10, 20, 30, 50, 100]


def recall_curve(fp: str, stage: str) -> List[Tuple[int, float]]:
    data = json.load(open(fp, encoding="utf-8"))
    stage_metrics = data["average_metrics"]["retrieval"][stage]
    return [(k, stage_metrics[f"Recall@{k}"]) for k in K_VALUES if f"Recall@{k}" in stage_metrics]


def main():
    ap = argparse.ArgumentParser(description="Recall@K curve comparing systems' evaluated results.")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--baseline-label", default="Dense (Kanon-2 cosine)")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--system-label", default="CaseRAG (Kanon-2 cosine)")
    ap.add_argument("--extra", action="append", default=[], metavar="LABEL=FILE",
                     help="additional system to overlay, repeatable")
    ap.add_argument("--stage", default="stage2", choices=["stage1", "stage2"])
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "recall_at_k_curve.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "recall_at_k_curve.csv"))
    args = ap.parse_args()

    series: List[Tuple[str, List[Tuple[int, float]]]] = [
        (args.baseline_label, recall_curve(args.baseline, args.stage)),
        (args.system_label, recall_curve(args.system, args.stage)),
    ]
    for extra in args.extra:
        label, fp = extra.split("=", 1)
        series.append((label, recall_curve(fp, args.stage)))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "k", "recall"])
        writer.writeheader()
        for label, curve in series:
            for k, r in curve:
                writer.writerow({"label": label, "k": k, "recall": r})
    print(f"CSV written -> {args.csv_out}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#8c8c8c", "#1f6feb", "#c0392b", "#27ae60", "#8e44ad"]
    for i, (label, curve) in enumerate(series):
        ks = [k for k, _ in curve]
        rs = [r for _, r in curve]
        ax.plot(ks, rs, marker="o", label=label, color=colors[i % len(colors)], linewidth=2)
        print(f"{label:<28} " + "  ".join(f"R@{k}={r:.3f}" for k, r in curve))

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"Recall@K ({args.stage}, final returned ranking)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
