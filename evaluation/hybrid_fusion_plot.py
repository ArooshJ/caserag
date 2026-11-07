"""
hybrid_fusion_plot.py - does fusing BM25 with a dense/structured retriever help? (T8)

Run with:
    python evaluation/hybrid_fusion_plot.py

WHY THIS EXISTS (T8 in docs/tooling/eval-runbook-multi-model.md)
    Lexical (BM25) and dense retrieval are often assumed to be complementary, so fusing them
    with Reciprocal Rank Fusion is a natural thing to try. The runs already exist
    (results/hybrid_rrf_*.json); T8 just asks that this null result be promoted into the paper
    as a reported finding rather than left as a pending run. This script makes the actual shape
    of the null result visible: for each of the 4 possible pairings (CaseRAG or Dense, x BGE or
    Kanon-2, each fused with BM25 5-gram), it plots BM25 alone, the dense/structured component
    alone, and the RRF-fused hybrid, side by side -- so it's visible whether fusion beats
    NEITHER component, ONE of them, or BOTH.

INPUT FILES (already evaluated; stage2 = final output; all multi-query fused, n=622)
    evaluation/results/bm25_ng5_multiquery_fused.json                    (BM25 alone)
    evaluation/results/caserag_emb1_multiquery_fused_k200k100.json       (CaseRAG BGE alone)
    evaluation/results/caserag_emb3_cosine_multiquery_fused_k200k100.json (CaseRAG Kanon-2 alone)
    evaluation/results/dense_emb1_multiquery_fused.json                  (Dense BGE alone)
    evaluation/results/dense_emb3_cosine_multiquery_fused.json           (Dense Kanon-2 alone)
    evaluation/results/hybrid_rrf_bm25ng5_caseragE1_multiquery_results.json
    evaluation/results/hybrid_rrf_bm25ng5_caseragE3cos_multiquery_results.json
    evaluation/results/hybrid_rrf_bm25ng5_denseE1_multiquery_results.json
    evaluation/results/hybrid_rrf_bm25ng5_denseE3cos_multiquery_results.json

OUTPUT
    research_paper/ablation/figures/hybrid_fusion_null_result.png
    research_paper/ablation/data/hybrid_fusion_null_result.csv

USAGE
    python evaluation/hybrid_fusion_plot.py
"""

import os
import csv
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

BM25_FILE = os.path.join(RESULTS_DIR, "bm25_ng5_multiquery_fused.json")

# (group label, component-alone file, hybrid file)
PAIRS = [
    ("CaseRAG\nBGE", os.path.join(RESULTS_DIR, "caserag_emb1_multiquery_fused_k200k100.json"),
                     os.path.join(RESULTS_DIR, "hybrid_rrf_bm25ng5_caseragE1_multiquery_results.json")),
    ("CaseRAG\nKanon-2", os.path.join(RESULTS_DIR, "caserag_emb3_cosine_multiquery_fused_k200k100.json"),
                        os.path.join(RESULTS_DIR, "hybrid_rrf_bm25ng5_caseragE3cos_multiquery_results.json")),
    ("Dense\nBGE", os.path.join(RESULTS_DIR, "dense_emb1_multiquery_fused.json"),
                   os.path.join(RESULTS_DIR, "hybrid_rrf_bm25ng5_denseE1_multiquery_results.json")),
    ("Dense\nKanon-2", os.path.join(RESULTS_DIR, "dense_emb3_cosine_multiquery_fused.json"),
                       os.path.join(RESULTS_DIR, "hybrid_rrf_bm25ng5_denseE3cos_multiquery_results.json")),
]


def load_map(fp: str) -> float:
    data = json.load(open(fp, encoding="utf-8"))
    return data["average_metrics"]["retrieval"]["stage2"]["MAP"]


def main():
    ap = argparse.ArgumentParser(description="Hybrid RRF fusion null-result plot (T8).")
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "hybrid_fusion_null_result.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "hybrid_fusion_null_result.csv"))
    args = ap.parse_args()

    bm25_map = load_map(BM25_FILE)
    print(f"BM25 (5-gram, multi-query fused) alone: MAP={bm25_map:.4f}\n")

    rows = []
    labels, component_vals, hybrid_vals, beats_component = [], [], [], []
    for label, component_fp, hybrid_fp in PAIRS:
        component_map = load_map(component_fp)
        hybrid_map = load_map(hybrid_fp)
        beats = hybrid_map > component_map
        labels.append(label.replace("\n", " "))
        component_vals.append(component_map)
        hybrid_vals.append(hybrid_map)
        beats_component.append(beats)
        print(f"{label.replace(chr(10), ' '):<18} component_alone={component_map:.4f}  "
              f"hybrid={hybrid_map:.4f}  hybrid_beats_component={beats}  "
              f"hybrid_beats_bm25={hybrid_map > bm25_map}")
        rows.append({"pair": label.replace("\n", " "), "bm25_alone": bm25_map,
                      "component_alone": component_map, "hybrid_rrf": hybrid_map,
                      "hybrid_beats_component": beats, "hybrid_beats_bm25": hybrid_map > bm25_map})

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "bm25_alone", "component_alone", "hybrid_rrf",
                                                "hybrid_beats_component", "hybrid_beats_bm25"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written -> {args.csv_out}")

    group_labels = [p[0] for p in PAIRS]
    x = np.arange(len(group_labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, [bm25_map] * len(group_labels), width, label="BM25 alone", color="#c0392b")
    ax.bar(x, component_vals, width, label="Dense/CaseRAG alone", color="#8c8c8c")
    bars_hybrid = ax.bar(x + width, hybrid_vals, width, label="Hybrid RRF (BM25 + component)", color="#1f6feb")

    for i, b in enumerate(bars_hybrid):
        h = b.get_height()
        marker = "" if beats_component[i] else " ✗ below component"
        ax.annotate(f"{h:.4f}{marker}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="#1f6feb" if beats_component[i] else "#c0392b")

    ax.set_ylabel("MAP")
    ax.set_title("Lexical+dense RRF fusion rarely beats the better component alone")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
