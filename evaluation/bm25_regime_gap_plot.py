"""
bm25_regime_gap_plot.py - how much does the query regime alone change BM25's score?

Run with:
    python evaluation/bm25_regime_gap_plot.py

WHY THIS EXISTS
    Section 5.2 of the paper departs from IL-PCSR's original protocol (the FULL query
    judgment document as the query) in favor of LLM-generated NATURAL-LANGUAGE questions, to
    model realistic legal research and avoid citation leakage. That choice means our absolute
    scores are not directly comparable to IL-PCSR's published numbers -- but "not comparable"
    is an abstract claim until it's quantified. This script isolates the query-regime effect
    on a single retriever (BM25, unaffected by embedding-model choice) so the size of the gap
    is visible on its own, uncontaminated by any architecture difference.

INPUT FILES (already evaluated; stage2 = final output)
    evaluation/results/bm25_ng5_fulltext_singlequery.json
        BM25 (5-gram) with the FULL query judgment document as the query -- the IL-PCSR
        original protocol, n=626 (close to the full 627-query test split).
    evaluation/results/bm25_ng5_multiquery_fused.json
        BM25 (5-gram) with our LLM-generated natural-language questions, multi-query fused --
        the protocol used everywhere else in this project, n=622.
    Both use the same BM25 variant (5-gram) so the only thing that differs is the query text
    itself.

OUTPUT
    research_paper/ablation/figures/bm25_regime_gap.png
    research_paper/ablation/data/bm25_regime_gap.csv

USAGE
    python evaluation/bm25_regime_gap_plot.py
"""

import os
import csv
import json
import argparse

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

CONFIGS = [
    ("Full-document query\n(IL-PCSR original protocol)", os.path.join(RESULTS_DIR, "bm25_ng5_fulltext_singlequery.json")),
    ("Natural-language query\n(this paper's protocol, multi-query fused)", os.path.join(RESULTS_DIR, "bm25_ng5_multiquery_fused.json")),
]


def load_stage2_metrics(fp: str) -> dict:
    data = json.load(open(fp, encoding="utf-8"))
    stage2 = data["average_metrics"]["retrieval"]["stage2"]
    n_queries = len(data.get("per_query_results", []))
    return {"MAP": stage2["MAP"], "MRR": stage2["MRR"], "n_queries": n_queries}


def main():
    ap = argparse.ArgumentParser(description="BM25 regime-gap plot: full-document vs natural-language query MAP.")
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "bm25_regime_gap.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "bm25_regime_gap.csv"))
    args = ap.parse_args()

    rows = []
    for label, fp in CONFIGS:
        m = load_stage2_metrics(fp)
        rows.append({"regime": label.replace("\n", " "), "file": os.path.basename(fp), **m})
        print(f"{label.splitlines()[0]:<45} MAP={m['MAP']:.4f}  MRR={m['MRR']:.4f}  n={m['n_queries']}")

    ratio = rows[0]["MAP"] / rows[1]["MAP"]
    print(f"\nFull-document / Natural-language MAP ratio: {ratio:.2f}x")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "file", "MAP", "MRR", "n_queries"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written -> {args.csv_out}")

    labels = [c[0] for c in CONFIGS]
    maps = [r["MAP"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bars = ax.bar(labels, maps, color=["#c0392b", "#1f6feb"], width=0.5)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

    ax.annotate(f"{ratio:.2f}x drop", xy=(0.5, max(maps) * 0.55), xycoords=("axes fraction", "data"),
                ha="center", fontsize=13, fontweight="bold", color="#c0392b")

    ax.set_ylabel("MAP")
    ax.set_title("BM25 score depends heavily on the query regime\n(same retriever, same corpus, different query text)")
    ax.set_ylim(0, max(maps) * 1.2)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
