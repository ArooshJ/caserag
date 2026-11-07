"""
k1_ablation_plot.py - how does Stage-1 pool size (k1) trade off precision vs deep recall?

Run with:
    python evaluation/k1_ablation_plot.py

WHY THIS EXISTS (T1 in docs/tooling/eval-runbook-multi-model.md)
    CaseRAG's Stage 1 prunes the corpus to the top k1 candidate cases before Stage 2 ever
    runs. k1=50/100/200 runs were started on Colab and were, as of the last check, recorded
    as interrupted/incomplete (partial files pending download from Google Drive). As of this
    script, all three are confirmed COMPLETE: evaluation/data/model=3_k1={50,100}_k2=100_cosine_multiquery.json
    have 0 unfilled records out of 2,199, and their fused+evaluated results
    (evaluation/results/model=3_k1={50,100}_k2=100_cosine_fused_results.json) score
    the full 622-query test set -- matching evaluation/results/caserag_emb3_cosine_multiquery_fused_k200k100.json
    (k1=200) which was already known complete. This script turns those three already-scored
    points into the k1 ablation the runbook asks for.

INPUT FILES (already evaluated; stage2 = final output; all emb3/Kanon-2, cosine, multi-query fused)
    evaluation/results/model=3_k1=50_k2=100_cosine_fused_results.json
    evaluation/results/model=3_k1=100_k2=100_cosine_fused_results.json
    evaluation/results/caserag_emb3_cosine_multiquery_fused_k200k100.json  (k1=200)

OUTPUT
    research_paper/ablation/figures/k1_ablation.png
    research_paper/ablation/data/k1_ablation.csv  (k1, metric, value)

NOTE ON SCOPE
    This script only reads and plots ALREADY-COMPUTED evaluation results. It does not run any
    new significance test between k1 settings (no k1-vs-k1 comparison exists yet in
    evaluation/results/significance.json as of this writing) -- if that is wanted,
    run evaluation/significance.py explicitly with the relevant --files, the same way
    the CaseRAG-vs-Dense tests were run for T2.

USAGE
    python evaluation/k1_ablation_plot.py
"""

import os
import csv
import json
import argparse

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

# (k1, file)
CONFIGS = [
    (50, os.path.join(RESULTS_DIR, "model=3_k1=50_k2=100_cosine_fused_results.json")),
    (100, os.path.join(RESULTS_DIR, "model=3_k1=100_k2=100_cosine_fused_results.json")),
    (200, os.path.join(RESULTS_DIR, "caserag_emb3_cosine_multiquery_fused_k200k100.json")),
]

METRICS_TO_PLOT = ["MAP", "MRR", "Recall@20", "Recall@100"]


def load_stage2(fp: str) -> dict:
    data = json.load(open(fp, encoding="utf-8"))
    n_queries = len(data.get("per_query_results", []))
    stage2 = data["average_metrics"]["retrieval"]["stage2"]
    return stage2, n_queries


def main():
    ap = argparse.ArgumentParser(description="k1 ablation: MAP/MRR/Recall@K vs Stage-1 pool size.")
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "k1_ablation.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "k1_ablation.csv"))
    args = ap.parse_args()

    rows = []
    series = {m: [] for m in METRICS_TO_PLOT}
    k1s = []
    for k1, fp in CONFIGS:
        stage2, n_queries = load_stage2(fp)
        k1s.append(k1)
        print(f"k1={k1:<4} n_queries={n_queries}  " + "  ".join(f"{m}={stage2[m]:.4f}" for m in METRICS_TO_PLOT))
        for m in METRICS_TO_PLOT:
            series[m].append(stage2[m])
            rows.append({"k1": k1, "metric": m, "value": stage2[m], "n_queries": n_queries})

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k1", "metric", "value", "n_queries"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written -> {args.csv_out}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    for m in ["MAP", "MRR"]:
        ax.plot(k1s, series[m], marker="o", linewidth=2, label=m)
    ax.set_xlabel("k1 (Stage-1 candidate pool size)")
    ax.set_ylabel("Score")
    ax.set_title("Ranking quality vs k1")
    ax.set_xticks(k1s)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    for m in ["Recall@20", "Recall@100"]:
        ax.plot(k1s, series[m], marker="o", linewidth=2, label=m)
    ax.set_xlabel("k1 (Stage-1 candidate pool size)")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs k1")
    ax.set_xticks(k1s)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("CaseRAG (Kanon-2, cosine, multi-query fused): effect of Stage-1 pool size k1")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
