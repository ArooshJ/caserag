"""
field_weight_ablation_plot.py - how sensitive is Stage-1 ranking to the 5 case-field weights?

Run with:
    python evaluation/field_weight_ablation_plot.py

WHY THIS EXISTS (T3 in docs/tooling/eval-runbook-multi-model.md)
    T3 asked whether zeroing out the oracle-identified weak fields (statutes, precedents) or
    isolating single fields changes Stage-1 ranking quality. The runbook and this session's own
    checklist (research_paper/ablation/CHECKLIST.md, as of 2026-07-04) both recorded T3 as
    "not yet run, needs a live DB pull." That was wrong: `git log` shows the matrix build and a
    9-point weight sweep were already run and committed on 2026-06-27 (commits a9371597 /
    0856513d / 819a89ba), and the full results table already exists in
    docs/findings/ablation_results_2026-06-24.md section 11. The intermediate .npz matrix artifact
    (field_sim_matrix_emb3.npz) is gone from disk, but the 9 already-scored, evaluate.py-shaped
    result files it produced are still present in evaluation/results/ and are not
    reproducible-from-scratch here -- they are the source of truth. This script only turns
    those into a paper-ready plot + tidy CSV; it re-runs nothing and touches no DB.

INPUT FILES (already evaluated; stage1-only -- this is a Stage-1 field-weighting diagnostic,
    not a Stage-2/final-pipeline number; all emb3/Kanon-2 cosine, full corpus, 2195 sub-queries)
    evaluation/results/weight_ablation_emb3_static.json               (CaseRAG baseline)
    evaluation/results/weight_ablation_emb3_facts_only.json
    evaluation/results/weight_ablation_emb3_issues_only.json
    evaluation/results/weight_ablation_emb3_holding_only.json
    evaluation/results/weight_ablation_emb3_uniform.json
    evaluation/results/weight_ablation_emb3_facts_issues.json
    evaluation/results/weight_ablation_emb3_facts_holding.json
    evaluation/results/weight_ablation_emb3_issues_holding.json
    evaluation/results/weight_ablation_emb3_facts_issues_holding.json  (best)

OUTPUT
    research_paper/ablation/figures/field_weight_ablation.png
    research_paper/ablation/data/field_weight_ablation.csv
        (config, weight_facts, weight_issues, weight_statutes, weight_precedents,
         weight_holding, metric, value)

NOTE ON SCOPE
    Full-corpus Stage-1 ranking only (not the fused k1=200 pipeline number from section 2/3 of
    the ablation doc) -- these numbers are not directly comparable to the headline MAP=0.1977.
    BPREF=1.0 for every row (an artifact: judged set == gold set, so there are no non-relevant
    judged cases) and is intentionally excluded from the plot.

USAGE
    python evaluation/field_weight_ablation_plot.py
"""

import os
import csv
import json
import argparse

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

FIELDS = ["facts", "issues", "statutes", "precedents", "holding"]

# (config label, weights (facts, issues, statutes, precedents, holding), file, short display name)
CONFIGS = [
    ("static",               (0.1, 0.3, 0.2, 0.1, 0.3), "weight_ablation_emb3_static.json",               "Static\n(baseline)"),
    ("facts_only",           (1.0, 0.0, 0.0, 0.0, 0.0), "weight_ablation_emb3_facts_only.json",            "Facts\nonly"),
    ("issues_only",          (0.0, 1.0, 0.0, 0.0, 0.0), "weight_ablation_emb3_issues_only.json",           "Issues\nonly"),
    ("holding_only",         (0.0, 0.0, 0.0, 0.0, 1.0), "weight_ablation_emb3_holding_only.json",          "Holding\nonly"),
    ("uniform",              (0.2, 0.2, 0.2, 0.2, 0.2), "weight_ablation_emb3_uniform.json",               "Uniform\n(all 5)"),
    ("facts_issues",         (0.5, 0.5, 0.0, 0.0, 0.0), "weight_ablation_emb3_facts_issues.json",          "Facts+\nIssues"),
    ("facts_holding",        (0.5, 0.0, 0.0, 0.0, 0.5), "weight_ablation_emb3_facts_holding.json",         "Facts+\nHolding"),
    ("issues_holding",       (0.0, 0.5, 0.0, 0.0, 0.5), "weight_ablation_emb3_issues_holding.json",        "Issues+\nHolding"),
    ("facts_issues_holding", (1/3, 1/3, 0.0, 0.0, 1/3), "weight_ablation_emb3_facts_issues_holding.json",  "Facts+Issues\n+Holding"),
]

METRICS_FOR_CSV = ["MAP", "MRR", "Recall@10", "Recall@20", "Recall@50", "Recall@100", "Recall@200",
                   "NDCG@10", "NDCG@20", "Hit@10", "Hit@20", "Hit@100"]
METRICS_TO_PLOT = ["MAP", "MRR"]


def load_stage1(fp: str) -> dict:
    data = json.load(open(fp, encoding="utf-8"))
    n_queries = len(data.get("per_query_results", []))
    stage1 = data["average_metrics"]["retrieval"]["stage1"]
    return stage1, n_queries


def main():
    ap = argparse.ArgumentParser(description="Field-weight ablation: MAP/MRR/Recall vs Stage-1 field weighting.")
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "field_weight_ablation.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "field_weight_ablation.csv"))
    args = ap.parse_args()

    rows = []
    metrics_by_config = {}
    for key, weights, fname, display in CONFIGS:
        fp = os.path.join(RESULTS_DIR, fname)
        stage1, n_queries = load_stage1(fp)
        metrics_by_config[key] = stage1
        print(f"{key:<22} n_queries={n_queries}  " +
              "  ".join(f"{m}={stage1[m]:.4f}" for m in ["MAP", "MRR", "Recall@200"]))
        for m in METRICS_FOR_CSV:
            rows.append({
                "config": key,
                "weight_facts": weights[0], "weight_issues": weights[1],
                "weight_statutes": weights[2], "weight_precedents": weights[3],
                "weight_holding": weights[4],
                "metric": m, "value": stage1[m], "n_queries": n_queries,
            })

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ["config", "weight_facts", "weight_issues", "weight_statutes",
                  "weight_precedents", "weight_holding", "metric", "value", "n_queries"]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written -> {args.csv_out}")

    # Order bars by MAP descending for readability; keep "static" recognizable via color.
    order = sorted((k for k, *_ in CONFIGS), key=lambda k: -metrics_by_config[k]["MAP"])
    display_by_key = {key: display for key, _, _, display in CONFIGS}
    best_key = order[0]

    def bar_colors():
        colors = []
        for k in order:
            if k == "static":
                colors.append("#4c72b0")   # blue = current pipeline default
            elif k == best_key:
                colors.append("#c0392b")   # red = best found
            else:
                colors.append("#b0b0b0")   # grey = other sweep points
        return colors

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    values = [metrics_by_config[k]["MAP"] for k in order]
    bars = ax.bar([display_by_key[k] for k in order], values, color=bar_colors())
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAP")
    ax.set_title("Stage-1 MAP by field-weight configuration")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    values = [metrics_by_config[k]["Recall@200"] for k in order]
    bars = ax.bar([display_by_key[k] for k in order], values, color=bar_colors())
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Recall@200")
    ax.set_title("Stage-1 Recall@200 by field-weight configuration\n(k1=200 = pipeline's Stage-1 pool size)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Field-weight ablation (Kanon-2 cosine, full corpus, Stage 1 only) --\n"
                 "blue = current static weights, red = best found, grey = other configs tested")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
