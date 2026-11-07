"""
embedding_contingency_plot.py - visualizes the paper's headline claim directly, with
significance annotations.

Run with:
    python evaluation/embedding_contingency_plot.py

WHY THIS EXISTS
    The paper's central finding is that CaseRAG's benefit over plain Dense retrieval is
    CONTINGENT on embedder quality -- same architecture, same corpus, same queries, different
    embedder, different-sized effect. That is a claim about an INTERACTION (architecture x
    embedder), which a single "CaseRAG beats Dense" bar chart cannot show. This script plots
    exactly the 2x2 grid the claim needs: {Dense, CaseRAG} x {BGE, Kanon-2}, MAP on the
    y-axis, so the reader sees the gap between the two embedder groups' bar-pairs, not just
    that CaseRAG is taller -- AND it annotates each pair with the actual significance test
    result (see SIGNIFICANCE below), so the figure doesn't overclaim a difference that isn't
    statistically distinguishable from noise.

SIGNIFICANCE (evaluation/results/significance.json)
    As of 2026-07-04 this file contains, among older entries, TWO paired-bootstrap tests
    (10,000 resamples, 95% CI, run by evaluation/significance.py) that directly cover
    the pairs this script plots:
      - CaseRAG vs Dense, Kanon-2, AP:  95% CI excludes 0 -> SIGNIFICANT, favors CaseRAG
      - CaseRAG vs Dense, BGE, AP:      95% CI includes 0 -> NOT significant
    This script looks these up automatically (matched by the exact pair of result filenames in
    each significance.json entry's config.files) and annotates "p=... (sig.)" or "n.s." under
    each embedder's lift percentage. If significance.json doesn't have an entry for a
    particular pair/metric, the annotation prints "not tested" rather than silently omitting
    it -- a missing test should be visible, not invisible.

INPUT FILES (already evaluated; stage2 = each system's final output)
    evaluation/results/dense_emb1_multiquery_fused.json
    evaluation/results/caserag_emb1_multiquery_fused_k200k100.json
    evaluation/results/dense_emb3_cosine_multiquery_fused.json
    evaluation/results/caserag_emb3_cosine_multiquery_fused_k200k100.json
    evaluation/results/significance.json (for the annotations; optional -- the plot
    still renders without it, just without significance labels)

OUTPUT
    research_paper/ablation/figures/embedding_contingency.png
    research_paper/ablation/data/embedding_contingency.csv
        (embedder, system, MAP, MRR, n_queries, sig_mean_diff, sig_ci_low, sig_ci_high,
         sig_p_le0, sig_significant, sig_n -- the sig_* columns are populated on the CaseRAG
         row only, since the test is CaseRAG-vs-Dense per embedder)

USAGE
    python evaluation/embedding_contingency_plot.py
    python evaluation/embedding_contingency_plot.py --metric MRR
"""

import os
import csv
import json
import argparse
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")
SIGNIFICANCE_PATH = os.path.join(RESULTS_DIR, "significance.json")

# (embedder label, dense file, system file)
CONFIGS = [
    ("BGE", os.path.join(RESULTS_DIR, "dense_emb1_multiquery_fused.json"),
             os.path.join(RESULTS_DIR, "caserag_emb1_multiquery_fused_k200k100.json")),
    ("Kanon-2", os.path.join(RESULTS_DIR, "dense_emb3_cosine_multiquery_fused.json"),
                os.path.join(RESULTS_DIR, "caserag_emb3_cosine_multiquery_fused_k200k100.json")),
]

# Per-query metric names (as used inside significance.json / retrieval_metrics.py) that the
# aggregate names in average_metrics correspond to. average_metrics.retrieval.stageN stores
# BOTH names with an identical value (MAP is literally the mean of per-query AP), so this is
# purely a lookup-key translation, not a different quantity.
METRIC_TO_SIG_KEY = {"MAP": "AP", "MRR": "RR"}


def load_stage2_metrics(fp: str) -> dict:
    data = json.load(open(fp, encoding="utf-8"))
    stage2 = data["average_metrics"]["retrieval"]["stage2"]
    n_queries = len(data.get("per_query_results", []))
    return {"MAP": stage2["MAP"], "MRR": stage2["MRR"], "n_queries": n_queries}


def find_significance(dense_fp: str, system_fp: str, metric: str) -> Optional[Dict]:
    """Look up the paired-bootstrap comparison for this exact (dense, system) file pair and
    metric in significance.json, matched by the pair's config.files set (order-independent).
    Returns None if significance.json is missing or has no matching entry -- callers must
    treat that as "not tested", never as "not significant"."""
    if not os.path.exists(SIGNIFICANCE_PATH):
        return None
    runs = json.load(open(SIGNIFICANCE_PATH, encoding="utf-8"))
    if isinstance(runs, dict):
        runs = [runs]
    target_files = {os.path.basename(dense_fp), os.path.basename(system_fp)}
    sig_metric = METRIC_TO_SIG_KEY.get(metric, metric)
    for run in runs:
        cfg = run.get("config", {})
        if set(cfg.get("files", [])) != target_files:
            continue
        for comp in run.get("comparisons", []):
            if comp.get("metric") == sig_metric:
                return comp
    return None


def main():
    ap = argparse.ArgumentParser(description="Embedding-contingency plot: Dense vs CaseRAG MAP across embedders.")
    ap.add_argument("--metric", default="MAP", choices=["MAP", "MRR"])
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "embedding_contingency.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "embedding_contingency.csv"))
    args = ap.parse_args()

    rows = []
    sig_by_embedder = {}
    for embedder, dense_fp, system_fp in CONFIGS:
        dense = load_stage2_metrics(dense_fp)
        system = load_stage2_metrics(system_fp)
        lift_pct = 100.0 * (system[args.metric] - dense[args.metric]) / dense[args.metric]
        sig = find_significance(dense_fp, system_fp, args.metric)
        sig_by_embedder[embedder] = sig

        dense_row = {"embedder": embedder, "system": "Dense", **dense,
                     "sig_mean_diff": "", "sig_ci_low": "", "sig_ci_high": "",
                     "sig_p_le0": "", "sig_significant": "", "sig_n": ""}
        system_row = {"embedder": embedder, "system": "CaseRAG", **system}
        if sig:
            system_row.update({
                "sig_mean_diff": sig["mean_diff"], "sig_ci_low": sig["ci_low"],
                "sig_ci_high": sig["ci_high"], "sig_p_le0": sig["p_le0"],
                "sig_significant": sig["significant"], "sig_n": sig["n"],
            })
        else:
            system_row.update({"sig_mean_diff": "", "sig_ci_low": "", "sig_ci_high": "",
                                "sig_p_le0": "", "sig_significant": "", "sig_n": ""})
        rows.append(dense_row)
        rows.append(system_row)

        sig_txt = "not tested"
        if sig:
            sig_txt = (f"{'SIGNIFICANT' if sig['significant'] else 'n.s.'} "
                       f"(95% CI [{sig['ci_low']:+.4f}, {sig['ci_high']:+.4f}], p(diff<=0)={sig['p_le0']:.4f}, n={sig['n']})")
        print(f"{embedder:<8} Dense {args.metric}={dense[args.metric]:.4f}  "
              f"CaseRAG {args.metric}={system[args.metric]:.4f}  lift={lift_pct:+.1f}%  "
              f"significance: {sig_txt}")

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ["embedder", "system", "MAP", "MRR", "n_queries",
                  "sig_mean_diff", "sig_ci_low", "sig_ci_high", "sig_p_le0", "sig_significant", "sig_n"]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written -> {args.csv_out}")

    embedders = [c[0] for c in CONFIGS]
    dense_vals = [r[args.metric] for r in rows if r["system"] == "Dense"]
    system_vals = [r[args.metric] for r in rows if r["system"] == "CaseRAG"]

    x = np.arange(len(embedders))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 6))
    bars1 = ax.bar(x - width / 2, dense_vals, width, label="Dense", color="#8c8c8c")
    bars2 = ax.bar(x + width / 2, system_vals, width, label="CaseRAG", color="#1f6feb")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    for i, (embedder, dv, sv) in enumerate(zip(embedders, dense_vals, system_vals)):
        lift = 100.0 * (sv - dv) / dv
        y = max(dv, sv) + 0.014
        ax.annotate(f"{lift:+.1f}%", xy=(x[i], y), ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#1f6feb" if lift > 0 else "#c0392b")

        sig = sig_by_embedder.get(embedder)
        if sig is None:
            sig_label, sig_color = "significance: not tested", "#999999"
        elif sig["significant"]:
            sig_label, sig_color = f"* significant (p(diff≤0)={sig['p_le0']:.3f})", "#1f6feb"
        else:
            sig_label, sig_color = f"n.s. (p(diff≤0)={sig['p_le0']:.3f})", "#c0392b"
        ax.annotate(sig_label, xy=(x[i], y), xytext=(0, 16), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, style="italic", color=sig_color)

    ax.set_ylabel(args.metric)
    ax.set_xlabel("Embedding model")
    ax.set_title(f"CaseRAG's benefit over Dense is contingent on embedder quality ({args.metric})")
    ax.set_xticks(x)
    ax.set_xticklabels(embedders)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(dense_vals + system_vals) * 1.35)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
