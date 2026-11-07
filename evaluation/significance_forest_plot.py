"""
significance_forest_plot.py - one headline figure for all three significance-tested claims.

Run with:
    python evaluation/significance_forest_plot.py

WHY THIS EXISTS
    T2 (docs/tooling/eval-runbook-multi-model.md) produced three paired-bootstrap significance tests,
    each already documented individually (embedding_contingency.md, recall_at_k_curve.md), but
    never plotted together. Section 6 will open with "does CaseRAG's benefit depend on the
    embedder, and does it come at a recall cost" -- both claims are visible at a glance in one
    forest plot, which existing per-analysis figures don't show side by side.

INPUT
    evaluation/results/significance.json (entries 1-3; entry 0 is an older, unrelated
    single-query BGE k1-sweep -- see research_paper/ablation/README.md gotcha #2 -- and is
    intentionally excluded here since it isn't one of the three headline claims).

OUTPUT
    research_paper/ablation/figures/significance_forest.png
    research_paper/ablation/data/significance_forest.csv

NOTE ON SCOPE
    Reads and plots already-computed significance.json entries. Runs no new bootstrap.

USAGE
    python evaluation/significance_forest_plot.py
"""

import os
import csv
import json
import argparse

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

SIGNIFICANCE_FILE = os.path.join(RESULTS_DIR, "significance.json")

# (entry index in significance.json, display label)
CLAIMS = [
    (1, "MAP -- Kanon-2\n(CaseRAG vs Dense)"),
    (2, "MAP -- BGE\n(CaseRAG vs Dense)"),
    (3, "Recall@100 -- Kanon-2\n(CaseRAG vs Dense)"),
]


def main():
    ap = argparse.ArgumentParser(description="Forest plot of the three T2 significance-tested headline claims.")
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "significance_forest.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "significance_forest.csv"))
    args = ap.parse_args()

    sig = json.load(open(SIGNIFICANCE_FILE, encoding="utf-8"))

    rows = []
    for entry_idx, label in CLAIMS:
        entry = sig[entry_idx]
        comp = entry["comparisons"][0]
        rows.append({
            "label": label.replace("\n", " "),
            "metric": comp["metric"],
            "mean_diff": comp["mean_diff"],
            "ci_low": comp["ci_low"],
            "ci_high": comp["ci_high"],
            "p_le0": comp["p_le0"],
            "significant": comp["significant"],
            "favors": comp["favors"],
            "n": comp["n"],
        })
        print(f"{label.replace(chr(10), ' '):<45} diff={comp['mean_diff']:+.4f}  "
              f"95% CI [{comp['ci_low']:+.4f}, {comp['ci_high']:+.4f}]  "
              f"p(diff<=0)={comp['p_le0']:.4f}  significant={comp['significant']}  favors={comp['favors']}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written -> {args.csv_out}")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    y_pos = list(range(len(rows)))[::-1]  # top-to-bottom in listed order

    for y, row in zip(y_pos, rows):
        favors_caserag = row["favors"].startswith("CaseRAG")
        if not row["significant"]:
            color = "#888888"
        elif favors_caserag:
            color = "#2e7d32"   # green = significant, favors CaseRAG
        else:
            color = "#c0392b"   # red = significant, favors Dense (a loss for CaseRAG)

        lo, hi, diff = row["ci_low"], row["ci_high"], row["mean_diff"]
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="butt")
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=color, linewidth=2)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=color, linewidth=2)
        ax.scatter([diff], [y], color=color, s=70, zorder=5)

        sig_txt = "significant" if row["significant"] else "not significant"
        ax.annotate(f"{diff:+.4f}  ({sig_txt}, p(diff≤0)={row['p_le0']:.3f})",
                    xy=(hi, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=color)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_xlabel("Mean difference (CaseRAG − Dense), 95% bootstrap CI")
    ax.set_title("All three T2 significance tests (paired bootstrap, n=621, 10,000 resamples)\n"
                 "green = significant gain for CaseRAG, red = significant loss, grey = not significant",
                 fontsize=11)
    ax.set_xlim(min(r["ci_low"] for r in rows) - 0.01, max(r["ci_high"] for r in rows) + 0.06)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
