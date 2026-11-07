"""
faithfulness_comparison_plot.py - query faithfulness vs corpus-summary faithfulness, side by side.

Run with:
    python evaluation/faithfulness_comparison_plot.py

WHY THIS EXISTS
    query_faithfulness.md (n=2,199, 89.9% faithful) and corpus_summary_faithfulness.md (n=500,
    78.6% faithful) each already report their own verdict distribution, but the size of the gap
    between them -- the retrieval substrate (corpus summaries) is noticeably less faithful than
    the query substrate -- has so far only been stated in prose, in two separate docs. This
    script is the direct visual comparison, for use as a single Section 6 figure.

INPUT
    research_paper/ablation/data/query_faithfulness.csv
    research_paper/ablation/data/corpus_faithfulness.csv
    (both already-computed audits; this script runs no new LLM-judge calls)

OUTPUT
    research_paper/ablation/figures/faithfulness_comparison.png
    research_paper/ablation/data/faithfulness_comparison.csv

NOTE ON SCOPE
    Verdict distribution (faithful/partial/unfaithful) only -- the two audits' flag taxonomies
    differ (query: leaks_identity/leaks_outcome/not_grounded/off_topic; corpus: not_grounded/
    issues_inaccurate/holding_inaccurate), sharing only "not_grounded", so a combined flag
    chart would compare non-comparable categories. See each analysis's own .md for its flags.
    Sample sizes differ by more than 4x (n=2,199 vs n=500) -- annotated on the plot, not
    glossed over, since the corpus estimate carries wider uncertainty.

USAGE
    python evaluation/faithfulness_comparison_plot.py
"""

import os
import csv
import argparse

import matplotlib.pyplot as plt

DATA_DIR = os.path.join("research_paper", "ablation", "data")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")

QUERY_CSV = os.path.join(DATA_DIR, "query_faithfulness.csv")
CORPUS_CSV = os.path.join(DATA_DIR, "corpus_faithfulness.csv")

VERDICT_ORDER = ["faithful", "partial", "unfaithful"]


def load_verdicts(fp: str) -> dict:
    out = {}
    with open(fp, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] == "verdict":
                out[row["type"]] = {"pct": float(row["pct"]), "count": int(row["count"]),
                                    "n_judged": int(row["n_judged"])}
    return out


def main():
    ap = argparse.ArgumentParser(description="Query vs corpus-summary faithfulness verdict comparison.")
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "faithfulness_comparison.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "faithfulness_comparison.csv"))
    args = ap.parse_args()

    query_v = load_verdicts(QUERY_CSV)
    corpus_v = load_verdicts(CORPUS_CSV)
    n_query = query_v["faithful"]["n_judged"]
    n_corpus = corpus_v["faithful"]["n_judged"]

    rows = []
    for v in VERDICT_ORDER:
        rows.append({"verdict": v, "audit": "query", "pct": query_v[v]["pct"],
                     "count": query_v[v]["count"], "n_judged": n_query})
        rows.append({"verdict": v, "audit": "corpus_summary", "pct": corpus_v[v]["pct"],
                     "count": corpus_v[v]["count"], "n_judged": n_corpus})
        print(f"{v:<12} query={query_v[v]['pct']:5.1f}%  corpus={corpus_v[v]['pct']:5.1f}%")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["verdict", "audit", "pct", "count", "n_judged"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written -> {args.csv_out}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = range(len(VERDICT_ORDER))
    width = 0.35

    query_vals = [query_v[v]["pct"] for v in VERDICT_ORDER]
    corpus_vals = [corpus_v[v]["pct"] for v in VERDICT_ORDER]

    bars_q = ax.bar([i - width / 2 for i in x], query_vals, width,
                    label=f"Query faithfulness (n={n_query})", color="#4c72b0")
    bars_c = ax.bar([i + width / 2 for i in x], corpus_vals, width,
                    label=f"Corpus-summary faithfulness (n={n_corpus})", color="#c0392b")

    for bars in (bars_q, bars_c):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([v.capitalize() for v in VERDICT_ORDER])
    ax.set_ylabel("% of judged sample")
    ax.set_ylim(0, 100)
    ax.set_title("Query faithfulness vs corpus-summary faithfulness\n"
                 f"(same judge, MiniMax M3; corpus is a {n_corpus}-case random sample, query is exhaustive)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
