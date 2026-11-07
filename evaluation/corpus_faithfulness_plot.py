"""
corpus_faithfulness_plot.py - visualize the corpus summary faithfulness audit (T6).

Run with:
    python evaluation/corpus_faithfulness_plot.py

WHY THIS EXISTS
    evaluation/results/corpus_summary_faithfulness_results.json holds the completed
    T6 audit (n=200 random sample of the 3,182 corpus cases with a usable Summary, judged by
    MiniMax M3 -- see ilpcsr_corpus_summary_faithfulness_judge.py) -- the numbers are quoted in
    prose in Section 5.4 of the paper (one validation sentence) but have no figure. This
    mirrors query_faithfulness_plot.py's two-panel layout (verdict distribution + per-flag
    rates) but for the corpus summaries, and explicitly annotates the finding that stands out:
    holding_inaccurate is far higher than the other flags, at a rate ~4x either of the other
    two -- worth visually distinguishing, not just listing alongside the others.

INPUT FILE
    evaluation/results/corpus_summary_faithfulness_results.json ({"_summary": {...},
    "_sampling": {...}, "records": [...]})

OUTPUT
    research_paper/ablation/figures/corpus_faithfulness.png
    research_paper/ablation/data/corpus_faithfulness.csv

USAGE
    python evaluation/corpus_faithfulness_plot.py
"""

import os
import csv
import json
import argparse

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("evaluation", "results")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

DEFAULT_INPUT = os.path.join(RESULTS_DIR, "corpus_summary_faithfulness_results.json")

VERDICT_ORDER = ["faithful", "partial", "unfaithful", "unparsed"]
FLAG_ORDER = ["not_grounded", "issues_inaccurate", "holding_inaccurate"]


def main():
    ap = argparse.ArgumentParser(description="Visualize the corpus summary faithfulness audit (T6).")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_DIR, "corpus_faithfulness.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_DIR, "corpus_faithfulness.csv"))
    args = ap.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))
    summary = data["_summary"]
    sampling = data.get("_sampling", {})
    n = summary["judged"]
    verdicts = summary["verdicts"]
    verdict_pct = summary["verdict_pct"]
    flags = summary["flag_counts"]
    flag_pct = summary["flag_pct"]

    print(f"Sample: {n} of {sampling.get('corpus_total', '?')} corpus cases "
          f"(seed={sampling.get('seed', '?')})")
    for v in VERDICT_ORDER:
        print(f"  {v:<12} {verdicts.get(v, 0):>5}  ({verdict_pct.get(v, 0)}%)")
    print("Flags:")
    for f in FLAG_ORDER:
        print(f"  {f:<18} {flags.get(f, 0):>5}  ({flag_pct.get(f, 0)}%)")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "type", "count", "pct", "n_judged"])
        for v in VERDICT_ORDER:
            writer.writerow(["verdict", v, verdicts.get(v, 0), verdict_pct.get(v, 0), n])
        for fl in FLAG_ORDER:
            writer.writerow(["flag", fl, flags.get(fl, 0), flag_pct.get(fl, 0), n])
    print(f"CSV written -> {args.csv_out}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    v_labels = [v for v in VERDICT_ORDER if verdicts.get(v, 0) > 0]
    v_vals = [verdicts[v] for v in v_labels]
    colors = {"faithful": "#1f6feb", "partial": "#f0ad4e", "unfaithful": "#c0392b", "unparsed": "#999999"}
    explode = [0.0 if v == "faithful" else 0.06 for v in v_labels]
    ax.pie(v_vals, labels=[f"{v}\n{verdicts[v]} ({verdict_pct[v]}%)" for v in v_labels],
           colors=[colors[v] for v in v_labels], startangle=90, explode=explode,
           labeldistance=1.12, wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax.set_title(f"Verdict distribution (n={n} corpus summaries)")

    ax = axes[1]
    f_vals = [flag_pct.get(fl, 0) for fl in FLAG_ORDER]
    bar_colors = ["#c0392b" if fl == "holding_inaccurate" else "#e08e6d" for fl in FLAG_ORDER]
    bars = ax.bar(FLAG_ORDER, f_vals, color=bar_colors)
    for b, fl in zip(bars, FLAG_ORDER):
        h = b.get_height()
        ax.annotate(f"{flags[fl]} ({h}%)", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of sampled summaries")
    ax.set_title("Specific failure-mode rates\n(holding_inaccurate highlighted)")
    ax.set_xticks(range(len(FLAG_ORDER)))
    ax.set_xticklabels(FLAG_ORDER, rotation=15, ha="right")
    ax.set_ylim(0, max(f_vals + [1]) * 1.35)

    fig.suptitle(f"Corpus summary faithfulness audit (n={n}/{sampling.get('corpus_total', '?')}, judge: MiniMax M3)")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"Plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
