"""
oracle_similarity_histogram_plot.py - Stage 1 vs Stage 2 gold-similarity distributions.

Run with:
    python evaluation/oracle_similarity_histogram_plot.py
    python evaluation/oracle_similarity_histogram_plot.py --input evaluation/data/gold_oracle_emb3_cosine_multiquery.json

WHY THIS EXISTS
    gold_oracle_similarity.py removes retrieval entirely and scores each query directly
    against only its own gold cases, separating two possible causes of a missed gold case:
    "the embedding cannot represent the match" (low similarity even with retrieval removed)
    vs "the two-stage funnel drops it" (similarity is fine, ranking loses it). The existing
    oracle_analysis_emb*.json files report only SUMMARY STATISTICS (mean/median/percentiles)
    of this comparison, not the full distribution. This script reads the RAW per-(sub-query,
    gold-case) pair scores and plots the actual Stage 1 vs Stage 2 similarity histograms,
    which is more informative than percentiles alone (e.g. it shows the shape/spread and any
    bimodality that summary stats would hide).

DATA-PROVENANCE FINDING (2026-07-04) -- WHY THIS SCRIPT IS EMBEDDER-AWARE
    The file this script originally read, evaluation/results/gold_oracle_similarity_multiquery.json
    (also duplicated in evaluation/data/), had NO embedding-model identifier anywhere
    in its filename or contents. Recomputing its summary statistics from the raw per-pair
    scores and comparing against the three existing oracle_analysis_emb*.json files showed it
    matched oracle_analysis_emb1.json EXACTLY (stage1/stage2 median and mean identical to
    several decimal places) -- i.e. it was the **BGE (emb1)** oracle run, not emb3/Kanon-2,
    despite the generic name. Both copies were renamed to gold_oracle_emb1_multiquery.json to
    make this permanent and unambiguous.

    This mattered beyond this script: Qualitative_Analysis.ipynb's first cell pairs a gold
    oracle file (GOLD_ORACLE_RESULTS_PATH) with emb3 (Kanon-2) RAG results
    (data/model=3_k1=200_k2=100_cosine_multiquery.json) to produce
    qualitative_semantic_gap_examples.json -- when it was pointed at the emb1 oracle file, that
    was an embedder mismatch. See research_paper/ablation/oracle_similarity_histogram.md and
    the ablation CHECKLIST.md §3 for the full story and what needs re-running.

    To prevent the same ambiguity recurring, this script now infers the embedder label
    directly from the --input filename (looking for "emb1"/"emb3" and "cosine"/"l2") instead
    of hardcoding it, and tags its own output filenames with that embedder so running it for
    a second embedder never silently overwrites the first embedder's figure/CSV.

INPUT FILE
    Default: evaluation/data/gold_oracle_emb1_multiquery.json (BGE).
    Pass --input evaluation/data/gold_oracle_emb3_cosine_multiquery.json once that
    file exists (produced by running gold_oracle_similarity.py --emb-model-id 3 --multi-query
    --distance-op cosine) to get the Kanon-2 version of this chart.

OUTPUT (filename auto-tagged with the inferred embedder, e.g. "_emb1" / "_emb3_cosine")
    research_paper/ablation/figures/oracle_similarity_histogram_<embedder_tag>.png
    research_paper/ablation/data/oracle_similarity_raw_scores_<embedder_tag>.csv

USAGE
    python evaluation/oracle_similarity_histogram_plot.py
    python evaluation/oracle_similarity_histogram_plot.py --input evaluation/data/gold_oracle_emb3_cosine_multiquery.json
"""

import os
import re
import csv
import json
import argparse
import statistics

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("evaluation", "results")
DATA_DIR_IN = os.path.join("evaluation", "data")
FIGURES_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_DIR = os.path.join("research_paper", "ablation", "data")

DEFAULT_INPUT = os.path.join(DATA_DIR_IN, "gold_oracle_emb1_multiquery.json")


def infer_embedder_label(input_path: str) -> str:
    """Derive a human-readable embedder label + a filesystem-safe tag from the input
    filename, e.g. 'gold_oracle_emb3_cosine_multiquery.json' -> ('Kanon-2 (emb3, cosine)',
    'emb3_cosine'). Falls back to the raw filename stem if the pattern isn't recognized,
    rather than silently mislabeling -- an unrecognized name should look unrecognized, not
    quietly default to the wrong embedder the way the pre-rename file did."""
    name = os.path.basename(input_path)
    m = re.search(r"emb(\d)(_cosine|_l2)?", name)
    if not m:
        stem = os.path.splitext(name)[0]
        return f"UNKNOWN embedder (unrecognized filename '{name}')", stem
    emb_id = m.group(1)
    op = (m.group(2) or "").lstrip("_") or "l2 (default)"
    friendly = {"1": "BGE", "3": "Kanon-2"}.get(emb_id, f"emb{emb_id}")
    tag = f"emb{emb_id}" + (f"_{m.group(2).lstrip('_')}" if m.group(2) else "")
    return f"{friendly} (emb{emb_id}, {op})", tag


def main():
    ap = argparse.ArgumentParser(description="Stage 1 vs Stage 2 gold-similarity histograms from raw oracle data.")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--embedder-label", default=None,
                     help="override the auto-inferred embedder label (use if the filename doesn't follow the emb<N>[_cosine|_l2] convention)")
    ap.add_argument("--plot-out", default=None, help="default: figures/oracle_similarity_histogram_<tag>.png")
    ap.add_argument("--csv-out", default=None, help="default: data/oracle_similarity_raw_scores_<tag>.csv")
    args = ap.parse_args()

    embedder_label, tag = infer_embedder_label(args.input)
    if args.embedder_label:
        embedder_label = args.embedder_label
    if embedder_label.startswith("UNKNOWN"):
        print(f"[WARN] {embedder_label} -- pass --embedder-label to label the plot correctly.")

    plot_out = args.plot_out or os.path.join(FIGURES_DIR, f"oracle_similarity_histogram_{tag}.png")
    csv_out = args.csv_out or os.path.join(DATA_DIR, f"oracle_similarity_raw_scores_{tag}.csv")

    print(f"Input: {args.input}")
    print(f"Embedder: {embedder_label}")
    data = json.load(open(args.input, encoding="utf-8"))
    records = data["records"]

    stage1_scores = [g["final_score"] for r in records for g in r.get("stage1", [])]
    stage2_scores = [g["best_chunk_sim"] for r in records for g in r.get("stage2_per_gold", [])]
    print(f"n gold pairs: stage1={len(stage1_scores)}  stage2={len(stage2_scores)}")

    for name, scores in [("Stage 1 (case-level, weighted field score)", stage1_scores),
                          ("Stage 2 (best chunk similarity)", stage2_scores)]:
        print(f"  {name}: mean={statistics.mean(scores):.4f}  median={statistics.median(scores):.4f}  "
              f"min={min(scores):.4f}  max={max(scores):.4f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "score"])
        for s in stage1_scores:
            writer.writerow(["stage1", s])
        for s in stage2_scores:
            writer.writerow(["stage2", s])
    print(f"CSV written -> {csv_out}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bins = 40
    ax.hist(stage1_scores, bins=bins, alpha=0.6, label=f"Stage 1 (n={len(stage1_scores)})", color="#8c8c8c")
    ax.hist(stage2_scores, bins=bins, alpha=0.6, label=f"Stage 2 (n={len(stage2_scores)})", color="#1f6feb")
    ax.axvline(statistics.median(stage1_scores), color="#555555", linestyle="--", linewidth=1)
    ax.axvline(statistics.median(stage2_scores), color="#144a99", linestyle="--", linewidth=1)
    ax.set_xlabel("Similarity to gold case (oracle, retrieval removed)")
    ax.set_ylabel("Count (sub-query, gold-case) pairs")
    ax.set_title(f"Gold-similarity distribution: Stage 1 vs Stage 2\nEmbedder: {embedder_label}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(plot_out, dpi=200)
    print(f"Plot saved -> {plot_out}")


if __name__ == "__main__":
    main()
