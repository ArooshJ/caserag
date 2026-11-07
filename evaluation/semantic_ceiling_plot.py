"""
semantic_ceiling_plot.py - for EVERY sub-query (not a curated top-20), how does the gold
case's own intrinsic (oracle) semantic similarity compare to what the pipeline actually
retrieves, and does that explain misses?

Run with:
    python evaluation/semantic_ceiling_plot.py

WHY THIS EXISTS
    Qualitative_Analysis.ipynb's first cell already computes a "semantic gap" (oracle
    similarity of the missed gold case vs. the top noise result actually retrieved), but only
    keeps the top-20 most dramatic examples on each side, for use as illustrative examples in
    the paper's qualitative section. That is a curated sample, not an unbiased population
    statistic, and it also silently drops any sub-query whose gold case lands between rank 11
    and rank 50 (neither "positive" (top-10) nor "negative" (missed from top-50) by that cell's
    own definition) -- roughly 19% of sub-queries fall into that gap and are excluded from the
    723+1058=1781 total examples.

    This script asks the same underlying question -- is the gold precedent actually
    semantically close to its query in embedding space, and does that predict whether the
    pipeline finds it -- but over the FULL population of sub-queries with an exhaustive,
    non-overlapping rank bucketing, to support a population-level claim rather than illustrate
    it with hand-picked examples. The intended use is different too: this is meant to speak to
    whether *retrieval misses reflect a genuine lack of semantic similarity between a legal
    question and its true precedent* (a property of the task/dataset, argued as a reason this
    paper is a diagnostic study of semantic retrieval's limits rather than a pitch for CaseRAG
    as a deployable system) as opposed to *the pipeline dropping something the embedding model
    itself would have ranked highly* (a pipeline/funnel defect, not a task property).

INPUT
    evaluation/data/gold_oracle_emb3_cosine_multiquery.json
        -- per sub-query, per-gold-case oracle similarity (no retrieval; direct query-to-known-
        gold-case comparison). 2,195 sub-queries.
    evaluation/data/model=3_k1=200_k2=100_cosine_multiquery.json
        -- per sub-query, the pipeline's ACTUAL Stage-2 output: up to k2=100 ranked paragraph
        hits (contexts), each carrying its own case_id and similarity score. 2,199 sub-queries
        (4 more than the oracle file -- those 4 have no oracle match and are excluded from the
        joined analysis; reported, not silently dropped).
    These two files are joined on `sub_query_id` -- the same join the notebook cell uses.

OUTPUT
    research_paper/ablation/figures/semantic_ceiling.png
    research_paper/ablation/data/semantic_ceiling.csv   (one row per sub-query, full population)

DEFINITIONS
    oracle_stage2_best  = max, over this sub-query's gold cases, of the gold case's own best
                          chunk-level cosine similarity to the query (the ceiling: how similar
                          the TRUE answer could possibly score, independent of retrieval).
    hit_rank            = 1-indexed position of the first returned context (paragraph) whose
                          case_id is a gold case, within the pipeline's actual up-to-100
                          returned contexts; None if no gold case appears at all.
    hit_bucket          = Hit@1-10 / Hit@11-50 / Hit@51-100 / Missed -- exhaustive, unlike the
                          notebook's positives (top-10 only) vs negatives (top-50 only), which
                          leaves an undefined ~19% middle band.
    ceiling_deficit     = (score of whatever the pipeline actually ranked #1) minus
                          (oracle_stage2_best). Positive: even the gold case's own best possible
                          chunk similarity does not beat what the pipeline put at rank 1 --
                          i.e. something else is genuinely more semantically similar to the
                          query than the correct legal answer is. Negative: the gold case's own
                          ceiling exceeds what got surfaced on top -- the pipeline underperformed
                          relative to what the embedding itself knows, a funnel/ranking issue
                          rather than a similarity-ceiling issue.

NOTE ON SCOPE
    Descriptive + one Mann-Whitney U test (Hit vs Missed oracle_stage2_best); no new embedding
    or retrieval calls. This is a Kanon-2/cosine-only analysis (the embedder used in the
    paper's headline configuration) -- not run for BGE.

USAGE
    python evaluation/semantic_ceiling_plot.py
"""

import os
import csv
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

DATA_DIR = os.path.join("evaluation", "data")
FIGURES_OUT_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_OUT_DIR = os.path.join("research_paper", "ablation", "data")

ORACLE_FILE = os.path.join(DATA_DIR, "gold_oracle_emb3_cosine_multiquery.json")
RAW_CASERAG_FILE = os.path.join(DATA_DIR, "model=3_k1=200_k2=100_cosine_multiquery.json")

HIT_BUCKETS = ["Hit@1-10", "Hit@11-50", "Hit@51-100", "Missed"]


def bucket_for_rank(hit_rank):
    if hit_rank is None:
        return "Missed"
    if hit_rank <= 10:
        return "Hit@1-10"
    if hit_rank <= 50:
        return "Hit@11-50"
    return "Hit@51-100"


def main():
    ap = argparse.ArgumentParser(description="Oracle (gold) similarity ceiling vs actual pipeline retrieval, full population.")
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_OUT_DIR, "semantic_ceiling.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_OUT_DIR, "semantic_ceiling.csv"))
    args = ap.parse_args()

    print(f"Loading oracle file: {ORACLE_FILE}")
    oracle_data = json.load(open(ORACLE_FILE, encoding="utf-8"))
    oracle_map = {r["sub_query_id"]: r for r in oracle_data["records"]}
    print(f"  {len(oracle_map)} oracle sub-query records")

    print(f"Loading raw CaseRAG file (large, ~460MB): {RAW_CASERAG_FILE}")
    rag_data = json.load(open(RAW_CASERAG_FILE, encoding="utf-8"))
    print(f"  {len(rag_data)} raw CaseRAG sub-query records")

    rows = []
    n_no_oracle_match = 0
    n_no_gold = 0
    for item in rag_data:
        sub_query_id = item.get("sub_query_id")
        gold = item.get("ground_truth_cases", [])
        if not sub_query_id or not gold:
            n_no_gold += 1
            continue
        oracle_record = oracle_map.get(sub_query_id)
        if not oracle_record:
            n_no_oracle_match += 1
            continue

        gold_set = set(gold)
        contexts = item.get("contexts", [])

        hit_rank = None
        for rank, ctx in enumerate(contexts, start=1):
            if ctx.get("case_id") in gold_set:
                hit_rank = rank
                break

        top1_score = contexts[0]["score"] if contexts else None
        top1_case_id = contexts[0]["case_id"] if contexts else None

        oracle_stage1_vals = [x.get("final_score") for x in oracle_record.get("stage1", [])]
        oracle_stage2_vals = [x.get("best_chunk_sim") for x in oracle_record.get("stage2_per_gold", [])]
        oracle_stage1_best = max(oracle_stage1_vals) if oracle_stage1_vals else None
        oracle_stage2_best = max(oracle_stage2_vals) if oracle_stage2_vals else None

        ceiling_deficit = None
        if top1_score is not None and oracle_stage2_best is not None:
            ceiling_deficit = top1_score - oracle_stage2_best

        rows.append({
            "sub_query_id": sub_query_id,
            "source_case_id": item.get("source_case_id"),
            "n_gold": len(gold_set),
            "n_contexts": len(contexts),
            "hit_rank": hit_rank if hit_rank is not None else "",
            "hit_bucket": bucket_for_rank(hit_rank),
            "oracle_stage1_best": oracle_stage1_best,
            "oracle_stage2_best": oracle_stage2_best,
            "retrieved_top1_score": top1_score,
            "retrieved_top1_case_id": top1_case_id,
            "ceiling_deficit": ceiling_deficit,
        })

    print(f"Joined rows: {len(rows)}  (skipped: {n_no_gold} with no ground_truth_cases, "
          f"{n_no_oracle_match} with no oracle match)")

    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    fieldnames = ["sub_query_id", "source_case_id", "n_gold", "n_contexts", "hit_rank",
                 "hit_bucket", "oracle_stage1_best", "oracle_stage2_best",
                 "retrieved_top1_score", "retrieved_top1_case_id", "ceiling_deficit"]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written -> {args.csv_out}  ({len(rows)} rows)")

    # ---- Aggregate stats ----
    bucket_counts = {b: 0 for b in HIT_BUCKETS}
    for r in rows:
        bucket_counts[r["hit_bucket"]] += 1
    n_total = len(rows)
    print("\nHit-bucket distribution (full population, n={}):".format(n_total))
    for b in HIT_BUCKETS:
        print(f"  {b:<12} {bucket_counts[b]:>5}  ({100*bucket_counts[b]/n_total:.1f}%)")

    hit_oracle = [r["oracle_stage2_best"] for r in rows
                 if r["hit_bucket"] != "Missed" and r["oracle_stage2_best"] is not None]
    miss_oracle = [r["oracle_stage2_best"] for r in rows
                  if r["hit_bucket"] == "Missed" and r["oracle_stage2_best"] is not None]
    print(f"\noracle_stage2_best -- Hit group   (n={len(hit_oracle)}): "
          f"median={np.median(hit_oracle):.4f}  mean={np.mean(hit_oracle):.4f}")
    print(f"oracle_stage2_best -- Missed group (n={len(miss_oracle)}): "
          f"median={np.median(miss_oracle):.4f}  mean={np.mean(miss_oracle):.4f}")

    u_stat, p_value = mannwhitneyu(hit_oracle, miss_oracle, alternative="greater")
    print(f"Mann-Whitney U (H1: Hit group's oracle_stage2_best > Missed group's): "
          f"U={u_stat:.1f}  p={p_value:.6g}")

    missed_rows = [r for r in rows if r["hit_bucket"] == "Missed" and r["ceiling_deficit"] is not None]
    n_missed_with_deficit = len(missed_rows)
    n_positive_deficit = sum(1 for r in missed_rows if r["ceiling_deficit"] > 0)
    n_nonpositive_deficit = n_missed_with_deficit - n_positive_deficit
    print(f"\nWithin Missed (n={n_missed_with_deficit}):")
    print(f"  ceiling_deficit > 0 (top result beats even gold's own ceiling): "
          f"{n_positive_deficit} ({100*n_positive_deficit/n_missed_with_deficit:.1f}%)")
    print(f"  ceiling_deficit <= 0 (gold's ceiling exceeds what surfaced -- funnel/rank issue): "
          f"{n_nonpositive_deficit} ({100*n_nonpositive_deficit/n_missed_with_deficit:.1f}%)")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Panel 1: oracle_stage2_best distribution, Hit vs Missed
    ax = axes[0, 0]
    bins = np.linspace(0, max(max(hit_oracle), max(miss_oracle)), 40)
    ax.hist(hit_oracle, bins=bins, alpha=0.55, label=f"Hit (n={len(hit_oracle)})", color="#2e7d32", density=True)
    ax.hist(miss_oracle, bins=bins, alpha=0.55, label=f"Missed (n={len(miss_oracle)})", color="#c0392b", density=True)
    ax.axvline(np.median(hit_oracle), color="#2e7d32", linestyle="--", linewidth=1.5)
    ax.axvline(np.median(miss_oracle), color="#c0392b", linestyle="--", linewidth=1.5)
    ax.set_xlabel("oracle_stage2_best (gold case's own best-chunk similarity to query)")
    ax.set_ylabel("Density")
    ax.set_title(f"Gold-case semantic ceiling: Hit vs Missed\nMann-Whitney p={p_value:.2g} (Hit > Missed)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Panel 2: hit-bucket distribution
    ax = axes[0, 1]
    bucket_vals = [bucket_counts[b] for b in HIT_BUCKETS]
    bucket_pcts = [100 * v / n_total for v in bucket_vals]
    colors = ["#2e7d32", "#7cb342", "#f9a825", "#c0392b"]
    bars = ax.bar(HIT_BUCKETS, bucket_pcts, color=colors)
    for bar, v, pct in zip(bars, bucket_vals, bucket_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.5, f"{pct:.1f}%\n(n={v})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of all sub-queries")
    ax.set_ylim(0, max(bucket_pcts) * 1.25)
    ax.set_title(f"Where the first gold hit lands (n={n_total} sub-queries, exhaustive)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Panel 3: within Missed, ceiling_deficit sign split
    ax = axes[1, 0]
    labels = ["Genuinely low similarity\n(deficit > 0):\ntop result beats gold's\nown ceiling",
             "Pipeline/funnel issue\n(deficit <= 0):\ngold's own ceiling was\nhigher than what surfaced"]
    sizes = [n_positive_deficit, n_nonpositive_deficit]
    wedge_colors = ["#c0392b", "#4c72b0"]
    ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%\n(n={int(round(p/100*n_missed_with_deficit))})",
          colors=wedge_colors, startangle=90, textprops={"fontsize": 9})
    ax.set_title(f"Within the Missed bucket (n={n_missed_with_deficit}):\nis the miss explained by low intrinsic similarity?")

    # Panel 4: scatter oracle ceiling vs actual top-1 score
    ax = axes[1, 1]
    hit_x = [r["oracle_stage2_best"] for r in rows if r["hit_bucket"] != "Missed" and r["oracle_stage2_best"] is not None and r["retrieved_top1_score"] is not None]
    hit_y = [r["retrieved_top1_score"] for r in rows if r["hit_bucket"] != "Missed" and r["oracle_stage2_best"] is not None and r["retrieved_top1_score"] is not None]
    miss_x = [r["oracle_stage2_best"] for r in rows if r["hit_bucket"] == "Missed" and r["oracle_stage2_best"] is not None and r["retrieved_top1_score"] is not None]
    miss_y = [r["retrieved_top1_score"] for r in rows if r["hit_bucket"] == "Missed" and r["oracle_stage2_best"] is not None and r["retrieved_top1_score"] is not None]
    ax.scatter(hit_x, hit_y, s=8, alpha=0.35, color="#2e7d32", label="Hit")
    ax.scatter(miss_x, miss_y, s=8, alpha=0.35, color="#c0392b", label="Missed")
    lims = [0, max(max(hit_x, default=0), max(miss_x, default=0), max(hit_y, default=0), max(miss_y, default=0)) * 1.05]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("oracle_stage2_best (gold's own ceiling)")
    ax.set_ylabel("Actual pipeline top-1 returned score")
    ax.set_title("Points above y=x: something scored higher\nthan the true answer's own ceiling")
    ax.legend(markerscale=2)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Semantic-similarity ceiling vs actual retrieval outcome (Kanon-2 cosine, all sub-queries, n={})".format(n_total),
                fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(FIGURES_OUT_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"\nPlot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
