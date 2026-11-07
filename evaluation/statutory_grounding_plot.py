"""
statutory_grounding_plot.py - do retrieved precedents share actual statutory grounding with
the query case, or just topical/semantic similarity?

Run with:
    python evaluation/statutory_grounding_plot.py

WHY THIS EXISTS
    semantic_ceiling.md already shows that most misses reflect genuinely low embedding-space
    similarity, even under the most charitable oracle comparison. This script asks a distinct,
    complementary question using a completely different, non-embedding signal: the paper's own
    Legal Evaluation Framework (Section 4) names "statutory consistency" as one of six
    jurisprudential dimensions of legal relevance that semantic similarity does not capture.
    IL-PCSR's case metadata records each case's actually-cited statutes with normalised IDs
    (`relevant_statute_ids`) -- a hard, court-verifiable signal, independent of any embedding
    model. This script checks whether the top result CaseRAG retrieves is more likely to share
    statutory grounding with the query case when the retrieval is correct (a cited precedent
    appears among the sub-query's 100 returned paragraphs) than when it misses (none does).
    The top result of a correct retrieval need not itself be a cited precedent.

INPUT
    il-pcsr_dataset/corpus/*.json and il-pcsr_dataset/test_queries/*.json
        -- per-case `court` and `relevant_statute_ids`/`relevant_statutes` metadata (not
        embeddings; independently recorded case facts). 3,807 cases total (3,183 corpus +
        622 query source cases -- query source cases are held out of the retrieval corpus, so
        their own metadata lives only in test_queries/).
    evaluation/data/model=3_k1=200_k2=100_cosine_multiquery.json
        -- the actual pipeline output (same file semantic_ceiling_plot.py uses).
    research_paper/ablation/data/semantic_ceiling.csv
        -- reused ONLY for its `hit_bucket` column, so this analysis's hit/miss split is
        identical to, and directly comparable with, the semantic-ceiling analysis (same 2,195
        sub-queries, same bucketing, different lens). Not required to exist for standalone
        overlap stats, but the script will skip the bucket cross-tab panel if missing.

DEFINITIONS
    Two overlap granularities, computed side by side because they can disagree:
      statute-ID overlap : exact match on `relevant_statute_ids` (e.g. "Section 32 of The
                           Income Tax Act, 1955" is a different ID from "Section 35 of The
                           Income Tax Act, 1955" -- strict, section-level).
      act-level overlap  : match on the Act name only, parsed from `relevant_statutes` text
                           (e.g. both sections above collapse to "The Income Tax Act, 1955" --
                           coarser, forgives citing a different section of the same law).
    Pairs where EITHER side has zero recorded statutes are excluded from that pair's
    denominator (28.7% of all cases have no `relevant_statute_ids` recorded -- counting these
    as "no overlap" would understate true overlap rates with a data-availability artifact, not
    a legal one; verified by checking the excluded count before trusting the final numbers).

OUTPUT
    research_paper/ablation/figures/statutory_grounding.png
    research_paper/ablation/data/statutory_grounding.csv  (one row per sub-query)

NOTE ON SCOPE
    Kanon-2/cosine pipeline only, same as semantic_ceiling.md. One chi-square test (Hit-top-1
    vs Missed-top-1 statute-ID overlap). No new embedding or retrieval calls -- reads existing
    case metadata and the existing raw retrieval file only.

USAGE
    python evaluation/statutory_grounding_plot.py
"""

import os
import csv
import json
import glob
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

DATA_DIR = os.path.join("evaluation", "data")
FIGURES_OUT_DIR = os.path.join("research_paper", "ablation", "figures")
DATA_OUT_DIR = os.path.join("research_paper", "ablation", "data")

RAW_CASERAG_FILE = os.path.join(DATA_DIR, "model=3_k1=200_k2=100_cosine_multiquery.json")
SEMANTIC_CEILING_CSV = os.path.join(DATA_OUT_DIR, "semantic_ceiling.csv")
CORPUS_GLOBS = [os.path.join("il-pcsr_dataset", "corpus", "*.json"),
               os.path.join("il-pcsr_dataset", "test_queries", "*.json")]

HIT_BUCKET_ORDER = ["Hit@1-10", "Hit@11-50", "Hit@51-100", "Missed"]


def act_name(statute_text: str) -> str:
    """'Section 32 of The Income Tax Act, 1955' -> 'The Income Tax Act, 1955'."""
    m = re.search(r"\bof\s+(.+)$", statute_text)
    return m.group(1).strip() if m else statute_text.strip()


def build_case_metadata() -> dict:
    meta = {}
    for pattern in CORPUS_GLOBS:
        for fp in glob.glob(pattern):
            d = json.load(open(fp, encoding="utf-8"))
            cid = str(d.get("caseID") or os.path.basename(fp).replace(".json", ""))
            meta[cid] = {
                "court": d.get("court", ""),
                "statute_ids": set(d.get("relevant_statute_ids", []) or []),
                "acts": set(act_name(s) for s in (d.get("relevant_statutes", []) or [])),
            }
    return meta


def load_hit_buckets() -> dict:
    if not os.path.exists(SEMANTIC_CEILING_CSV):
        print(f"[WARN] {SEMANTIC_CEILING_CSV} not found -- bucket cross-tab panel will be skipped.")
        return {}
    out = {}
    with open(SEMANTIC_CEILING_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["sub_query_id"]] = row["hit_bucket"]
    return out


def main():
    ap = argparse.ArgumentParser(description="Statutory/jurisdictional grounding of retrieved vs gold precedents.")
    ap.add_argument("--plot-out", default=os.path.join(FIGURES_OUT_DIR, "statutory_grounding.png"))
    ap.add_argument("--csv-out", default=os.path.join(DATA_OUT_DIR, "statutory_grounding.csv"))
    args = ap.parse_args()

    print("Building case metadata (court + statute_ids + acts)...")
    meta = build_case_metadata()
    print(f"  {len(meta)} cases with metadata")
    n_empty_statutes = sum(1 for m in meta.values() if not m["statute_ids"])
    print(f"  {n_empty_statutes} ({100*n_empty_statutes/len(meta):.1f}%) have zero relevant_statute_ids recorded")

    print(f"Loading raw CaseRAG file: {RAW_CASERAG_FILE}")
    rag_data = json.load(open(RAW_CASERAG_FILE, encoding="utf-8"))
    print(f"  {len(rag_data)} records")

    hit_buckets = load_hit_buckets()

    rows = []
    for item in rag_data:
        src = item.get("source_case_id")
        sub_query_id = item.get("sub_query_id")
        gold = item.get("ground_truth_cases", [])
        if not src or not gold or src not in meta:
            continue
        q_statute_ids = meta[src]["statute_ids"]
        q_acts = meta[src]["acts"]
        q_court = meta[src]["court"]
        contexts = item.get("contexts", [])

        n_gold_meta = n_gold_statute_overlap = n_gold_act_overlap = 0
        for gid in gold:
            if gid not in meta:
                continue
            g = meta[gid]
            if not q_statute_ids or not g["statute_ids"]:
                continue
            n_gold_meta += 1
            if q_statute_ids & g["statute_ids"]:
                n_gold_statute_overlap += 1
            if q_acts & g["acts"]:
                n_gold_act_overlap += 1

        gold_set = set(gold)
        hit_rank = None
        for rank, ctx in enumerate(contexts, start=1):
            if ctx.get("case_id") in gold_set:
                hit_rank = rank
                break
        top1_case_id = contexts[0]["case_id"] if contexts else None

        top1_court = top1_is_sc = top1_statute_overlap = top1_act_overlap = ""
        if top1_case_id and top1_case_id in meta:
            t = meta[top1_case_id]
            top1_court = t["court"]
            top1_is_sc = t["court"] == "Supreme Court of India"
            if q_statute_ids and t["statute_ids"]:
                top1_statute_overlap = bool(q_statute_ids & t["statute_ids"])
            if q_acts and t["acts"]:
                top1_act_overlap = bool(q_acts & t["acts"])

        rows.append({
            "sub_query_id": sub_query_id,
            "source_case_id": src,
            "query_court": q_court,
            "hit_bucket": hit_buckets.get(sub_query_id, ""),
            "n_gold": len(gold),
            "n_gold_with_meta": n_gold_meta,
            "n_gold_statute_overlap": n_gold_statute_overlap,
            "n_gold_act_overlap": n_gold_act_overlap,
            "top1_case_id": top1_case_id,
            "top1_court": top1_court,
            "top1_is_supreme_court": top1_is_sc,
            "top1_statute_overlap": top1_statute_overlap,
            "top1_act_overlap": top1_act_overlap,
        })

    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written -> {args.csv_out}  ({len(rows)} rows)")

    # ---- Aggregate: Gold vs Hit-top1 vs Missed-top1, both granularities ----
    total_gold_meta = sum(r["n_gold_with_meta"] for r in rows)
    total_gold_statute_overlap = sum(r["n_gold_statute_overlap"] for r in rows)
    total_gold_act_overlap = sum(r["n_gold_act_overlap"] for r in rows)

    hit_rows = [r for r in rows if r["hit_bucket"] and r["hit_bucket"] != "Missed" and r["top1_statute_overlap"] != ""]
    miss_rows = [r for r in rows if r["hit_bucket"] == "Missed" and r["top1_statute_overlap"] != ""]

    def rate(rs, key):
        vals = [r[key] for r in rs if r[key] != ""]
        return (sum(vals), len(vals))

    gold_statute_pct = 100 * total_gold_statute_overlap / total_gold_meta
    gold_act_pct = 100 * total_gold_act_overlap / total_gold_meta
    hit_statute_n, hit_statute_d = rate(hit_rows, "top1_statute_overlap")
    miss_statute_n, miss_statute_d = rate(miss_rows, "top1_statute_overlap")
    hit_act_n, hit_act_d = rate(hit_rows, "top1_act_overlap")
    miss_act_n, miss_act_d = rate(miss_rows, "top1_act_overlap")

    print(f"\n[statute-ID level] Gold: {total_gold_statute_overlap}/{total_gold_meta} = {gold_statute_pct:.1f}%")
    print(f"[statute-ID level] Hit-top1: {hit_statute_n}/{hit_statute_d} = {100*hit_statute_n/hit_statute_d:.1f}%")
    print(f"[statute-ID level] Missed-top1: {miss_statute_n}/{miss_statute_d} = {100*miss_statute_n/miss_statute_d:.1f}%")

    print(f"\n[act level] Gold: {total_gold_act_overlap}/{total_gold_meta} = {gold_act_pct:.1f}%")
    print(f"[act level] Hit-top1: {hit_act_n}/{hit_act_d} = {100*hit_act_n/hit_act_d:.1f}%")
    print(f"[act level] Missed-top1: {miss_act_n}/{miss_act_d} = {100*miss_act_n/miss_act_d:.1f}%")

    table = np.array([[hit_statute_n, hit_statute_d - hit_statute_n],
                      [miss_statute_n, miss_statute_d - miss_statute_n]])
    chi2, p_value, dof, _ = chi2_contingency(table)
    print(f"\nChi-square (Hit-top1 vs Missed-top1, statute-ID overlap): chi2={chi2:.2f}, p={p_value:.3g}")

    hit_sc_n, hit_sc_d = rate(hit_rows, "top1_is_supreme_court")
    miss_sc_n, miss_sc_d = rate(miss_rows, "top1_is_supreme_court")
    print(f"\nTop-1 is Supreme Court: Hit {100*hit_sc_n/hit_sc_d:.1f}%  Missed {100*miss_sc_n/miss_sc_d:.1f}%  "
          f"(court corpus base rate ~89% SC -- interpret relative to that, not to 50%)")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

    ax = axes[0, 0]
    groups = ["Gold\n(true citations)", "Hit-top1\n(correct retrieval)", "Missed-top1\n(wrong retrieval)"]
    vals = [gold_statute_pct, 100*hit_statute_n/hit_statute_d, 100*miss_statute_n/miss_statute_d]
    colors = ["#2e7d32", "#4c72b0", "#c0392b"]
    bars = ax.bar(groups, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("% sharing >=1 statute ID with query case")
    ax.set_ylim(0, max(vals)*1.3)
    ax.set_title(f"Statute-ID (section-level) overlap\nchi2 p={p_value:.2g} (Hit-top1 vs Missed-top1)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[0, 1]
    vals2 = [gold_act_pct, 100*hit_act_n/hit_act_d, 100*miss_act_n/miss_act_d]
    bars = ax.bar(groups, vals2, color=colors)
    for bar, v in zip(bars, vals2):
        ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("% sharing >=1 Act with query case")
    ax.set_ylim(0, max(vals2)*1.3)
    ax.set_title("Act-level (coarser) overlap\nsame direction, same rank order as section-level")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[1, 0]
    sc_vals = [100*hit_sc_n/hit_sc_d, 100*miss_sc_n/miss_sc_d]
    bars = ax.bar(["Hit-top1", "Missed-top1"], sc_vals, color=["#4c72b0", "#c0392b"])
    for bar, v in zip(bars, sc_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.axhline(89.2, color="black", linestyle="--", linewidth=1, label="corpus base rate (89.2% SC)")
    ax.set_ylabel("% of top-1 results that are Supreme Court")
    ax.set_ylim(0, 100)
    ax.set_title("Court hierarchy: NOT discriminating\n(both track the corpus's own SC-heavy base rate)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[1, 1]
    if hit_buckets:
        bucket_statute_rate = []
        bucket_ns = []
        for b in HIT_BUCKET_ORDER:
            brs = [r for r in rows if r["hit_bucket"] == b and r["top1_statute_overlap"] != ""]
            n, d = rate(brs, "top1_statute_overlap")
            bucket_statute_rate.append(100*n/d if d else 0)
            bucket_ns.append(d)
        bars = ax.bar(HIT_BUCKET_ORDER, bucket_statute_rate, color=["#2e7d32", "#7cb342", "#f9a825", "#c0392b"])
        for bar, v, n in zip(bars, bucket_statute_rate, bucket_ns):
            ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
        ax.axhline(gold_statute_pct, color="green", linestyle="--", linewidth=1, label=f"Gold rate ({gold_statute_pct:.1f}%)")
        ax.set_ylabel("% top-1 sharing >=1 statute ID with query")
        ax.set_ylim(0, max(bucket_statute_rate)*1.3)
        ax.set_title("Statute overlap by hit-rank bucket\n(finer view of the same pattern)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "semantic_ceiling.csv not found\n-- run semantic_ceiling_plot.py first",
               ha="center", va="center")

    fig.suptitle("Statutory grounding: retrieved precedents vs the query case's own cited statutes (Kanon-2 cosine)",
                fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(FIGURES_OUT_DIR, exist_ok=True)
    fig.savefig(args.plot_out, dpi=200)
    print(f"\nPlot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
