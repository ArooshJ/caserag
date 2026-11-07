"""
extract_all_metrics.py - flatten every evaluate.py result file into one tidy CSV.

Run with:
    python evaluation/extract_all_metrics.py

WHY THIS EXISTS
    evaluation/results/ has ~30 evaluate.py output files (one per system/embedder/
    query-mode/k1-k2 configuration), each a nested JSON blob
    {"average_metrics": {"retrieval": {"stage1": {...}, "stage2": {...}}, "generation": {...}},
     "per_query_results": [...]}. Comparing systems means opening several of these by hand
     every time. This script walks every result file once, pulls out average_metrics.retrieval
    for each stage, and writes ONE long-format ("tidy") CSV: one row per
    (file, stage, metric, k, value). Long format survives files having different K-cutoff
    lists, is trivially filterable/pivotable in Excel/pandas/R, and is the shared foundation
    every other plot script in this folder builds on (see embedding_contingency_plot.py,
    recall_at_k_curve_plot.py) -- so a fix here propagates everywhere instead of needing N
    copies of the same JSON-parsing logic.

WHICH FILES ARE INCLUDED
    Only files whose top level has an "average_metrics" key (i.e. evaluate.py's own output
    shape). This automatically EXCLUDES files that serve a different purpose and have a
    different shape:
      - query_faithfulness_results.json, gold_oracle_emb1_multiquery.json, gold_oracle_emb3_cosine_multiquery.json ({_summary, records})
      - oracle_analysis_emb*.json ({_headline, metric1.., metric2.., ...})
      - error_analysis.json, significance.json, pld_complete_evaluation_dataset_k=5.json (LISTs)
      - qualitative_semantic_gap_examples.json ({metadata, top_20_*})
    These are real, useful files for other analyses -- they're just not "a system's scored
    metrics" and don't belong in a metrics table. Skipped files are printed, not silently
    dropped, so nothing is invisibly lost.

    A NOTE ON significance.json SPECIFICALLY: it exists and is real, but it is a significance
    test over an EARLIER single-query BGE (emb1) k1-sweep vs BM25/Dense (616 aligned queries,
    files results_model=1_k1=*_k2=*.json) -- NOT a test of the current headline claim
    (CaseRAG vs Dense, emb3/Kanon-2, cosine, multi-query fused). Do not read it as answering
    the paper's pending T2 significance gate; that test has not been run yet as of this
    writing (see research_paper/ablation/README.md).

OUTPUT
    research_paper/ablation/data/all_systems_metrics_long.csv, columns:
        file, label, stage, metric, k, value
    "label" is a best-effort human-readable system tag guessed from the filename (system,
    embedder, distance op, query mode, fusion) -- always double-check it against the filename
    before quoting it in the paper; it is a convenience, not an authoritative name.

USAGE
    python evaluation/extract_all_metrics.py
    python evaluation/extract_all_metrics.py --results-dir evaluation/results --out research_paper/ablation/data/all_systems_metrics_long.csv
"""

import os
import re
import csv
import json
import glob
import argparse
from typing import Dict, List, Optional

RESULTS_DIR = os.path.join("evaluation", "results")
DEFAULT_OUT = os.path.join("research_paper", "ablation", "data", "all_systems_metrics_long.csv")

SCALAR_METRICS = ["AP", "RR", "BPREF", "MAP", "MRR"]
CUTOFF_METRIC_PREFIXES = ["P@", "Recall@", "NDCG@", "Hit@"]


HYBRID_SUBSYSTEM = {
    "caseragE1": "CaseRAG-emb1(BGE)", "caseragE3cos": "CaseRAG-emb3(Kanon-2)cosine",
    "denseE1": "Dense-emb1(BGE)", "denseE3cos": "Dense-emb3(Kanon-2)cosine",
}


def guess_label(filename: str) -> str:
    """Best-effort human-readable tag from the filename. Convenience only -- see docstring.

    NOTE: "hybrid_rrf" must be checked BEFORE "bm25" -- every hybrid filename (e.g.
    hybrid_rrf_bm25ng5_caseragE1_multiquery_results.json) contains "bm25" as a substring, so
    checking bm25 first would mislabel every hybrid file as plain BM25."""
    name = filename.replace(".json", "")
    parts = []
    if "hybrid_rrf" in name:
        parts.append("Hybrid-RRF(BM25-5gram")
        for tag, label in HYBRID_SUBSYSTEM.items():
            if tag in name:
                parts[-1] += f"+{label}"
                break
        parts[-1] += ")"
    elif "bm25" in name:
        parts.append("BM25")
        if "ng1" in name:
            parts.append("unigram")
        if "ng5" in name:
            parts.append("5-gram")
        if "fulltext" in name:
            parts.append("full-text-query")
    elif name.startswith("caserag") or name.startswith("results_model") or name.startswith("model="):
        parts.append("CaseRAG")
    elif name.startswith("dense"):
        parts.append("Dense")

    m = re.search(r"emb(\d)", name)
    if m:
        parts.append(f"emb{m.group(1)}" + (" (BGE)" if m.group(1) == "1" else " (Kanon-2)" if m.group(1) == "3" else ""))
    m = re.search(r"model=(\d)", name)
    if m and not any(p.startswith("emb") for p in parts):
        parts.append(f"emb{m.group(1)}")
    if "cosine" in name:
        parts.append("cosine")
    if "multiquery" in name:
        parts.append("multi-query")
    elif "singlequery" in name:
        parts.append("single-query")
    if "fused" in name:
        parts.append("fused")

    m = re.search(r"k1[=_](\d+)", name)
    k1 = m.group(1) if m else None
    m = re.search(r"k2[=_](\d+)", name)
    k2 = m.group(1) if m else None
    if k1 is None and k2 is None:
        # Older filenames concatenate the two cutoffs with no separator, e.g.
        # "..._k10k100" (k1=10, k2=100), "..._k5k30" (k1=5, k2=30), "..._k5k5", "..._k200k100" --
        # confirmed against significance.json's file list, which spells out the same four
        # configs with explicit k1=/k2= names.
        m = re.search(r"k(\d+)k(\d+)$", name)
        if m:
            k1, k2 = m.group(1), m.group(2)
    if k1 is not None:
        parts.append(f"k1={k1}")
    if k2 is not None:
        parts.append(f"k2={k2}")

    if name.startswith("old_") or "duplicate" in name:
        parts.append("[DEPRECATED COPY]")
    return " ".join(parts) if parts else name


def flatten_stage(rows: List[Dict], file: str, label: str, stage: str, stage_metrics: Dict) -> None:
    for metric_key, value in stage_metrics.items():
        if metric_key in SCALAR_METRICS:
            rows.append({"file": file, "label": label, "stage": stage,
                         "metric": metric_key, "k": "", "value": value})
            continue
        for prefix in CUTOFF_METRIC_PREFIXES:
            if metric_key.startswith(prefix):
                metric_name = prefix.rstrip("@")
                k = metric_key[len(prefix):]
                rows.append({"file": file, "label": label, "stage": stage,
                             "metric": metric_name, "k": k, "value": value})
                break


def extract_file(fp: str) -> Optional[List[Dict]]:
    try:
        data = json.load(open(fp, encoding="utf-8"))
    except Exception as e:
        print(f"  [SKIP] {os.path.basename(fp)} -- unreadable JSON ({e})")
        return None
    if not isinstance(data, dict) or "average_metrics" not in data:
        return None

    filename = os.path.basename(fp)
    label = guess_label(filename)
    retrieval = data["average_metrics"].get("retrieval", {})
    rows = []
    for stage in ("stage1", "stage2"):
        stage_metrics = retrieval.get(stage) or {}
        if not stage_metrics:
            continue
        flatten_stage(rows, filename, label, stage, stage_metrics)

    generation = data["average_metrics"].get("generation") or {}
    if generation:
        for metric_key, value in generation.items():
            if isinstance(value, (int, float)):
                rows.append({"file": filename, "label": label, "stage": "generation",
                             "metric": metric_key, "k": "", "value": value})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Flatten every evaluate.py result file into one tidy CSV.")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    included, skipped = [], []
    all_rows: List[Dict] = []
    for fp in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        rows = extract_file(fp)
        if rows is None:
            skipped.append(os.path.basename(fp))
            continue
        included.append(os.path.basename(fp))
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "stage", "metric", "k", "value"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Included ({len(included)} files, {len(all_rows)} rows):")
    for name in included:
        print(f"  + {name}")
    print(f"\nSkipped (no average_metrics key -- different purpose, see module docstring; "
          f"{len(skipped)} files):")
    for name in skipped:
        print(f"  - {name}")
    print(f"\nCSV written -> {args.out}")


if __name__ == "__main__":
    main()
