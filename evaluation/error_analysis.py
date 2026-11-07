#!/usr/bin/env python
"""
error_analysis.py - Why the two-stage filter loses deep recall (the mechanism).

WHY THIS EXISTS
    significance.py tells you WHETHER systems differ; it does not explain CaseRAG's
    deep-recall deficit. The two-stage engine keeps only the top-k1 CASES after
    stage 1, then searches chunks WITHIN those cases at stage 2. So any ground-truth
    precedent that is not in stage-1's top-k1 is unrecoverable - stage 2 can never
    surface it. This script measures that loss: per config, what fraction of relevant
    precedents never reach stage 2, and how the stage-1 recall ceiling rises with k1.
    That measured number is the paper's central mechanism - it turns "our numbers
    were flat" into "here is the binding constraint, and here is the design lesson."

WHAT IT DOES
    Reads the CaseRAG INPUT files in data/ (model=1_k1=*.json), which retain the raw
    per-query lists (ground_truth_cases, stage1_case_ids, retrieved_cases). For each
    query it computes the stage-1 recall ceiling and stage-2 recall (stage 2 is a
    subset of stage 1, so it is bounded by it), aggregates per config, and lists the
    worst queries (every relevant precedent dropped at stage 1) for qualitative use.

USAGE (from repo root)
    python evaluation/error_analysis.py
    python evaluation/error_analysis.py --worst-n 30
    python evaluation/error_analysis.py --configs data/model=1_k1=200_k2=100.json
"""

import re
import sys
import json
import argparse
from pathlib import Path

# Reuse the evaluator's chunk->case collapse so stage-2 case counting is IDENTICAL
# to evaluate.py (a case repeated once per matching chunk counts once, at its best
# rank). Re-implementing it here would risk drifting from the reported metrics.
from retrieval_metrics import RetrievalMetrics

_COLLAPSE = RetrievalMetrics(collapse="first")


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "evaluation" / "data"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "error_analysis.json"
DEFAULT_WORST_MD = RESULTS_DIR / "error_analysis_worst.md"


def parse_ks(name: str):
    """Pull (k1, k2) out of a filename like 'model=1_k1=200_k2=100.json' for labels
    and for sorting configs by k1 (the variable that drives the recall ceiling)."""
    k1 = re.search(r"k1=(\d+)", name)
    k2 = re.search(r"k2=(\d+)", name)
    return (int(k1.group(1)) if k1 else None,
            int(k2.group(1)) if k2 else None)


def analyze_file(path: Path, worst_n: int) -> dict:
    """Compute stage-1 / stage-2 recall stats for one CaseRAG input file.

    Per query: relevant = GT precedents; stage1 = top-k1 cases; stage2 = unique cases
    surfaced by stage-2 chunks (collapsed). Returns aggregate recall numbers plus the
    worst queries where stage-1 recall is 0 (all relevant precedents filtered out).
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    k1, k2 = parse_ks(path.name)

    n_q = 0
    sum_s1_recall = 0.0
    sum_s2_recall = 0.0
    total_gt = 0
    gt_lost_s1 = 0
    n_total_loss = 0            # queries where every GT precedent was dropped at stage 1
    no_stage1 = 0              # records missing stage1_case_ids (diagnostic)
    worst = []

    for item in data:
        relevant = {str(x) for x in item.get("ground_truth_cases", [])}
        if not relevant:
            continue            # the 1 query with no ground truth is skipped
        stage1_list = item.get("stage1_case_ids", [])
        if not stage1_list:
            no_stage1 += 1
        stage1 = {str(x) for x in stage1_list}
        stage2 = set(_COLLAPSE._collapse_to_cases([str(x) for x in item.get("retrieved_cases", [])]))

        s1_recall = len(relevant & stage1) / len(relevant)
        s2_recall = len(relevant & stage2) / len(relevant)

        n_q += 1
        sum_s1_recall += s1_recall
        sum_s2_recall += s2_recall
        total_gt += len(relevant)
        gt_lost_s1 += len(relevant - stage1)
        if s1_recall == 0.0:
            n_total_loss += 1
            worst.append({
                "query_id": str(item.get("source_case_id", "")),
                "question": (item.get("question") or "")[:140],
                "n_gt": len(relevant),
            })

    worst.sort(key=lambda w: w["n_gt"], reverse=True)
    return {
        "file": path.name,
        "k1": k1, "k2": k2,
        "n_queries": n_q,
        "mean_stage1_recall_ceiling": (sum_s1_recall / n_q) if n_q else 0.0,
        "mean_stage2_recall": (sum_s2_recall / n_q) if n_q else 0.0,
        "pct_gt_lost_at_stage1": (gt_lost_s1 / total_gt) if total_gt else 0.0,
        "queries_total_stage1_loss": n_total_loss,
        "records_missing_stage1_ids": no_stage1,
        "worst_queries": worst[:worst_n],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Quantify stage-1 recall loss in the CaseRAG two-stage engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--data-dir", default=str(DATA_DIR),
                    help="Directory of CaseRAG input files (default: evaluation/data)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Explicit input files (default: all data/model=1_k1=*.json)")
    ap.add_argument("--worst-n", type=int, default=20,
                    help="How many worst (full stage-1 loss) queries to record per config")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to save the JSON report")
    ap.add_argument("--worst-md", default=str(DEFAULT_WORST_MD), help="Where to save the worst-query markdown")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.configs:
        files = [Path(c) for c in args.configs]
    else:
        files = sorted(data_dir.glob("model=1_k1=*.json"))
    files = [f for f in files if f.exists()]
    if not files:
        print(f"[FAIL] No CaseRAG input files found in {data_dir} (model=1_k1=*.json)")
        sys.exit(1)

    reports = [analyze_file(f, args.worst_n) for f in files]
    reports.sort(key=lambda r: (r["k1"] if r["k1"] is not None else 0))

    print("\n" + "=" * 92)
    print("STAGE-1 RECALL LOSS BY CONFIG  (stage-2 recall is bounded above by the stage-1 ceiling)")
    print("=" * 92)
    print(f"{'config':22s} {'k1':>5s} {'queries':>8s} {'S1 ceiling':>11s} {'S2 recall':>10s} "
          f"{'%GT lost@S1':>12s} {'all-lost q':>11s}")
    print("-" * 92)
    for r in reports:
        cfg = f"k1={r['k1']}_k2={r['k2']}"
        print(f"{cfg:22s} {str(r['k1']):>5s} {r['n_queries']:>8d} "
              f"{r['mean_stage1_recall_ceiling']:>11.4f} {r['mean_stage2_recall']:>10.4f} "
              f"{r['pct_gt_lost_at_stage1']:>11.1%} {r['queries_total_stage1_loss']:>11d}")
    missing = sum(r["records_missing_stage1_ids"] for r in reports)
    if missing:
        print(f"\n[WARN] {missing} record(s) across configs had no stage1_case_ids "
              f"(treated as empty stage 1).")

    # Save JSON report.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] saved per-config report -> {out_path}")

    # Save a readable worst-query markdown for the qualitative paragraph in the paper.
    md_lines = ["# Stage-1 total-loss queries (every relevant precedent filtered out)\n"]
    for r in reports:
        md_lines.append(f"\n## config k1={r['k1']} k2={r['k2']}  "
                        f"({r['queries_total_stage1_loss']} fully-lost queries)\n")
        for w in r["worst_queries"]:
            md_lines.append(f"- `{w['query_id']}` (n_gt={w['n_gt']}): {w['question']}")
    md_path = Path(args.worst_md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[OK] saved worst-query list -> {md_path}")


if __name__ == "__main__":
    main()
