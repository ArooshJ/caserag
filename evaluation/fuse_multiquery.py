#!/usr/bin/env python
"""
fuse_multiquery.py - Reciprocal Rank Fusion (RRF) of per-sub-query rankings into
one fused ranked list per source case.

WHY THIS EXISTS
    The multi-query datasets carry ONE record per sub-query (~3.5 per source case).
    evaluate.py scores each record independently and macro-averages, so feeding it the
    raw multi-query file yields PER-SUB-QUERY metrics — not the per-CASE number that is
    comparable to the single-query study (which has one record per case). The IL-PCSR
    protocol is per query-case, so the multi-query approach must collapse its sub-query
    rankings back to ONE ranked list per case before scoring. This script does that
    collapse via RRF and writes an evaluate.py-compatible per-case file.

    It is SYSTEM-AGNOSTIC by design: it fuses whatever `retrieved_cases` / `stage1_case_ids`
    are in the input. Run it identically on the CaseRAG multi-query fill-eval output, the
    BM25 multi-query output, and the dense multi-query output so all three systems are
    fused the same way and remain fairly comparable.

WHAT IT DOES
    1. Load the multi-query input (list of sub-query records).
    2. Group records by `source_case_id`.
    3. Per case: RRF-fuse the sub-queries' `retrieved_cases` lists (stage 2) and, separately,
       their `stage1_case_ids` lists (stage 1) into two fused, deduplicated ranked lists.
    4. Emit ONE record per case: source_case_id + the case's gold + the two fused lists,
       in the same shape evaluate.py reads.
    Then:
        python evaluation/evaluate.py --input <fused.json> --retrieval-only --output <results.json>
    gives per-case MAP/MRR/Recall@K directly comparable to the single-query results.

RRF
    score(case c) = Σ_subqueries  1 / (k + rank_i(c)),  rank_i 1-based, only where c appears.
    k (default 60) damps the contribution of low ranks; a case ranked high in ANY sub-query,
    or middling in SEVERAL, floats up. Dedup is automatic (one accumulator per case id).

Usage (from repo root):
    python evaluation/fuse_multiquery.py
    python evaluation/fuse_multiquery.py --input evaluation/data/bm25_multiquery_top100.json
    python evaluation/fuse_multiquery.py --rrf-k 60 --top-n 100
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "evaluation" / "data" / "model=1_k1=200_k2=100_multiquery.json"


def rrf_fuse(ranked_lists, k: int, top_n: int):
    """Fuse several ranked lists of case ids into one via Reciprocal Rank Fusion.

    Each input list is a ranked list of case ids (strings) from one sub-query. A case's
    fused score sums 1/(k+rank) across every list it appears in. Returns the fused case
    ids sorted by score (desc), truncated to top_n (0 = no cap). Ids are stringified so
    int/str id mismatches across sub-queries collapse to one accumulator.
    """
    scores = defaultdict(float)
    for lst in ranked_lists:
        for rank, case_id in enumerate(lst, start=1):
            scores[str(case_id)] += 1.0 / (k + rank)
    fused = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return fused[:top_n] if top_n and top_n > 0 else fused


def first_nonempty_gold(records):
    """Return the gold list for a case. Gold is per source case, so it is identical across
    a case's sub-queries; we take the first non-empty list and warn if any sibling differs
    (a difference would signal a data bug in dataset construction, not normal operation)."""
    gold = []
    seen = None
    mismatch = False
    for r in records:
        g = r.get("ground_truth_cases", []) or []
        if g and not gold:
            gold = g
        if g:
            key = tuple(sorted(str(x) for x in g))
            if seen is None:
                seen = key
            elif key != seen:
                mismatch = True
    return gold, mismatch


def main():
    ap = argparse.ArgumentParser(
        description="RRF-fuse per-sub-query rankings into one ranked list per source case."
    )
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="multi-query file with retrieved_cases/stage1_case_ids filled "
                         "(CaseRAG fill-eval, BM25-multi, or dense-multi output)")
    ap.add_argument("--output", default=None,
                    help="output path (default: input stem with _multiquery -> _fused)")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF damping constant (default 60)")
    ap.add_argument("--top-n", type=int, default=100,
                    help="cap fused list length (default 100; 0 = no cap)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[FAIL] input not found: {in_path}")
        return
    if args.output:
        out_path = Path(args.output)
    else:
        # mirror fill-eval naming: ..._multiquery.json -> ..._fused.json
        stem = in_path.stem
        stem = stem.replace("_multiquery", "").replace("_top100", "")
        out_path = in_path.with_name(f"{stem}_fused.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} sub-query records from {in_path.name}")

    # Group sub-query records by their source case, preserving first-seen case order.
    by_case = OrderedDict()
    skipped_no_id = 0
    for r in records:
        sid = str(r.get("source_case_id", "")).strip()
        if not sid:
            skipped_no_id += 1
            continue
        by_case.setdefault(sid, []).append(r)

    fused_records = []
    n_empty_fusion = 0
    n_gold_mismatch = 0
    for sid, recs in by_case.items():
        stage2_lists = [r.get("retrieved_cases", []) or [] for r in recs]
        stage1_lists = [r.get("stage1_case_ids", []) or [] for r in recs]
        fused_stage2 = rrf_fuse(stage2_lists, args.rrf_k, args.top_n)
        fused_stage1 = rrf_fuse(stage1_lists, args.rrf_k, args.top_n)

        gold, mismatch = first_nonempty_gold(recs)
        if mismatch:
            n_gold_mismatch += 1
        if not fused_stage2 and not fused_stage1:
            n_empty_fusion += 1

        # `question` is cosmetic for retrieval-only eval; join the sub-query issues for
        # traceability so the per-case record still says what was asked.
        issues = [r.get("issue", "") for r in recs if r.get("issue")]
        fused_records.append({
            "source_case_id":     sid,
            "question":           " | ".join(issues),
            # Stringify gold to match the stringified fused lists — otherwise an int/str
            # type mismatch makes evaluate.py's set-intersection silently score all-zeros.
            "ground_truth_cases": [str(x) for x in gold],
            "stage1_case_ids":    fused_stage1,
            "retrieved_cases":    fused_stage2,
            "generated_answer":   "",
            "contexts":           [],
            "n_subqueries":       len(recs),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fused_records, f, indent=2, ensure_ascii=False)

    n_cases = len(fused_records)
    avg_sq = round(len(records) / n_cases, 2) if n_cases else 0
    print(f"\nFused -> {out_path}")
    print(f"  cases (output records) : {n_cases}")
    print(f"  avg sub-queries/case   : {avg_sq}")
    print(f"  rrf_k={args.rrf_k}  top_n={args.top_n}")
    if skipped_no_id:
        print(f"  [WARN] {skipped_no_id} input records had no source_case_id (skipped)")
    if n_empty_fusion:
        print(f"  [WARN] {n_empty_fusion} cases fused to an EMPTY list "
              f"(all sub-queries returned nothing) — evaluate.py skips these")
    if n_gold_mismatch:
        print(f"  [WARN] {n_gold_mismatch} cases had differing gold across sub-queries "
              f"(used first non-empty; check dataset construction)")
    print("\n[OK] Done. To score per-case (comparable to single-query):")
    print(f"  python evaluation/evaluate.py --input {out_path} "
          f"--retrieval-only --output evaluation/results/{out_path.stem}_results.json")


if __name__ == "__main__":
    main()
