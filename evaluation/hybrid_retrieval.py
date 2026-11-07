#!/usr/bin/env python
"""
hybrid_retrieval.py - Fuse two retrieval system outputs per sub-query.

Reads two evaluate.py-compatible JSONs (BM25, Dense, or CaseRAG) and produces
a fused ranked list per sub-query. Two fusion methods:

  rrf          Reciprocal Rank Fusion — rank-based, no scores needed, parameter-free.
  score-fusion Min-max normalised score combination with a configurable alpha weight.
               alpha=1.0 => pure system-A, alpha=0.0 => pure system-B.
               Supports an alpha sweep (comma-separated list) producing one output per alpha.

Output files slot directly into the existing fuse_multiquery.py -> evaluate.py chain.

Usage examples:
  # RRF: BM25 ngram5 + Dense emb1 (existing files, no re-run needed)
  python evaluation/hybrid_retrieval.py \\
      --system-a evaluation/data/bm25_multiquery_ngram5_top100.json \\
      --system-b evaluation/data/dense_emb1_multiquery_top100.json \\
      --label-a bm25ng5 --label-b dense_emb1

  # Score fusion sweep: BM25 + CaseRAG, alpha 0.1..0.9
  python evaluation/hybrid_retrieval.py \\
      --system-a evaluation/data/bm25_multiquery_ngram5_top100.json \\
      --system-b evaluation/data/model=1_k1=200_k2=100_multiquery.json \\
      --label-a bm25ng5 --label-b caserag_emb1 \\
      --method score-fusion --alpha 0.1,0.3,0.5,0.7,0.9
"""

import os
import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "evaluation" / "data"


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def extract_scores(record: dict) -> dict:
    """Return {doc_id: float} scores from a record.

    Priority:
      1. retrieved_scores field  (BM25 / Dense after this change)
      2. sources field           (CaseRAG fill-eval — max chunk score per case)
      3. rank-based proxy        (legacy files without scores — warns once per system)
    Returns an empty dict only when retrieved_cases is also empty.
    """
    if record.get("retrieved_scores"):
        return {str(k): float(v) for k, v in record["retrieved_scores"].items()}

    sources = record.get("sources")
    if sources:
        case_scores: dict = {}
        for chunk in sources:
            cid = str(chunk.get("case_id", chunk.get("doc_id", "")))
            sc  = float(chunk.get("score", chunk.get("similarity_score", 0.0)))
            if cid and sc > case_scores.get(cid, float("-inf")):
                case_scores[cid] = sc
        if case_scores:
            return case_scores

    # Rank-based fallback: score = (n - rank) / n so rank-0 = 1.0, last = ~0
    ranked = record.get("retrieved_cases", [])
    n = len(ranked)
    if n == 0:
        return {}
    return {str(doc_id): (n - rank) / n for rank, doc_id in enumerate(ranked)}


# ---------------------------------------------------------------------------
# Fusion methods
# ---------------------------------------------------------------------------

def fuse_rrf(list_a: list, list_b: list, k: int, top_n: int) -> list:
    """Reciprocal Rank Fusion. score(c) = 1/(k+rank_a) + 1/(k+rank_b).
    Cases appearing in only one list contribute only that term."""
    scores: dict = {}
    for rank, doc_id in enumerate(list_a):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(list_b):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores, key=lambda d: scores[d], reverse=True)
    return ranked[:top_n]


def fuse_scores(raw_a: dict, raw_b: dict, alpha: float, top_n: int) -> list:
    """Min-max normalise each system's scores independently, then combine.
    alpha * norm_a + (1-alpha) * norm_b.
    Cases absent from a system get that system's normalised score = 0.0."""
    eps = 1e-9
    all_ids = set(raw_a) | set(raw_b)

    def normalise(raw: dict) -> dict:
        if not raw:
            return {}
        mn, mx = min(raw.values()), max(raw.values())
        return {d: (raw.get(d, mn) - mn) / (mx - mn + eps) for d in all_ids}

    norm_a = normalise(raw_a)
    norm_b = normalise(raw_b)

    combined = {
        d: alpha * norm_a.get(d, 0.0) + (1 - alpha) * norm_b.get(d, 0.0)
        for d in all_ids
    }
    ranked = sorted(combined, key=lambda d: combined[d], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Record matching
# ---------------------------------------------------------------------------

def build_index(records: list) -> dict:
    """Index records by (source_case_id, sub_query_id|question) for O(1) lookup."""
    idx = {}
    for rec in records:
        key = (str(rec.get("source_case_id", "")),
               str(rec.get("sub_query_id", rec.get("question", ""))))
        idx[key] = rec
    return idx


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def save_atomic(records: list, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fuse two retrieval system outputs via RRF or score fusion."
    )
    ap.add_argument("--system-a", required=True,
                    help="Path to first system output JSON (any evaluate.py-compatible format)")
    ap.add_argument("--system-b", required=True,
                    help="Path to second system output JSON")
    ap.add_argument("--method", default="rrf", choices=["rrf", "score-fusion"],
                    help="Fusion method: rrf (default) or score-fusion")
    ap.add_argument("--alpha", default="0.5",
                    help="For score-fusion: alpha weight for system-A (0.0–1.0). "
                         "Comma-separated for sweep, e.g. '0.1,0.3,0.5,0.7,0.9'. "
                         "alpha=1.0 => pure A, alpha=0.0 => pure B. (default: 0.5)")
    ap.add_argument("--rrf-k", type=int, default=60,
                    help="RRF damping constant (default: 60, same as fuse_multiquery.py)")
    ap.add_argument("--top-n", type=int, default=100,
                    help="Max cases in fused output per query (default: 100)")
    ap.add_argument("--label-a", default=None,
                    help="Short name for system-A used in output filename (default: input stem)")
    ap.add_argument("--label-b", default=None,
                    help="Short name for system-B used in output filename (default: input stem)")
    ap.add_argument("--output-dir", default=None,
                    help=f"Output directory (default: {DATA_DIR})")
    args = ap.parse_args()

    path_a = Path(args.system_a)
    path_b = Path(args.system_b)
    label_a = args.label_a or path_a.stem
    label_b = args.label_b or path_b.stem
    out_dir = Path(args.output_dir) if args.output_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (path_a, path_b):
        if not p.exists():
            print(f"[FAIL] not found: {p}")
            sys.exit(1)

    print(f"Loading system-A: {path_a}")
    with open(path_a, encoding="utf-8") as f:
        records_a = json.load(f)
    print(f"Loading system-B: {path_b}")
    with open(path_b, encoding="utf-8") as f:
        records_b = json.load(f)

    print(f"system-A: {len(records_a)} records  |  system-B: {len(records_b)} records")

    idx_b = build_index(records_b)

    # Parse alpha values for score-fusion
    alphas = []
    if args.method == "score-fusion":
        try:
            alphas = [float(a.strip()) for a in args.alpha.split(",")]
        except ValueError:
            print(f"[FAIL] --alpha must be a float or comma-separated floats, got: {args.alpha}")
            sys.exit(1)
        for a in alphas:
            if not 0.0 <= a <= 1.0:
                print(f"[FAIL] alpha values must be in [0.0, 1.0], got: {a}")
                sys.exit(1)

    # Track whether any record used the rank-based fallback (warn once)
    fallback_warned_a = fallback_warned_b = False

    # Collect fused records per alpha (rrf = single output, score-fusion = one per alpha)
    # For rrf we use alphas=[] and produce one output.
    output_batches = {None: []} if args.method == "rrf" else {a: [] for a in alphas}

    unmatched = 0
    for rec_a in records_a:
        key = (str(rec_a.get("source_case_id", "")),
               str(rec_a.get("sub_query_id", rec_a.get("question", ""))))
        rec_b = idx_b.get(key)

        if rec_b is None:
            unmatched += 1
            # Emit system-A record unchanged into all output batches
            for batch in output_batches.values():
                batch.append(dict(rec_a))
            continue

        list_a = rec_a.get("retrieved_cases", [])
        list_b = rec_b.get("retrieved_cases", [])

        base = {
            "source_case_id":     str(rec_a.get("source_case_id", "")),
            "question":           rec_a.get("question", ""),
            "answer":             rec_a.get("answer", ""),
            "ground_truth_cases": rec_a.get("ground_truth_cases", []),
            "stage1_case_ids":    [],
            "generated_answer":   "",
            "contexts":           [],
        }
        if "sub_query_id" in rec_a:
            base["sub_query_id"] = rec_a["sub_query_id"]
        if "issue" in rec_a:
            base["issue"] = rec_a["issue"]

        if args.method == "rrf":
            fused = fuse_rrf(list_a, list_b, args.rrf_k, args.top_n)
            output_batches[None].append({**base, "retrieved_cases": fused})
        else:
            raw_a = extract_scores(rec_a)
            raw_b = extract_scores(rec_b)

            # Warn once if falling back to rank proxy
            if not fallback_warned_a and not rec_a.get("retrieved_scores") and not rec_a.get("sources"):
                print(f"  [WARN] system-A has no retrieved_scores or sources — "
                      f"using rank-based proxy for score fusion. "
                      f"Re-run with updated bm25/dense scripts for actual scores.")
                fallback_warned_a = True
            if not fallback_warned_b and not rec_b.get("retrieved_scores") and not rec_b.get("sources"):
                print(f"  [WARN] system-B has no retrieved_scores or sources — "
                      f"using rank-based proxy for score fusion.")
                fallback_warned_b = True

            for alpha in alphas:
                fused = fuse_scores(raw_a, raw_b, alpha, args.top_n)
                output_batches[alpha].append({**base, "retrieved_cases": fused})

    if unmatched:
        print(f"  [WARN] {unmatched} system-A records had no matching system-B record "
              f"— emitted unchanged.")

    # Write output files atomically
    output_paths = []
    for alpha_key, records in output_batches.items():
        if args.method == "rrf":
            fname = f"hybrid_rrf_{label_a}_{label_b}_multiquery.json"
        else:
            fname = f"hybrid_sf_a{alpha_key:.2f}_{label_a}_{label_b}_multiquery.json"
        out_path = out_dir / fname
        save_atomic(records, out_path)
        output_paths.append(out_path)
        print(f"  Saved {len(records)} records -> {out_path}")

    print(f"\n[OK] Done. Fused {len(records_a) - unmatched} queries "
          f"({'RRF k=' + str(args.rrf_k) if args.method == 'rrf' else 'score-fusion'}).")
    print("\nNext steps:")
    for out_path in output_paths:
        stem = out_path.stem
        print(f"\n  # fuse sub-queries to per-case then evaluate:")
        print(f"  python evaluation/fuse_multiquery.py \\")
        print(f"      --input {out_path} \\")
        print(f"      --output evaluation/data/{stem}_fused.json")
        print(f"  python evaluation/evaluate.py \\")
        print(f"      --input evaluation/data/{stem}_fused.json \\")
        print(f"      --retrieval-only \\")
        print(f"      --output evaluation/results/{stem}_results.json")


if __name__ == "__main__":
    main()
