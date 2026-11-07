#!/usr/bin/env python
"""
bm25_retrieval.py - BM25 baseline retrieval over the ILPCSR corpus.

Produces a JSON file in the same format as ilpcsr_evaluation_dataset.json
with retrieved_cases (and stage1_case_ids) populated by BM25 ranking, so
evaluate.py --retrieval-only can consume it directly.

Usage (from repo root):
    pip install rank_bm25
    python evaluation/bm25_retrieval.py
    python evaluation/bm25_retrieval.py --top-k 100 --output evaluation/data/bm25_top100.json

Colab (clone repo, cd to root):
    !python evaluation/bm25_retrieval.py --output evaluation/data/bm25_top100.json
    # then evaluate:
    !python evaluation/evaluate.py --input evaluation/data/bm25_top100.json --retrieval-only --output evaluation/data/bm25_results.json

Note on query field:
    Default uses the LLM-generated 'question' field — same protocol as CaseRAG evaluation.
    The published IL-PCSR BM25 baseline (MAP=43.98) uses the full source case text as query
    (different protocol); to approximate it use --query-field source_case_id with a custom
    loader, or compare only within our query protocol.
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("[FAIL] rank_bm25 not installed. Run: pip install rank_bm25")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "il-pcsr_dataset" / "corpus"
EVAL_DATASET = REPO_ROOT / "evaluation" / "ilpcsr_evaluation_dataset.json"
MULTIQUERY_DATASET = REPO_ROOT / "evaluation" / "ilpcsr_multiquery_evaluation_dataset.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "data" / "bm25_top100.json"
DEFAULT_OUTPUT_MULTI = REPO_ROOT / "evaluation" / "data" / "bm25_multiquery_top100.json"

PARA_FIELDS = [
    "Fact", "Issue", "Petitioner's Argument", "Respondent's Argument",
    "Analysis of the law", "Precedent Analysis", "Court's Reasoning", "Conclusion",
]


def tokenize(text: str, max_n: int = 1):
    """Lowercase word tokenizer. Returns unigrams plus all n-grams up to max_n."""
    words = re.findall(r"[a-z]+", text.lower())
    if max_n == 1:
        return words
    tokens = list(words)
    for n in range(2, max_n + 1):
        tokens.extend("_".join(words[i:i+n]) for i in range(len(words) - n + 1))
    return tokens


def case_full_text(data: dict) -> str:
    """Concatenate all paragraph fields into one document string."""
    parts = []
    for field in PARA_FIELDS:
        paras = data.get(field, [])
        if isinstance(paras, list):
            parts.extend(p for p in paras if isinstance(p, str))
        elif isinstance(paras, str):
            parts.append(paras)
    return " ".join(parts)


def build_index(corpus_dir: Path, max_n: int = 1):
    """
    Load all corpus JSONs, build BM25Okapi index.
    Returns (case_ids: list[str], bm25: BM25Okapi).
    """
    files = sorted(corpus_dir.glob("*.json"))
    n = len(files)
    if n == 0:
        print(f"[FAIL] No JSON files found in {corpus_dir}")
        sys.exit(1)
    print(f"Loading corpus: {n} files from {corpus_dir}")

    case_ids = []
    tokenized_docs = []
    for i, fp in enumerate(files, 1):
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        case_id = str(data.get("caseID", fp.stem))
        text = case_full_text(data)
        case_ids.append(case_id)
        tokenized_docs.append(tokenize(text, max_n))
        if i % 500 == 0:
            print(f"  tokenized {i}/{n}...")

    print(f"Building BM25Okapi index (ngram={max_n}) over {n} docs...")
    bm25 = BM25Okapi(tokenized_docs)
    print("[OK] Index ready.")
    return case_ids, bm25


def bm25_top_k(query: str, bm25: BM25Okapi, case_ids: list, top_k: int, max_n: int = 1):
    """Return (ranked_ids, scores_dict) for the top-k cases by BM25 score."""
    tokens = tokenize(query, max_n)
    if not tokens:
        return [], {}
    score_arr = bm25.get_scores(tokens)
    ranked_idx = sorted(range(len(score_arr)), key=lambda i: score_arr[i], reverse=True)[:top_k]
    ranked_ids = [case_ids[i] for i in ranked_idx]
    scores = {case_ids[i]: float(score_arr[i]) for i in ranked_idx}
    return ranked_ids, scores


def _load_existing(out_path: Path) -> list:
    """Load a partial output file for resume. Returns [] if absent or corrupt."""
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            print(f"  [WARN] existing output unreadable — starting fresh: {out_path}")
    return []


def _save_atomic(records: list, out_path: Path):
    """Write records to a .tmp file then rename — crash-safe atomic save."""
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser(
        description="BM25 baseline retrieval — outputs evaluate.py-compatible JSON"
    )
    ap.add_argument("--corpus-dir", default=str(CORPUS_DIR),
                    help="Path to il-pcsr_dataset/corpus/ (default: auto-detected from repo root)")
    ap.add_argument("--eval-dataset", default=None,
                    help="Path to eval dataset (auto: single-query or multi-query based on --multi-query flag)")
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: bm25_top100.json or bm25_multiquery_top100.json)")
    ap.add_argument("--multi-query", action="store_true",
                    help="Input is the multi-query dataset — carry through sub_query_id/issue fields")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Cases to retrieve per query (default: 100)")
    ap.add_argument("--query-field", default="question",
                    help="Field in eval dataset to use as BM25 query text (default: 'question')")
    ap.add_argument("--ngram", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help="Max n-gram size for tokenization (default: 1 = unigrams). "
                         "IL-PCSR best BM25 used 5; 3 is a reasonable middle ground.")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if args.eval_dataset is None:
        args.eval_dataset = str(MULTIQUERY_DATASET if args.multi_query else EVAL_DATASET)
    eval_path = Path(args.eval_dataset)
    if args.output:
        output_path = Path(args.output)
    else:
        _data_dir = REPO_ROOT / "evaluation" / "data"
        if args.multi_query:
            output_path = _data_dir / f"bm25_multiquery_ngram{args.ngram}_top{args.top_k}.json"
        else:
            output_path = _data_dir / f"bm25_ngram{args.ngram}_top{args.top_k}.json"

    if not corpus_dir.exists():
        print(f"[FAIL] corpus dir not found: {corpus_dir}")
        sys.exit(1)
    if not eval_path.exists():
        print(f"[FAIL] eval dataset not found: {eval_path}")
        sys.exit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_ids, bm25 = build_index(corpus_dir, args.ngram)

    print(f"\nLoading eval dataset: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as fh:
        eval_data = json.load(fh)
    print(f"  {len(eval_data)} queries, query_field='{args.query_field}', top_k={args.top_k}")

    output_records = _load_existing(output_path)
    n_resume = len(output_records)
    if n_resume:
        print(f"Resuming from record {n_resume}/{len(eval_data)} (skipping already-done)")
        eval_data = eval_data[n_resume:]

    print("\nRunning BM25 retrieval...")
    skipped = 0
    total = n_resume + len(eval_data)
    for i, item in enumerate(eval_data, n_resume + 1):
        query = (item.get(args.query_field) or "").strip()
        if not query:
            skipped += 1
        ranked, scores = bm25_top_k(query, bm25, case_ids, args.top_k, args.ngram)
        rec = {
            "source_case_id":     item.get("source_case_id", ""),
            "question":           item.get("question", ""),
            "answer":             item.get("answer", ""),
            "ground_truth_cases": item.get("ground_truth_cases", []),
            # Both fields get the same BM25 ranked list — single-stage baseline.
            "stage1_case_ids":    [],     # single-stage baseline: no stage 1
            "retrieved_cases":    ranked,
            "retrieved_scores":   scores,
            "generated_answer":   "",
            "contexts":           [],
        }
        # Carry through multi-query fields so fusion eval can group by source_case_id.
        if args.multi_query:
            if "sub_query_id" in item:
                rec["sub_query_id"] = item["sub_query_id"]
            if "issue" in item:
                rec["issue"] = item["issue"]
        output_records.append(rec)
        if i % 100 == 0:
            _save_atomic(output_records, output_path)
            print(f"  {i}/{total}")

    if skipped:
        print(f"  [WARN] {skipped} queries had empty '{args.query_field}' field — zero results for those.")

    print(f"\nSaving {len(output_records)} records -> {output_path}")
    _save_atomic(output_records, output_path)

    stem = output_path.stem
    print("\n[OK] Done. To evaluate:")
    print(f"  python evaluation/evaluate.py \\")
    print(f"      --input {output_path} \\")
    print(f"      --retrieval-only \\")
    print(f"      --output evaluation/data/{stem}_results.json")


if __name__ == "__main__":
    main()
