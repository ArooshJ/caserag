#!/usr/bin/env python
"""
full_text_bm25_retrieval.py - BM25 retrieval using the full source case text as query,
replicating the IL-PCSR paper's original protocol (MAP=43.98% with --ngram 5).

Index  : il-pcsr_dataset/corpus/       (3,183 candidate precedent cases)
Queries: il-pcsr_dataset/test_queries/ (627 pre-masked query cases, default)
         il-pcsr_dataset/dev_queries/  (627, for hyperparameter search)
         il-pcsr_dataset/train_queries/(5017, for large-scale ablation)

Citation masking ([PRECEDENT], [SECTION], [ACT]) is already applied in the query
case JSONs by the IL-PCSR authors — no post-processing needed.

Usage (from repo root):
    python evaluation/full_text_bm25_retrieval.py --ngram 5   # paper replication
    python evaluation/full_text_bm25_retrieval.py --ngram 1   # unigram baseline
    python evaluation/full_text_bm25_retrieval.py --ngram 5 --query-split dev --k1 2.0

Then evaluate:
    python evaluation/evaluate.py \\
        --input evaluation/data/full_text_bm25_ngram5_top100_test.json \\
        --retrieval-only \\
        --output evaluation/results/full_text_bm25_ngram5_top100_test_results.json
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
DATASET_ROOT = REPO_ROOT / "il-pcsr_dataset"

PARA_FIELDS = [
    "Fact", "Issue", "Petitioner's Argument", "Respondent's Argument",
    "Analysis of the law", "Precedent Analysis", "Court's Reasoning", "Conclusion",
]

QUERY_SPLITS = {
    "test":  "test_queries",
    "dev":   "dev_queries",
    "train": "train_queries",
}


def tokenize(text: str, max_n: int = 1) -> list:
    words = re.findall(r"[a-z]+", text.lower())
    if max_n == 1:
        return words
    tokens = list(words)
    for n in range(2, max_n + 1):
        tokens.extend("_".join(words[i:i+n]) for i in range(len(words) - n + 1))
    return tokens


def case_text(data: dict, fields: list) -> str:
    parts = []
    for field in fields:
        paras = data.get(field, [])
        if isinstance(paras, list):
            parts.extend(p for p in paras if isinstance(p, str) and p.strip())
        elif isinstance(paras, str) and paras.strip():
            parts.append(paras)
    return " ".join(parts)


def build_index(corpus_dir: Path, fields: list, max_n: int, k1: float, b: float):
    files = sorted(corpus_dir.glob("*.json"))
    n = len(files)
    if n == 0:
        print(f"[FAIL] No JSON files in {corpus_dir}")
        sys.exit(1)
    print(f"Loading corpus: {n} files ...")

    case_ids, tokenized_docs = [], []
    for i, fp in enumerate(files, 1):
        with open(fp, encoding="utf-8") as fh:
            data = json.load(fh)
        case_ids.append(str(data.get("caseID", fp.stem)))
        tokenized_docs.append(tokenize(case_text(data, fields), max_n))
        if i % 500 == 0:
            print(f"  tokenized {i}/{n} ...")

    print(f"Building BM25Okapi (ngram={max_n}, k1={k1}, b={b}) over {n} docs ...")
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    print("[OK] Index ready.")
    return case_ids, bm25


def load_queries(query_dir: Path, min_golds: int) -> list:
    files = sorted(query_dir.glob("*.json"))
    queries, skipped = [], 0
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            data = json.load(fh)
        if len(data.get("relevant_precedent_ids", [])) < min_golds:
            skipped += 1
            continue
        queries.append(data)
    print(f"Query cases: {len(queries)} loaded, {skipped} skipped (< {min_golds} gold precedents)")
    return queries


def _load_jsonl(jsonl_path: Path) -> list:
    records = []
    if not jsonl_path.exists():
        return records
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if records:
        print(f"Resuming from JSONL: {len(records)} records done ({jsonl_path.name})")
    return records


def _save_final(records: list, out_path: Path, jsonl_path: Path):
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
    if jsonl_path.exists():
        jsonl_path.unlink()


def bm25_top_k(query_text: str, bm25: BM25Okapi, case_ids: list, top_k: int, max_n: int):
    tokens = tokenize(query_text, max_n)
    if not tokens:
        return [], {}
    scores = bm25.get_scores(tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    ranked_ids = [case_ids[i] for i in ranked_idx]
    scores_dict = {case_ids[i]: float(scores[i]) for i in ranked_idx}
    return ranked_ids, scores_dict


def main():
    ap = argparse.ArgumentParser(
        description="Full-text BM25 retrieval — replicates IL-PCSR paper protocol"
    )
    ap.add_argument("--corpus-dir", default=str(CORPUS_DIR),
                    help="Path to il-pcsr_dataset/corpus/ (default: auto-detected from repo root)")
    ap.add_argument("--query-split", default="test", choices=["test", "dev", "train"],
                    help="Which split to use as queries (default: test — 627 queries, matches paper)")
    ap.add_argument("--query-dir", default=None,
                    help="Override query directory path directly (overrides --query-split)")
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: data/full_text_bm25_ngram{N}_top{K}_{split}.json)")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Cases to retrieve per query (default: 100)")
    ap.add_argument("--ngram", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help="Max n-gram size (default: 1 = unigrams). Use --ngram 5 to replicate paper MAP 43.98%%.")
    ap.add_argument("--k1", type=float, default=1.5,
                    help="BM25 k1 — term frequency saturation ceiling (default: 1.5)")
    ap.add_argument("--b", type=float, default=0.75,
                    help="BM25 b — document length normalisation (default: 0.75)")
    ap.add_argument("--min-golds", type=int, default=1,
                    help="Skip query cases with fewer than this many gold precedents (default: 1)")
    ap.add_argument("--fields", default=None,
                    help=(
                        "Comma-separated paragraph fields to include in query and index text. "
                        "Default: all 8 PARA_FIELDS. Ablation e.g.: 'Fact,Issue' or "
                        "'Fact,Issue,Court\\'s Reasoning,Conclusion' (exclude Precedent Analysis)."
                    ))
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if args.query_dir:
        query_dir = Path(args.query_dir)
        split_label = query_dir.name
    else:
        split_label = args.query_split
        query_dir = DATASET_ROOT / QUERY_SPLITS[args.query_split]

    fields = [f.strip() for f in args.fields.split(",")] if args.fields else PARA_FIELDS

    if args.output:
        output_path = Path(args.output)
    else:
        _data_dir = REPO_ROOT / "evaluation" / "data"
        tag = f"ngram{args.ngram}_k1{args.k1}_b{args.b}" if (args.k1 != 1.5 or args.b != 0.75) else f"ngram{args.ngram}"
        output_path = _data_dir / f"full_text_bm25_{tag}_top{args.top_k}_{split_label}.json"

    for d, label in [(corpus_dir, "--corpus-dir"), (query_dir, "--query-dir / --query-split")]:
        if not d.exists():
            print(f"[FAIL] {label} not found: {d}")
            sys.exit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nProtocol : full-text BM25 | split={split_label} | ngram={args.ngram} | k1={args.k1} | b={args.b}")
    print(f"Fields   : {', '.join(fields)}")
    print(f"Output   : {output_path}\n")

    case_ids, bm25 = build_index(corpus_dir, fields, args.ngram, args.k1, args.b)
    queries = load_queries(query_dir, args.min_golds)

    jsonl_path = output_path.with_suffix(".jsonl")
    existing = _load_jsonl(jsonl_path)
    done_ids = {r["source_case_id"] for r in existing}
    pending = [q for q in queries if str(q.get("caseID", "")) not in done_ids]
    print(f"Pending  : {len(pending)} / {len(queries)}\n")

    all_records = list(existing)
    total = len(queries)

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for i, q in enumerate(pending, len(existing) + 1):
            case_id = str(q.get("caseID", ""))
            q_text  = case_text(q, fields)

            if not q_text.strip():
                rec = {
                    "source_case_id":    case_id,
                    "question":          f"[full-text-query] {q.get('title', case_id)}",
                    "answer":            "",
                    "ground_truth_cases": [str(x) for x in q.get("relevant_precedent_ids", [])],
                    "stage1_case_ids":   [],
                    "retrieved_cases":   [],
                    "retrieved_scores":  {},
                    "generated_answer":  "",
                    "contexts":          [],
                    "query_protocol":    "full-text-bm25",
                    "skipped":           True,
                }
            else:
                ranked, scores = bm25_top_k(q_text, bm25, case_ids, args.top_k, args.ngram)
                rec = {
                    "source_case_id":    case_id,
                    "question":          f"[full-text-query] {q.get('title', case_id)}",
                    "answer":            "",
                    "ground_truth_cases": [str(x) for x in q.get("relevant_precedent_ids", [])],
                    "stage1_case_ids":   [],
                    "retrieved_cases":   ranked,
                    "retrieved_scores":  scores,
                    "generated_answer":  "",
                    "contexts":          [],
                    "query_protocol":    "full-text-bm25",
                }

            all_records.append(rec)
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            if i % 100 == 0:
                print(f"  {i}/{total}")

    print(f"\nSaving {len(all_records)} records -> {output_path}")
    _save_final(all_records, output_path, jsonl_path)

    stem = output_path.stem
    print("\n[OK] Done. To evaluate:")
    print(f"  python evaluation/evaluate.py \\")
    print(f"      --input {output_path} \\")
    print(f"      --retrieval-only \\")
    print(f"      --output evaluation/results/{stem}_results.json")


if __name__ == "__main__":
    main()
