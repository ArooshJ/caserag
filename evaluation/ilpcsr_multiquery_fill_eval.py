#!/usr/bin/env python3
"""
ilpcsr_multiquery_fill_eval.py

Runs each sub-query in the multi-query evaluation dataset through the
CaseRAG pipeline and populates retrieved_cases / contexts / stage1_case_ids.

Adapts ILPCSR_FILL_EVAL_JSON.ipynb for the multi-query record template
({source_case_id, sub_query_id, issue, question, ground_truth_cases, ...}).

Usage (from repo root):
    python evaluation/ilpcsr_multiquery_fill_eval.py
    python evaluation/ilpcsr_multiquery_fill_eval.py --k1 200 --k2 100
    python evaluation/ilpcsr_multiquery_fill_eval.py --emb-model-id 2 --k1 50 --k2 25
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Put repo root on sys.path so ilpcsr_caserag_pipeline_functions can be found
# regardless of whether this script is run from evaluation/ or repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ilpcsr_caserag_pipeline_functions import query_rag_pipeline  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "evaluation" / "ilpcsr_multiquery_evaluation_dataset.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "data"


def output_path(output_dir: Path, emb_model_id: int, k1: int, k2: int, distance_op: str = "l2") -> Path:
    """Deterministic filename encoding model/k config and distance op."""
    suffix = "_cosine" if distance_op == "cosine" else ""
    return output_dir / f"model={emb_model_id}_k1={k1}_k2={k2}{suffix}_multiquery.json"


def load_or_resume(input_path: Path, out_path: Path) -> list:
    """Prefer a partial output file over the template — keeps crash resume exact.
    Same pattern as ILPCSR_FILL_EVAL_JSON.ipynb."""
    if out_path.exists():
        print(f"Resuming from: {out_path}")
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    print(f"Starting fresh from: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


def validate_schema(dataset: list) -> None:
    """Guard: abort early if this looks like a single-query dataset (missing source_case_id)
    so users don't accidentally overwrite the wrong fill-eval results."""
    if dataset and "source_case_id" not in dataset[0]:
        raise ValueError(
            "Input does not look like a multi-query dataset (no 'source_case_id' key). "
            "For the single-query dataset use ILPCSR_FILL_EVAL_JSON.ipynb."
        )


def is_filled(item: dict) -> bool:
    """Record already processed when retrieved_cases is non-empty — matches notebook convention."""
    return len(item.get("retrieved_cases", [])) > 0


def save(dataset: list, out_path: Path) -> None:
    """Atomic incremental save — write to .tmp then rename so a crash never corrupts the output."""
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill multi-query eval dataset with CaseRAG pipeline results."
    )
    parser.add_argument(
        "--emb-model-id", type=int, default=1,
        help="Row id in ilpcsr_rag_config (default: 1)",
    )
    parser.add_argument(
        "--k1", type=int, default=200,
        help="Stage-1 case retrieval limit (default: 200, matches deep-recall config)",
    )
    parser.add_argument(
        "--k2", type=int, default=100,
        help="Stage-2 passage retrieval limit (default: 100)",
    )
    parser.add_argument(
        "--distance-op", default="l2", choices=["l2", "cosine"],
        help="pgvector distance operator: l2 uses <-> (default), cosine uses <=> (Kanon-correct, uses HNSW index)",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to multi-query evaluation dataset JSON",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for output file (default: evaluation/data/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = output_path(args.output_dir, args.emb_model_id, args.k1, args.k2, args.distance_op)

    print(f"EMB model ID : {args.emb_model_id}")
    print(f"k1={args.k1}  k2={args.k2}  distance_op={args.distance_op}")
    print(f"Output       : {out}")

    dataset = load_or_resume(args.input, out)
    validate_schema(dataset)

    total = len(dataset)
    n_done = sum(1 for r in dataset if is_filled(r))
    print(f"Records: {total} total | {n_done} already filled | {total - n_done} to process\n")

    errors = 0
    for i, item in enumerate(tqdm(dataset, desc="Sub-queries")):
        question = item.get("question", "").strip()
        if not question:
            continue

        if is_filled(item):
            continue

        try:
            result = query_rag_pipeline(
                test_query=question,
                emb_model_id=args.emb_model_id,
                top_k_stage1=args.k1,
                top_k_stage2=args.k2,
                distance_op=args.distance_op,
            )

            if "error" in result:
                print(f"\n[WARN] Pipeline error on sub_query_id={item.get('sub_query_id')}: {result['error']}")
                errors += 1
                continue

            sources = result.get("sources", [])
            item["retrieved_cases"]  = [s["case_id"] for s in sources]
            item["contexts"]         = sources
            item["stage1_case_ids"]  = result.get("stage1_case_ids", [])
            item["generated_answer"] = result.get("generated_answer", "")

        except Exception as exc:
            print(f"\n[ERROR] index={i} sub_query_id={item.get('sub_query_id')}: {exc}")
            item["retrieved_cases"]  = []
            item["contexts"]         = []
            item["stage1_case_ids"]  = []
            item["generated_answer"] = f"ERROR: {exc}"
            errors += 1

        save(dataset, out)

    print(f"\nDone. Output: {out}")
    n_filled_final = sum(1 for r in dataset if is_filled(r))
    print(f"Filled: {n_filled_final}/{total}  |  Errors: {errors}")


if __name__ == "__main__":
    main()
