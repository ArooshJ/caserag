#!/usr/bin/env python
"""
dense_chunk_retrieval.py - Naive single-stage dense baseline over ALL chunks.

WHY THIS EXISTS
    Ablation against CaseRAG's two-stage engine. CaseRAG stage-2 is ALREADY
    max-pool over chunks (retrieval_metrics._collapse_to_cases keeps each case at
    its best/highest-similarity chunk), but restricted to stage-1's top-k1 cases
    and capped at LIMIT 300. This baseline does the SAME max-pool over chunks with
    NO stage-1 restriction and NO LIMIT-300 cap. The only variable that changes is
    the stage-1 filter, so:
        naive-dense BEATS two-stage  -> stage-1 is starving recall
        naive-dense LOSES to two-stage -> stage-1 is buying precision
    Either outcome is a publishable finding.

WHAT IT DOES
    For each eval query:
      1. encode the query with the SAME model the engine uses (from ilpcsr_rag_config)
      2. score EVERY case by the max similarity of any of its chunks (max-pool)
      3. rank cases, take top-k, write doc_ids into `retrieved_cases`
    Output JSON is evaluate.py-compatible (same shape as ilpcsr_evaluation_dataset.json).
    `retrieved_cases` is filled (lands in the stage2 metric slot); `stage1_case_ids`
    is left empty (single-stage baseline has no stage 1) - evaluate.py handles empty
    stage1 cleanly.

COLAB
    Set NEON_DB_* secrets in the sidebar (key icon) - same names the ingest script uses.
    Then from a notebook cell:
      !pip install sentence-transformers psycopg2-binary python-dotenv
      !python evaluation/dense_chunk_retrieval.py --output evaluation/data/dense_top100.json
      !python evaluation/evaluate.py --input evaluation/data/dense_top100.json --retrieval-only --output evaluation/data/dense_results.json

PERFORMANCE NOTE
    The max-pool query GROUP BYs over all chunks, so the HNSW index can't accelerate
    it (it's a full scan per query). The script times the FIRST query and prints it.
    If that's pathologically slow (or hits a statement timeout), tell the author -
    the fast fallback is an ANN top-N-chunks pull collapsed in Python (not built yet,
    because it trades exactness for speed and needs hnsw.ef_search tuning).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import psycopg2

try:
    from dotenv import load_dotenv
except ImportError:
    print("[FAIL] python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from ilpcsr_caserag_pipeline_functions import encode_query  # noqa: E402

EVAL_DATASET = REPO_ROOT / "evaluation" / "ilpcsr_evaluation_dataset.json"
MULTIQUERY_DATASET = REPO_ROOT / "evaluation" / "ilpcsr_multiquery_evaluation_dataset.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "data" / "dense_top100.json"
DEFAULT_OUTPUT_MULTI = REPO_ROOT / "evaluation" / "data" / "dense_multiquery_top100.json"

# Colab secret names (NEON_DB_*) -> env var names the script reads (DB_*).
# Mirrors ingestion/ingest_ilpcsr_to_db.py so existing Colab secrets just work.
_COLAB_SECRET_MAP = {
    "NEON_DB_NAME":     "DB_NAME",
    "NEON_DB_USER":     "DB_USER",
    "NEON_DB_PASSWORD": "DB_PASSWORD",
    "NEON_DB_HOST":     "DB_HOST",
    "NEON_DB_PORT":     "DB_PORT",
}


def load_credentials():
    """In Colab: read NEON_DB_* secrets -> DB_* env vars. Else: load .env."""
    try:
        from google.colab import userdata
        loaded = []
        for colab_key, env_key in _COLAB_SECRET_MAP.items():
            val = userdata.get(colab_key)
            if val:
                os.environ[env_key] = val
                loaded.append(colab_key)
        print(f"Colab detected. Secrets loaded: {loaded or 'NONE - set NEON_DB_* in sidebar (key icon)'}")
    except ImportError:
        load_dotenv()
    except Exception as e:
        print(f"Colab userdata bridge unavailable: {e}; falling back to .env / env vars")
        load_dotenv()


def get_db_conn(sslmode: str):
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT", "5432"),
        sslmode=sslmode,
    )


def fetch_emb_model_name(cur, emb_model_id: int) -> str:
    """Read which embedding model this config id used, exactly as ingest_ilpcsr_to_db.py
    does. Fetching from rag_config (not hardcoding) keeps the query encoder in sync with
    whatever model actually embedded the chunks for this emb_model_id."""
    cur.execute(
        "SELECT emb_model_name FROM ilpcsr_rag_config WHERE emb_model_id = %s;",
        (emb_model_id,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"emb_model_id {emb_model_id} not found in ilpcsr_rag_config")
    return row[0]


DISTANCE_OPS = {"l2": "<->", "cosine": "<=>"}  # pgvector operator tokens


def _build_maxpool_sql(op: str) -> str:
    """Build the max-pool SQL with the chosen pgvector distance operator."""
    return f"""
    SELECT p.doc_id,
           MAX(1 - (ce.embedding {op} %(emb)s::vector)) AS sim
    FROM ilpcsr_content_chunks c
    JOIN ilpcsr_chunk_embeddings ce ON c.chunk_id = ce.chunk_id
    JOIN ilpcsr_paragraphs p ON c.paragraph_id = p.paragraph_id
    WHERE ce.emb_model_id = %(mid)s
    GROUP BY p.doc_id
    ORDER BY sim DESC
    LIMIT %(k)s;
"""


def dense_top_k(cur, query_vec_str: str, emb_model_id: int, top_k: int, sql: str):
    """Return (ranked_ids, scores_dict) for top-k cases by max-pool chunk similarity."""
    cur.execute(sql, {"emb": query_vec_str, "mid": emb_model_id, "k": top_k})
    rows = cur.fetchall()
    ranked_ids = [str(r[0]) for r in rows]
    scores = {str(r[0]): float(r[1]) for r in rows}
    return ranked_ids, scores


def _load_jsonl(jsonl_path: Path) -> list:
    """Load a partial JSONL working file for resume. Skips any corrupt last line (crash mid-write)."""
    records = []
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # truncated last line from crash — skip and re-process
    return records


def _finalize(records: list, out_path: Path, jsonl_path: Path):
    """Atomic JSON write from accumulated records, then clean up the JSONL working file."""
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
    try:
        jsonl_path.unlink()
    except Exception:
        print(f"  [WARN] could not delete JSONL working file: {jsonl_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Naive single-stage dense baseline over all chunks (max-pool)"
    )
    ap.add_argument("--emb-model-id", type=int, default=1,
                    help="config id in ilpcsr_rag_config (default 1, matches results_model=1)")
    ap.add_argument("--top-k-cases", type=int, default=100,
                    help="cases to return per query (default 100)")
    ap.add_argument("--query-field", default="question",
                    help="eval-dataset field used as query text (default 'question')")
    ap.add_argument("--eval-dataset", default=None,
                    help="eval dataset path (auto: single-query or multi-query based on --multi-query flag)")
    ap.add_argument("--output", default=None,
                    help="output path (default: dense_top100.json or dense_multiquery_top100.json)")
    ap.add_argument("--multi-query", action="store_true",
                    help="input is the multi-query dataset — carry through sub_query_id/issue")
    ap.add_argument("--distance-op", default="l2", choices=["l2", "cosine"],
                    help="pgvector distance operator: l2 uses <-> (default), cosine uses <=> (Kanon-correct, uses HNSW index)")
    ap.add_argument("--no-ssl", action="store_true", help="connect without SSL (local Postgres)")
    ap.add_argument("--timeout", default="120s", help="per-statement timeout (default 120s)")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="only run the first N queries (0 = all; for quick timing tests)")
    args = ap.parse_args()

    if args.eval_dataset is None:
        args.eval_dataset = str(MULTIQUERY_DATASET if args.multi_query else EVAL_DATASET)
    eval_path = Path(args.eval_dataset)
    _data_dir = REPO_ROOT / "evaluation" / "data"
    _op_suffix = "_cosine" if args.distance_op == "cosine" else ""
    output_path = Path(args.output) if args.output else (
        _data_dir / f"dense_emb{args.emb_model_id}{_op_suffix}_multiquery_top{args.top_k_cases}.json"
        if args.multi_query else
        _data_dir / f"dense_emb{args.emb_model_id}{_op_suffix}_top{args.top_k_cases}.json"
    )
    if not eval_path.exists():
        print(f"[FAIL] eval dataset not found: {eval_path}")
        sys.exit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    load_credentials()
    missing = [k for k in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST") if not os.environ.get(k)]
    if missing:
        print(f"[FAIL] missing DB credentials: {', '.join(missing)} (set NEON_DB_* secrets or .env)")
        sys.exit(1)

    sslmode = "disable" if args.no_ssl else os.environ.get("DB_SSLMODE", "require")
    print(f"Connecting: db={os.environ.get('DB_NAME')} sslmode={sslmode} (host/creds hidden)")
    conn = get_db_conn(sslmode)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute(f"SET statement_timeout = '{args.timeout}';")
    except Exception:
        pass

    emb_model_name = fetch_emb_model_name(cur, args.emb_model_id)
    maxpool_sql = _build_maxpool_sql(DISTANCE_OPS[args.distance_op])
    print(f"emb_model_id={args.emb_model_id} -> '{emb_model_name}'  top_k_cases={args.top_k_cases}  distance_op={args.distance_op}")

    with open(eval_path, "r", encoding="utf-8") as fh:
        eval_data = json.load(fh)
    if args.limit_queries > 0:
        eval_data = eval_data[:args.limit_queries]
    print(f"{len(eval_data)} queries, query_field='{args.query_field}'")

    jsonl_path = output_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        output_records = _load_jsonl(jsonl_path)
        n_resume = len(output_records)
        if n_resume:
            print(f"Resuming from JSONL: {n_resume}/{len(eval_data)} already done")
    elif output_path.exists():
        # Completed run — load JSON, loop will skip all records
        try:
            with open(output_path, encoding="utf-8") as f:
                output_records = json.load(f)
        except Exception:
            output_records = []
        n_resume = len(output_records)
        if n_resume:
            print(f"Output already complete ({n_resume} records). Re-run will be a no-op.")
    else:
        output_records = []
        n_resume = 0

    if n_resume:
        eval_data = eval_data[n_resume:]

    skipped = 0
    total = n_resume + len(eval_data)
    t_start = time.time()
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for i, item in enumerate(eval_data, n_resume + 1):
            query = (item.get(args.query_field) or "").strip()
            if not query:
                skipped += 1
                ranked, scores = [], {}
            else:
                vec_str = str(encode_query(query, emb_model_name))
                t0 = time.time()
                ranked, scores = dense_top_k(cur, vec_str, args.emb_model_id, args.top_k_cases, maxpool_sql)
                if i == n_resume + 1:
                    dt = time.time() - t0
                    est = dt * len(eval_data) / 60.0
                    print(f"  [timing] first query DB scan = {dt:.2f}s  "
                          f"-> est ~{est:.1f} min for {len(eval_data)} remaining queries")
                    print(f"  [timing] first query returned {len(ranked)} distinct cases "
                          f"(need >= {args.top_k_cases} for full Recall@{args.top_k_cases})")
            rec = {
                "source_case_id":     item.get("source_case_id", ""),
                "question":           item.get("question", ""),
                "answer":             item.get("answer", ""),
                "ground_truth_cases": item.get("ground_truth_cases", []),
                "stage1_case_ids":    [],
                "retrieved_cases":    ranked,
                "retrieved_scores":   scores,
                "generated_answer":   "",
                "contexts":           [],
            }
            if args.multi_query:
                if "sub_query_id" in item:
                    rec["sub_query_id"] = item["sub_query_id"]
                if "issue" in item:
                    rec["issue"] = item["issue"]
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()
            output_records.append(rec)
            if i % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  {i}/{total}  ({elapsed:.0f}s elapsed)")

    if skipped:
        print(f"  [WARN] {skipped} queries had empty '{args.query_field}' -> zero results.")

    print(f"\nFinalizing {len(output_records)} records -> {output_path}")
    _finalize(output_records, output_path, jsonl_path)

    cur.close()
    conn.close()
    print("\n[OK] Done. To evaluate:")
    print(f"  python evaluation/evaluate.py \\")
    print(f"      --input {output_path} \\")
    print(f"      --retrieval-only \\")
    print(f"      --output evaluation/data/dense_results.json")


if __name__ == "__main__":
    main()
