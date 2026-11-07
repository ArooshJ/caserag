#!/usr/bin/env python3
"""
pool_field_dump.py

For each sub-query in a multiquery fill-eval JSON (which already has stage1_case_ids),
fetches the 5 per-field similarity scores from the DB for those exact case IDs.

No full corpus scan — one small WHERE doc_id IN (...) query per sub-query.
Output is the artifact consumed by weight_sensitivity.py.

Usage:
    python evaluation/pool_field_dump.py
    python evaluation/pool_field_dump.py --input evaluation/data/my_fill_eval.json
    python evaluation/pool_field_dump.py --emb-model-id 3 --distance-op cosine --no-ssl
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ilpcsr_caserag_pipeline_functions import (  # noqa: E402
    encode_query, STATIC_WEIGHTS, DISTANCE_OPS,
    _query_emb_cache, _load_cache, fetch_rag_config,
)

DATA_DIR = REPO_ROOT / "evaluation" / "data"

_COLAB_SECRET_MAP = {
    "NEON_DB_NAME":     "DB_NAME",
    "NEON_DB_USER":     "DB_USER",
    "NEON_DB_PASSWORD": "DB_PASSWORD",
    "NEON_DB_HOST":     "DB_HOST",
    "NEON_DB_PORT":     "DB_PORT",
}


def load_credentials():
    try:
        from google.colab import userdata
        loaded = []
        for colab_key, env_key in _COLAB_SECRET_MAP.items():
            val = userdata.get(colab_key)
            if val:
                os.environ[env_key] = val
                loaded.append(colab_key)
        print(f"Colab detected. Secrets loaded: {loaded or 'NONE'}")
    except ImportError:
        load_dotenv()
    except Exception as e:
        print(f"Colab userdata unavailable ({e}); falling back to .env")
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


def build_pool_sql(op: str) -> str:
    """Per-field similarity scores for a fixed set of doc_ids (the stage-1 pool).
    Sorted by default weighted score so the artifact is human-readable; Script 2
    re-sorts under its own weights and ignores this order."""
    return f"""
    WITH similarity_scores AS (
        SELECT
            c.doc_id,
            COALESCE(1 - (ce.facts_emb      {op} %(emb)s::vector), 0) AS score_facts,
            COALESCE(1 - (ce.issues_emb     {op} %(emb)s::vector), 0) AS score_issues,
            COALESCE(1 - (ce.statutes_emb   {op} %(emb)s::vector), 0) AS score_statutes,
            COALESCE(1 - (ce.precedents_emb {op} %(emb)s::vector), 0) AS score_precedents,
            COALESCE(1 - (ce.holding_emb    {op} %(emb)s::vector), 0) AS score_holding
        FROM ilpcsr_cases c
        JOIN ilpcsr_cases_embeddings ce ON c.doc_id = ce.doc_id
        WHERE ce.emb_model_id = %(mid)s
          AND c.doc_id IN %(pool_ids)s
    )
    SELECT
        doc_id,
        score_facts, score_issues, score_statutes, score_precedents, score_holding,
        (score_facts      * %(wf)s
         + score_issues   * %(wi)s
         + score_statutes * %(ws)s
         + score_precedents * %(wp)s
         + score_holding  * %(wh)s) AS default_score
    FROM similarity_scores
    ORDER BY default_score DESC;
    """


def fetch_pool(cur, query_vec_str: str, pool_ids: list, emb_model_id: int, sql: str) -> list:
    """Run pool SQL; return list of per-case field-score dicts in default-weight order."""
    params = {
        "emb":  query_vec_str,
        "mid":  emb_model_id,
        "pool_ids": tuple(int(x) for x in pool_ids),
        "wf":   STATIC_WEIGHTS["facts"],
        "wi":   STATIC_WEIGHTS["issues"],
        "ws":   STATIC_WEIGHTS["statutes"],
        "wp":   STATIC_WEIGHTS["precedents"],
        "wh":   STATIC_WEIGHTS["holding"],
    }
    cur.execute(sql, params)
    pool = []
    for doc_id, sf, si, ss, sp, sh, _default in cur.fetchall():
        pool.append({
            "case_id":    str(doc_id),
            "facts":      float(sf),
            "issues":     float(si),
            "statutes":   float(ss),
            "precedents": float(sp),
            "holding":    float(sh),
        })
    return pool


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
        print(f"Resuming from JSONL checkpoint: {len(records)} records done ({jsonl_path.name})")
    return records


def _save_final(payload: dict, out_path: Path, jsonl_path: Path):
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
    try:
        if jsonl_path.exists():
            jsonl_path.unlink()
    except Exception as e:
        print(f"  [WARN] could not delete JSONL checkpoint: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Dump per-field similarity scores for the stage-1 pool per sub-query. "
                    "Reads stage1_case_ids from fill-eval output; no full corpus scan."
    )
    ap.add_argument(
        "--input", type=Path,
        default=DATA_DIR / "model=3_k1=200_k2=100_cosine_multiquery (1).json",
        help="Multiquery fill-eval JSON with stage1_case_ids. "
             "Default: emb3 cosine k1=200 multiquery file.",
    )
    ap.add_argument(
        "--emb-model-id", type=int, default=3,
        help="Embedding model id in ilpcsr_rag_config (default 3 = Kanon-2)",
    )
    ap.add_argument(
        "--distance-op", default="cosine", choices=["l2", "cosine"],
        help="pgvector distance operator used when the corpus was embedded (default cosine)",
    )
    ap.add_argument("--no-ssl", action="store_true",
                    help="Connect without SSL (local Postgres, not NeonDB)")
    ap.add_argument(
        "--output-dir", type=Path, default=DATA_DIR,
        help="Directory for output file (default: evaluation/data/)",
    )
    ap.add_argument("--timeout", default="120s",
                    help="Per-statement DB timeout (default 120s)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"pool_field_dump_emb{args.emb_model_id}_{args.distance_op}.json"
    out_path  = args.output_dir / out_name
    jsonl_path = out_path.with_suffix(".jsonl")

    print(f"Input        : {args.input}")
    print(f"Output       : {out_path}")
    print(f"emb_model_id : {args.emb_model_id}   distance_op : {args.distance_op}")

    if not args.input.exists():
        print(f"[FAIL] Input file not found: {args.input}")
        print("       Pass --input <path> to specify the fill-eval multiquery JSON.")
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        dataset = json.load(f)

    sample = next((r for r in dataset), None)
    if sample and "stage1_case_ids" not in sample:
        print("[FAIL] Input records lack 'stage1_case_ids'.")
        print("       Run ilpcsr_multiquery_fill_eval.py first, then pass its output here.")
        sys.exit(1)

    load_credentials()
    missing_creds = [k for k in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST")
                     if not os.environ.get(k)]
    if missing_creds:
        print(f"[FAIL] Missing DB credentials: {', '.join(missing_creds)}")
        print("       Set NEON_DB_* secrets (Colab) or DB_* vars in .env (local).")
        sys.exit(1)

    sslmode = "disable" if args.no_ssl else os.environ.get("DB_SSLMODE", "require")
    print(f"Connecting: db={os.environ.get('DB_NAME')}  sslmode={sslmode}")
    conn = get_db_conn(sslmode)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute(f"SET statement_timeout = '{args.timeout}';")
    except Exception:
        pass

    config = fetch_rag_config(args.emb_model_id)
    if not config:
        print(f"[FAIL] emb_model_id {args.emb_model_id} not found in ilpcsr_rag_config")
        sys.exit(1)
    emb_model_name = config["emb_model_name"]
    print(f"emb_model_id={args.emb_model_id} -> '{emb_model_name}'")

    if emb_model_name not in _query_emb_cache:
        _query_emb_cache[emb_model_name] = _load_cache(emb_model_name)
    n_cached = len(_query_emb_cache[emb_model_name])
    print(f"Query embedding cache: {n_cached} entries pre-loaded for '{emb_model_name}'")

    sql = build_pool_sql(DISTANCE_OPS[args.distance_op])

    records = _load_jsonl(jsonl_path)
    done_keys = {(r["source_case_id"], r["sub_query_id"]) for r in records}

    skipped_empty = 0
    n_cache_miss  = 0
    pool_sizes    = []
    t_start = time.time()
    first_query = True

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for i, item in enumerate(dataset, 1):
            src_id = str(item.get("source_case_id", ""))
            sub_id = str(item.get("sub_query_id", item.get("question", f"idx_{i}")))
            key    = (src_id, sub_id)

            if key in done_keys:
                continue

            pool_ids = item.get("stage1_case_ids") or []
            question = (item.get("question") or "").strip()

            if not pool_ids or not question:
                skipped_empty += 1
                continue

            cache = _query_emb_cache.get(emb_model_name, {})
            if question not in cache:
                n_cache_miss += 1

            vec_str = str(encode_query(question, emb_model_name))

            t0 = time.time()
            pool = fetch_pool(cur, vec_str, pool_ids, args.emb_model_id, sql)
            if first_query:
                dt = time.time() - t0
                remaining = sum(1 for r in dataset
                                if (str(r.get("source_case_id", "")),
                                    str(r.get("sub_query_id", r.get("question", "")))) not in done_keys)
                print(f"  [timing] first query = {dt:.3f}s  "
                      f"-> est ~{dt * remaining / 60:.1f} min for {remaining} remaining")
                first_query = False

            pool_sizes.append(len(pool))

            rec = {
                "source_case_id":     src_id,
                "sub_query_id":       sub_id,
                "question":           question,
                "ground_truth_cases": [str(g) for g in (item.get("ground_truth_cases") or [])],
                "pool":               pool,
            }
            records.append(rec)
            done_keys.add(key)
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()

            if i % 100 == 0:
                elapsed = time.time() - t_start
                ps = sorted(pool_sizes)
                print(f"  {i}/{len(dataset)}  ({elapsed:.0f}s)  "
                      f"pool: min={ps[0]} median={ps[len(ps)//2]} max={ps[-1]}")

    elapsed = time.time() - t_start
    print(f"\nDone. {len(records)} records in {elapsed:.0f}s")
    if skipped_empty:
        print(f"  Skipped (empty stage1_case_ids or question): {skipped_empty}")
    if n_cache_miss:
        print(f"  Cache misses (encoded live via API/model): {n_cache_miss}")

    pool_size_stats = {}
    if pool_sizes:
        ps = sorted(pool_sizes)
        pool_size_stats = {"min": ps[0], "median": ps[len(ps)//2], "max": ps[-1]}
        print(f"  Pool sizes: {pool_size_stats}")

    meta = {
        "emb_model_id":    args.emb_model_id,
        "emb_model_name":  emb_model_name,
        "distance_op":     args.distance_op,
        "input_file":      str(args.input),
        "n_records":       len(records),
        "n_skipped_empty": skipped_empty,
        "pool_size_stats": pool_size_stats,
        "static_weights":  STATIC_WEIGHTS,
    }
    _save_final({"_meta": meta, "records": records}, out_path, jsonl_path)
    print(f"Saved: {out_path}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
