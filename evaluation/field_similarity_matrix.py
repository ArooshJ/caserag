#!/usr/bin/env python3
"""
field_similarity_matrix.py

Computes the full n_queries × n_cases × 5_fields raw dot-product matrix for
ILPCSR queries vs corpus cases. One bulk DB pull + 5 local matmuls = minutes.

Stores raw dot + norms (operator-agnostic): both cosine and L2 derive on the fly.
  cosine[q,c,f] = dot[q,c,f] / (qnorm[q] * cnorm[c,f])
  L2sq[q,c,f]   = qnorm[q]^2 + cnorm[c,f]^2 - 2*dot[q,c,f]

Artifacts:
  data/field_sim_matrix_emb{id}.npz       (dot, qnorm, cnorm, doc_ids, query_ids, source_case_ids)
  data/field_sim_matrix_emb{id}_gold.json  {sub_query_id: [gold_case_ids]}
  data/field_sim_matrix_emb{id}_meta.json  (provenance)

Usage:
    python evaluation/field_similarity_matrix.py              # emb3 (Kanon-2)
    python evaluation/field_similarity_matrix.py --emb-model-id 1  # emb1 (BGE)
    python evaluation/field_similarity_matrix.py --no-ssl          # local Postgres
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
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
    encode_query, _query_emb_cache, _load_cache, fetch_rag_config,
)

DATA_DIR = REPO_ROOT / "evaluation" / "data"
FIELDS   = ["facts", "issues", "statutes", "precedents", "holding"]

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
        for ck, ek in _COLAB_SECRET_MAP.items():
            v = userdata.get(ck)
            if v:
                os.environ[ek] = v
                loaded.append(ck)
        print(f"Colab secrets loaded: {loaded or 'NONE'}")
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


def _parse_vec(s) -> np.ndarray:
    """Parse pgvector string '[v1,v2,...]' or list/array to float32."""
    if isinstance(s, (list, np.ndarray)):
        return np.asarray(s, dtype=np.float32)
    return np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)


def pull_case_vectors(cur, emb_model_id: int, dim: int) -> tuple:
    """Bulk pull all case field vectors for emb_model_id.
    Returns (doc_ids list, {field: (n_cases, dim) float32 array})."""
    print("Pulling case vectors from DB (one query)...")
    t0 = time.time()
    cur.execute(
        """
        SELECT doc_id, facts_emb, issues_emb, statutes_emb, precedents_emb, holding_emb
        FROM ilpcsr_cases_embeddings
        WHERE emb_model_id = %s
        ORDER BY doc_id
        """,
        (emb_model_id,),
    )
    rows = cur.fetchall()
    print(f"  Fetched {len(rows)} rows in {time.time()-t0:.1f}s")

    doc_ids = []
    C = {f: [] for f in FIELDS}
    skipped = 0
    null_padded = 0
    for row in rows:
        doc_id = row[0]
        vecs   = row[1:]
        parsed = []
        skip = False
        has_null = False
        for v in vecs:
            if v is None:
                parsed.append(np.zeros(dim, dtype=np.float32))
                has_null = True
            else:
                p = _parse_vec(v)
                if len(p) != dim:
                    print(f"  [WARN] Skipping doc_id={doc_id}: dim mismatch "
                          f"(expected {dim}, got {len(p)})")
                    skipped += 1
                    skip = True
                    break
                parsed.append(p)
        if skip:
            continue
        if has_null:
            null_padded += 1
        doc_ids.append(str(doc_id))
        for i, f in enumerate(FIELDS):
            C[f].append(parsed[i])

    if skipped or null_padded:
        print(f"  Skipped {skipped} rows (dim mismatch). {null_padded} rows had NULL fields (zero-padded). n_cases={len(doc_ids)}")

    return doc_ids, {f: np.stack(C[f]).astype(np.float32) for f in FIELDS}


def build_query_matrix(dataset: list, emb_model_name: str) -> tuple:
    """Build Q (n_q, dim) from dataset sub-queries. Returns (Q, query_ids, source_case_ids, gold_map)."""
    cache = _query_emb_cache.get(emb_model_name, {})
    query_ids, source_case_ids, questions = [], [], []
    gold_map = {}
    skipped = 0

    for i, item in enumerate(dataset):
        q  = (item.get("question") or "").strip()
        gt = item.get("ground_truth_cases") or []
        if not q or not gt:
            skipped += 1
            continue
        sub_id = str(item.get("sub_query_id", item.get("question", f"idx_{i}")))
        src_id = str(item.get("source_case_id", ""))
        query_ids.append(sub_id)
        source_case_ids.append(src_id)
        questions.append(q)
        gold_map[sub_id] = [str(g) for g in gt]

    n_miss = sum(1 for q in questions if q not in cache)
    print(f"Queries: {len(questions)} valid, {skipped} skipped (empty question or gold)")
    if n_miss:
        print(f"  [INFO] {n_miss} cache misses — encoding live via API")
    if not questions:
        print("[FAIL] No valid queries.")
        sys.exit(1)

    vecs = []
    for q in questions:
        v = encode_query(q, emb_model_name)
        vecs.append(np.asarray(v, dtype=np.float32))

    Q = np.stack(vecs).astype(np.float32)
    return Q, query_ids, source_case_ids, gold_map


def main():
    ap = argparse.ArgumentParser(
        description="Build full n_q × n_cases × 5 field-similarity matrix via matmul. "
                    "One DB pull + local matmul. Artifact consumed by weight_sensitivity.py."
    )
    ap.add_argument(
        "--input", type=Path,
        default=DATA_DIR / "model=3_k1=200_k2=100_cosine_multiquery (1).json",
        help="Fill-eval multiquery JSON (sub-query list + ground_truth_cases). "
             "Default: emb3 cosine k1=200 file.",
    )
    ap.add_argument(
        "--emb-model-id", type=int, default=3,
        help="emb_model_id in ilpcsr_rag_config (default 3 = Kanon-2; 1 = BGE)",
    )
    ap.add_argument(
        "--no-ssl", action="store_true",
        help="Connect without SSL (local Postgres, not NeonDB)",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=DATA_DIR,
        help="Directory for artifacts (default: evaluation/data/)",
    )
    ap.add_argument("--timeout", default="300s",
                    help="Per-statement DB timeout (default 300s; bulk pull can be slow)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem     = f"field_sim_matrix_emb{args.emb_model_id}"
    out_npz  = args.output_dir / f"{stem}.npz"
    out_gold = args.output_dir / f"{stem}_gold.json"
    out_meta = args.output_dir / f"{stem}_meta.json"

    print(f"Input      : {args.input}")
    print(f"Output     : {out_npz}")
    print(f"emb_model_id : {args.emb_model_id}")

    if not args.input.exists():
        print(f"[FAIL] Input not found: {args.input}")
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        dataset = json.load(f)

    # ── DB connection ──────────────────────────────────────────────────────
    load_credentials()
    missing = [k for k in ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST")
               if not os.environ.get(k)]
    if missing:
        print(f"[FAIL] Missing DB credentials: {', '.join(missing)}")
        print("       Set NEON_DB_* (Colab) or DB_* in .env (local).")
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

    # ── Config ────────────────────────────────────────────────────────────
    config = fetch_rag_config(args.emb_model_id)
    if not config:
        print(f"[FAIL] emb_model_id {args.emb_model_id} not in ilpcsr_rag_config")
        sys.exit(1)
    emb_model_name = config["emb_model_name"]
    dim = int(config.get("dimensions", 1536))
    print(f"emb_model_id={args.emb_model_id} -> '{emb_model_name}'  dim={dim}")

    # ── Query embedding cache ──────────────────────────────────────────────
    if emb_model_name not in _query_emb_cache:
        _query_emb_cache[emb_model_name] = _load_cache(emb_model_name)
    print(f"Query cache: {len(_query_emb_cache[emb_model_name])} entries pre-loaded")

    # ── Pull case vectors (one bulk query) ────────────────────────────────
    doc_ids, C = pull_case_vectors(cur, args.emb_model_id, dim)
    cur.close()
    conn.close()
    n_cases = len(doc_ids)
    print(f"n_cases={n_cases}  (each C_f shape: ({n_cases}, {dim}))")

    # ── Build query matrix ─────────────────────────────────────────────────
    Q, query_ids, source_case_ids, gold_map = build_query_matrix(dataset, emb_model_name)
    n_q = len(Q)
    print(f"Q shape: ({n_q}, {dim})")

    # Assert dims consistent
    for f in FIELDS:
        if C[f].shape != (n_cases, dim):
            print(f"[FAIL] Dim mismatch for field {f}: {C[f].shape}")
            sys.exit(1)

    # ── Compute norms ─────────────────────────────────────────────────────
    qnorm = np.linalg.norm(Q, axis=1).astype(np.float32)                                    # (n_q,)
    cnorm = np.stack([np.linalg.norm(C[f], axis=1) for f in FIELDS], axis=1).astype(np.float32)  # (n_cases, 5)

    # ── Matmul per field → dot (n_q, n_cases, 5) ──────────────────────────
    print(f"\nRunning 5 matmuls: ({n_q}×{dim}) @ ({dim}×{n_cases}) per field...")
    t0 = time.time()
    dot_slices = []
    for i, f in enumerate(FIELDS):
        df = (Q @ C[f].T).astype(np.float32)   # (n_q, n_cases)
        dot_slices.append(df)
        print(f"  [{i+1}/5] {f:<12} done  {time.time()-t0:.1f}s")
    dot = np.stack(dot_slices, axis=2)          # (n_q, n_cases, 5)
    print(f"Matmul complete: {time.time()-t0:.1f}s  "
          f"dot shape={dot.shape}  uncompressed={dot.nbytes/1e6:.0f}MB")

    # ── Save ──────────────────────────────────────────────────────────────
    print("\nSaving artifacts (npz compressed)...")
    t_save = time.time()
    np.savez_compressed(
        out_npz,
        dot=dot,
        qnorm=qnorm,
        cnorm=cnorm,
        doc_ids=np.array(doc_ids),
        query_ids=np.array(query_ids),
        source_case_ids=np.array(source_case_ids),
    )
    print(f"  Saved npz in {time.time()-t_save:.1f}s  ({out_npz.stat().st_size/1e6:.0f}MB on disk)")

    with open(out_gold, "w", encoding="utf-8") as f:
        json.dump(gold_map, f, ensure_ascii=False)

    meta = {
        "emb_model_id":    args.emb_model_id,
        "emb_model_name":  emb_model_name,
        "dim":             dim,
        "field_order":     FIELDS,
        "n_q":             n_q,
        "n_cases":         n_cases,
        "dot_shape":       list(dot.shape),
        "uncompressed_mb": round((dot.nbytes + qnorm.nbytes + cnorm.nbytes) / 1e6, 1),
        "input_file":      str(args.input),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nDone.")
    print(f"  {out_npz}")
    print(f"  {out_gold}")
    print(f"  {out_meta}")
    print(f"\nNext step — run weight ablations on full corpus:")
    print(f"  python evaluation/weight_sensitivity.py --matrix {out_npz}")
    print(f"  python evaluation/weight_sensitivity.py --matrix {out_npz} --weights 0 1 0 0 0")


if __name__ == "__main__":
    main()
