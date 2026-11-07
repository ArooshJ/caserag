#!/usr/bin/env python
"""
gold_oracle_similarity.py - "Oracle retrieval" diagnostic: how similar is each
query to its OWN ground-truth precedents, in embedding space, with retrieval
removed from the equation entirely.

WHY THIS EXISTS
    error_analysis.py shows the two-stage engine MISSES most gold precedents
    (stage-1 recall caps the whole pipeline). But "misses" could mean two very
    different things:
        (a) the pipeline is at fault   - golds are close to the query in embedding
            space, the ranker just buries them under distractors; or
        (b) the embeddings are at fault - golds simply are NOT close to the query,
            so NO similarity-ranked retriever could ever surface them.
    This script isolates (b). It does NOT search the corpus. For each query it
    takes ONLY the gold cases and scores them exactly the way the engine would,
    so the number we get is the *intrinsic* query<->gold similarity, free of any
    competition from non-gold cases. If those numbers are low, the bottleneck is
    similarity != relevance (the motivation for relevance-at-scale methods), not
    a tunable pipeline knob.

WHAT IT DOES (per eval query, gold cases only - NO corpus-wide scan)
    Stage-1 oracle : score each gold case with the engine's exact 5-field weighted
                     similarity (facts/issues/statutes/precedents/holding, same
                     STATIC_WEIGHTS). Restricted to the gold doc_ids -> a handful
                     of indexed rows, no full scan.
    Stage-2 oracle : restrict the chunk search to `p.doc_id IN (gold_ids)` so every
                     returned chunk is guaranteed to belong to a gold case; record
                     per-gold best-chunk and mean-chunk similarity. This answers
                     "ASSUMING stage 1 handed stage 2 the right cases, how well do
                     the chunks inside them match the query?"
    Missing golds  : a gold precedent absent from the corpus / not ingested has no
                     embedding -> it can never be retrieved. Those are COUNTED and
                     reported, never silently dropped, because they are themselves
                     a hard recall ceiling.

WHAT IT DELIBERATELY DOES NOT DO
    No rank/percentile of golds vs the whole corpus. That contrast (where a gold
    lands among all cases) is derived LATER and for free from the existing
    data/model=1_k1=200_k2=100.json rankings - no need to re-run a full similarity
    search here.

MULTI-QUERY MODE  (--multi-query)
    Point the script at the multi-query dataset (one record PER SUB-QUERY, many sub-
    queries sharing one source case + its gold). Each sub-query is encoded and scored on
    its own, so the similarity distributions analyse EVERY query, not one per case. Case-
    level gold inventory counts (total / missing / no-chunks) are deduped by source case
    so they are NOT multiplied by the sub-query count. An extra "best per (case, gold)
    across the case's sub-queries" lens tests the multi-query premise: do different issue-
    queries between them pull a gold closer than any single query would? On the single-
    query dataset both lenses coincide (one sub-query per case), so output is unchanged.

RUNNING IT (same environment as dense_chunk_retrieval.py)
    Colab : set NEON_DB_* secrets in the sidebar (key icon), then
        !python evaluation/gold_oracle_similarity.py                  # single-query
        !python evaluation/gold_oracle_similarity.py --multi-query    # per sub-query
    Local : DB_* in .env, add --no-ssl for local Postgres
        python evaluation/gold_oracle_similarity.py --no-ssl
    Quick timing test: --limit-queries 5
"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path

import numpy as np
import psycopg2

try:
    from dotenv import load_dotenv
except ImportError:
    print("[FAIL] python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from ilpcsr_caserag_pipeline_functions import encode_query, ISAACUS_MODELS, _query_emb_cache, _load_cache  # noqa: E402

EVAL_DATASET = REPO_ROOT / "evaluation" / "ilpcsr_evaluation_dataset.json"
# Multi-query input/output (one record per sub-query); kept on separate paths so a
# --multi-query run never overwrites the single-query oracle results, and vice versa.
MULTIQUERY_DATASET = REPO_ROOT / "evaluation" / "ilpcsr_multiquery_evaluation_dataset.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "data" / "gold_oracle_similarity.json"
DEFAULT_OUTPUT_MULTI = REPO_ROOT / "evaluation" / "data" / "gold_oracle_similarity_multiquery.json"
# Corpus JSONs — summary field texts live here, NOT in the DB (ingest only stores embeddings).
ILPCSR_CORPUS = REPO_ROOT / "il-pcsr_dataset" / "corpus"

# The engine's case-level field weights, copied verbatim from
# ilpcsr_caserag_pipeline_functions.STATIC_WEIGHTS so the oracle stage-1 score is
# identical to what retrieve_top_cases would assign these same gold cases.
STATIC_WEIGHTS = {
    "facts": 0.1, "issues": 0.3, "statutes": 0.2,
    "precedents": 0.1, "holding": 0.3,
}

# Colab secret names (NEON_DB_*) -> env var names this script reads (DB_*).
# Mirrors dense_chunk_retrieval.py / ingest_ilpcsr_to_db.py so existing secrets work.
_COLAB_SECRET_MAP = {
    "NEON_DB_NAME":     "DB_NAME",
    "NEON_DB_USER":     "DB_USER",
    "NEON_DB_PASSWORD": "DB_PASSWORD",
    "NEON_DB_HOST":     "DB_HOST",
    "NEON_DB_PORT":     "DB_PORT",
}


def load_credentials():
    """In Colab: read NEON_DB_* secrets -> DB_* env vars. Else: load .env.
    Why: lets the same script run unchanged on Colab (where the ilpcsr_* tables
    live for most runs) and locally, without editing connection code."""
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
    """Open the pgvector connection holding the ilpcsr_* tables. Separated so the
    SSL mode (require for NeonDB, disable for local) is decided by one CLI flag."""
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT", "5432"),
        sslmode=sslmode,
    )


def fetch_emb_model_name(cur, emb_model_id: int) -> str:
    """Read the embedding model name for this config id from ilpcsr_rag_config.
    Why: the query encoder MUST match the model that embedded the gold cases for
    this emb_model_id, or the cosine numbers are meaningless. Same lookup the
    engine and the dense baseline use, so the encoder stays in sync."""
    cur.execute(
        "SELECT emb_model_name FROM ilpcsr_rag_config WHERE emb_model_id = %s;",
        (emb_model_id,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"emb_model_id {emb_model_id} not found in ilpcsr_rag_config")
    return row[0]


DISTANCE_OPS = {"l2": "<->", "cosine": "<=>"}  # pgvector operator tokens


def _build_stage1_sql(op: str) -> str:
    """Stage-1 oracle SQL with the chosen pgvector distance operator.
    Same 5-field weighted scoring as retrieve_top_cases, restricted to gold doc_ids."""
    return f"""
    WITH similarity_scores AS (
        SELECT
            c.doc_id,
            c.title,
            COALESCE(1 - (ce.facts_emb      {op} %(emb)s::vector), 0) AS score_facts,
            COALESCE(1 - (ce.issues_emb     {op} %(emb)s::vector), 0) AS score_issues,
            COALESCE(1 - (ce.statutes_emb   {op} %(emb)s::vector), 0) AS score_statutes,
            COALESCE(1 - (ce.precedents_emb {op} %(emb)s::vector), 0) AS score_precedents,
            COALESCE(1 - (ce.holding_emb    {op} %(emb)s::vector), 0) AS score_holding
        FROM ilpcsr_cases c
        JOIN ilpcsr_cases_embeddings ce ON c.doc_id = ce.doc_id
        WHERE ce.emb_model_id = %(mid)s
          AND c.doc_id IN %(gold)s
    )
    SELECT
        doc_id, title,
        score_facts, score_issues, score_statutes, score_precedents, score_holding,
        ( score_facts      * %(w_facts)s
        + score_issues     * %(w_issues)s
        + score_statutes   * %(w_statutes)s
        + score_precedents * %(w_precedents)s
        + score_holding    * %(w_holding)s ) AS final_score
    FROM similarity_scores;
"""


def _build_stage2_sql(op: str) -> str:
    """Stage-2 oracle SQL with the chosen pgvector distance operator.
    Same chunk scoring as retrieve_detailed_context, restricted to gold doc_ids, no LIMIT."""
    return f"""
    SELECT
        p.doc_id,
        1 - (ce.embedding {op} %(emb)s::vector) AS sim
    FROM ilpcsr_content_chunks c
    JOIN ilpcsr_chunk_embeddings ce ON c.chunk_id = ce.chunk_id
    JOIN ilpcsr_paragraphs p ON c.paragraph_id = p.paragraph_id
    WHERE p.doc_id IN %(gold)s
      AND ce.emb_model_id = %(mid)s
    ORDER BY sim DESC;
"""

# Fetch raw chunk texts for a set of doc_ids — used by --cross-emb to re-embed with a
# second model locally. chunk_text is model-agnostic (no emb_model_id filter needed).
CHUNK_TEXTS_SQL = """
    SELECT p.doc_id, c.chunk_text
    FROM ilpcsr_content_chunks c
    JOIN ilpcsr_paragraphs p ON c.paragraph_id = p.paragraph_id
    WHERE p.doc_id IN %(gold)s
    ORDER BY c.chunk_id;
"""


def stage1_gold_scores(cur, query_vec_str, gold_ids, emb_model_id, sql):
    """Score the gold cases at case level (oracle stage 1). Returns a list of
    per-gold dicts. Golds with no row (no embedding for this model) are simply
    absent from the result -> the caller diffs against the requested set to count
    the missing ones."""
    params = {
        "emb": query_vec_str, "mid": emb_model_id, "gold": tuple(gold_ids),
        "w_facts": STATIC_WEIGHTS["facts"], "w_issues": STATIC_WEIGHTS["issues"],
        "w_statutes": STATIC_WEIGHTS["statutes"], "w_precedents": STATIC_WEIGHTS["precedents"],
        "w_holding": STATIC_WEIGHTS["holding"],
    }
    cur.execute(sql, params)
    out = []
    for doc_id, title, sf, si, sst, sp, sh, final in cur.fetchall():
        out.append({
            "doc_id": str(doc_id),
            "title": title or "",
            "final_score": float(final),
            "fields": {
                "facts": float(sf), "issues": float(si), "statutes": float(sst),
                "precedents": float(sp), "holding": float(sh),
            },
        })
    return out


def stage2_gold_chunk_scores(cur, query_vec_str, gold_ids, emb_model_id, sql):
    """Collapse gold-case chunk similarities to per-gold best/mean (oracle stage 2).
    'best' (max-pool) mirrors how the engine's _collapse_to_cases scores a case by
    its single nearest chunk; 'mean' shows whether the match is one lucky paragraph
    or broadly present in the case."""
    cur.execute(sql, {"emb": query_vec_str, "gold": tuple(gold_ids), "mid": emb_model_id})
    per_gold = {}
    for doc_id, sim in cur.fetchall():
        per_gold.setdefault(str(doc_id), []).append(float(sim))
    out = []
    for did, sims in per_gold.items():
        out.append({
            "doc_id": did,
            "best_chunk_sim": max(sims),
            "mean_chunk_sim": sum(sims) / len(sims),
            "n_chunks": len(sims),
        })
    out.sort(key=lambda r: r["best_chunk_sim"], reverse=True)
    return out


def _dist(vals):
    """Descriptive stats dict for a list of floats. Module-level so both summarize() and
    run_cross_emb_analysis() produce identical-shaped distribution objects."""
    if not vals:
        return None
    sv = sorted(vals)
    return {
        "n": len(sv),
        "mean": round(statistics.fmean(sv), 4),
        "median": round(statistics.median(sv), 4),
        "p10": round(_pct(sv, 10), 4),
        "p25": round(_pct(sv, 25), 4),
        "p75": round(_pct(sv, 75), 4),
        "p90": round(_pct(sv, 90), 4),
        "min": round(sv[0], 4),
        "max": round(sv[-1], 4),
    }


def _cosine(a, b) -> float:
    """Cosine similarity on numpy vectors. WHY NOT 1-L2: L2 distance is not comparable across
    models with different native vector scales; cosine on unit vectors is model-agnostic."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _pct(sorted_vals, p):
    """Nearest-rank percentile on an already-sorted list (small helper so the
    summary needs no numpy)."""
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _load_jsonl(jsonl_path: Path) -> list:
    """Load in-progress records from JSONL checkpoint. Skips corrupt/truncated last line."""
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
        print(f"Resuming from JSONL: {len(records)} records already done ({jsonl_path.name})")
    return records


def _save_final(payload: dict, out_path: Path, jsonl_path: Path):
    """Atomically write final output then delete JSONL checkpoint."""
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
    try:
        if jsonl_path.exists():
            jsonl_path.unlink()
    except Exception as e:
        print(f"  [WARN] could not delete JSONL checkpoint: {e}")


def _case_key(rec):
    """Identity of the SOURCE CASE a record belongs to, for grouping sub-queries of the same
    case. Falls back to per-record uniqueness (sub_query_id / question / object id) when there
    is no source_case_id, so a missing id never collapses unrelated records OR fakes pooling."""
    return rec.get("source_case_id") or rec.get("sub_query_id") or rec.get("question") or id(rec)


def summarize(records):
    """Two lenses, kept separate so the multi-query dataset is read honestly:

      (1) PER-(sub-query, gold) similarity — every sub-query is its own data point. This is the
          "analyse each query" view: stage1_gold_weighted_sim / stage2_gold_best_chunk_sim pool
          one point per (sub-query, gold).
      (2) BEST-per-case similarity (*_best_per_case) — for each (case, gold) the MAX similarity
          across that case's sub-queries. Tests the multi-query premise: do different issue-
          queries between them bring a gold closer than any single query would?

    Case-level gold INVENTORY (how many golds exist / are missing / have no chunks) is a property
    of the CASE, so it is deduped to one record per source case — otherwise a case with 4 sub-
    queries counts its golds 4x. Gold membership/chunk-presence is query-independent, so the
    first record seen for a case carries its whole inventory. On a single-query dataset every
    case has exactly one record: dedup is a no-op and (1) == (2), so this stays backward-
    compatible with the original single-query headline numbers."""
    s1_all, s2_best_all = [], []
    field_all = {k: [] for k in STATIC_WEIGHTS}
    s1_best_cg, s2_best_cg = {}, {}   # (case, gold_doc_id) -> best score across sub-queries

    for rec in records:
        cid = _case_key(rec)
        for g in rec["stage1"]:
            s1_all.append(g["final_score"])
            for k in field_all:
                field_all[k].append(g["fields"][k])
            key = (cid, g["doc_id"])
            s1_best_cg[key] = max(s1_best_cg.get(key, -1.0), g["final_score"])
        for c in rec["stage2_per_gold"]:
            s2_best_all.append(c["best_chunk_sim"])
            key = (cid, c["doc_id"])
            s2_best_cg[key] = max(s2_best_cg.get(key, -1.0), c["best_chunk_sim"])

    # Case-level inventory, deduped to one record per distinct source case.
    by_case = {}
    for rec in records:
        by_case.setdefault(_case_key(rec), rec)

    n_gold_total = n_gold_in_db = n_missing = n_no_chunks = 0
    cases_with_any_missing = 0
    for rec in by_case.values():
        n_gold_total += rec["n_gold"]
        n_gold_in_db += rec["n_gold_in_db"]
        n_missing += len(rec["missing_gold_ids"])
        if rec["missing_gold_ids"]:
            cases_with_any_missing += 1
        chunk_ids = {c["doc_id"] for c in rec["stage2_per_gold"]}
        for g in rec["stage1"]:
            if g["doc_id"] not in chunk_ids:
                n_no_chunks += 1  # gold case in DB at case level but with no chunk rows

    distinct_cases = len(by_case)
    return {
        "subqueries": len(records),
        "distinct_cases": distinct_cases,
        "avg_subqueries_per_case": round(len(records) / distinct_cases, 2) if distinct_cases else 0,
        "gold_total": n_gold_total,
        "gold_in_db_case_level": n_gold_in_db,
        "gold_missing_from_db": n_missing,
        "cases_with_any_missing_gold": cases_with_any_missing,
        "gold_in_db_but_no_chunks": n_no_chunks,
        "stage1_gold_weighted_sim": _dist(s1_all),
        "stage2_gold_best_chunk_sim": _dist(s2_best_all),
        "stage1_gold_weighted_sim_best_per_case": _dist(list(s1_best_cg.values())),
        "stage2_gold_best_chunk_sim_best_per_case": _dist(list(s2_best_cg.values())),
        "stage1_per_field_sim": {k: _dist(v) for k, v in field_all.items()},
    }


def load_corpus_summary_fields(doc_id: str):
    """Read the 5 LLM-summary text fields from il-pcsr_dataset/corpus/<doc_id>*.json.
    Replicates the exact string construction in ingest_ilpcsr_to_db.py so the text seen by
    the compare model is identical to what the primary model originally embedded."""
    matches = list(ILPCSR_CORPUS.glob(f"{doc_id}_*.json"))
    if not matches:
        matches = list(ILPCSR_CORPUS.glob(f"{doc_id}.json"))
    if not matches:
        return None
    try:
        with open(matches[0], encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    summary = data.get("Summary", {}) or {}
    return {
        "facts":      summary.get("key_facts", "") or "",
        "issues":     "; ".join(summary.get("legal_issues", []) or []),
        "statutes":   "; ".join(summary.get("key_statutes_cited", []) or []),
        "precedents": "; ".join(summary.get("key_precedents_cited", []) or []),
        "holding":    summary.get("holding", "") or "",
    }


def run_cross_emb_analysis(records, primary_model, primary_model_name, compare_model_name, cur):
    """Post-pass: re-embed gold case summary fields + chunks locally with a second
    SentenceTransformer and compare cosine distributions side-by-side with the primary model.
    WHY: isolates whether low oracle scores are model-specific (swap the encoder to fix it) vs
    inherent to the text (no encoder bridges the gap). Uses cosine on unit vectors for BOTH
    models so distributions are comparable regardless of native vector scale. Mutates each
    record in-place to add a 'cross_emb' key with per-gold per-model scores."""
    from sentence_transformers import SentenceTransformer

    print(f"\n[cross-emb] Loading compare model '{compare_model_name}' ...")
    cmp_model = SentenceTransformer(compare_model_name)
    print("[cross-emb] Model loaded.")

    FIELD_KEYS = ["facts", "issues", "statutes", "precedents", "holding"]

    # Collect all gold doc_ids that appear in stage1 results (= have DB embeddings for this model)
    all_gold_ids = set()
    for rec in records:
        for g in rec["stage1"]:
            all_gold_ids.add(g["doc_id"])

    # Load corpus JSON text fields for each gold doc_id (primary model embedded these at ingest)
    print(f"[cross-emb] Loading corpus JSONs for {len(all_gold_ids)} gold cases ...")
    case_fields = {}
    missing_corpus = []
    for doc_id in sorted(all_gold_ids):
        fields = load_corpus_summary_fields(doc_id)
        if fields:
            case_fields[doc_id] = fields
        else:
            missing_corpus.append(doc_id)
    if missing_corpus:
        print(f"[cross-emb] WARN: {len(missing_corpus)} gold cases not found in corpus dir "
              f"(skipped): {missing_corpus[:5]}{'...' if len(missing_corpus) > 5 else ''}")

    # Fetch stored chunk texts from DB in one round-trip (chunk_text is model-agnostic)
    chunk_texts_by_doc = {}
    if all_gold_ids:
        print("[cross-emb] Fetching chunk texts from DB ...")
        cur.execute(CHUNK_TEXTS_SQL, {"gold": tuple(all_gold_ids)})
        for doc_id, chunk_text in cur.fetchall():
            chunk_texts_by_doc.setdefault(str(doc_id), []).append(chunk_text)

    # Embed all gold case fields once per doc_id (cache avoids re-encoding shared gold cases)
    print(f"[cross-emb] Encoding {len(case_fields)} case field-sets with both models ...")
    m1_case_embs = {}   # doc_id -> {field: np.array}
    m2_case_embs = {}
    for doc_id, fields in case_fields.items():
        texts = [fields[k] for k in FIELD_KEYS]
        m1_case_embs[doc_id] = dict(zip(FIELD_KEYS, primary_model.encode(texts)))
        m2_case_embs[doc_id] = dict(zip(FIELD_KEYS, cmp_model.encode(texts)))

    # Embed all gold chunks once per doc_id
    n_chunks = sum(len(v) for v in chunk_texts_by_doc.values())
    print(f"[cross-emb] Encoding {n_chunks} chunks across {len(chunk_texts_by_doc)} cases ...")
    m1_chunk_embs = {}   # doc_id -> list[np.array]
    m2_chunk_embs = {}
    for doc_id, texts in chunk_texts_by_doc.items():
        if texts:
            m1_chunk_embs[doc_id] = primary_model.encode(texts)
            m2_chunk_embs[doc_id] = cmp_model.encode(texts)

    # Pre-encode all unique query texts (sub-queries sharing the same text re-use the cached vec)
    unique_queries = list({rec["question"] for rec in records})
    print(f"[cross-emb] Encoding {len(unique_queries)} unique sub-query texts ...")
    m1_q = {q: primary_model.encode(q) for q in unique_queries}
    m2_q = {q: cmp_model.encode(q) for q in unique_queries}

    m1_s1_all, m2_s1_all = [], []
    m1_s2_all, m2_s2_all = [], []

    for rec in records:
        q = rec["question"]
        q1, q2 = m1_q[q], m2_q[q]
        m1_entries, m2_entries = [], []

        for g in rec["stage1"]:
            doc_id = g["doc_id"]
            if doc_id not in case_fields:
                continue

            m1_s1 = sum(_cosine(q1, m1_case_embs[doc_id][k]) * STATIC_WEIGHTS[k] for k in FIELD_KEYS)
            m2_s1 = sum(_cosine(q2, m2_case_embs[doc_id][k]) * STATIC_WEIGHTS[k] for k in FIELD_KEYS)
            m1_s1_all.append(m1_s1)
            m2_s1_all.append(m2_s1)

            m1_s2 = max((_cosine(q1, ce) for ce in m1_chunk_embs[doc_id]), default=None) \
                if doc_id in m1_chunk_embs else None
            m2_s2 = max((_cosine(q2, ce) for ce in m2_chunk_embs[doc_id]), default=None) \
                if doc_id in m2_chunk_embs else None
            if m1_s2 is not None:
                m1_s2_all.append(m1_s2)
            if m2_s2 is not None:
                m2_s2_all.append(m2_s2)

            m1_entries.append({"doc_id": doc_id, "stage1_cosine": round(m1_s1, 4),
                                "stage2_best_cosine": round(m1_s2, 4) if m1_s2 is not None else None})
            m2_entries.append({"doc_id": doc_id, "stage1_cosine": round(m2_s1, 4),
                                "stage2_best_cosine": round(m2_s2, 4) if m2_s2 is not None else None})

        rec["cross_emb"] = {"model1_local": m1_entries, "model2": m2_entries}

    return {
        "primary_model": primary_model_name,
        "compare_model": compare_model_name,
        "n_gold_with_corpus_text": len(case_fields),
        "n_gold_missing_corpus": len(missing_corpus),
        "note": ("model1_local uses cosine on unit vectors, not the DB's 1-L2 score — small "
                 "numerical differences from the DB values are expected; rank should match."),
        "model1_local_stage1_cosine": _dist(m1_s1_all),
        "model2_stage1_cosine": _dist(m2_s1_all),
        "model1_local_stage2_best_cosine": _dist(m1_s2_all),
        "model2_stage2_best_cosine": _dist(m2_s2_all),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Oracle (gold-only) query<->ground-truth similarity diagnostic."
    )
    ap.add_argument("--emb-model-id", type=int, default=1,
                    help="config id in ilpcsr_rag_config (default 1, matches model=1 results)")
    ap.add_argument("--query-field", default="question",
                    help="eval-dataset field used as query text (default 'question')")
    ap.add_argument("--multi-query", action="store_true",
                    help="read the multi-query dataset (one record per sub-query) and analyse "
                         "EACH sub-query; case-level gold counts are deduped by source case")
    ap.add_argument("--eval-dataset", default=None,
                    help="override input path (default: single-query dataset, or the multi-query "
                         "dataset when --multi-query is set)")
    ap.add_argument("--output", default=None,
                    help="override output path (default switches with --multi-query)")
    ap.add_argument("--no-ssl", action="store_true", help="connect without SSL (local Postgres)")
    ap.add_argument("--timeout", default="120s", help="per-statement timeout (default 120s)")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="only run the first N queries (0 = all; for quick timing tests)")
    ap.add_argument("--distance-op", default="l2", choices=["l2", "cosine"],
                    help="pgvector distance operator: l2 uses <-> (default), cosine uses <=> (Kanon-correct, uses HNSW index)")
    ap.add_argument("--cross-emb", default=None, metavar="MODEL_NAME",
                    help="(multi-query only) post-pass: re-embed gold case summary fields + "
                         "chunks with this SentenceTransformer model and compare cosine "
                         "distributions against the primary model side-by-side. Isolates "
                         "whether low oracle scores are model-specific or text-inherent. "
                         "Example: 'sentence-transformers/all-mpnet-base-v2'")
    args = ap.parse_args()

    eval_path = Path(args.eval_dataset) if args.eval_dataset else (
        MULTIQUERY_DATASET if args.multi_query else EVAL_DATASET)
    _data_dir = REPO_ROOT / "evaluation" / "data"
    _op_suffix = "_cosine" if args.distance_op == "cosine" else ""
    output_path = Path(args.output) if args.output else (
        _data_dir / f"gold_oracle_emb{args.emb_model_id}{_op_suffix}_multiquery.json"
        if args.multi_query else
        _data_dir / f"gold_oracle_emb{args.emb_model_id}{_op_suffix}.json"
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
    stage1_sql = _build_stage1_sql(DISTANCE_OPS[args.distance_op])
    stage2_sql = _build_stage2_sql(DISTANCE_OPS[args.distance_op])
    print(f"emb_model_id={args.emb_model_id} -> '{emb_model_name}'  distance_op={args.distance_op}")

    if emb_model_name not in _query_emb_cache:
        _query_emb_cache[emb_model_name] = _load_cache(emb_model_name)
    print(f"Embedding cache: {len(_query_emb_cache[emb_model_name])} queries pre-loaded for '{emb_model_name}'")

    with open(eval_path, "r", encoding="utf-8") as fh:
        eval_data = json.load(fh)
    if args.limit_queries > 0:
        eval_data = eval_data[:args.limit_queries]
    # Auto-detect the dataset shape so the readout is honest even if --multi-query was omitted
    # but a multi-query file was passed via --eval-dataset (sub_query_id only exists there).
    is_multi = any("sub_query_id" in it for it in eval_data)
    n_cases_in = len({str(it.get("source_case_id", "")) for it in eval_data})
    print(f"{'MULTI-QUERY' if is_multi else 'single-query'}: {len(eval_data)} queries across "
          f"{n_cases_in} distinct cases, query_field='{args.query_field}'")

    jsonl_path = output_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        records = _load_jsonl(jsonl_path)
    elif output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as _f:
                _d = json.load(_f)
            records = _d.get("records", []) if isinstance(_d, dict) else _d
            print(f"Output already complete ({len(records)} records). Re-run will be a no-op.")
        except Exception:
            records = []
    else:
        records = []
    done_keys = {
        (r.get("source_case_id", ""), r.get("sub_query_id", r.get("question", "")))
        for r in records
    }

    _first_pending_q = None
    for _it in eval_data:
        _q = (_it.get(args.query_field) or "").strip()
        if not _q:
            continue
        _key = (str(_it.get("source_case_id", "")), str(_it.get("sub_query_id", _q)))
        if _key not in done_keys:
            _first_pending_q = _q
            break
    if _first_pending_q is not None:
        _hit = _first_pending_q in _query_emb_cache.get(emb_model_name, {})
        print(f"  → first pending query: {'CACHE HIT' if _hit else 'CACHE MISS (will encode)'}")

    skipped_no_query = skipped_no_gold = 0
    t_start = time.time()
    first_processed = True
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for i, item in enumerate(eval_data, 1):
            query = (item.get(args.query_field) or "").strip()
            gold_ids = [str(g) for g in (item.get("ground_truth_cases") or [])]
            gold_ids = list(dict.fromkeys(gold_ids))  # dedup, preserve order
            if not query:
                skipped_no_query += 1
                continue
            if not gold_ids:
                skipped_no_gold += 1
                continue

            key = (str(item.get("source_case_id", "")), str(item.get("sub_query_id", query)))
            if key in done_keys:
                continue  # already processed in a previous run

            vec_str = str(encode_query(query, emb_model_name))
            t0 = time.time()
            s1 = stage1_gold_scores(cur, vec_str, gold_ids, args.emb_model_id, stage1_sql)
            s2 = stage2_gold_chunk_scores(cur, vec_str, gold_ids, args.emb_model_id, stage2_sql)
            if first_processed:
                dt = time.time() - t0
                remaining = len(eval_data) - len(done_keys)
                print(f"  [timing] first query (both oracle stages) = {dt:.3f}s "
                      f"-> est ~{dt * remaining / 60.0:.1f} min for {remaining} remaining queries")
                first_processed = False

            found_ids = {g["doc_id"] for g in s1}
            missing_gold = [g for g in gold_ids if g not in found_ids]

            rec = {
                "source_case_id":   str(item.get("source_case_id", "")),
                "question":         query,
                "n_gold":           len(gold_ids),
                "n_gold_in_db":     len(found_ids),
                "missing_gold_ids": missing_gold,
                "stage1":           s1,
                "stage2_per_gold":  s2,
            }
            # Carry the sub-query identity through so each query stays traceable in the records and
            # downstream grouping/fusion can tie a sub-query back to its source case.
            if "sub_query_id" in item:
                rec["sub_query_id"] = item["sub_query_id"]
            if "issue" in item:
                rec["issue"] = item["issue"]
            records.append(rec)
            done_keys.add(key)
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(eval_data)}  ({time.time() - t_start:.0f}s elapsed)")

    if skipped_no_query:
        print(f"  [WARN] {skipped_no_query} queries had empty '{args.query_field}' -> skipped.")
    if skipped_no_gold:
        print(f"  [WARN] {skipped_no_gold} queries had no ground_truth_cases -> skipped.")

    summary = summarize(records)

    cross_emb_result = None
    if args.cross_emb:
        if emb_model_name in ISAACUS_MODELS:
            print(f"[cross-emb] WARN: --cross-emb requires a SentenceTransformer primary model. "
                  f"'{emb_model_name}' is an API-backed Isaacus model — skipping cross-emb analysis.")
        elif not is_multi:
            print("[cross-emb] WARN: --cross-emb is only supported on multi-query datasets "
                  "(auto-detected single-query shape). Skipping.")
        else:
            from sentence_transformers import SentenceTransformer
            print(f"Loading primary model '{emb_model_name}' for cross-emb analysis ...")
            primary_model = SentenceTransformer(emb_model_name)
            cross_emb_result = run_cross_emb_analysis(
                records, primary_model, emb_model_name, args.cross_emb, cur
            )

    payload = {"_summary": summary, "records": records}
    if cross_emb_result:
        payload["cross_emb_analysis"] = cross_emb_result

    print(f"\nSaving {len(records)} records -> {output_path}")
    _save_final(payload, output_path, jsonl_path)

    # Headline readout
    def _line(d):
        d = d or {}
        return (f"median {d.get('median')}   mean {d.get('mean')}   "
                f"p10 {d.get('p10')}   p90 {d.get('p90')}   range [{d.get('min')}, {d.get('max')}]")

    print("\n========== ORACLE SIMILARITY SUMMARY ==========")
    print(f"sub-queries: {summary['subqueries']}   distinct cases: {summary['distinct_cases']}"
          f"   (avg {summary['avg_subqueries_per_case']} sub-queries/case)")
    print(f"gold precedents (per case, deduped): {summary['gold_total']}")
    print(f"  golds with case-level embedding : {summary['gold_in_db_case_level']}")
    print(f"  golds MISSING from DB (no emb)  : {summary['gold_missing_from_db']}"
          f"  (in {summary['cases_with_any_missing_gold']} cases)")
    print(f"  golds in DB but with no chunks  : {summary['gold_in_db_but_no_chunks']}")
    print(f"\nStage-1 gold weighted similarity (per sub-query x gold, 5-field engine scoring):")
    print(f"  {_line(summary['stage1_gold_weighted_sim'])}")
    print(f"Stage-2 gold best-chunk similarity (per sub-query x gold, max-pool inside golds):")
    print(f"  {_line(summary['stage2_gold_best_chunk_sim'])}")
    if is_multi:
        print(f"\n[multi-query lens] BEST similarity per (case, gold) across the case's sub-queries:")
        print(f"  stage-1  {_line(summary['stage1_gold_weighted_sim_best_per_case'])}")
        print(f"  stage-2  {_line(summary['stage2_gold_best_chunk_sim_best_per_case'])}")
    print(f"\nPer-field gold similarity (median, per sub-query x gold):")
    for k, d in (summary["stage1_per_field_sim"] or {}).items():
        if d:
            print(f"  {k:<11} median {d['median']}   mean {d['mean']}   (weight {STATIC_WEIGHTS[k]})")
    if cross_emb_result:
        ce = cross_emb_result
        print(f"\n---------- CROSS-EMBEDDING COMPARISON ----------")
        print(f"primary : {ce['primary_model']} (cosine, local)")
        print(f"compare : {ce['compare_model']}")
        print(f"corpus coverage: {ce['n_gold_with_corpus_text']} gold cases found, "
              f"{ce['n_gold_missing_corpus']} missing")
        print(f"Stage-1 weighted cosine (per sub-query x gold):")
        print(f"  primary (local)  {_line(ce['model1_local_stage1_cosine'])}")
        print(f"  compare model    {_line(ce['model2_stage1_cosine'])}")
        print(f"Stage-2 best-chunk cosine (per sub-query x gold):")
        print(f"  primary (local)  {_line(ce['model1_local_stage2_best_cosine'])}")
        print(f"  compare model    {_line(ce['model2_stage2_best_cosine'])}")
        print(f"NOTE: {ce['note']}")
    print("===============================================")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
