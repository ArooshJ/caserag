from pathlib import Path
from sentence_transformers import SentenceTransformer
import psycopg2
import os
import json

from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

DB_NAME = os.environ.get("DB_NAME", "neondb")
DB_USER = os.environ.get("DB_USER", "neondb_owner")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

EMBEDDING_MODEL_CACHE = {}

# Isaacus API-backed models — cannot be loaded via SentenceTransformer
ISAACUS_MODELS = {"kanon-2-embedder"}

# Lazy Isaacus client singleton — only constructed when a Kanon 2 query is actually made
_isaacus_client = None

# In-memory query embedding cache: {model_name: {query_text: [float, ...]}}
# Canonical on-disk format: query_emb_cache_{model}.jsonl (one {"q":..,"e":..} per line, O(1) append).
# Legacy fallback: query_emb_cache_{model}.json (old full-dict format, read-only).
_query_emb_cache: dict = {}
_CACHE_DIR = Path(__file__).resolve().parent / "evaluation" / "data"


def _get_isaacus_client():
    global _isaacus_client
    if _isaacus_client is None:
        from isaacus import Isaacus
        _isaacus_client = Isaacus()
    return _isaacus_client


def _cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return _CACHE_DIR / f"query_emb_cache_{safe}.jsonl"


def _load_cache(model_name: str) -> dict:
    """Load cache from JSONL (canonical) + legacy JSON (read-only fallback). Merges both."""
    result = {}
    # Legacy JSON dict — existing Isaacus cache; read-only, never written going forward
    legacy = _CACHE_DIR / f"query_emb_cache_{model_name.replace('/', '_')}.json"
    if legacy.exists():
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                result.update(json.load(f))
        except Exception:
            pass
    # JSONL — canonical new format; takes precedence on key collision
    p = _cache_path(model_name)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    result[entry["q"]] = entry["e"]
                except Exception:
                    pass
    return result


def _save_cache(model_name: str, text: str, embedding: list):
    """Append one new cache entry to the JSONL file — O(1), no full rewrite."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(model_name), "a", encoding="utf-8") as f:
        f.write(json.dumps({"q": text, "e": embedding}, ensure_ascii=False) + "\n")


def encode_query(text: str, model_name: str) -> list:
    """Encode a query string, returning from disk cache if available.

    Both Isaacus and SentenceTransformer models use the same JSONL cache:
    one line appended per new unique query — O(1) per save, no full rewrite.
    Legacy JSON caches are read on first load and transparently merged.
    """
    if model_name not in _query_emb_cache:
        _query_emb_cache[model_name] = _load_cache(model_name)
    if text in _query_emb_cache[model_name]:
        return _query_emb_cache[model_name][text]

    if model_name in ISAACUS_MODELS:
        response = _get_isaacus_client().embeddings.create(
            model=model_name,
            texts=[text],
            task="retrieval/query",
        )
        embedding = response.embeddings[0].embedding
    else:
        embedding = get_embedding_model(model_name).encode(text).tolist()

    _query_emb_cache[model_name][text] = embedding
    _save_cache(model_name, text, embedding)
    return embedding

STATIC_WEIGHTS = {
    "facts": 0.1,
    "issues": 0.3,
    "statutes": 0.2,
    "precedents": 0.1,
    "holding": 0.3,
}

DISTANCE_OPS = {"l2": "<->", "cosine": "<=>"}  # pgvector operator tokens


def get_db_conn(sslmode: str = "require"):
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        sslmode=sslmode,
    )


def fetch_rag_config(emb_model_id: int) -> dict:
    """
    Fetches the ablation configuration for the given ID.
    Returns a dictionary with the emb_model_name and llm_model.
    """
    db_conn = None
    try:
        db_conn = get_db_conn()
        cursor = db_conn.cursor()

        query = """
            SELECT emb_model_name, llm_model, dimensions
            FROM ilpcsr_rag_config
            WHERE emb_model_id = %s;
        """
        cursor.execute(query, (emb_model_id,))
        result = cursor.fetchone()

        if not result:
            raise ValueError(f"Configuration for emb_model_id {emb_model_id} not found.")

        return {
            "emb_model_name": result[0],
            "llm_model": result[1],
            "dimensions": result[2],
        }
    except psycopg2.Error as e:
        print(f"Database error fetching config: {e}")
        return None
    finally:
        if db_conn:
            db_conn.close()


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Retrieves the model from the global cache, loading it if it doesn't exist.
    """
    if model_name not in EMBEDDING_MODEL_CACHE:
        print(f"Loading embedding model '{model_name}' into cache for the first time...")
        EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        print(f"Model '{model_name}' loaded successfully.")

    return EMBEDDING_MODEL_CACHE[model_name]


def retrieve_top_cases(
    query_text: str,
    emb_model_id: int,
    top_k: int = 10,
    distance_op: str = "l2",
):
    """
    Stage 1 Retrieval: Finds the top K most relevant cases using a weighted 
    multi-vector search, isolated to a specific embedding model configuration.
    """
    db_conn = None
    try:
        config = fetch_rag_config(emb_model_id)
        if not config:
            print(f"Error: Could not retrieve config for emb_model_id {emb_model_id}")
            return []

        query_embedding = encode_query(query_text, config["emb_model_name"])
        embedding_str = str(query_embedding)

        db_conn = get_db_conn()
        cursor = db_conn.cursor()

        op = DISTANCE_OPS[distance_op]
        query_sql = f"""
            WITH similarity_scores AS (
                SELECT
                    c.doc_id,
                    c.title,
                    COALESCE(1 - (ce.facts_emb {op} %(embedding)s::vector), 0) AS score_facts,
                    COALESCE(1 - (ce.issues_emb {op} %(embedding)s::vector), 0) AS score_issues,
                    COALESCE(1 - (ce.statutes_emb {op} %(embedding)s::vector), 0) AS score_statutes,
                    COALESCE(1 - (ce.precedents_emb {op} %(embedding)s::vector), 0) AS score_precedents,
                    COALESCE(1 - (ce.holding_emb {op} %(embedding)s::vector), 0) AS score_holding
                FROM
                    ilpcsr_cases c
                JOIN
                    ilpcsr_cases_embeddings ce ON c.doc_id = ce.doc_id
                WHERE
                    ce.emb_model_id = %(emb_model_id)s
            )
            SELECT
                doc_id,
                title,
                (
                    (score_facts * %(w_facts)s) +
                    (score_issues * %(w_issues)s) +
                    (score_statutes * %(w_statutes)s) +
                    (score_precedents * %(w_precedents)s) +
                    (score_holding * %(w_holding)s)
                ) AS final_score
            FROM
                similarity_scores
            ORDER BY
                final_score DESC
            LIMIT
                %(limit)s;
        """

        params = {
            "embedding": embedding_str,
            "emb_model_id": emb_model_id,
            "w_facts": STATIC_WEIGHTS["facts"],
            "w_issues": STATIC_WEIGHTS["issues"],
            "w_statutes": STATIC_WEIGHTS["statutes"],
            "w_precedents": STATIC_WEIGHTS["precedents"],
            "w_holding": STATIC_WEIGHTS["holding"],
            "limit": top_k,
        }

        cursor.execute(query_sql, params)
        results = cursor.fetchall()
        return results

    except Exception as e:
        print(f"Error in retrieve_top_cases: {e}")
        return []
    finally:
        if db_conn:
            db_conn.close()


def retrieve_detailed_context(
    query_text: str,
    case_ids: list,
    emb_model_id: int,
    top_k: int = 10,
    distance_op: str = "l2",
):
    """
    Stage 2 Retrieval: Finds the most relevant specific paragraphs from the 
    pre-filtered list of top cases, strictly using the specified embedding model.
    """
    if not case_ids:
        return []

    db_conn = None
    try:
        config = fetch_rag_config(emb_model_id)
        if not config:
            return []

        query_embedding = encode_query(query_text, config["emb_model_name"])
        embedding_str = str(query_embedding)

        db_conn = get_db_conn()
        cursor = db_conn.cursor()

        op = DISTANCE_OPS[distance_op]
        query_sql = f"""
            SELECT
                p.full_text,
                p.metadata_label,
                p.doc_id,
                cs.title,
                cs.source_url,
                1 - (ce.embedding {op} %s::vector) AS similarity_score
            FROM
                ilpcsr_content_chunks c
            JOIN
                ilpcsr_chunk_embeddings ce ON c.chunk_id = ce.chunk_id
            JOIN
                ilpcsr_paragraphs p ON c.paragraph_id = p.paragraph_id
            JOIN
                ilpcsr_cases cs ON p.doc_id = cs.doc_id
            WHERE
                p.doc_id IN %s 
                AND ce.emb_model_id = %s
            ORDER BY
                similarity_score DESC
            LIMIT 300;
        """

        cursor.execute(query_sql, (embedding_str, tuple(case_ids), emb_model_id))
        results = cursor.fetchall()

        unique_paragraphs = {}
        for row in results:
            full_text, label, doc_id, title, url, score = row
            if full_text not in unique_paragraphs or score > unique_paragraphs[full_text]["score"]:
                unique_paragraphs[full_text] = {
                    "label": label,
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "score": score,
                }

        sorted_paragraphs = sorted(
            unique_paragraphs.items(), key=lambda item: item[1]["score"], reverse=True
        )
        return sorted_paragraphs[:top_k]

    except Exception as e:
        print(f"Error in retrieve_detailed_context: {e}")
        return []
    finally:
        if db_conn:
            db_conn.close()


def query_rag_pipeline(
    test_query: str,
    emb_model_id: int,
    top_k_stage1: int = 5,
    top_k_stage2: int = 5,
    distance_op: str = "l2",
):
    """
    Executes the two-stage retrieval pipeline for a specific ablation configuration.
    The output keeps "generated_answer" as None so its layout matches the saved results.
    """
    config = fetch_rag_config(emb_model_id)
    if not config:
        return {"error": f"Configuration for emb_model_id {emb_model_id} not found."}

    llm_model_name = config["llm_model"]

    #print(f"\nExecuting Stage 1 (Cases) for model ID: {emb_model_id}...")
    top_cases = retrieve_top_cases(
        test_query,
        emb_model_id,
        top_k=top_k_stage1,
        distance_op=distance_op,
    )

    case_ids_to_search = [case[0] for case in top_cases] if top_cases else []
    
    #print("stage 1 cases ids len = ",len(case_ids_to_search))
    output_sources = []
    #print("Top Cases Retrieved:", len(top_cases))

    if case_ids_to_search:
        #print(f"Executing Stage 2 (Paragraphs) for {len(case_ids_to_search)} cases...")
        detailed_context = retrieve_detailed_context(
            test_query,
            case_ids_to_search,
            emb_model_id,
            top_k=top_k_stage2,
            distance_op=distance_op,
        )

        #print("Detailed Context Passages Retrieved:", len(detailed_context))
        if detailed_context:
            for text, data in detailed_context:
                source_entry = {
                    "case_id": data["doc_id"],
                    "title": data["title"],
                    "url": data["url"],
                    "label": data["label"],
                    "text": text,
                    "score": float(data["score"]),
                }
                output_sources.append(source_entry)

    final_output = {
        "question": test_query,
        "ablation_config_id": emb_model_id,
        "embedding_model_used": config["emb_model_name"],
        "llm_model_used": llm_model_name,
        "stage1_case_ids": case_ids_to_search,
        "sources_retrieved": len(output_sources),
        "sources": output_sources,
        "generated_answer": None,
    }

    return final_output
