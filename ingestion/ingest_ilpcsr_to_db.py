import csv
import os
import json
import datetime
import psycopg2
from psycopg2.extras import execute_values
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Colab secret names (NEON_DB_*) → env var names the rest of the script reads (DB_*)
_COLAB_SECRET_MAP = {
    "NEON_DB_NAME":     "DB_NAME",
    "NEON_DB_USER":     "DB_USER",
    "NEON_DB_PASSWORD": "DB_PASSWORD",
    "NEON_DB_HOST":     "DB_HOST",
    "NEON_DB_PORT":     "DB_PORT",
}

def _load_credentials():
    """
    In Colab: reads NEON_DB_* secrets (sidebar key icon), maps to DB_* env vars.
    Not in Colab: reads DB_* from .env as normal.
    Colab terminal caveat: userdata's JS bridge is unavailable from a shell —
    run via notebook cell (`!python ingestion/ingest_ilpcsr_to_db.py`) instead,
    or manually export DB_* vars in the terminal before running.
    """
    try:
        from google.colab import userdata
        loaded = []
        for colab_key, env_key in _COLAB_SECRET_MAP.items():
            val = userdata.get(colab_key)
            if val:
                os.environ[env_key] = val
                loaded.append(colab_key)
        print(f"Colab detected. Secrets loaded: {loaded or 'NONE — set secrets in sidebar (key icon)'}")
    except ImportError:
        load_dotenv()
    except Exception as e:
        print(f"Colab userdata bridge unavailable: {e}")
        print("Either run via notebook cell (!python ingestion/ingest_ilpcsr_to_db.py)")
        print("or manually: export DB_NAME=... DB_PASSWORD=... etc. before running from terminal.")

_load_credentials()

DB_NAME     = os.environ.get("DB_NAME", "neondb")
DB_USER     = os.environ.get("DB_USER", "neondb_owner")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST     = os.environ.get("DB_HOST", "localhost")
DB_PORT     = os.environ.get("DB_PORT", "5432")

# Relative to this script: ingestion/ -> repo root -> il-pcsr_dataset/corpus
SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR.parent / "il-pcsr_dataset" / "corpus"

TEXT_KEYS = [
    'Fact', 'Issue', "Petitioner's Argument", "Respondent's Argument",
    'Analysis of the law', 'Precedent Analysis', "Court's Reasoning",
    'Conclusion', 'Court Disclosure', 'Uncategorized'
]


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT, sslmode='require'
    )


def detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {name} — using CUDA")
        return "cuda"
    print("No GPU detected — falling back to CPU")
    return "cpu"


def batch_ingest_from_directory(dataset_path: Path, emb_model_id: int, device: str):
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = log_dir / f"ilpcsr_DB_Ingestion_error_Log_{timestamp}.txt"
    csv_filepath = log_dir / f"ilpcsr_inserted_ids_{timestamp}.csv"

    db_conn  = None
    csv_file = None
    try:
        db_conn = get_db_conn()
        cursor  = db_conn.cursor()

        csv_file = open(csv_filepath, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['case_id', 'paragraph_id', 'chunk_id'])

        print(f"Fetching config for emb_model_id={emb_model_id}...")
        cursor.execute(
            "SELECT emb_model_name, chunk_size, chunk_overlap FROM ilpcsr_rag_config WHERE emb_model_id = %s",
            (emb_model_id,)
        )
        config = cursor.fetchone()
        if not config:
            print(f"No config found for emb_model_id={emb_model_id}. Aborting.")
            return

        emb_model_name, chunk_size, chunk_overlap = config
        print(f"Config: model='{emb_model_name}', chunk_size={chunk_size}, overlap={chunk_overlap}")

        print(f"Loading embedding model on {device}...")
        model        = SentenceTransformer(emb_model_name, device=device)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        cursor.execute("SELECT doc_id FROM ilpcsr_cases;")
        existing_cases = {row[0] for row in cursor.fetchall()}

        cursor.execute("SELECT doc_id FROM ilpcsr_cases_embeddings WHERE emb_model_id = %s;", (emb_model_id,))
        existing_embeddings = {row[0] for row in cursor.fetchall()}

        print(f"Scanning '{dataset_path}'...")
        json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files. {len(existing_cases)} cases already in DB.")

        for filename in json_files:
            try:
                with open(dataset_path / filename, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                case_id = case_data.get("caseID")
                summary  = case_data.get("Summary")

                if not case_id or not summary:
                    print(f"-> Skipping '{filename}' (missing caseID or Summary)")
                    continue

                print(f"\n=> {filename} (caseID: {case_id})")

                # para_chunk_to_cid: (paragraph_id, chunk_text) → chunk_id, built during text
                # insert and reused in embed phase (avoids a SELECT). None = text already existed.
                para_chunk_to_cid = None

                # --- PHASE 1: TEXT (write-once) ---
                if case_id not in existing_cases:
                    source_url = f"https://indiankanoon.org/doc/{case_id}/"
                    cursor.execute("""
                        INSERT INTO ilpcsr_cases (doc_id, source_url, title, court, case_date, relevant_statutes, relevant_precedents)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, (
                        case_id, source_url,
                        case_data.get('title'), case_data.get('court'),
                        case_data.get('date') or None,
                        case_data.get('relevant_statutes', []),
                        case_data.get('relevant_precedents', [])
                    ))

                    # Collect all paragraphs for this case
                    para_rows = [
                        (case_id, category, paragraph_text)
                        for category in TEXT_KEYS
                        for paragraph_text in case_data.get(category, [])
                        if paragraph_text.strip()
                    ]

                    if not para_rows:
                        print(f"   [Text] No paragraph content — committing case row, skipping")
                        existing_cases.add(case_id)
                        db_conn.commit()
                        continue

                    # Batch insert paragraphs — RETURNING full_text lets us map UUID→text
                    # without relying on result ordering (UUID PKs have no order guarantee)
                    para_result = execute_values(
                        cursor,
                        "INSERT INTO ilpcsr_paragraphs (doc_id, metadata_label, full_text) VALUES %s RETURNING paragraph_id, full_text",
                        para_rows, fetch=True
                    )
                    full_text_to_pid = {row[1]: row[0] for row in para_result}

                    # Build chunk rows using the paragraph_id lookup
                    chunk_rows = [
                        (full_text_to_pid[paragraph_text], chunk_text)
                        for _, _, paragraph_text in para_rows
                        for chunk_text in text_splitter.split_text(paragraph_text)
                    ]

                    if chunk_rows:
                        # RETURNING paragraph_id lets us key by (paragraph_id, chunk_text) so
                        # same text in different paragraphs stays separate (not collapsed as dupes)
                        chunk_result = execute_values(
                            cursor,
                            "INSERT INTO ilpcsr_content_chunks (paragraph_id, chunk_text) VALUES %s RETURNING chunk_id, paragraph_id, chunk_text",
                            chunk_rows, fetch=True
                        )
                        para_chunk_to_cid = {(str(row[1]), row[2]): row[0] for row in chunk_result}

                        # Log inserted IDs to CSV (one row per deduplicated chunk)
                        for (para_id, chunk_text), chunk_id in para_chunk_to_cid.items():
                            csv_writer.writerow([case_id, para_id, str(chunk_id)])

                    print(f"   [Text] Inserted {len(chunk_rows)} chunks across {len(para_rows)} paragraphs")
                    existing_cases.add(case_id)
                else:
                    print("   [Text] Already exists, skipping")

                # --- PHASE 2: EMBEDDINGS (write-many) ---
                if case_id in existing_embeddings:
                    print(f"   [Embed] Already embedded for model {emb_model_id}, skipping")
                    continue

                facts_text      = summary.get('key_facts', '')
                issues_text     = "; ".join(summary.get('legal_issues', []))
                statutes_text   = "; ".join(summary.get('key_statutes_cited', []))
                precedents_text = "; ".join(summary.get('key_precedents_cited', []))
                holding_text    = summary.get('holding', '')

                # Batch encode all 5 summary fields in one GPU call
                summary_texts = [facts_text, issues_text, statutes_text, precedents_text, holding_text]
                summary_embs  = model.encode(
                    summary_texts, batch_size=8, device=device, show_progress_bar=False
                )
                facts_emb, issues_emb, statutes_emb, precedents_emb, holding_emb = [
                    e.tolist() if text else None
                    for e, text in zip(summary_embs, summary_texts)
                ]

                cursor.execute("""
                    INSERT INTO ilpcsr_cases_embeddings
                        (doc_id, emb_model_id, facts_emb, issues_emb, statutes_emb, precedents_emb, holding_emb)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (case_id, emb_model_id, facts_emb, issues_emb, statutes_emb, precedents_emb, holding_emb))

                # Use chunk data from text phase if available; otherwise fall back to SELECT
                # (fallback path: text was already in DB from a previous interrupted run)
                if para_chunk_to_cid is not None:
                    chunk_ids   = list(para_chunk_to_cid.values())
                    chunk_texts = [ct for (_, ct) in para_chunk_to_cid.keys()]
                else:
                    cursor.execute("""
                        SELECT c.chunk_id, c.chunk_text
                        FROM ilpcsr_content_chunks c
                        JOIN ilpcsr_paragraphs p ON c.paragraph_id = p.paragraph_id
                        WHERE p.doc_id = %s;
                    """, (case_id,))
                    rows        = cursor.fetchall()
                    chunk_ids   = [r[0] for r in rows]
                    chunk_texts = [r[1] for r in rows]

                if chunk_texts:
                    chunk_embs = model.encode(
                        chunk_texts, batch_size=64, device=device, show_progress_bar=False
                    )
                    execute_values(
                        cursor,
                        "INSERT INTO ilpcsr_chunk_embeddings (chunk_id, emb_model_id, embedding) VALUES %s",
                        [(chunk_ids[i], emb_model_id, chunk_embs[i].tolist()) for i in range(len(chunk_ids))]
                    )

                print(f"   [Embed] Inserted {len(chunk_texts)} chunk embeddings")
                db_conn.commit()

            except Exception as e:
                if db_conn:
                    db_conn.rollback()
                msg = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {filename} - ERROR: {e}\n"
                print(f"   ERROR: {e} — skipping")
                with open(log_filepath, 'a', encoding='utf-8') as log_f:
                    log_f.write(msg)

        print(f"\nIngestion complete. Error log: {log_filepath}")
        print(f"Inserted IDs log:  {csv_filepath}")

    except psycopg2.Error as e:
        print(f"Critical DB error: {e}")
        if db_conn:
            db_conn.rollback()
    finally:
        if db_conn:
            db_conn.close()
        if csv_file:
            csv_file.close()


def build_vector_indexes(emb_model_id: int):
    db_conn = None
    try:
        db_conn = get_db_conn()
        cursor  = db_conn.cursor()

        cursor.execute("SELECT dimensions FROM ilpcsr_rag_config WHERE emb_model_id = %s", (emb_model_id,))
        result = cursor.fetchone()
        if not result:
            print("Config not found, cannot build indexes.")
            return

        dimensions = result[0]
        print(f"Building HNSW indexes (dims={dimensions})...")

        for col in ['facts_emb', 'issues_emb', 'statutes_emb', 'precedents_emb', 'holding_emb']:
            index_name = f"idx_cases_{col}_model_{emb_model_id}"
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON ilpcsr_cases_embeddings
                USING hnsw (({col}::vector({dimensions})) vector_cosine_ops)
                WHERE emb_model_id = {emb_model_id};
            """)
            print(f"   -> {index_name}")

        chunk_index = f"idx_chunks_emb_model_{emb_model_id}"
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {chunk_index}
            ON ilpcsr_chunk_embeddings
            USING hnsw ((embedding::vector({dimensions})) vector_cosine_ops)
            WHERE emb_model_id = {emb_model_id};
        """)
        print(f"   -> {chunk_index}")

        db_conn.commit()
        print("All indexes built.")

    except psycopg2.Error as e:
        print(f"DB error building indexes: {e}")
        if db_conn:
            db_conn.rollback()
    finally:
        if db_conn:
            db_conn.close()


if __name__ == "__main__":
    TARGET_EMB_MODEL_ID = 1
    DATASET_DIR         = DEFAULT_DATA   # override here if needed

    if not DATASET_DIR.is_dir():
        print(f"Dataset directory not found: {DATASET_DIR}")
        exit(1)

    device = detect_device()
    batch_ingest_from_directory(DATASET_DIR, TARGET_EMB_MODEL_ID, device)
    build_vector_indexes(TARGET_EMB_MODEL_ID)
