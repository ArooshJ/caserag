import os
import torch
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

_COLAB_SECRET_MAP = {
    "NEON_DB_NAME":     "DB_NAME",
    "NEON_DB_USER":     "DB_USER",
    "NEON_DB_PASSWORD": "DB_PASSWORD",
    "NEON_DB_HOST":     "DB_HOST",
    "NEON_DB_PORT":     "DB_PORT",
}

def _load_credentials():
    try:
        from google.colab import userdata
        for colab_key, env_key in _COLAB_SECRET_MAP.items():
            val = userdata.get(colab_key)
            if val:
                os.environ[env_key] = val
    except ImportError:
        load_dotenv()
    except Exception as e:
        print(f"Colab userdata unavailable: {e} — falling back to .env")
        load_dotenv()

_load_credentials()

DB_NAME     = os.environ.get("DB_NAME", "neondb")
DB_USER     = os.environ.get("DB_USER", "neondb_owner")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST     = os.environ.get("DB_HOST", "localhost")
DB_PORT     = os.environ.get("DB_PORT", "5432")


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
    print("No GPU — using CPU")
    return "cpu"


def embed_missing_chunks(emb_model_id: int, batch_size: int = 64, device: str = "cpu"):
    conn = get_db_conn()
    cur  = conn.cursor()

    cur.execute(
        "SELECT emb_model_name FROM ilpcsr_rag_config WHERE emb_model_id = %s",
        (emb_model_id,)
    )
    row = cur.fetchone()
    if not row:
        print(f"No config for emb_model_id={emb_model_id}. Aborting.")
        conn.close()
        return
    emb_model_name = row[0]
    print(f"Model: {emb_model_name}")

    # Content-grain repair: a piece of content is identified by (doc_id, full_text, chunk_text).
    # The invariant is that EVERY such content group has >= 1 embedded copy. Duplicate copies
    # (same content under another paragraph/case, or repeated within a paragraph) are allowed to
    # stay unembedded. So we find every content group with ZERO embedded copies and embed exactly
    # ONE representative chunk_id per group. This subsumes sole-copy chunks (a group of size one).
    cur.execute("""
        SELECT DISTINCT ON (p.doc_id, p.full_text, cc.chunk_text)
               cc.chunk_id, cc.chunk_text
        FROM   ilpcsr_content_chunks cc
        JOIN   ilpcsr_paragraphs p ON p.paragraph_id = cc.paragraph_id
        WHERE  (p.doc_id, p.full_text, cc.chunk_text) IN (
                   SELECT p2.doc_id, p2.full_text, cc2.chunk_text
                   FROM   ilpcsr_content_chunks cc2
                   JOIN   ilpcsr_paragraphs p2 ON p2.paragraph_id = cc2.paragraph_id
                   LEFT JOIN ilpcsr_chunk_embeddings ce
                          ON ce.chunk_id = cc2.chunk_id AND ce.emb_model_id = %s
                   GROUP  BY p2.doc_id, p2.full_text, cc2.chunk_text
                   HAVING COUNT(ce.chunk_emb_id) = 0
               )
        ORDER  BY p.doc_id, p.full_text, cc.chunk_text, cc.chunk_id
    """, (emb_model_id,))
    rows = cur.fetchall()

    if not rows:
        print("No zero-embedding content groups found — invariant already holds. Nothing to do.")
        conn.close()
        return

    chunk_ids   = [r[0] for r in rows]
    chunk_texts = [r[1] for r in rows]
    total       = len(chunk_ids)
    print(f"Zero-embedding content groups to repair (1 chunk each): {total}")

    print(f"Loading {emb_model_name} on {device}...")
    model = SentenceTransformer(emb_model_name, device=device)

    embedded = 0
    num_batches = (total + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_ids   = chunk_ids[i * batch_size : (i + 1) * batch_size]
        batch_texts = chunk_texts[i * batch_size : (i + 1) * batch_size]

        embs = model.encode(batch_texts, batch_size=batch_size, device=device, show_progress_bar=False)
        execute_values(
            cur,
            "INSERT INTO ilpcsr_chunk_embeddings (chunk_id, emb_model_id, embedding) VALUES %s",
            [(batch_ids[j], emb_model_id, embs[j].tolist()) for j in range(len(batch_ids))]
        )
        conn.commit()
        embedded += len(batch_ids)
        print(f"  Batch {i + 1}/{num_batches}: embedded {len(batch_ids)} chunks ({embedded}/{total} total)")

    print(f"\nDone. Embedded {embedded} chunks.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    TARGET_EMB_MODEL_ID = 1
    device = detect_device()
    embed_missing_chunks(TARGET_EMB_MODEL_ID, batch_size=64, device=device)
