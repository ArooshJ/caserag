"""
ilpcsr_multiquery_prep_pipeline.py - MULTI-QUERY generation for IL-PCSR.

WHY THIS EXISTS
    The single-query pipelines (ilpcsr_query_prep_pipeline.py / ..._batch_...) emit
    ONE blended query per case. But a judgment usually adjudicates SEVERAL distinct
    legal issues, each with its own (subset of) cited precedents. This script asks the
    LLM to enumerate the case's distinct legal issues and emit ONE focused search query
    PER ISSUE, so a case yields MULTIPLE queries that all share the same case-level
    precedent gold. That enables a multi-query / rank-fusion retrieval evaluation
    (see docs/methods/ilpcsr_paper_evaluation_protocol.md S8: union the gold = case citations,
    fuse the sub-query rankings, then score the paper's MAP/MRR/F1@K).

SCOPE
    Queries ONLY - no reference answers. Multi-query is a *retrieval* study; the gold is
    the case-level cited precedents, so per-sub-query reference answers aren't needed
    (a Phase-2 answer pass could be added later if generation eval is ever wanted).

PROVIDERS  (--provider)  — both use a Batch API by default ("batch if possible")
    gemini : **Gemini Batch API** (google-genai, JSONL) — same machinery as
             ilpcsr_batch_prep_pipeline.py. One request per case returns a JSON object of
             N issue-queries.
    novita : default. **Novita Batch API** (OpenAI-compatible, base .../openai/v1, endpoint
             /v1/batches, <=24h, cheaper + higher rate limit). Default model
             google/gemma-4-31b-it, as used for the paper's questions. `--no-batch` switches to
             online per-case calls (handy for a small smoke test without the 24h wait).

OUTPUT  (separate file - the single-query dataset is left untouched)
    evaluation/ilpcsr_multiquery_evaluation_dataset.json
    One entry PER SUB-QUERY: {source_case_id, sub_query_id, issue, question,
    ground_truth_cases (= case-level precedents), retrieved_cases, generated_answer,
    contexts}. Grouping by source_case_id is what later enables cluster-aware /
    fusion aggregation.

RESUME
    Each processed case is checkpointed by writing `multi_queries` into its
    test_queries/<id>.json; re-runs skip cases that already have it.

DRY RUN
    --dry-run previews the prompt / JSONL without any LLM call or upload.
"""

import os
import re
import sys
import json
import time
import argparse
from typing import List, Dict, Optional

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

# --- Field definitions (identical to the single-query pipelines) ---

QUERY_FIELDS = [
    "Fact",
    "Issue",
    "Petitioner's Argument",
    "Respondent's Argument",
]

# Fallback fields used when QUERY_FIELDS yield fewer than this many characters.
# Court's reasoning / precedent analysis carry the real legal issues even when the
# situation fields are scrubbed or procedural-only (11 of the 622 IL-PCSR test cases with questions).
FALLBACK_FIELDS = [
    "Analysis of the law",
    "Precedent Analysis",
    "Court's Reasoning",
    "Conclusion",
]
FALLBACK_THRESHOLD = 200  # chars — below this, try FALLBACK_FIELDS

# --- Model / provider configuration ---

GOOGLE_MODELS = [
    "gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3-flash-preview",
]

# Example Novita model ids — NOT authoritative. `--model` accepts any string and the
# provider is chosen by `--provider`, so unknown ids still work. VERIFY exact ids and
# the base URL against Novita's live catalog before a real run.
NOVITA_MODELS_EXAMPLES = [
    "google/gemma-4-31b-it",   # the model used for the paper's questions
    "deepseek/deepseek-v3-0324",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
]
# Novita is OpenAI-compatible AND exposes an OpenAI-style Batch API. Confirmed base for
# both chat + batch is .../openai/v1 (batches endpoint = .../openai/v1/batches).
# Source: novita.ai/docs/guides/llm-batch-api. Env-overridable.
NOVITA_BASE_URL = os.environ.get("NOVITA_BASE_URL", "https://api.novita.ai/openai/v1")

DEFAULT_NOVITA_MODEL = "google/gemma-4-31b-it"   # Gemma 4 (31B), as used for the paper's questions
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.3   # slight headroom for distinct-issue diversity

# --- Path / limit defaults ---

QUERIES_DIR     = os.path.join("il-pcsr_dataset", "test_queries")
CORPUS_DIR      = os.path.join("il-pcsr_dataset", "corpus")
OUTPUT_FILE     = os.path.join("evaluation", "ilpcsr_multiquery_evaluation_dataset.json")
AGENT_WORKSPACE = ".agent_workspace"
CHECKPOINT_KEY  = "multi_queries"

# BAAI/bge-large-en-v1.5 has a 512-token window (~365 words); keep each query concise.
MAX_QUERY_WORDS    = 100
MAX_QUERIES_PER_CASE_DEFAULT = 6   # cost/quality cap; most cases raise <=4 distinct issues


# --- Prompt builder (adjusted for multi-issue enumeration) ---

def build_multiquery_messages(query_field_text: str, max_q: int) -> List[Dict]:
    """Build the multi-issue prompt. Asks for a STRICT JSON object
    {"queries": [{"issue": "...", "query": "..."}]} so the one response parses into N
    entries, and the object shape is compatible with both Gemini JSON mode and OpenAI
    json_object mode. Keeps the single-query pipeline's anonymization rules so queries
    can't leak party/case identities."""
    system_prompt = (
        "You are a legal research assistant. A single judgment usually decides SEVERAL "
        "DISTINCT legal issues. Identify each distinct legal issue or question in the "
        "case material below, and for EACH one produce a concise natural-language query "
        "a lawyer would use to search for relevant prior cases on THAT issue.\n"
        "Rules: "
        "(1) Produce one query per genuinely distinct legal issue. If the case raises "
        f"only one issue, return one query. Return AT MOST {max_q} queries; merge minor "
        "overlapping points rather than padding the count. "
        "(2) Do NOT hint at the court's eventual decision or outcome. "
        "(3) Do NOT include names of any parties, companies, institutions, or prior "
        "cases/judgments — refer to their role or the legal principle instead "
        "(e.g. 'an earlier precedent on dishonored cheques'). "
        "(4) Each query frames an OPEN legal situation where multiple cases could "
        f"plausibly be relevant, and is under {MAX_QUERY_WORDS} words STRICTLY. "
        "(5) If the text has placeholder tokens like [SECTION], [ACT] or [jurisdiction], "
        "do NOT echo them and do NOT invent names/numbers; describe the provision "
        "functionally (e.g. 'the applicable income tax penalty section'). "
        "(6) Return ONLY strict JSON, no markdown, no prose, in exactly this shape: "
        '{"queries": [{"issue": "<short issue label>", "query": "<query text>"}]}'
    )
    user_prompt = (
        "Enumerate the distinct legal issues in the following case material and emit one "
        "search query per issue as specified:\n\n"
        f"{query_field_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


# --- Output sanitisation (placeholder cleanup; copied from the batch pipeline) ---

_SANITISE_RULES = [
    (re.compile(r"\[SECTION\](?:\s*,\s*\[SECTION\])+\s+of\s+(?:the\s+)?\[ACT\]"),
     "the relevant statutory provisions"),
    (re.compile(r"\[SECTION\]\s+of\s+(?:the\s+)?\[ACT\]"), "the relevant statutory provision"),
    (re.compile(r"\[SECTION\]\s+\[ACT\]"), "the relevant statutory provision"),
    (re.compile(r"\[ACT\]\s+\[SECTION\]"), "the relevant statutory provision"),
    (re.compile(r"[Ss]ections?\s+\[SECTION\]"), "the relevant statutory provision"),
    (re.compile(r"s\.\s*\[SECTION\]"), "the relevant statutory provision"),
    (re.compile(r"\[SECTION\]"), "the relevant statutory provision"),
    (re.compile(r"\[ACT\]"), "the applicable act"),
    (re.compile(r"\[jurisdiction\]"), "the applicable jurisdiction"),
    (re.compile(r"\[ENTITY\]"), "a party"),
    (re.compile(r"\[PRECEDENT\]"), "a cited precedent"),
    (re.compile(r"\[CASE NUMBER\]"), ""),
]

def _sanitise_output(text: str) -> str:
    for pat, repl in _SANITISE_RULES:
        text = pat.sub(repl, text)
    text = re.sub(r"\bthe\s+the\b", "the", text, flags=re.IGNORECASE)
    return text.strip()


# --- Response parsing ---

def parse_queries(raw_text: str, max_q: int) -> List[Dict]:
    """Turn a model response into a clean list of {"issue","query"} dicts. Robust to
    ```json fences, a bare array instead of the object, or a plain-text single query —
    a parse failure must never crash a batch, only yield fewer/zero entries for one case."""
    if not raw_text:
        return []
    txt = raw_text.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt).strip()  # strip code fences

    items = None
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
            items = obj["queries"]
        elif isinstance(obj, list):
            items = obj
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", txt, re.DOTALL)  # last-ditch: grab a JSON blob
        if m:
            try:
                obj = json.loads(m.group())
                items = obj.get("queries") if isinstance(obj, dict) else obj
            except Exception:
                items = None

    out: List[Dict] = []
    seen = set()
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                q = str(it.get("query", "")).strip()
                issue = str(it.get("issue", "")).strip()
            else:
                q, issue = str(it).strip(), ""
            q = _sanitise_output(q)
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append({"issue": issue, "query": q})
    else:
        # Not JSON at all — treat the whole thing as a single query (degraded fallback).
        q = _sanitise_output(txt)
        if q:
            out.append({"issue": "", "query": q})

    return out[:max_q]


# --- Field / file helpers ---

def extract_query_text(doc: Dict) -> str:
    """Extract situation text for the query generator. Primary: QUERY_FIELDS (Fact/Issue/Arguments).
    Fallback: reasoning fields when primary yields < FALLBACK_THRESHOLD chars — handles cases
    where situation fields are scrubbed/procedural-only but court reasoning has real legal content."""
    def _join(keys):
        parts = []
        for key in keys:
            val = doc.get(key)
            if isinstance(val, list):
                parts.extend(val)
            elif isinstance(val, str) and val.strip():
                parts.append(val.strip())
        return " ".join(parts).strip()

    primary = _join(QUERY_FIELDS)
    if len(primary) >= FALLBACK_THRESHOLD:
        return primary
    fallback = _join(FALLBACK_FIELDS)
    return (primary + " " + fallback).strip() if primary else fallback


def build_entries(doc: Dict, parsed: List[Dict]) -> List[Dict]:
    """One eval entry per sub-query, all sharing the case-level precedent gold and the
    source_case_id (the grouping key for later fusion). sub_query_id keeps them distinct."""
    case_id = str(doc.get("caseID", ""))
    gold = [str(pid) for pid in doc.get("relevant_precedent_ids", [])]
    entries = []
    for i, pq in enumerate(parsed):
        entries.append({
            "source_case_id":     case_id,
            "sub_query_id":       f"{case_id}::{i}",
            "issue":              pq.get("issue", ""),
            "question":           pq["query"],
            "ground_truth_cases": gold,
            "retrieved_cases":    [],
            "generated_answer":   "",
            "contexts":           [],
        })
    return entries


def append_entries(entries: List[Dict], output_file: str) -> int:
    """Append sub-query entries, deduped by (source_case_id, question). Returns count added."""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except Exception:
            dataset = []
    else:
        dataset = []

    existing = {(e.get("source_case_id"), e.get("question")) for e in dataset}
    added = 0
    for e in entries:
        key = (e["source_case_id"], e["question"])
        if key in existing:
            continue
        existing.add(key)
        dataset.append(e)
        added += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    return added


def checkpoint_case(filepath: str, parsed: List[Dict]) -> None:
    """Persist the parsed multi-queries into the source doc so re-runs skip this case."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc[CHECKPOINT_KEY] = parsed
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [WARN] checkpoint write failed for {os.path.basename(filepath)}: {e}")


def scan_unprocessed(queries_dir: str, n: int) -> List[str]:
    """Cases without a non-empty multi_queries checkpoint, first n (sorted)."""
    all_files = sorted(f for f in os.listdir(queries_dir) if f.endswith(".json"))
    todo = []
    for fn in all_files:
        try:
            with open(os.path.join(queries_dir, fn), "r", encoding="utf-8") as f:
                doc = json.load(f)
            if not doc.get(CHECKPOINT_KEY):
                todo.append(fn)
        except Exception:
            todo.append(fn)
    return todo[:n]


# ============================ GEMINI BATCH-API PATH ============================
# Mirrors ilpcsr_batch_prep_pipeline.py: JSONL keyed by filename, matched by key dict
# (not position), single phase (queries only).

def _gemini_request(messages: List[Dict]) -> Dict:
    """Fold system+user into one user turn and force JSON output. JSONL batch requests
    have no top-level systemInstruction, so the system text is prepended."""
    system_text = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_text   = next((m["content"] for m in messages if m["role"] == "user"),   "")
    combined = f"{system_text}\n\n{user_text}" if system_text else user_text
    return {
        "contents": [{"role": "user", "parts": [{"text": combined}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def _poll_until_done(client, job):
    while not job.done:
        print(f"  [poll] {job.state} — waiting 30s...")
        time.sleep(30)
        job = client.batches.get(name=job.name)
    return job


def _run_gemini_batch(model_name: str, batch: List[str], queries_dir: str,
                      output_file: str, max_q: int, dry_run: bool) -> int:
    from google import genai

    keyed_requests: Dict[str, Dict] = {}
    for fn in batch:
        try:
            with open(os.path.join(queries_dir, fn), "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {fn}: load error — {e}")
            continue
        text = extract_query_text(doc)
        if not text:
            print(f"  [SKIP] {fn}: no content in query fields.")
            continue
        keyed_requests[fn] = _gemini_request(build_multiquery_messages(text, max_q))

    if not keyed_requests:
        print("  Nothing to submit.")
        return 0

    jsonl_path = os.path.join(AGENT_WORKSPACE, "ilpcsr-multiquery.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for key, request in keyed_requests.items():
            f.write(json.dumps({"key": key, "request": request}) + "\n")

    first_key = next(iter(keyed_requests))
    print(f"  JSONL written: {jsonl_path}  ({len(keyed_requests)} requests)")
    print(f"  Preview key '{first_key}':\n"
          f"  {keyed_requests[first_key]['contents'][0]['parts'][0]['text'][:240]}...")
    if dry_run:
        print(f"  [DRY RUN] not uploading. Inspect: {jsonl_path}")
        return 0

    client = genai.Client()
    print(f"  Uploading...")
    uploaded = client.files.upload(file=jsonl_path, config={"mime_type": "application/jsonl"})
    job = client.batches.create(model=f"models/{model_name}", src=uploaded.name,
                                config={"display_name": "ilpcsr-multiquery"})
    print(f"  Job: {job.name}")
    job = _poll_until_done(client, job)
    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"  [ERROR] job state: {job.state}")
        return 0

    result_file = client.files.get(name=job.dest.file_name)
    content = client.files.download(file=result_file).decode("utf-8")
    total_added = 0
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = obj.get("key", "")
        if not key or "response" not in obj:
            if "error" in obj:
                print(f"  [WARN] '{key}' failed: {obj['error'].get('message')}")
            continue
        try:
            text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            print(f"  [WARN] no text for '{key}'")
            continue
        parsed = parse_queries(text, max_q)
        if not parsed:
            print(f"  [WARN] '{key}' parsed 0 queries")
            continue
        checkpoint_case(os.path.join(queries_dir, key), parsed)
        added = append_entries(build_entries(json.load(open(os.path.join(queries_dir, key), encoding="utf-8")), parsed), output_file)
        total_added += added
        print(f"  [✓] {key}: {len(parsed)} queries, +{added} entries")

    try:
        os.remove(jsonl_path)
    except Exception:
        pass
    return total_added


# ==================== NOVITA PATH (OpenAI-compatible chat + Batch API) ====================
# Novita exposes online chat AND an OpenAI-style Batch API (base .../openai/v1). Batch is the
# default (cheaper, higher rate limit, <=24h); --no-batch uses online per-case calls.

def _get_novita_client():
    """Build the OpenAI client pointed at Novita. Lazy import so a Gemini-only env needn't
    have `openai` installed; returns None (with a printed reason) on any missing piece."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  [FAIL] `openai` not installed (needed for --provider novita).")
        return None
    api_key = os.environ.get("NOVITA_API_KEY")
    if not api_key:
        print("  [FAIL] NOVITA_API_KEY not set.")
        return None
    return OpenAI(base_url=NOVITA_BASE_URL, api_key=api_key)


def _run_novita_direct(model_name: str, batch: List[str], queries_dir: str,
                       output_file: str, max_q: int, dry_run: bool) -> int:
    """Online per-case calls — the --no-batch fallback (good for a 2-case smoke test without
    the 24h batch wait). json_object response format pairs with the prompt's {"queries":[...]}."""
    client = None if dry_run else _get_novita_client()
    if not dry_run and client is None:
        return 0
    total_added = 0
    for i, fn in enumerate(batch, 1):
        filepath = os.path.join(queries_dir, fn)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {fn}: load error — {e}")
            continue
        text = extract_query_text(doc)
        if not text:
            print(f"  [SKIP] {fn}: no query-field content.")
            continue
        messages = build_multiquery_messages(text, max_q)
        print(f"  [{i}/{len(batch)}] {fn}")
        if dry_run:
            print(f"    [DRY RUN] prompt excerpt: {messages[0]['content'][:160]}...")
            continue
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages, temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            print(f"    [WARN] Novita call failed: {e}")
            continue
        parsed = parse_queries(raw or "", max_q)
        if not parsed:
            print(f"    [WARN] parsed 0 queries.")
            continue
        checkpoint_case(filepath, parsed)
        added = append_entries(build_entries(doc, parsed), output_file)
        total_added += added
        print(f"    [✓] {len(parsed)} queries, +{added} entries")
    return total_added


def _run_novita_batch(model_name: str, batch: List[str], queries_dir: str,
                      output_file: str, max_q: int, dry_run: bool) -> int:
    """Novita Batch API (OpenAI-compatible JSONL): one /v1/chat/completions request per case,
    keyed by custom_id = filename and matched by id on return. Mirrors the Gemini batch
    contract. NOTE: field shapes below follow the documented OpenAI batch standard; validate
    the response wrapper on the first real run (providers occasionally vary it)."""
    lines = []
    for fn in batch:
        try:
            with open(os.path.join(queries_dir, fn), "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {fn}: load error — {e}")
            continue
        text = extract_query_text(doc)
        if not text:
            print(f"  [SKIP] {fn}: no query-field content.")
            continue
        lines.append({
            "custom_id": fn,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": build_multiquery_messages(text, max_q),
                "temperature": TEMPERATURE,
                "response_format": {"type": "json_object"},
            },
        })

    if not lines:
        print("  Nothing to submit.")
        return 0

    jsonl_path = os.path.join(AGENT_WORKSPACE, "ilpcsr-multiquery-novita.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    print(f"  JSONL written: {jsonl_path}  ({len(lines)} requests)")
    print(f"  Preview custom_id '{lines[0]['custom_id']}':\n"
          f"  {lines[0]['body']['messages'][1]['content'][:200]}...")
    if dry_run:
        print(f"  [DRY RUN] not uploading. Inspect: {jsonl_path}")
        return 0

    client = _get_novita_client()
    if client is None:
        return 0

    with open(jsonl_path, "rb") as fh:
        batch_file = client.files.create(file=fh, purpose="batch")
    job = client.batches.create(input_file_id=batch_file.id,
                                endpoint="/v1/chat/completions",
                                completion_window="24h")
    print(f"  Batch job: {job.id}  (status {job.status})")
    terminal = {"completed", "failed", "expired", "cancelled"}
    while job.status not in terminal:
        print(f"  [poll] {job.status} — waiting 60s...")
        time.sleep(60)
        job = client.batches.retrieve(job.id)
    if job.status != "completed":
        print(f"  [ERROR] batch ended: {job.status}")
        return 0

    resp = client.files.content(job.output_file_id)
    content = getattr(resp, "text", None) or resp.read().decode("utf-8")
    total_added = 0
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = obj.get("custom_id", "")
        try:
            raw = obj["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            print(f"  [WARN] no text for '{cid}' (error: {obj.get('error')})")
            continue
        parsed = parse_queries(raw, max_q)
        if not parsed:
            print(f"  [WARN] '{cid}' parsed 0 queries")
            continue
        checkpoint_case(os.path.join(queries_dir, cid), parsed)
        doc = json.load(open(os.path.join(queries_dir, cid), encoding="utf-8"))
        added = append_entries(build_entries(doc, parsed), output_file)
        total_added += added
        print(f"  [✓] {cid}: {len(parsed)} queries, +{added} entries")

    try:
        os.remove(jsonl_path)
    except Exception:
        pass
    return total_added


# --- Entry point ---

def main():
    ap = argparse.ArgumentParser(description="Multi-query (per-issue) generation for IL-PCSR.")
    ap.add_argument("--provider", choices=["gemini", "novita"], default="novita")
    ap.add_argument("--model", type=str, default=None,
                    help="model id (default: google/gemma-4-31b-it for novita, as in the paper; "
                         "gemini-2.5-flash-lite for gemini)")
    ap.add_argument("--n", type=int, default=2, help="cases to process this run")
    ap.add_argument("--max-queries", type=int, default=MAX_QUERIES_PER_CASE_DEFAULT,
                    help="max distinct queries per case")
    ap.add_argument("--output", type=str, default=OUTPUT_FILE)
    ap.add_argument("--queries-dir", type=str, default=QUERIES_DIR)
    ap.add_argument("--no-batch", action="store_true",
                    help="gemini: use direct calls instead of the Batch API")
    ap.add_argument("--dry-run", action="store_true",
                    help="preview prompt/JSONL without any LLM call or upload")
    args = ap.parse_args()

    model = args.model or (DEFAULT_GEMINI_MODEL if args.provider == "gemini" else DEFAULT_NOVITA_MODEL)

    batch = scan_unprocessed(args.queries_dir, args.n)
    if not batch:
        print("No unprocessed cases found.")
        return
    print(f"Provider={args.provider}  model={model}  cases={len(batch)}  "
          f"max_queries/case={args.max_queries}  output={args.output}")
    if args.dry_run:
        print("[DRY RUN] no LLM calls / uploads.\n")

    a = (model, batch, args.queries_dir, args.output, args.max_queries, args.dry_run)
    if args.provider == "novita":
        added = _run_novita_direct(*a) if args.no_batch else _run_novita_batch(*a)
    else:  # gemini
        if args.no_batch:
            print("[gemini] --no-batch not implemented; omit it to use the Batch API, "
                  "or use `--provider novita --no-batch` for online calls.")
            return
        added = _run_gemini_batch(*a)

    if not args.dry_run:
        print(f"\nDone: +{added} sub-query entries appended to {args.output}")


if __name__ == "__main__":
    main()
