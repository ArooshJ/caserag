"""
ilpcsr_query_faithfulness_judge.py - LLM-as-judge for QUERY FAITHFULNESS.

WHY THIS EXISTS
    CaseRAG's eval queries are LLM-generated from each case's situation fields. Threat
    #2 to validity is: are those queries FAITHFUL to the source, or do they hallucinate
    issues / leak the outcome / leak a party's identity? The leakage audit already proved
    0/615 *identity* leakage lexically; this script judges the harder, semantic half:
    does each query stay grounded in the situation it was generated from, raise a legal
    issue genuinely present, and avoid telegraphing the court's decision?

    A bare LLM-judge is the WEAKEST kind of evidence on its own. So this tool is built to
    be *calibrated*: export a hand-rating sample, rate ~30 yourself, then score the judge's
    agreement with you. The agreement number is what makes the judge credible in the paper.

WHAT IT JUDGES (per query)
    Input the judge sees = the case's situation fields (Fact / Issue / Petitioner's /
    Respondent's Argument — exactly what the generator saw) + the generated query.
    Output per query (strict JSON):
      verdict        : faithful | partial | unfaithful
      grounded       : every issue/fact in the query is supported by the situation
      leaks_outcome  : query reveals/hints the court's decision or holding
      leaks_identity : query names a specific party / case / precedent
      on_topic       : query asks about an issue genuinely present in the case
      justification  : <= 25 words
    `leaks_identity` is an independent cross-check on the lexical leakage audit (expect ~0).

DATASET-AGNOSTIC
    --input is either ilpcsr_evaluation_dataset.json (single-query) or the multi-query
    dataset; both carry source_case_id + question, and situation fields come from
    test_queries/<source_case_id>.json. sub_query_id / issue are carried through if present.

PROVIDERS  (--provider) — use a DIFFERENT family than the generator (Gemma 4 31B in the paper)
    novita : default. Novita Batch API (OpenAI-compatible, base .../openai/v1). Default model
             minimax/minimax-m3, the judge used in the paper (a different family from the
             Gemma generator). openai/gpt-oss-120b is a non-reasoning alternative. A reasoning
             model (e.g. deepseek-r1-0528) is available via --model but emits far more output
             tokens; deliberately NOT the default.
    gemini : Gemini Batch API (default gemini-2.5-flash), for a second-opinion / cross-judge run.
    --no-batch : online per-case calls (novita) for a quick smoke test.

MODES
    (default)               judge --input, write results + summary.
    --calibration-sample N  export N random (situation, query) pairs with blank human fields
                            for you to hand-rate (no LLM call).
    --agreement HUMAN_FILE  compare a filled hand-rated file against a judge-results file and
                            print agreement (exact + collapsed faithful-vs-not).
"""

import os
import re
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Optional

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

# Situation fields the query was generated from — the judge must see exactly these to
# assess groundedness (same list the generators use).
QUERY_FIELDS = ["Fact", "Issue", "Petitioner's Argument", "Respondent's Argument"]

# Kept in sync with ilpcsr_multiquery_prep_pipeline.py so both scripts surface the same model
# menu. Any id works via --model; the provider is chosen by --provider.
GOOGLE_MODELS = [
    "gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3-flash-preview",
]
# Novita ids. The default BATCH path needs a CONFIRMED batch-eligible id (per the Novita
# batch-API docs) — those are NOVITA_BATCH_MODELS, listed first and all NON-reasoning (a verdict
# task needs no chain-of-thought; reasoning ids like deepseek-r1-0528 emit far more output
# tokens, so they are never the default). The remaining ids mirror the multi-query script's menu
# and are usable via --no-batch (online); verify their batch-eligibility before batch use.
NOVITA_BATCH_MODELS = [
    "minimax/minimax-m3", "openai/gpt-oss-120b", "qwen/qwen3-vl-235b-a22b-instruct", "qwen/qwen3-4b-fp8",
]
NOVITA_MODELS_EXAMPLES = NOVITA_BATCH_MODELS + [
    "deepseek/deepseek-v3-0324", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b",
]
NOVITA_BASE_URL = os.environ.get("NOVITA_BASE_URL", "https://api.novita.ai/openai/v1")

DEFAULT_NOVITA_MODEL = "minimax/minimax-m3"   # the judge used in the paper; same as the corpus-summary judge
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0   # a judge should be as deterministic as the API allows

QUERIES_DIR     = os.path.join("il-pcsr_dataset", "test_queries")
DEFAULT_INPUT   = os.path.join("evaluation", "ilpcsr_evaluation_dataset.json")
DEFAULT_OUTPUT  = os.path.join("evaluation", "results", "query_faithfulness_results.json")
CALIB_TEMPLATE  = os.path.join("evaluation", "results", "query_faithfulness_calibration_template.json")
AGENT_WORKSPACE = ".agent_workspace"

MAX_SITUATION_CHARS = 24000   # ~6k tokens; bounds outlier cases (p95 situation ~4k tokens)
VALID_VERDICTS = {"faithful", "partial", "unfaithful"}


# --- Source loading ---

def load_situation(source_case_id: str, queries_dir: str, max_chars: int) -> Optional[str]:
    """Pull the situation fields (Fact/Issue/Arguments) from test_queries/<id>.json and
    join them. Returns None if the source doc is missing or empty — the caller counts those
    as skipped rather than judging a query against nothing."""
    fp = os.path.join(queries_dir, f"{source_case_id}.json")
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return None
    parts = []
    for k in QUERY_FIELDS:
        v = doc.get(k)
        if isinstance(v, list):
            parts.append(f"## {k}\n" + " ".join(str(x) for x in v))
    text = "\n\n".join(parts).strip()
    if not text:
        return None
    return text[:max_chars]


# --- Prompt ---

def build_judge_messages(situation_text: str, query: str) -> List[Dict]:
    """Strict-evaluator prompt. Asks for one JSON object so it pairs with both Gemini JSON
    mode and OpenAI json_object mode and parses to exactly one verdict per query."""
    system_prompt = (
        "You are a strict legal-NLP evaluator. You are given a case's SITUATION fields "
        "(Fact, Issue, and the parties' arguments) and a SEARCH QUERY that was "
        "auto-generated from them for retrieving relevant prior cases. Judge whether the "
        "query is FAITHFUL to the situation.\n"
        "Assess: (a) grounded — every legal issue or fact asserted in the query is supported "
        "by the situation, with nothing invented; (b) on_topic — the query targets a legal "
        "issue genuinely present in the case; (c) leaks_outcome — whether the query reveals "
        "or hints at the court's eventual decision/holding (it must NOT); (d) leaks_identity "
        "— whether the query names a specific party, company, or prior case/precedent (it "
        "must NOT; describing a role or principle is fine).\n"
        "verdict = 'faithful' if grounded and on_topic with no leaks; 'partial' if it mostly "
        "holds but has minor drift or one weak issue; 'unfaithful' if it invents content, "
        "leaks the outcome, names an identity, or is off-topic.\n"
        "Return ONLY strict JSON, no prose, exactly: "
        '{"verdict":"faithful|partial|unfaithful","grounded":true,"on_topic":true,'
        '"leaks_outcome":false,"leaks_identity":false,"justification":"<=25 words"}'
    )
    user_prompt = (
        f"SITUATION:\n{situation_text}\n\n"
        f"GENERATED QUERY:\n{query}\n\n"
        "Judge the query now."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


# --- Response parsing ---

def parse_verdict(raw: str) -> Dict:
    """Coerce a model response into the verdict dict; a parse failure yields
    verdict='unparsed' (counted, never crashes a batch)."""
    base = {"verdict": "unparsed", "grounded": None, "on_topic": None,
            "leaks_outcome": None, "leaks_identity": None, "justification": ""}
    if not raw:
        return base
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()).strip()
    obj = None
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return base
    v = str(obj.get("verdict", "")).strip().lower()
    base["verdict"] = v if v in VALID_VERDICTS else "unparsed"
    for k in ("grounded", "on_topic", "leaks_outcome", "leaks_identity"):
        base[k] = obj.get(k)
    base["justification"] = str(obj.get("justification", ""))[:300]
    return base


# --- Entry iteration / output ---

def iter_judgeable(records: List[Dict], queries_dir: str, max_chars: int):
    """Yield (index, record, situation_text) for records with a question and a loadable
    situation; print/skip the rest."""
    for i, item in enumerate(records):
        q = (item.get("question") or "").strip()
        sid = str(item.get("source_case_id", ""))
        if not q or not sid:
            continue
        sit = load_situation(sid, queries_dir, max_chars)
        if not sit:
            continue
        yield i, item, sit


def make_record(item: Dict, verdict: Dict) -> Dict:
    """One output row: carry source/sub-query identifiers + the verdict fields."""
    out = {
        "source_case_id": str(item.get("source_case_id", "")),
        "question":       item.get("question", ""),
    }
    if "sub_query_id" in item:
        out["sub_query_id"] = item["sub_query_id"]
    if "issue" in item:
        out["issue"] = item["issue"]
    out.update(verdict)
    return out


def summarize(records: List[Dict]) -> Dict:
    """Verdict distribution + flag rates — the headline the threats-section quotes."""
    n = len(records)
    vc = {"faithful": 0, "partial": 0, "unfaithful": 0, "unparsed": 0}
    flags = {"leaks_outcome": 0, "leaks_identity": 0, "not_grounded": 0, "off_topic": 0}
    for r in records:
        vc[r.get("verdict", "unparsed")] = vc.get(r.get("verdict", "unparsed"), 0) + 1
        if r.get("leaks_outcome") is True:
            flags["leaks_outcome"] += 1
        if r.get("leaks_identity") is True:
            flags["leaks_identity"] += 1
        if r.get("grounded") is False:
            flags["not_grounded"] += 1
        if r.get("on_topic") is False:
            flags["off_topic"] += 1
    pct = lambda x: round(100.0 * x / n, 1) if n else 0.0
    return {
        "judged": n,
        "verdicts": vc,
        "verdict_pct": {k: pct(v) for k, v in vc.items()},
        "flag_counts": flags,
        "flag_pct": {k: pct(v) for k, v in flags.items()},
    }


def write_output(records: List[Dict], output_file: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    payload = {"_summary": summarize(records), "records": records}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_summary(records: List[Dict]) -> None:
    s = summarize(records)
    print("\n========== QUERY FAITHFULNESS SUMMARY ==========")
    print(f"judged: {s['judged']}")
    for k in ("faithful", "partial", "unfaithful", "unparsed"):
        print(f"  {k:<11} {s['verdicts'][k]:>5}  ({s['verdict_pct'][k]}%)")
    print("flags (count / %):")
    for k, v in s["flag_counts"].items():
        print(f"  {k:<14} {v:>5}  ({s['flag_pct'][k]}%)")
    print("================================================")


# ============================ GEMINI BATCH-API PATH ============================

def _gemini_request(messages: List[Dict]) -> Dict:
    system_text = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_text   = next((m["content"] for m in messages if m["role"] == "user"),   "")
    combined = f"{system_text}\n\n{user_text}" if system_text else user_text
    return {
        "contents": [{"role": "user", "parts": [{"text": combined}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": TEMPERATURE},
    }


def _run_gemini_batch(model_name, work, output_file, dry_run) -> int:
    """work = list of (key, item, messages). Key = sub_query_id or source_case_id."""
    from google import genai
    jsonl_path = os.path.join(AGENT_WORKSPACE, "ilpcsr-faithfulness.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
    item_by_key = {k: it for k, it, _ in work}
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for key, _, messages in work:
            f.write(json.dumps({"key": key, "request": _gemini_request(messages)}) + "\n")
    print(f"  JSONL written: {jsonl_path}  ({len(work)} requests)")
    if dry_run:
        print(f"  [DRY RUN] not uploading. Inspect: {jsonl_path}")
        return 0

    client = genai.Client()
    uploaded = client.files.upload(file=jsonl_path, config={"mime_type": "application/jsonl"})
    job = client.batches.create(model=f"models/{model_name}", src=uploaded.name,
                                config={"display_name": "ilpcsr-faithfulness"})
    print(f"  Job: {job.name}")
    while not job.done:
        print(f"  [poll] {job.state} — waiting 30s...")
        time.sleep(30)
        job = client.batches.get(name=job.name)
    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"  [ERROR] job state: {job.state}")
        return 0

    result_file = client.files.get(name=job.dest.file_name)
    content = client.files.download(file=result_file).decode("utf-8")
    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = obj.get("key", "")
        try:
            text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            continue
        if key in item_by_key:
            records.append(make_record(item_by_key[key], parse_verdict(text)))
    write_output(records, output_file)
    try:
        os.remove(jsonl_path)
    except Exception:
        pass
    print_summary(records)
    return len(records)


# ============================ NOVITA PATH (OpenAI-compatible) ============================

def _get_novita_client():
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


def _run_novita_batch(model_name, work, output_file, dry_run) -> int:
    item_by_key = {k: it for k, it, _ in work}
    jsonl_path = os.path.join(AGENT_WORKSPACE, "ilpcsr-faithfulness-novita.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for key, _, messages in work:
            f.write(json.dumps({
                "custom_id": key, "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": model_name, "messages": messages,
                         "temperature": TEMPERATURE, "response_format": {"type": "json_object"}},
            }) + "\n")
    print(f"  JSONL written: {jsonl_path}  ({len(work)} requests)")
    if dry_run:
        print(f"  [DRY RUN] not uploading. Inspect: {jsonl_path}")
        return 0

    client = _get_novita_client()
    if client is None:
        return 0
    with open(jsonl_path, "rb") as fh:
        batch_file = client.files.create(file=fh, purpose="batch")
    job = client.batches.create(input_file_id=batch_file.id,
                                endpoint="/v1/chat/completions", completion_window="24h")
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
    records = []
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
            continue
        if cid in item_by_key:
            records.append(make_record(item_by_key[cid], parse_verdict(raw)))
    write_output(records, output_file)
    try:
        os.remove(jsonl_path)
    except Exception:
        pass
    print_summary(records)
    return len(records)


def _run_novita_direct(model_name, work, output_file, dry_run) -> int:
    client = None if dry_run else _get_novita_client()
    if not dry_run and client is None:
        return 0
    records = []
    for i, (key, item, messages) in enumerate(work, 1):
        print(f"  [{i}/{len(work)}] {key}")
        if dry_run:
            print(f"    [DRY RUN] {messages[1]['content'][:120]}...")
            continue
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages, temperature=TEMPERATURE,
                response_format={"type": "json_object"})
            raw = resp.choices[0].message.content
        except Exception as e:
            print(f"    [WARN] call failed: {e}")
            continue
        records.append(make_record(item, parse_verdict(raw)))
    if not dry_run:
        write_output(records, output_file)
        print_summary(records)
    return len(records)


# --- Calibration helpers ---

def export_calibration(records: List[Dict], queries_dir: str, n: int, max_chars: int) -> None:
    """Write N random (situation, query) pairs with blank human fields for hand-rating —
    the gold-standard half of the calibration the paper needs."""
    pool = list(iter_judgeable(records, queries_dir, max_chars))
    random.seed(42)
    sample = random.sample(pool, min(n, len(pool)))
    rows = []
    for _, item, sit in sample:
        rows.append({
            "source_case_id": str(item.get("source_case_id", "")),
            "sub_query_id":   item.get("sub_query_id", ""),
            "question":       item.get("question", ""),
            "situation":      sit,
            "human_verdict":  "",      # fill: faithful | partial | unfaithful
            "human_notes":    "",
        })
    os.makedirs(os.path.dirname(os.path.abspath(CALIB_TEMPLATE)), exist_ok=True)
    with open(CALIB_TEMPLATE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} calibration rows -> {CALIB_TEMPLATE}\n"
          f"Hand-rate `human_verdict`, then: --agreement {CALIB_TEMPLATE}")


def score_agreement(human_file: str, judge_file: str) -> None:
    """Compare hand ratings against judge verdicts (matched on source_case_id+question):
    exact 3-way agreement and collapsed faithful-vs-not. This is the credibility number."""
    human = json.load(open(human_file, encoding="utf-8"))
    judged = json.load(open(judge_file, encoding="utf-8")).get("records", [])
    jmap = {(r["source_case_id"], r["question"]): r.get("verdict") for r in judged}

    matched = exact = collapsed = 0
    collapse = lambda v: "faithful" if v == "faithful" else "not"
    for h in human:
        hv = str(h.get("human_verdict", "")).strip().lower()
        if hv not in VALID_VERDICTS:
            continue
        jv = jmap.get((str(h.get("source_case_id", "")), h.get("question", "")))
        if jv is None:
            continue
        matched += 1
        if hv == jv:
            exact += 1
        if collapse(hv) == collapse(jv):
            collapsed += 1
    if not matched:
        print("No matched + rated rows. Did you fill human_verdict and pass the right judge file?")
        return
    print(f"Matched rated rows : {matched}")
    print(f"Exact agreement    : {exact}/{matched}  ({100.0*exact/matched:.1f}%)")
    print(f"Collapsed (faithful vs not): {collapsed}/{matched}  ({100.0*collapsed/matched:.1f}%)")


# --- Entry point ---

def main():
    ap = argparse.ArgumentParser(description="LLM-as-judge for IL-PCSR query faithfulness.")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="eval dataset (single- or multi-query)")
    ap.add_argument("--queries-dir", default=QUERIES_DIR)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--provider", choices=["novita", "gemini"], default="novita",
                    help="novita (default; a different model family from the generator) or gemini")
    ap.add_argument("--model", default=None,
                    help=("model id (default: %s for novita, %s for gemini). "
                          "novita batch-eligible: %s | gemini: %s"
                          % (DEFAULT_NOVITA_MODEL, DEFAULT_GEMINI_MODEL,
                             ", ".join(NOVITA_BATCH_MODELS), ", ".join(GOOGLE_MODELS))))
    ap.add_argument("--limit", type=int, default=0, help="judge only first N (0 = all)")
    ap.add_argument("--max-situation-chars", type=int, default=MAX_SITUATION_CHARS)
    ap.add_argument("--no-batch", action="store_true", help="novita: online per-case calls")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--calibration-sample", type=int, default=0,
                    help="export N random rows for hand-rating instead of judging")
    ap.add_argument("--agreement", default=None, metavar="HUMAN_FILE",
                    help="score a filled hand-rated file against --output and exit")
    args = ap.parse_args()

    if args.agreement:
        score_agreement(args.agreement, args.output)
        return

    if not os.path.exists(args.input):
        print(f"[FAIL] input not found: {args.input}")
        sys.exit(1)
    records = json.load(open(args.input, encoding="utf-8"))
    if args.limit > 0:
        records = records[:args.limit]

    if args.calibration_sample > 0:
        export_calibration(records, args.queries_dir, args.calibration_sample, args.max_situation_chars)
        return

    model = args.model or (DEFAULT_NOVITA_MODEL if args.provider == "novita" else DEFAULT_GEMINI_MODEL)

    # Build the work list once (key, item, messages); key = sub_query_id if present else case id.
    work, skipped = [], 0
    for i, item, sit in iter_judgeable(records, args.queries_dir, args.max_situation_chars):
        key = str(item.get("sub_query_id") or item.get("source_case_id") or i)
        work.append((key, item, build_judge_messages(sit, item["question"])))
    skipped = len(records) - len(work)

    print(f"Provider={args.provider}  model={model}  to_judge={len(work)}  skipped={skipped}  "
          f"output={args.output}")
    if not work:
        print("Nothing to judge.")
        return
    if args.dry_run:
        print("[DRY RUN] no LLM calls / uploads.")

    if args.provider == "novita":
        _run_novita_direct(model, work, args.output, args.dry_run) if args.no_batch \
            else _run_novita_batch(model, work, args.output, args.dry_run)
    else:
        if args.no_batch:
            print("[gemini] --no-batch not implemented; omit it for the Batch API.")
            return
        _run_gemini_batch(model, work, args.output, args.dry_run)


if __name__ == "__main__":
    main()
