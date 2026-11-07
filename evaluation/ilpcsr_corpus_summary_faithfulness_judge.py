"""
Run with :
python evaluation/ilpcsr_corpus_summary_faithfulness_judge.py --no-batch

ilpcsr_corpus_summary_faithfulness_judge.py - LLM-as-judge for CORPUS SUMMARY FAITHFULNESS.

WHY THIS EXISTS
    CaseRAG embeds each of the 3,183 candidate cases via an LLM-generated 5-field summary
    (key_facts, legal_issues, key_statutes_cited, key_precedents_cited, holding), produced by
    meta-llama/llama-4-scout-17b-16e-instruct from the case's raw rhetorical-role text. The
    query-faithfulness audit (ilpcsr_query_faithfulness_judge.py) already validates the
    generated QUERIES; it says nothing about whether the SUMMARIES that are actually embedded
    and searched are faithful to their source judgments. If the summarizer hallucinates a fact,
    invents an issue, or gets the holding backwards, that error propagates directly into the
    embedding substrate and is invisible to the query audit. This script closes that gap.

    Mirrors the design of ilpcsr_query_faithfulness_judge.py: a strict LLM-as-judge prompt,
    a DIFFERENT model family from the generator, and a calibration workflow (hand-rate a
    sample, measure agreement) so the audit is credible rather than a bare LLM-judging-an-LLM
    claim.

WHY A RANDOM SUBSET, NOT ALL 3,183
    Auditing all cases is unnecessary cost for the same statistical purpose a sample serves.
    We draw a fixed-seed random sample (default n=200) from the corpus and report faithfulness
    over that sample, exactly as an opinion-poll style estimate. Pass --sample-size 0 to judge
    every case with a usable Summary instead.

    GROWING THE SAMPLE (e.g. 200 -> 400)
    --sample-size is a TARGET TOTAL, not "how many to draw this run". If --output already
    contains N judged cases, re-running with a larger --sample-size draws only the
    (target - N) additional cases needed, sampled at random (same fixed seed) from the
    corpus EXCLUDING every case_id already present in --output. Existing records are never
    re-judged, re-ordered, or dropped -- the new cases are appended and the file is rewritten
    atomically, so the final sample is the union of every case ever judged for this --output.
    There is no mechanism to filter which verdicts get kept: whatever the judge returns for the
    newly drawn cases is written and counted, the same as for the original sample.

WHAT IT JUDGES (per sampled case)
    Input the judge sees = the case's raw rhetorical-role segments (Fact, Issue, Petitioner's
    Argument, Respondent's Argument, Analysis of the law, Precedent Analysis, Court's Reasoning,
    Conclusion, Court Disclosure -- i.e. everything EXCEPT the "Uncategorized" boilerplate/headnote
    block and the "Summary" itself) + the three LLM-GENERATED summary fields (key_facts,
    legal_issues, holding). key_statutes_cited / key_precedents_cited are NOT judged: they are
    copied verbatim from case metadata (relevant_statutes / relevant_precedents) at generation
    time, not produced by the LLM, so there is nothing to audit for hallucination there.

    Output per case (strict JSON):
      verdict          : faithful | partial | unfaithful
      grounded         : every fact asserted in key_facts/legal_issues/holding is supported by
                          the source text, with nothing invented
      issues_accurate  : every issue listed in legal_issues is genuinely litigated in the source
                          (not fabricated or off-topic)
      holding_accurate : the holding field correctly reflects the court's actual decision/outcome
                          (not reversed, not confused with a different case's outcome)
      justification    : <= 25 words

DATASET
    --corpus-dir defaults to il-pcsr_dataset/corpus/ (one JSON file per case, each already
    containing both the raw rhetorical fields and a "Summary" block written by the summarizer
    notebook). No external query-set file is needed -- this script reads the corpus directly.

PROVIDERS (--provider) -- default matches the query-faithfulness judge for consistency: a
DIFFERENT model family from the summarizer (meta-llama/llama-4-scout-17b-16e-instruct via Groq).
    novita : default. Novita's OpenAI-compatible API. Default model minimax/minimax-m3 (same
             judge used for query faithfulness).
    gemini : Gemini Batch API, for a second-opinion / cross-judge run.
    --no-batch : online per-case calls (novita) for a quick smoke test.

MODES
    (default)               sample --sample-size cases, judge them, write results + summary.
    --calibration-sample N  export N random (source, summary) pairs with blank human fields
                            for hand-rating (no LLM call). Uses the SAME sample as the main run
                            when --sample-size matches, so the calibration set is a true subset.
    --agreement HUMAN_FILE  compare a filled hand-rated file against a judge-results file and
                            print agreement (exact + collapsed faithful-vs-not).
"""

import os
import re
import sys
import json
import time
import glob
import random
import argparse
from typing import List, Dict, Optional

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

# Raw rhetorical-role fields that make up the source judgment text. "Uncategorized" (headnote /
# citation / bench-listing boilerplate) and "Summary" (the thing being judged) are deliberately
# excluded -- same principle as the query judge only showing the generator's actual input fields.
SOURCE_FIELDS = [
    "Fact", "Issue", "Petitioner's Argument", "Respondent's Argument",
    "Analysis of the law", "Precedent Analysis", "Court's Reasoning",
    "Conclusion", "Court Disclosure",
]

# Summary fields that are actually LLM-generated and therefore judged. key_statutes_cited /
# key_precedents_cited are injected from metadata at generation time, not generated -- excluded.
JUDGED_SUMMARY_FIELDS = ["key_facts", "legal_issues", "holding"]

GOOGLE_MODELS = [
    "gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3-flash-preview",
]
NOVITA_BATCH_MODELS = [
    "minimax/minimax-m3", "openai/gpt-oss-120b", "qwen/qwen3-vl-235b-a22b-instruct", "qwen/qwen3-4b-fp8",
]
NOVITA_MODELS_EXAMPLES = NOVITA_BATCH_MODELS + [
    "deepseek/deepseek-v3-0324", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b",
]
NOVITA_BASE_URL = os.environ.get("NOVITA_BASE_URL", "https://api.novita.ai/openai/v1")

DEFAULT_NOVITA_MODEL = "minimax/minimax-m3"   # same judge family used for query faithfulness
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0   # a judge should be as deterministic as the API allows

CORPUS_DIR      = os.path.join("il-pcsr_dataset", "corpus")
DEFAULT_OUTPUT  = os.path.join("evaluation", "results", "corpus_summary_faithfulness_results.json")
CALIB_TEMPLATE  = os.path.join("evaluation", "results", "corpus_summary_faithfulness_calibration_template.json")
AGENT_WORKSPACE = ".agent_workspace"

def _batch_state_path(output_file: str) -> str:
    return output_file + ".batch_state.json"


DEFAULT_SAMPLE_SIZE = 200     # "a handful" per user instruction, not all 3,183; --sample-size 0 = judge all
DEFAULT_SEED = 42             # fixed seed -> the sample is reproducible and is a superset-consistent
                               # subset across runs with the same --sample-size
MAX_SOURCE_CHARS = 32000      # ~8k tokens; full judgments run longer than the query "situation" fields
VALID_VERDICTS = {"faithful", "partial", "unfaithful"}


# --- Source loading ---

def iter_corpus_files(corpus_dir: str):
    """Yield (case_id, filepath) for every corpus_dir/*.json file, sorted for reproducibility."""
    for fp in sorted(glob.glob(os.path.join(corpus_dir, "*.json"))):
        case_id = os.path.splitext(os.path.basename(fp))[0]
        yield case_id, fp


def load_case(fp: str) -> Optional[Dict]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_source_text(doc: Dict, max_chars: int) -> str:
    """Concatenate the raw rhetorical-role fields the summarizer saw, capped at max_chars."""
    parts = []
    for k in SOURCE_FIELDS:
        v = doc.get(k)
        if isinstance(v, list) and v:
            parts.append(f"## {k}\n" + " ".join(str(x) for x in v))
    text = "\n\n".join(parts).strip()
    return text[:max_chars]


def build_summary_text(summary: Dict) -> str:
    """Render only the LLM-generated summary fields (statutes/precedents are metadata, not judged)."""
    facts = summary.get("key_facts", "")
    issues = summary.get("legal_issues", [])
    if isinstance(issues, list):
        issues_str = "; ".join(str(x) for x in issues)
    else:
        issues_str = str(issues)
    holding = summary.get("holding", "")
    return (
        f"key_facts: {facts}\n"
        f"legal_issues: {issues_str}\n"
        f"holding: {holding}"
    )


def has_usable_summary(doc: Dict) -> bool:
    s = doc.get("Summary")
    return isinstance(s, dict) and bool(s.get("key_facts") or s.get("holding"))


# --- Prompt ---

def build_judge_messages(source_text: str, summary_text: str) -> List[Dict]:
    """Strict-evaluator prompt for corpus-summary faithfulness (hallucination / holding accuracy),
    parallel in structure to the query-faithfulness judge's prompt."""
    system_prompt = (
        "You are a strict legal-NLP evaluator. You are given the SOURCE TEXT of a court "
        "judgment (its Facts, Issues, Arguments, Analysis, Precedent discussion, Reasoning, and "
        "Conclusion) and a GENERATED SUMMARY that was auto-produced from it to represent the "
        "case in a retrieval system. Judge whether the summary is FAITHFUL to the source.\n"
        "Assess: (a) grounded -- every fact or claim asserted in key_facts, legal_issues, or "
        "holding is supported by the source text, with nothing invented or drawn from outside "
        "knowledge; (b) issues_accurate -- every issue listed in legal_issues is an issue "
        "genuinely litigated or decided in the source (not fabricated, not off-topic); "
        "(c) holding_accurate -- the holding field correctly states the court's actual decision "
        "or outcome as it appears in the source, in the correct direction (it must NOT reverse, "
        "invert, or confuse the outcome).\n"
        "verdict = 'faithful' if grounded and issues_accurate and holding_accurate; 'partial' if "
        "it mostly holds but has minor drift, an imprecise phrasing, or one weak issue; "
        "'unfaithful' if it invents facts, fabricates an issue, or misstates the holding/outcome.\n"
        "Return ONLY strict JSON, no prose, exactly: "
        '{"verdict":"faithful|partial|unfaithful","grounded":true,"issues_accurate":true,'
        '"holding_accurate":true,"justification":"<=25 words"}'
    )
    user_prompt = (
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"GENERATED SUMMARY:\n{summary_text}\n\n"
        "Judge the summary now."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


# --- Response parsing ---

def parse_verdict(raw: str) -> Dict:
    """Coerce a model response into the verdict dict; a parse failure yields
    verdict='unparsed' (counted, never crashes a batch)."""
    base = {"verdict": "unparsed", "grounded": None, "issues_accurate": None,
            "holding_accurate": None, "justification": ""}
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
    for k in ("grounded", "issues_accurate", "holding_accurate"):
        base[k] = obj.get(k)
    base["justification"] = str(obj.get("justification", ""))[:300]
    return base


# --- Sampling / iteration ---

def sample_cases(corpus_dir: str, sample_size: int, seed: int, exclude_ids: Optional[set] = None):
    """Return a fixed-seed random sample of (case_id, doc) pairs restricted to cases with a
    usable Summary, excluding any case_id in `exclude_ids` (cases already judged in a prior
    run against the same --output). sample_size=0 means: use every remaining eligible case
    (no sampling). sample_size otherwise means "how many to draw from what's left", i.e. the
    caller is responsible for passing (target_total - already_judged) when topping up an
    existing sample -- see main()."""
    exclude_ids = exclude_ids or set()
    pool = []
    skipped_no_summary = 0
    for case_id, fp in iter_corpus_files(corpus_dir):
        if case_id in exclude_ids:
            continue
        doc = load_case(fp)
        if doc is None:
            continue
        if not has_usable_summary(doc):
            skipped_no_summary += 1
            continue
        pool.append((case_id, doc))
    if sample_size and sample_size < len(pool):
        random.seed(seed)
        pool = random.sample(pool, sample_size)
        pool.sort(key=lambda x: x[0])
    return pool, skipped_no_summary


def make_record(case_id: str, doc: Dict, verdict: Dict) -> Dict:
    out = {
        "case_id": case_id,
        "title": doc.get("title", ""),
    }
    out.update(verdict)
    return out


def summarize(records: List[Dict]) -> Dict:
    """Verdict distribution + flag rates -- the number the paper's faithfulness paragraph quotes."""
    n = len(records)
    vc = {"faithful": 0, "partial": 0, "unfaithful": 0, "unparsed": 0}
    flags = {"not_grounded": 0, "issues_inaccurate": 0, "holding_inaccurate": 0}
    for r in records:
        vc[r.get("verdict", "unparsed")] = vc.get(r.get("verdict", "unparsed"), 0) + 1
        if r.get("grounded") is False:
            flags["not_grounded"] += 1
        if r.get("issues_accurate") is False:
            flags["issues_inaccurate"] += 1
        if r.get("holding_accurate") is False:
            flags["holding_inaccurate"] += 1
    pct = lambda x: round(100.0 * x / n, 1) if n else 0.0
    return {
        "judged": n,
        "verdicts": vc,
        "verdict_pct": {k: pct(v) for k, v in vc.items()},
        "flag_counts": flags,
        "flag_pct": {k: pct(v) for k, v in flags.items()},
    }


def write_output(records: List[Dict], output_file: str, sample_size: int, seed: int,
                  corpus_total: int, skipped_no_summary: int) -> None:
    """Rewrite the full output file from the current in-memory records list. Called after
    EVERY judged case in the direct/online path (see _run_novita_direct), so the file on disk
    always reflects everything judged so far -- an interruption never loses completed work.
    A rewrite-in-place is used rather than append-only JSONL to keep the on-disk shape
    identical to the query-faithfulness judge's {_summary, records} output; at a few hundred
    records this is a negligible cost per case."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    payload = {
        "_summary": summarize(records),
        "_sampling": {
            "corpus_total": corpus_total,
            "skipped_no_summary": skipped_no_summary,
            "sample_size": sample_size,
            "seed": seed,
        },
        "records": records,
    }
    # Write to a temp file then atomically replace, so a crash mid-write can never corrupt
    # the last good result (the previous write_output's content survives such a crash).
    tmp_path = output_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_file)


def load_existing_records(output_file: str) -> List[Dict]:
    """Load previously-judged records from output_file, if it exists and parses. Used to
    resume a run: cases already present are skipped rather than re-judged (and re-paid-for)."""
    if not os.path.exists(output_file):
        return []
    try:
        payload = json.load(open(output_file, encoding="utf-8"))
        return payload.get("records", [])
    except Exception:
        print(f"  [WARN] could not parse existing {output_file}; starting fresh.")
        return []


def print_summary(records: List[Dict]) -> None:
    s = summarize(records)
    print("\n========== CORPUS SUMMARY FAITHFULNESS ==========")
    print(f"judged: {s['judged']}")
    for k in ("faithful", "partial", "unfaithful", "unparsed"):
        print(f"  {k:<11} {s['verdicts'][k]:>5}  ({s['verdict_pct'][k]}%)")
    print("flags (count / %):")
    for k, v in s["flag_counts"].items():
        print(f"  {k:<18} {v:>5}  ({s['flag_pct'][k]}%)")
    print("===================================================")


# ============================ GEMINI BATCH-API PATH ============================

def _gemini_request(messages: List[Dict]) -> Dict:
    system_text = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_text   = next((m["content"] for m in messages if m["role"] == "user"),   "")
    combined = f"{system_text}\n\n{user_text}" if system_text else user_text
    return {
        "contents": [{"role": "user", "parts": [{"text": combined}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": TEMPERATURE},
    }


def _run_gemini_batch(model_name, work, output_file, dry_run, sample_size, seed,
                       corpus_total, skipped_no_summary, overwrite=False, resume_batch=False) -> int:
    """work = list of (key, case_id, doc, messages). Resume semantics mirror _run_novita_batch:
    skip already-judged cases on a fresh submission; --resume-batch reattaches to a saved job
    name instead of resubmitting after an interrupted poll."""
    from google import genai
    existing_records = [] if overwrite else load_existing_records(output_file)
    done_ids = {r["case_id"] for r in existing_records}
    state_path = _batch_state_path(output_file)

    if resume_batch:
        if not os.path.exists(state_path):
            print(f"  [FAIL] --resume-batch given but no state file at {state_path}.")
            return len(existing_records)
        state = json.load(open(state_path, encoding="utf-8"))
        job_name = state["job_id"]
        item_by_key = {k: (cid, d) for k, cid, d, _ in work if k in set(state["case_ids"])}
        client = genai.Client()
        job = client.batches.get(name=job_name)
        print(f"  Resuming batch job: {job.name}  (status {job.state})")
    else:
        remaining = [(k, cid, doc, msgs) for k, cid, doc, msgs in work if cid not in done_ids]
        if done_ids:
            print(f"  Resuming: {len(done_ids)} case(s) already judged in {output_file} -- "
                  f"excluding from this batch. {len(remaining)} remaining.")
        if not remaining:
            print("  Nothing left to judge -- all sampled cases already have a saved verdict.")
            print_summary(existing_records)
            return len(existing_records)

        item_by_key = {k: (cid, d) for k, cid, d, _ in remaining}
        jsonl_path = os.path.join(AGENT_WORKSPACE, "ilpcsr-corpus-faithfulness.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for key, _, _, messages in remaining:
                f.write(json.dumps({"key": key, "request": _gemini_request(messages)}) + "\n")
        print(f"  JSONL written: {jsonl_path}  ({len(remaining)} requests)")
        if dry_run:
            print(f"  [DRY RUN] not uploading. Inspect: {jsonl_path}")
            return len(existing_records)

        client = genai.Client()
        uploaded = client.files.upload(file=jsonl_path, config={"mime_type": "application/jsonl"})
        job = client.batches.create(model=f"models/{model_name}", src=uploaded.name,
                                    config={"display_name": "ilpcsr-corpus-faithfulness"})
        print(f"  Job: {job.name}")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"job_id": job.name, "case_ids": list(item_by_key.keys())}, f)
        print(f"  Job name saved to {state_path} -- if this run is interrupted, "
              f"rerun with --resume-batch instead of resubmitting.")

    try:
        while not job.done:
            print(f"  [poll] {job.state} — waiting 30s... (safe to Ctrl+C; rerun with --resume-batch)")
            time.sleep(30)
            job = client.batches.get(name=job.name)
    except KeyboardInterrupt:
        print(f"\n  [STOPPED] Batch job {job.name} is still running server-side. "
              f"Rerun with --resume-batch to reattach once it completes.")
        return len(existing_records)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"  [ERROR] job state: {job.state}  (job name preserved in {state_path})")
        return len(existing_records)

    result_file = client.files.get(name=job.dest.file_name)
    content = client.files.download(file=result_file).decode("utf-8")
    records = list(existing_records)
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
            cid, d = item_by_key[key]
            records.append(make_record(cid, d, parse_verdict(text)))
    write_output(records, output_file, sample_size, seed, corpus_total, skipped_no_summary)
    try:
        os.remove(state_path)
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


def _run_novita_batch(model_name, work, output_file, dry_run, sample_size, seed,
                       corpus_total, skipped_no_summary, overwrite=False, resume_batch=False) -> int:
    """Batch path: one bulk submission scored asynchronously. Resumability here works
    differently from the direct path, since a batch job has no partial-progress API:
      - Cases already saved in output_file (from a prior completed run) are skipped from the
        NEW submission, so re-running never re-pays for cases you already have a verdict for.
      - The submitted job id is written to a sidecar `<output>.batch_state.json` immediately
        after submission. If the polling loop is interrupted (Ctrl+C, terminal closed, machine
        sleep), rerun with --resume-batch to reattach to that same job instead of submitting
        (and paying for) a duplicate batch.
    """
    existing_records = [] if overwrite else load_existing_records(output_file)
    done_ids = {r["case_id"] for r in existing_records}
    state_path = _batch_state_path(output_file)

    if resume_batch:
        if not os.path.exists(state_path):
            print(f"  [FAIL] --resume-batch given but no state file at {state_path}.")
            return len(existing_records)
        state = json.load(open(state_path, encoding="utf-8"))
        job_id = state["job_id"]
        item_by_key = {k: (cid, d) for k, cid, d, _ in work if k in set(state["case_ids"])}
        client = _get_novita_client()
        if client is None:
            return len(existing_records)
        job = client.batches.retrieve(job_id)
        print(f"  Resuming batch job: {job.id}  (status {job.status})")
    else:
        remaining = [(k, cid, doc, msgs) for k, cid, doc, msgs in work if cid not in done_ids]
        if done_ids:
            print(f"  Resuming: {len(done_ids)} case(s) already judged in {output_file} -- "
                  f"excluding from this batch. {len(remaining)} remaining.")
        if not remaining:
            print("  Nothing left to judge -- all sampled cases already have a saved verdict.")
            print_summary(existing_records)
            return len(existing_records)

        item_by_key = {k: (cid, d) for k, cid, d, _ in remaining}
        jsonl_path = os.path.join(AGENT_WORKSPACE, "ilpcsr-corpus-faithfulness-novita.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for key, _, _, messages in remaining:
                f.write(json.dumps({
                    "custom_id": key, "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": model_name, "messages": messages,
                             "temperature": TEMPERATURE, "response_format": {"type": "json_object"}},
                }) + "\n")
        print(f"  JSONL written: {jsonl_path}  ({len(remaining)} requests)")
        if dry_run:
            print(f"  [DRY RUN] not uploading. Inspect: {jsonl_path}")
            return len(existing_records)

        client = _get_novita_client()
        if client is None:
            return len(existing_records)
        with open(jsonl_path, "rb") as fh:
            batch_file = client.files.create(file=fh, purpose="batch")
        job = client.batches.create(input_file_id=batch_file.id,
                                    endpoint="/v1/chat/completions", completion_window="24h")
        print(f"  Batch job: {job.id}  (status {job.status})")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"job_id": job.id, "case_ids": list(item_by_key.keys())}, f)
        print(f"  Job id saved to {state_path} -- if this run is interrupted, "
              f"rerun with --resume-batch instead of resubmitting.")

    terminal = {"completed", "failed", "expired", "cancelled"}
    try:
        while job.status not in terminal:
            print(f"  [poll] {job.status} — waiting 60s... (safe to Ctrl+C; rerun with --resume-batch)")
            time.sleep(60)
            job = client.batches.retrieve(job.id)
    except KeyboardInterrupt:
        print(f"\n  [STOPPED] Batch job {job.id} is still running server-side. "
              f"Rerun with --resume-batch to reattach once it completes.")
        return len(existing_records)

    if job.status != "completed":
        print(f"  [ERROR] batch ended: {job.status}  (job id preserved in {state_path})")
        return len(existing_records)

    resp = client.files.content(job.output_file_id)
    content = getattr(resp, "text", None) or resp.read().decode("utf-8")
    records = list(existing_records)
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid_key = obj.get("custom_id", "")
        try:
            raw = obj["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            continue
        if cid_key in item_by_key:
            cid, d = item_by_key[cid_key]
            records.append(make_record(cid, d, parse_verdict(raw)))
    write_output(records, output_file, sample_size, seed, corpus_total, skipped_no_summary)
    try:
        os.remove(state_path)
    except Exception:
        pass
    print_summary(records)
    return len(records)


def _run_novita_direct(model_name, work, output_file, dry_run, sample_size, seed,
                        corpus_total, skipped_no_summary, overwrite=False) -> int:
    """Online, one-call-per-case path. Resumable by construction: every case already present
    in output_file is skipped (unless --overwrite), and the file is rewritten to disk after
    EVERY successfully judged case -- not just at the end. So: a crash, a killed process, a
    rate-limit error, or a Ctrl+C never costs you previously-paid-for judgments; simply rerun
    the exact same command and it continues from where it stopped."""
    existing_records = [] if overwrite else load_existing_records(output_file)
    done_ids = {r["case_id"] for r in existing_records}
    if done_ids:
        print(f"  Resuming: {len(done_ids)} case(s) already judged in {output_file} -- skipping those.")

    remaining = [(key, cid, doc, msgs) for key, cid, doc, msgs in work if cid not in done_ids]
    if not remaining:
        print("  Nothing left to judge -- all sampled cases already have a saved verdict.")
        if not dry_run:
            print_summary(existing_records)
        return len(existing_records)

    client = None if dry_run else _get_novita_client()
    if not dry_run and client is None:
        return 0

    records = list(existing_records)
    try:
        for i, (key, case_id, doc, messages) in enumerate(remaining, 1):
            print(f"  [{i}/{len(remaining)}] {key}")
            if dry_run:
                print(f"    [DRY RUN] {messages[1]['content'][:120]}...")
                continue
            try:
                resp = client.chat.completions.create(
                    model=model_name, messages=messages, temperature=TEMPERATURE,
                    response_format={"type": "json_object"})
                raw = resp.choices[0].message.content
            except Exception as e:
                print(f"    [WARN] call failed: {e} -- will retry this case on the next run.")
                continue
            records.append(make_record(case_id, doc, parse_verdict(raw)))
            # Incremental save -- the whole point of the resumable design. Cheap at this scale
            # (a few hundred small records), so we pay the full-rewrite cost every iteration
            # rather than risk losing paid-for API calls to an interruption.
            write_output(records, output_file, sample_size, seed, corpus_total, skipped_no_summary)
    except KeyboardInterrupt:
        print(f"\n  [STOPPED] {len(records) - len(existing_records)} new case(s) judged and saved "
              f"this run ({len(records)}/{len(existing_records) + len(remaining)} total). "
              "Rerun the same command to continue.")
        return len(records)

    if not dry_run:
        print_summary(records)
    return len(records)


# --- Calibration helpers ---

def export_calibration(pool, n: int, max_chars: int) -> None:
    """Write N random (source, summary) pairs with blank human fields for hand-rating -- the
    gold-standard half of the calibration the paper needs. Drawn from the same eligible pool
    the main run samples from, with the same fixed seed for reproducibility."""
    random.seed(DEFAULT_SEED)
    sample = random.sample(pool, min(n, len(pool)))
    rows = []
    for case_id, doc in sample:
        rows.append({
            "case_id": case_id,
            "title": doc.get("title", ""),
            "source_text": build_source_text(doc, max_chars),
            "generated_summary": build_summary_text(doc.get("Summary", {})),
            "human_verdict": "",      # fill: faithful | partial | unfaithful
            "human_notes": "",
        })
    os.makedirs(os.path.dirname(os.path.abspath(CALIB_TEMPLATE)), exist_ok=True)
    with open(CALIB_TEMPLATE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} calibration rows -> {CALIB_TEMPLATE}\n"
          f"Hand-rate `human_verdict`, then: --agreement {CALIB_TEMPLATE}")


def score_agreement(human_file: str, judge_file: str) -> None:
    """Compare hand ratings against judge verdicts (matched on case_id): exact 3-way agreement
    and collapsed faithful-vs-not. This is the credibility number the paper cites."""
    human = json.load(open(human_file, encoding="utf-8"))
    judged = json.load(open(judge_file, encoding="utf-8")).get("records", [])
    jmap = {r["case_id"]: r.get("verdict") for r in judged}

    matched = exact = collapsed = 0
    collapse = lambda v: "faithful" if v == "faithful" else "not"
    for h in human:
        hv = str(h.get("human_verdict", "")).strip().lower()
        if hv not in VALID_VERDICTS:
            continue
        jv = jmap.get(str(h.get("case_id", "")))
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
    ap = argparse.ArgumentParser(description="LLM-as-judge for IL-PCSR corpus summary faithfulness.")
    ap.add_argument("--corpus-dir", default=CORPUS_DIR)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                    help=("TARGET TOTAL random subset size (0 = judge every case with a usable "
                          "Summary). If --output already has judged cases, only enough NEW "
                          "never-judged cases are drawn to reach this total; prior results are "
                          "kept as-is."))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--provider", choices=["novita", "gemini"], default="novita",
                    help="novita (default; same judge family as the query-faithfulness audit) or gemini")
    ap.add_argument("--model", default=None,
                    help=("model id (default: %s for novita, %s for gemini). "
                          "novita batch-eligible: %s | gemini: %s"
                          % (DEFAULT_NOVITA_MODEL, DEFAULT_GEMINI_MODEL,
                             ", ".join(NOVITA_BATCH_MODELS), ", ".join(GOOGLE_MODELS))))
    ap.add_argument("--limit", type=int, default=0, help="judge only first N of the sample (0 = all sampled)")
    ap.add_argument("--max-source-chars", type=int, default=MAX_SOURCE_CHARS)
    ap.add_argument("--no-batch", action="store_true", help="novita: online per-case calls")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--calibration-sample", type=int, default=0,
                    help="export N random rows for hand-rating instead of judging")
    ap.add_argument("--agreement", default=None, metavar="HUMAN_FILE",
                    help="score a filled hand-rated file against --output and exit")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore any existing --output and re-judge every sampled case from scratch")
    ap.add_argument("--resume-batch", action="store_true",
                    help="reattach to a previously-submitted batch job (see <output>.batch_state.json) "
                         "instead of submitting a new one -- use after an interrupted batch run")
    args = ap.parse_args()

    if args.agreement:
        score_agreement(args.agreement, args.output)
        return

    if not os.path.isdir(args.corpus_dir):
        print(f"[FAIL] corpus dir not found: {args.corpus_dir}")
        sys.exit(1)

    corpus_total = len(glob.glob(os.path.join(args.corpus_dir, "*.json")))

    if args.calibration_sample > 0:
        pool, skipped_no_summary = sample_cases(args.corpus_dir, 0, args.seed)  # full eligible pool
        export_calibration(pool, args.calibration_sample, args.max_source_chars)
        return

    existing_records = [] if args.overwrite else load_existing_records(args.output)
    done_ids = {r["case_id"] for r in existing_records}

    if args.sample_size and done_ids:
        needed = max(args.sample_size - len(done_ids), 0)
        if needed == 0:
            print(f"  {len(done_ids)} case(s) already judged in {args.output} -- "
                  f"already >= requested --sample-size {args.sample_size}. Nothing new to sample.")
            print_summary(existing_records)
            return
        print(f"  {len(done_ids)} case(s) already judged in {args.output}; drawing {needed} "
              f"NEW, never-judged case(s) at random (seed={args.seed}) to reach a total of "
              f"{args.sample_size}.")
        pool, skipped_no_summary = sample_cases(args.corpus_dir, needed, args.seed, exclude_ids=done_ids)
    else:
        pool, skipped_no_summary = sample_cases(args.corpus_dir, args.sample_size, args.seed,
                                                  exclude_ids=done_ids)
    if args.limit > 0:
        pool = pool[:args.limit]

    model = args.model or (DEFAULT_NOVITA_MODEL if args.provider == "novita" else DEFAULT_GEMINI_MODEL)

    work = []
    for case_id, doc in pool:
        source_text = build_source_text(doc, args.max_source_chars)
        summary_text = build_summary_text(doc.get("Summary", {}))
        if not source_text or not summary_text.strip():
            continue
        work.append((case_id, case_id, doc, build_judge_messages(source_text, summary_text)))

    print(f"Provider={args.provider}  model={model}  corpus_total={corpus_total}  "
          f"skipped_no_summary={skipped_no_summary}  target_sample_size={args.sample_size or 'ALL'}  "
          f"already_judged={len(done_ids)}  to_judge={len(work)}  output={args.output}")
    if not work:
        print("Nothing to judge.")
        return
    if args.dry_run:
        print("[DRY RUN] no LLM calls / uploads.")

    if args.provider == "novita":
        if args.no_batch:
            _run_novita_direct(model, work, args.output, args.dry_run, args.sample_size, args.seed,
                               corpus_total, skipped_no_summary, overwrite=args.overwrite)
        else:
            _run_novita_batch(model, work, args.output, args.dry_run, args.sample_size, args.seed,
                              corpus_total, skipped_no_summary,
                              overwrite=args.overwrite, resume_batch=args.resume_batch)
    else:
        if args.no_batch:
            print("[gemini] --no-batch not implemented; omit it for the Batch API.")
            return
        _run_gemini_batch(model, work, args.output, args.dry_run, args.sample_size, args.seed,
                          corpus_total, skipped_no_summary,
                          overwrite=args.overwrite, resume_batch=args.resume_batch)


if __name__ == "__main__":
    main()
