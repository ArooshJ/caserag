"""
Audit ILPCSR query splits for populated fields.
Checks: query, reference_answer, relevant_precedents, relevant_statutes.
Run from repo root or ingestion/.
"""

import os, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERY_ROOT = REPO_ROOT / "il-pcsr_dataset"

SPLITS = ["test_queries", "dev_queries", "train_queries"]
FIELDS = ["query", "reference_answer", "relevant_precedents", "relevant_statutes"]


def _is_populated(val) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (list, dict)):
        return len(val) > 0
    return bool(val)


def audit_split(split_name: str) -> None:
    split_dir = QUERY_ROOT / split_name
    if not split_dir.exists():
        print(f"  [missing] {split_dir}")
        return

    files = [f for f in os.listdir(split_dir) if f.endswith(".json")]
    total = len(files)
    counts = {field: 0 for field in FIELDS}
    all_core = 0  # both query + reference_answer populated

    for fname in files:
        with open(split_dir / fname, encoding="utf-8") as fh:
            try:
                d = json.load(fh)
            except json.JSONDecodeError:
                continue

        field_pop = {f: _is_populated(d.get(f)) for f in FIELDS}
        for f in FIELDS:
            if field_pop[f]:
                counts[f] += 1
        if field_pop["query"] and field_pop["reference_answer"]:
            all_core += 1

    print(f"\n{'='*50}")
    print(f"Split : {split_name}")
    print(f"Total : {total}")
    for f in FIELDS:
        pct = counts[f] / total * 100 if total else 0
        print(f"  {f:<25} {counts[f]:>5} / {total}  ({pct:.1f}%)")
    core_pct = all_core / total * 100 if total else 0
    print(f"  {'query + reference_answer':<25} {all_core:>5} / {total}  ({core_pct:.1f}%)")


if __name__ == "__main__":
    for split in SPLITS:
        audit_split(split)
    print()
