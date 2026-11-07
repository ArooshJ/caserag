"""
Run to know how many cases in corpus have summary fields 
"""

import os, json
from pathlib import Path
keys = ['key_facts','legal_issues','holding','key_statutes_cited','key_precedents_cited']
REPO_ROOT = Path(__file__).resolve().parent.parent
corpus_dir = REPO_ROOT / "il-pcsr_dataset" / "corpus"
total = has_any = has_all5 = 0
sub = {k: 0 for k in keys}
for f in os.listdir(corpus_dir):
    if not f.endswith('.json'): continue
    total += 1
    with open(os.path.join(corpus_dir, f), encoding='utf-8') as fh:
        d = json.load(fh)
    s = d.get('Summary')
    if not isinstance(s, dict) or not s: continue
    has_any += 1
    all5 = True
    for k in keys:
        if s.get(k): sub[k] += 1
        else: all5 = False
    if all5: has_all5 += 1
print(f'Total: {total} | Any Summary: {has_any} | All 5: {has_all5}')
for k,v in sub.items(): print(f'  {k}: {v}')
