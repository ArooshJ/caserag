#!/usr/bin/env python
"""
oracle_analysis.py - Post-processing diagnostic over a completed gold oracle JSON.

Computes 5 story-tellable metrics for a research paper without touching the DB,
embedding models, or APIs. Reads oracle JSON + optional fill-eval JSON, writes
an analysis JSON with headline numbers.

Usage (from repo root):
    python evaluation/oracle_analysis.py --oracle evaluation/data/gold_oracle_emb1_multiquery.json

    # With gold rank in actual retrieval (metric 5):
    python evaluation/oracle_analysis.py \
        --oracle evaluation/data/gold_oracle_emb1_multiquery.json \
        --fill-eval evaluation/data/model=1_k1=200_k2=100_multiquery.json

    # emb3 (Kanon-2, cosine) once evaluation/gold_oracle_similarity.py has been run
    # with --emb-model-id 3 --multi-query --distance-op cosine (see
    # research_paper/ablation/oracle_similarity_histogram.md):
    python evaluation/oracle_analysis.py --oracle evaluation/data/gold_oracle_emb3_cosine_multiquery.json
"""

import os
import sys
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict


FIELD_KEYS = ["facts", "issues", "statutes", "precedents", "holding"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(vals):
    """Descriptive stats dict for a list of floats. Returns None on empty input."""
    if not vals:
        return None
    sv = sorted(vals)
    n = len(sv)
    def pct(p):
        k = max(0, min(n - 1, int(round(p / 100.0 * (n - 1)))))
        return round(sv[k], 4)
    return {
        "n": n,
        "mean":   round(statistics.fmean(sv), 4),
        "median": round(statistics.median(sv), 4),
        "p10": pct(10), "p25": pct(25), "p75": pct(75), "p90": pct(90),
        "min": round(sv[0], 4), "max": round(sv[-1], 4),
    }


def _pct(n, total):
    return round(n / total * 100, 2) if total else None


def load_oracle(path: Path) -> list:
    """Handle both flat-list and {"_summary":..., "records":[...]} oracle shapes."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("records", [])


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

def build_pairs(records: list) -> list:
    """
    Flatten nested oracle records into a flat list of (sub-query, gold) pair dicts.
    Each pair is one gold doc evaluated against one sub-query.
    WHY flat: all 5 metrics iterate the same list — avoids re-parsing nesting 5x.
    """
    pairs = []
    for rec in records:
        src  = rec.get("source_case_id", "")
        sqid = rec.get("sub_query_id", rec.get("question", ""))

        # Build stage2 lookup by doc_id for O(1) join
        s2_by_id = {c["doc_id"]: c for c in rec.get("stage2_per_gold", [])}

        seen = set()
        for g in rec.get("stage1", []):
            doc_id = g["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)

            s2 = s2_by_id.get(doc_id)
            field_scores = g.get("fields", {})

            # argmax among positive fields; "none_positive" if all fields <= 0
            pos = {k: v for k, v in field_scores.items() if v > 0}
            field_winner = max(pos, key=pos.get) if pos else "none_positive"

            pairs.append({
                "source_case_id": src,
                "sub_query_id":   sqid,
                "gold_doc_id":    doc_id,
                "stage1_score":   g["final_score"],
                "field_scores":   field_scores,
                "field_winner":   field_winner,
                "stage2_best":    s2["best_chunk_sim"] if s2 else None,
                "stage2_mean":    s2["mean_chunk_sim"] if s2 else None,
                "n_chunks":       s2["n_chunks"]       if s2 else None,
            })
    return pairs


# ---------------------------------------------------------------------------
# Metric 1 — Stage-2 vs Stage-1 continuous delta analysis
# ---------------------------------------------------------------------------

def _pearson(xs, ys):
    """Pearson r between two equal-length lists. Returns None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return round(num / den, 4) if den else None


def metric1_stage_delta(pairs: list) -> dict:
    """
    Threshold-free analysis of how stage-2 (chunk) similarity relates to
    stage-1 (case-level) similarity across all (sub-query, gold) pairs.

    Reports:
      - Distribution of delta = stage2_best - stage1_score
      - Pearson correlation between stage1 and stage2 scores
      - Fraction where stage2 > stage1 (chunks outperform case embedding)
      - Fraction where stage2 > stage1 by >0.05 (meaningful gap)
      - Fraction where stage1 < 0 but stage2 > 0 (natural geometric anchor:
        negative cosine = opposite-direction vectors; positive = same half-space)

    These metrics are threshold-free and scale-invariant — comparable across
    embedding models without recalibration.
    """
    s1, s2, deltas, n_excl = [], [], [], 0

    for p in pairs:
        if p["stage2_best"] is None:
            n_excl += 1
            continue
        s1.append(p["stage1_score"])
        s2.append(p["stage2_best"])
        deltas.append(p["stage2_best"] - p["stage1_score"])

    n = len(deltas)
    return {
        "n_pairs_with_stage2":            n,
        "n_excluded_no_chunks":           n_excl,
        "pearson_r_stage1_vs_stage2":     _pearson(s1, s2),
        "pct_stage2_gt_stage1":           _pct(sum(1 for d in deltas if d > 0),    n),
        "pct_stage2_gt_stage1_by_005":    _pct(sum(1 for d in deltas if d > 0.05), n),
        "pct_stage1_neg_stage2_pos":      _pct(sum(1 for a, b in zip(s1, s2) if a < 0 and b > 0), n),
        "pct_both_negative":              _pct(sum(1 for a, b in zip(s1, s2) if a < 0 and b < 0), n),
        "delta_stage2_minus_stage1":      _dist(deltas),
        "stage1_distribution":            _dist(s1),
        "stage2_distribution":            _dist(s2),
    }


# ---------------------------------------------------------------------------
# Metric 2 — Hard gold fraction
# ---------------------------------------------------------------------------

def metric2_hard_gold(pairs: list) -> dict:
    """
    Fraction of gold pairs with very low or negative case-level similarity.
    Low stage1 scores = these golds cannot be surfaced by case-level retrieval
    regardless of pipeline tuning.
    """
    n_total = len(pairs)
    n_neg   = sum(1 for p in pairs if p["stage1_score"] < 0)
    n_lo5   = sum(1 for p in pairs if p["stage1_score"] < 0.05)
    n_lo10  = sum(1 for p in pairs if p["stage1_score"] < 0.10)

    cases_any_neg = len({p["source_case_id"] for p in pairs if p["stage1_score"] < 0})
    n_cases       = len({p["source_case_id"] for p in pairs})

    return {
        "n_total_pairs":         n_total,
        "n_negative_stage1":     n_neg,
        "n_below_0.05":          n_lo5,
        "n_below_0.10":          n_lo10,
        "pct_negative":          _pct(n_neg,   n_total),
        "pct_below_0.05":        _pct(n_lo5,   n_total),
        "pct_below_0.10":        _pct(n_lo10,  n_total),
        "n_cases_total":         n_cases,
        "n_cases_with_any_negative_gold": cases_any_neg,
        "pct_cases_with_any_negative":    _pct(cases_any_neg, n_cases),
        "stage1_score_distribution":      _dist([p["stage1_score"] for p in pairs]),
    }


# ---------------------------------------------------------------------------
# Metric 3 — Field dominance
# ---------------------------------------------------------------------------

def metric3_field_dominance(pairs: list) -> dict:
    """
    Which summary field (facts/issues/statutes/precedents/holding) carries the
    most signal per gold pair? Diagnoses dead-weight fields and validates weights.
    """
    n_total       = len(pairs)
    win_counts    = defaultdict(int)
    field_scores  = defaultdict(list)
    field_pos     = defaultdict(int)

    for p in pairs:
        win_counts[p["field_winner"]] += 1
        for k in FIELD_KEYS:
            v = p["field_scores"].get(k, 0.0)
            field_scores[k].append(v)
            if v > 0:
                field_pos[k] += 1

    dominant = max((k for k in FIELD_KEYS), key=lambda k: win_counts.get(k, 0))

    fields_detail = {
        k: {
            "win_count": win_counts.get(k, 0),
            "win_pct":                    _pct(win_counts.get(k, 0), n_total),
            "pct_positive_contribution":  _pct(field_pos[k], n_total),
            "score_distribution":         _dist(field_scores[k]),
        }
        for k in FIELD_KEYS
    }

    return {
        "n_total_pairs":     n_total,
        "dominant_field":    dominant,
        "none_positive_count": win_counts.get("none_positive", 0),
        "pct_none_positive": _pct(win_counts.get("none_positive", 0), n_total),
        "fields":            fields_detail,
    }


# ---------------------------------------------------------------------------
# Metric 4 — Multi-query oracle uplift
# ---------------------------------------------------------------------------

def metric4_multiquery_uplift(pairs: list) -> dict:
    """
    Per (case, gold): compare the best sub-query's score vs the mean sub-query score.
    Uplift % = (best - mean) / |mean| * 100.
    Quantifies the theoretical gain from asking multiple sub-questions.
    WHY per-(case, gold): different golds within a case respond to different sub-queries;
    averaging across golds within a sub-query would obscure this selectivity.
    """
    # (source_case_id, gold_doc_id) -> sub_query_id -> stage1_score
    cg_subq_s1 = defaultdict(dict)
    cg_subq_s2 = defaultdict(dict)

    for p in pairs:
        key = (p["source_case_id"], p["gold_doc_id"])
        cg_subq_s1[key][p["sub_query_id"]] = p["stage1_score"]
        if p["stage2_best"] is not None:
            cg_subq_s2[key][p["sub_query_id"]] = p["stage2_best"]

    s1_uplifts, s1_abs = [], []
    s2_uplifts, s2_abs = [], []
    eps = 1e-9

    for key, subq_map in cg_subq_s1.items():
        if len(subq_map) <= 1:
            continue  # single sub-query: uplift = 0 by definition, exclude
        vals = list(subq_map.values())
        best, mean = max(vals), statistics.fmean(vals)
        s1_uplifts.append((best - mean) / (abs(mean) + eps) * 100)
        s1_abs.append(best - mean)

    for key, subq_map in cg_subq_s2.items():
        if len(subq_map) <= 1:
            continue
        vals = list(subq_map.values())
        best, mean = max(vals), statistics.fmean(vals)
        s2_uplifts.append((best - mean) / (abs(mean) + eps) * 100)
        s2_abs.append(best - mean)

    n_multi_cg = sum(1 for v in cg_subq_s1.values() if len(v) > 1)

    def _uplift_block(uplifts, abs_gains):
        return {
            "n_case_gold_pairs": len(uplifts),
            "uplift_pct_distribution": _dist(uplifts),
            "abs_gain_distribution":   _dist(abs_gains),
            "mean_uplift_pct":         round(statistics.fmean(uplifts), 2) if uplifts else None,
            "median_uplift_pct":       round(statistics.median(uplifts), 2) if uplifts else None,
        }

    return {
        "note": "uplift = (best_subquery_score - mean_subquery_score) / |mean| * 100, per (case, gold) pair",
        "n_case_gold_pairs_with_multiple_subqueries": n_multi_cg,
        "stage1": _uplift_block(s1_uplifts, s1_abs),
        "stage2": _uplift_block(s2_uplifts, s2_abs),
    }


# ---------------------------------------------------------------------------
# Metric 5 — Gold rank in actual retrieval
# ---------------------------------------------------------------------------

def metric5_gold_rank(fill_eval_path: Path) -> dict:
    """
    Cross-reference fill-eval output to find where each gold case actually ranked.
    Uses fill-eval's own ground_truth_cases (same source as oracle gold_ids).
    retrieved_cases is deduped before ranking — fill-eval accumulates duplicate
    doc_ids from stage-2 chunk collapse, and the first occurrence is the true rank.
    """
    with open(fill_eval_path, encoding="utf-8") as f:
        fill_data = json.load(f)

    all_ranks        = []
    first_gold_ranks = []
    n_not_retrieved  = 0
    n_total_gold_pairs = 0

    for rec in fill_data:
        raw = rec.get("retrieved_cases", [])
        seen_dedup = set()
        retrieved = [x for x in raw if x not in seen_dedup and not seen_dedup.add(x)]
        ret_rank  = {doc_id: (i + 1) for i, doc_id in enumerate(retrieved)}

        gold_ids = [str(g) for g in (rec.get("ground_truth_cases") or [])]
        n_total_gold_pairs += len(gold_ids)

        query_ranks = []
        for gid in gold_ids:
            if gid in ret_rank:
                r = ret_rank[gid]
                all_ranks.append(r)
                query_ranks.append(r)
            else:
                n_not_retrieved += 1

        if query_ranks:
            first_gold_ranks.append(min(query_ranks))

    hit_at_k = {
        f"Hit@{k}": _pct(sum(1 for r in all_ranks if r <= k), n_total_gold_pairs)
        for k in [1, 5, 10, 20, 50, 100]
    }

    return {
        "fill_eval_path":             str(fill_eval_path),
        "n_queries":                  len(fill_data),
        "n_total_gold_pairs":         n_total_gold_pairs,
        "n_retrieved":                len(all_ranks),
        "n_not_retrieved":            n_not_retrieved,
        "pct_not_retrieved":          _pct(n_not_retrieved, n_total_gold_pairs),
        "rank_distribution":          _dist(all_ranks),
        "first_gold_rank_per_query":  _dist(first_gold_ranks),
        "hit_at_k":                   hit_at_k,
    }


# ---------------------------------------------------------------------------
# Headline + print
# ---------------------------------------------------------------------------

def _headline(m1, m2, m3, m4, m5) -> dict:
    h = {}
    if m1:
        h["pearson_r_stage1_vs_stage2"]  = m1["pearson_r_stage1_vs_stage2"]
        h["pct_stage2_gt_stage1"]        = m1["pct_stage2_gt_stage1"]
        h["pct_stage2_gt_stage1_by_005"] = m1["pct_stage2_gt_stage1_by_005"]
        h["pct_stage1_neg_stage2_pos"]   = m1["pct_stage1_neg_stage2_pos"]
        h["pct_both_negative"]           = m1["pct_both_negative"]
        h["delta_median"]                = (m1["delta_stage2_minus_stage1"] or {}).get("median")
    if m2:
        h["pct_golds_negative_case_sim"]  = m2["pct_negative"]
        h["pct_golds_below_median_0.05"]  = m2["pct_below_0.05"]
    if m3:
        h["dominant_field"]               = m3["dominant_field"]
        h["dominant_field_win_pct"]       = m3["fields"][m3["dominant_field"]]["win_pct"]
        h["precedents_pct_positive"]      = m3["fields"]["precedents"]["pct_positive_contribution"]
    if m4:
        h["multiquery_stage1_median_uplift_pct"] = m4["stage1"]["median_uplift_pct"]
        h["multiquery_stage2_median_uplift_pct"] = m4["stage2"]["median_uplift_pct"]
    if m5:
        fgr = m5.get("first_gold_rank_per_query") or {}
        h["median_first_gold_rank"]       = fgr.get("median")
        h["pct_gold_not_retrieved"]       = m5["pct_not_retrieved"]
        h["hit_at_10"]                    = m5["hit_at_k"].get("Hit@10")
        h["hit_at_100"]                   = m5["hit_at_k"].get("Hit@100")
    return h


def _print_headlines(hl, m1, m2, m3, m4, m5):
    print("\n========== ORACLE ANALYSIS — HEADLINE NUMBERS ==========")
    if m1:
        print(f"[M1] Stage-2 > Stage-1 in {hl.get('pct_stage2_gt_stage1')}% of pairs"
              f" (by >0.05: {hl.get('pct_stage2_gt_stage1_by_005')}%)."
              f" Pearson r={hl.get('pearson_r_stage1_vs_stage2')}."
              f" Delta median={hl.get('delta_median')}."
              f" Stage1<0 but Stage2>0: {hl.get('pct_stage1_neg_stage2_pos')}%."
              f" Both negative: {hl.get('pct_both_negative')}%.")
    if m2:
        print(f"[M2] {hl.get('pct_golds_negative_case_sim')}% of gold pairs have NEGATIVE case-level similarity."
              f" {hl.get('pct_golds_below_median_0.05')}% fall below stage-1 median threshold (0.05).")
    if m3:
        print(f"[M3] Dominant field: '{hl.get('dominant_field')}' wins {hl.get('dominant_field_win_pct')}% of gold pairs."
              f" 'precedents' contributes positively in only {hl.get('precedents_pct_positive')}% of pairs.")
    if m4:
        print(f"[M4] Multi-query best-sub-query lifts stage-1 gold sim by median +{hl.get('multiquery_stage1_median_uplift_pct')}%"
              f", stage-2 by +{hl.get('multiquery_stage2_median_uplift_pct')}%.")
    if m5:
        print(f"[M5] Median rank of first gold in retrieval: {hl.get('median_first_gold_rank')}."
              f" {hl.get('pct_gold_not_retrieved')}% of golds not retrieved at all."
              f" Hit@10={hl.get('hit_at_10')}%  Hit@100={hl.get('hit_at_100')}%.")
    print("=========================================================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Post-processing diagnostic over a completed gold oracle JSON. No DB or API needed."
    )
    ap.add_argument("--oracle", required=True,
                    help="Path to gold oracle JSON (output of gold_oracle_similarity.py)")
    ap.add_argument("--fill-eval", default=None,
                    help="Path to fill-eval multiquery JSON — enables metric 5 (gold rank in retrieval)")
    ap.add_argument("--output", default=None,
                    help="Output path (default: oracle_analysis_<stem>.json next to oracle file)")
    args = ap.parse_args()

    oracle_path = Path(args.oracle)
    if not oracle_path.exists():
        print(f"[FAIL] oracle file not found: {oracle_path}")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        stem = oracle_path.stem
        for prefix in ("gold_oracle_similarity_", "gold_oracle_"):
            stem = stem.replace(prefix, "")
        out_path = oracle_path.parent / f"oracle_analysis_{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading oracle: {oracle_path}")
    records = load_oracle(oracle_path)
    print(f"  {len(records)} records")

    print("Building (sub-query, gold) pair list...")
    pairs = build_pairs(records)
    print(f"  {len(pairs)} pairs")

    print("\nMetric 1: Stage-2 vs Stage-1 delta analysis...")
    m1 = metric1_stage_delta(pairs)
    print(f"  stage2>stage1: {m1['pct_stage2_gt_stage1']}%  |  "
          f"pearson r={m1['pearson_r_stage1_vs_stage2']}  |  "
          f"delta median={( m1['delta_stage2_minus_stage1'] or {}).get('median')}")

    print("Metric 2: Hard gold fraction...")
    m2 = metric2_hard_gold(pairs)
    print(f"  negative: {m2['pct_negative']}%  |  "
          f"below 0.05: {m2['pct_below_0.05']}%  |  "
          f"below 0.10: {m2['pct_below_0.10']}%")

    print("Metric 3: Field dominance...")
    m3 = metric3_field_dominance(pairs)
    wins = "  ".join(f"{k}={m3['fields'][k]['win_pct']}%" for k in FIELD_KEYS)
    print(f"  dominant: {m3['dominant_field']}  |  none_positive: {m3['pct_none_positive']}%  |  {wins}")

    print("Metric 4: Multi-query oracle uplift...")
    m4 = metric4_multiquery_uplift(pairs)
    print(f"  stage1 median uplift: +{m4['stage1']['median_uplift_pct']}%  |  "
          f"stage2 median uplift: +{m4['stage2']['median_uplift_pct']}%")

    m5 = None
    if args.fill_eval:
        fp = Path(args.fill_eval)
        if not fp.exists():
            print(f"[WARN] fill-eval not found: {fp} — skipping metric 5")
        else:
            print("Metric 5: Gold rank in retrieval...")
            m5 = metric5_gold_rank(fp)
            fgr = m5.get("first_gold_rank_per_query") or {}
            print(f"  median first-gold rank: {fgr.get('median')}  |  "
                  f"not retrieved: {m5['pct_not_retrieved']}%  |  "
                  f"Hit@10: {m5['hit_at_k'].get('Hit@10')}%  "
                  f"Hit@100: {m5['hit_at_k'].get('Hit@100')}%")
    else:
        print("Metric 5: skipped (pass --fill-eval to enable)")

    hl = _headline(m1, m2, m3, m4, m5)
    _print_headlines(hl, m1, m2, m3, m4, m5)

    payload = {
        "_headline": hl,
        "metric1_two_stage_rescue":    m1,
        "metric2_hard_gold_fraction":  m2,
        "metric3_field_dominance":     m3,
        "metric4_multiquery_uplift":   m4,
    }
    if m5:
        payload["metric5_gold_rank_in_retrieval"] = m5

    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
    print(f"\n[OK] Written -> {out_path}")


if __name__ == "__main__":
    main()
