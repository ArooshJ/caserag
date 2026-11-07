# evaluate.py
"""
Legal Precedent Retrieval Evaluator
Computes retrieval metrics for stage 1 (case shortlist) and stage 2 (final ranking).

The output keeps empty "generation" fields so its layout matches the saved results in
evaluation/results/; answer generation is not part of this release.

Usage:
    python evaluate.py --input data.json --output results.json

    # Python
    from evaluate import LegalPrecedentEvaluator
    evaluator = LegalPrecedentEvaluator()
    results = evaluator.evaluate_from_file("data.json", "results.json")
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List
import argparse
import sys

from retrieval_metrics import RetrievalMetrics


class LegalPrecedentEvaluator:
    """Retrieval evaluator for legal precedent retrieval systems."""

    def __init__(self, k1_values: List[int] = [1,3,5,10,20,30,50,100],
                 k2_values: List[int] = [1,3,5,10,20,30,50,100],
                 collapse: str = 'first'):
        """
        Args:
            k1_values: K values for stage 1 retrieval metrics (P@K, Recall@K, NDCG@K)
            k2_values: K values for stage 2 retrieval metrics
            collapse: case-collapse policy for chunk-derived rankings ('first' = best chunk,
                      'count' = chunk frequency); applied to both stages.
        """
        self.retrieval_metrics_s1 = RetrievalMetrics(k1_values, collapse=collapse)
        self.retrieval_metrics_s2 = RetrievalMetrics(k2_values, collapse=collapse)

    def load_data(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        print(f"[OK] Loaded {len(data)} evaluation samples")
        return data

    def _compute_retrieval(self, retrieved: List, ground_truth: List, metrics: RetrievalMetrics) -> Dict:
        """Run retrieval metrics for one retrieved list against ground truth."""
        if not retrieved or not ground_truth:
            return {}
        relevant = set(ground_truth)
        judged = set(retrieved) | relevant
        grades = {case: 1 for case in ground_truth}
        return metrics.compute_all(retrieved, relevant, grades, judged)

    def evaluate_single_item(self, item: Dict) -> Dict:
        """Evaluates retrieval for a single item."""
        ground_truth = item.get("ground_truth_cases", [])
        return {
            "query_id": item.get("source_case_id", "unknown"),
            "question": item.get("question", ""),
            "retrieval_metrics": {
                "stage1": self._compute_retrieval(
                    item.get("stage1_case_ids", []), ground_truth, self.retrieval_metrics_s1
                ),
                "stage2": self._compute_retrieval(
                    item.get("retrieved_cases", []), ground_truth, self.retrieval_metrics_s2
                ),
            },
            "generation_metrics": {}
        }

    def _finalize_retrieval(self, agg: defaultdict) -> Dict:
        """Mean-aggregate a retrieval accumulator and add MAP/MRR aliases."""
        if not agg:
            return {}
        out = {m: np.mean(s) for m, s in agg.items()}
        if "AP" in out:
            out["MAP"] = out["AP"]
        if "RR" in out:
            out["MRR"] = out["RR"]
        return out

    def evaluate_complete(self, data: List[Dict]) -> Dict:
        """Evaluates retrieval across the dataset."""
        print("\n" + "=" * 75)
        print("LEGAL PRECEDENT RETRIEVAL EVALUATION")
        print("Metric families: Retrieval (stage1 + stage2)")
        print("=" * 75)

        print("\n[1/2] Evaluating each query individually...")
        all_per_query_results = [self.evaluate_single_item(item) for item in data]
        print(f"[OK] Evaluated {len(all_per_query_results)} queries.")

        avg_stage1 = defaultdict(list)
        avg_stage2 = defaultdict(list)
        for res in all_per_query_results:
            for metric, score in res["retrieval_metrics"]["stage1"].items():
                avg_stage1[metric].append(score)
            for metric, score in res["retrieval_metrics"]["stage2"].items():
                avg_stage2[metric].append(score)

        print("\n[2/2] Averaging scores across all queries...")
        final_avg_retrieval = {
            "stage1": self._finalize_retrieval(avg_stage1),
            "stage2": self._finalize_retrieval(avg_stage2),
        }
        print("[OK] Metrics computed.")

        return {
            "average_metrics": {
                "retrieval": final_avg_retrieval,
                "generation": {},
            },
            "per_query_results": all_per_query_results
        }

    def print_results(self, results: Dict):
        """Pretty print evaluation results."""
        print("\n" + "=" * 75)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 75)

        avg_retrieval = results["average_metrics"]["retrieval"]
        priority = ["MAP", "MRR", "BPREF"]
        for stage_name in ("stage1", "stage2"):
            stage_metrics = avg_retrieval.get(stage_name, {})
            if not stage_metrics:
                continue
            print("\n" + "-" * 75)
            print(f"AVERAGE RETRIEVAL METRICS - {stage_name.upper()} (ACROSS ALL QUERIES)")
            print("-" * 75)
            for metric in priority:
                if metric in stage_metrics:
                    print(f"  {metric:20s}: {stage_metrics[metric]:.4f}")
            print()
            for metric, score in sorted(stage_metrics.items()):
                if metric not in priority and metric not in ["AP", "RR"]:
                    print(f"  {metric:20s}: {score:.4f}")

        print("\n" + "=" * 75)

    def evaluate_from_file(self, filepath: str, output_file: str = None) -> Dict:
        """Load data, evaluate, print, and optionally save results."""
        data = self.load_data(filepath)
        results = self.evaluate_complete(data)
        self.print_results(results)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Complete results saved to: {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Legal Precedent Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python evaluate.py --input data.json --output results.json
        """
    )

    parser.add_argument("--input", "-i", required=True, help="Input JSON file with evaluation data")
    parser.add_argument("--output", "-o", default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--k1-values", nargs="+", type=int, default=[1,3,5,10,20,30,50,100], help="K values for stage 1 retrieval metrics")
    parser.add_argument("--k2-values", nargs="+", type=int, default=[1,3,5,10,20,30,50,100], help="K values for stage 2 retrieval metrics")
    parser.add_argument("--collapse", choices=["first", "count"], default="first",
                        help="Case-collapse policy for chunk-derived stage-2 rankings: "
                             "'first' = best chunk (max-pool, default), 'count' = chunk frequency")

    args = parser.parse_args()

    evaluator = LegalPrecedentEvaluator(
        k1_values=args.k1_values,
        k2_values=args.k2_values,
        collapse=args.collapse
    )
    evaluator.evaluate_from_file(args.input, args.output)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python evaluate.py --input data.json [--output results.json]")
        print("For all options: python evaluate.py --help")
