import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import argparse
import sys
from sentence_transformers import SentenceTransformer, util



class RetrievalMetrics:
    """Compute all retrieval evaluation metrics."""
    
    def __init__(self, k_values: List[int] = [1,2,3,5], semantic_model: str = 'all-MiniLM-L6-v2',
                 collapse: str = 'first'):
        """
        k_values: K cutoffs for rank-based metrics (P@K, Recall@K, NDCG@K, Hit@K).
        semantic_model: HuggingFace model name for semantic_recall_at_k (loaded lazily on first call).
        collapse: how a chunk-derived ranking (a case repeated once per matching chunk) is
                  folded into a ranking of unique cases before scoring.
                  'first' (default) = keep each case at its best/highest-ranked chunk (max-pool);
                  'count' = order cases by how many chunks they contributed (frequency-pool).
        """
        self.k_values = k_values
        self._semantic_model_name = semantic_model
        self.collapse = collapse if collapse in ('first', 'count') else 'first'
        self.embed_model = None  # loaded lazily in semantic_recall_at_k

    def _collapse_to_cases(self, retrieved: List[str]) -> List[str]:
        """Collapse a chunk-derived ranking with repeated case IDs into a ranking of UNIQUE cases.

        Why this exists: stage-2 `retrieved_cases` repeats a case once per matching chunk
        (~100 chunk entries -> ~10 distinct cases). Case-level metrics (P@K, NDCG@K, BPREF)
        must count each case once; counting repeats inflates them (NDCG even exceeds 1).
        Called once at the top of compute_all so every metric shares ONE case ranking.
        Without it, dcg_at_k / precision_at_k double-count duplicate cases.

        Non-destructive: returns a NEW list; the input `retrieved` is never mutated and no
        record is discarded from the caller's data - a case seen at ranks 1..6 simply scores
        once, at its best rank.
        """
        if not retrieved:
            return []
        first_index = {}  # case_id -> earliest (best) position in the original ranking
        for i, cid in enumerate(retrieved):
            if cid not in first_index:
                first_index[cid] = i
        if self.collapse == 'count':
            from collections import Counter
            counts = Counter(retrieved)
            # frequency desc, tie broken by best (earliest) chunk position
            return sorted(first_index.keys(), key=lambda c: (-counts[c], first_index[c]))
        # 'first': dict preserves insertion order, so keys are already in best-chunk order
        return list(first_index.keys())


    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Precision@K: Proportion of retrieved docs in top-K that are relevant."""
        # --- THIS FUNCTION IS CORRECT ---
        if k == 0 or len(retrieved) == 0:
            return 0.0
        retrieved_at_k = retrieved[:k]
        relevant_at_k = sum(1 for case_id in retrieved_at_k if case_id in relevant)
        return relevant_at_k / k
    
    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Recall@K: Proportion of relevant docs found in top-K."""
        # --- THIS FUNCTION IS NOW FIXED ---
        if len(relevant) == 0:
            return 0.0
        
        # FIX: Convert to set to handle duplicate retrieved IDs.
        # This prevents counting the same relevant doc multiple times.
        retrieved_at_k_unique = set(retrieved[:k])
        
        # Count based on the *unique* set
        relevant_at_k = sum(1 for case_id in retrieved_at_k_unique if case_id in relevant)
        return relevant_at_k / len(relevant)
    
    def average_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Average Precision: Precision at each relevant document position."""
        # --- THIS FUNCTION IS NOW FIXED ---
        if len(relevant) == 0:
            return 0.0
        
        num_relevant = 0
        sum_precisions = 0.0
        
        # FIX: Keep track of relevant docs we've already found
        seen_relevant = set()
        
        for i, case_id in enumerate(retrieved, 1):
            
            # FIX: Only count if it's relevant AND we haven't seen it before
            if case_id in relevant and case_id not in seen_relevant:
                
                num_relevant += 1
                precision_at_i = num_relevant / i
                sum_precisions += precision_at_i
                
                # Add this doc to our 'seen' list
                seen_relevant.add(case_id)
        
        return sum_precisions / len(relevant)
    
    def reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Reciprocal Rank: 1 / position of first relevant document."""
        # --- THIS FUNCTION IS CORRECT ---
        for i, case_id in enumerate(retrieved, 1):
            if case_id in relevant:
                return 1.0 / i
        return 0.0
    
    def dcg_at_k(self, retrieved: List[str], relevance_grades: Dict[str, int], k: int) -> float:
        """Discounted Cumulative Gain at K."""
        # --- THIS FUNCTION IS CORRECT ---
        dcg = 0.0
        for i, case_id in enumerate(retrieved[:k], 1):
            grade = relevance_grades.get(case_id, 0)
            dcg += (2 ** grade - 1) / np.log2(i + 1)
        return dcg
    
    def ndcg_at_k(self, retrieved: List[str], relevance_grades: Dict[str, int], k: int) -> float:
        """Normalized Discounted Cumulative Gain at K."""
        # --- THIS FUNCTION IS CORRECT ---
        dcg = self.dcg_at_k(retrieved, relevance_grades, k)
        
        # Ideal DCG: sort by relevance grade descending
        ideal_retrieved = sorted(relevance_grades.keys(), 
                                key=lambda x: relevance_grades[x], 
                                reverse=True)
        idcg = self.dcg_at_k(ideal_retrieved, relevance_grades, k)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    def bpref(self, retrieved: List[str], relevant: Set[str], judged: Set[str]) -> float:
        """BPREF: For incomplete relevance judgments (citation-based evaluation)."""
        # --- THIS FUNCTION IS CORRECT ---
        if len(relevant) == 0:
            return 0.0
        
        R = len(relevant)
        non_relevant_judged = judged - relevant
        
        score = 0.0
        for case_id in relevant:
            if case_id not in retrieved:
                continue
            
            # Count non-relevant judged docs ranked before this relevant doc
            try:
                rank = retrieved.index(case_id)
            except ValueError:
                # Should not happen if case_id is in retrieved, but as a safeguard
                continue 
                
            n_before = sum(1 for i in range(rank) if retrieved[i] in non_relevant_judged)
            
            score += 1 - min(n_before, R) / R
        
        return score / R

    def hit_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Hit@K: 1 if any relevant document appears in top-K, else 0."""
        # --- THIS FUNCTION IS CORRECT ---
        if len(relevant) == 0:
            return 0.0
        retrieved_at_k = retrieved[:k]
        return 1.0 if any(case_id in relevant for case_id in retrieved_at_k) else 0.0

    def semantic_recall_at_k(
        self, retrieved_texts: List[str], gold_text: str, k: int, threshold: float = 0.75
        ) -> float:
        
        """Semantic Recall@K: considers retrieved chunks semantically similar to gold as relevant."""
        # --- THIS FUNCTION IS CORRECT ---
        if not gold_text or len(retrieved_texts) == 0:
            return 0.0
        if self.embed_model is None:
            print(f"  -> Loading semantic model '{self._semantic_model_name}'...")
            self.embed_model = SentenceTransformer(self._semantic_model_name)
        gold_emb = self.embed_model.encode(gold_text, convert_to_tensor=True)
        retrieved_embs = self.embed_model.encode(retrieved_texts[:k], convert_to_tensor=True)
        cosine_scores = util.cos_sim(gold_emb, retrieved_embs)[0]
        relevant_hits = (cosine_scores > threshold).sum().item()
        return relevant_hits / 1.0  # one gold per query

    
    def compute_all(self, retrieved: List[str], relevant: Set[str], 
                   grades: Dict[str, int], judged: Set[str]) -> Dict[str, float]:
        """Compute all retrieval metrics for a single query."""
        # Collapse the chunk-derived ranking to unique cases ONCE so every metric below
        # scores each case a single time (fixes duplicate-inflated NDCG@K / P@K / BPREF).
        retrieved = self._collapse_to_cases(retrieved)
        metrics = {
            "AP": self.average_precision(retrieved, relevant),
            "RR": self.reciprocal_rank(retrieved, relevant),
            "BPREF": self.bpref(retrieved, relevant, judged)
        }
        
        for k in self.k_values:
            metrics[f"P@{k}"] = self.precision_at_k(retrieved, relevant, k)
            metrics[f"Recall@{k}"] = self.recall_at_k(retrieved, relevant, k)
            metrics[f"NDCG@{k}"] = self.ndcg_at_k(retrieved, grades, k)
            metrics[f"Hit@{k}"] = self.hit_at_k(retrieved, relevant, k)
            
        
        return metrics