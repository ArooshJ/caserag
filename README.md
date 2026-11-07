# Evaluating the Limits of Semantic Similarity in Indian Legal Precedent Retrieval

Code and results behind the paper *Evaluating the Limits of Semantic Similarity in Indian Legal Precedent Retrieval*.

This is not a new retrieval system. The retrieval pipeline here (called CaseRAG in the paper) is an instrument for one question: when similarity-based retrieval misses the precedents that judges cited, is that a loss in the pipeline or a limit of similarity itself? The paper studies this on Indian precedent retrieval, using [IL-PCSR](https://huggingface.co/datasets/Exploration-Lab/IL-PCSR) with 2,199 issue-based questions generated from 622 of its test cases.

## What is in this repository

| Path | Contents |
|---|---|
| `ilpcsr_caserag_pipeline_functions.py` | The two-stage pipeline: Stage 1 ranks cases by a weighted sum of five field similarities; Stage 2 ranks paragraph chunks within the top cases |
| `evaluation/` | Question generation, retrieval runs, fusion, metrics, the diagnostic analyses, the LLM audits, and the plot scripts |
| `evaluation/ilpcsr_multiquery_evaluation_dataset.json` | The 2,199 generated questions with their gold precedents |
| `evaluation/results/` | Saved results behind every table and figure (see below) |
| `ingestion/` | Loading cases into Postgres/pgvector, chunking and embedding, summary generation, data checks |
| `notebooks/` | Dataset export from Hugging Face, and the deduplication, embedding-gap and summary-quality checks |

**Not included:** the IL-PCSR dataset itself (get it from Hugging Face; it is gated), the database, the generated case summaries, and the raw per-question rankings (about 3 GB). The paper's numbers can be checked from the saved results without them.

## Paper to file map

All paths are under `evaluation/results/`. Plot scripts are in `evaluation/`.

| Paper | Saved results | Script |
|---|---|---|
| Table 1, Fig. 2: Dense vs CaseRAG, both embedders | `caserag_emb3_cosine_multiquery_fused_k200k100.json`, `dense_emb3_cosine_multiquery_fused.json` (Kanon-2); `caserag_emb1_multiquery_fused_k200k100.json`, `dense_emb1_multiquery_fused.json` (BGE); `significance.json` | `significance_forest_plot.py`, `significance.py` |
| Table 2: Stage-1 pool size k1 | `model=3_k1=50_k2=100_cosine_fused_results.json`, `model=3_k1=100_k2=100_cosine_fused_results.json`, and the k1 = 200 file above | `k1_ablation_plot.py` |
| Table 3: field-weight sensitivity | `weight_ablation_emb3_*.json` | `field_weight_ablation_plot.py` |
| Table 4: BM25 hybrids | `hybrid_rrf_bm25ng5_*_multiquery_results.json`, `bm25_ng5_multiquery_fused.json` | `hybrid_fusion_plot.py` |
| §6.2: query-regime gap for BM25 | `bm25_ng5_fulltext_singlequery.json`, `bm25_ng5_multiquery_fused.json` | `bm25_regime_gap_plot.py` |
| Fig. 3: Recall@K | the Kanon-2 fused files above, `bm25_ng5_multiquery_fused.json` | `recall_at_k_curve_plot.py` |
| §6.3, Fig. 4: similarity ceiling | `semantic_ceiling.csv` | `semantic_ceiling_plot.py` (see note) |
| §6.4, Table 5, Fig. 5: statutory grounding | `statutory_grounding.csv` | `statutory_grounding_plot.py` (see note) |
| §6.5, Tables 6–7, Fig. 6: faithfulness audits | `query_faithfulness_results.json`, `corpus_summary_faithfulness_results.json` | `query_faithfulness_plot.py`, `corpus_faithfulness_plot.py`, then `faithfulness_comparison_plot.py` |
| Appendix A: qualitative examples | `qualitative_examples/subqueries/`, `qualitative_examples/manifest.json`, `qualitative_semantic_gap_examples.json` | — |

Other files in `results/` are supporting runs: per-question (unfused) results, BM25 unigram, Kanon-2 with L2 distance, oracle analyses, error analysis and rank stratification.

**Note on §6.3 and §6.4.** The two scripts compute these analyses from the raw rankings, which are not included. `semantic_ceiling.csv` and `statutory_grounding.csv` are their saved outputs, one row per question with gold (2,195 rows): IDs, hit rank, scores, courts and statute-overlap flags, with no case text. Every number in §6.3 and §6.4 can be recomputed from these two tables. Table 5, for example, compares `top1_statute_overlap` between rows with `hit_bucket` = `Missed` and the rest.

## Reproducing the figures from saved results

Requires Python with numpy, scipy and matplotlib (`requirements.txt` lists everything). Run the scripts from the repository root:

```bash
python evaluation/recall_at_k_curve_plot.py
python evaluation/significance_forest_plot.py
python evaluation/k1_ablation_plot.py
python evaluation/query_faithfulness_plot.py
```

Each writes a figure to `research_paper/ablation/figures/` and a CSV to `research_paper/ablation/data/`, creating the folders if needed; pass `--plot-out` and `--csv-out` to change this. Run `query_faithfulness_plot.py` and `corpus_faithfulness_plot.py` before `faithfulness_comparison_plot.py`, which reads their CSVs.

To recompute the significance tests in Table 1, for example for Kanon-2:

```bash
python evaluation/significance.py \
  --files evaluation/results/caserag_emb3_cosine_multiquery_fused_k200k100.json \
          evaluation/results/dense_emb3_cosine_multiquery_fused.json \
  --metrics AP --output significance_check.json
```

## Re-running the pipeline (not tested from this repository)

The full pipeline needs services the saved results do not: Postgres with pgvector, the [Isaacus](https://isaacus.com) API for Kanon-2 embeddings, BGE (`BAAI/bge-large-en-v1.5`, run locally with sentence-transformers), and an LLM provider for question generation, summaries and the audits. The paper used Gemma 4 (31B) for questions, Llama 4 Scout for case summaries and MiniMax M3 as the judge, all through Novita. The script defaults match these runs (`--provider novita`; `google/gemma-4-31b-it` for questions, `minimax/minimax-m3` for both judges); other models can be chosen with `--model`. LLM outputs and API embeddings will not reproduce exactly, so expect small differences from the paper.

The order the experiments ran in:

1. Export IL-PCSR from Hugging Face to local JSON (`notebooks/ilpcsr_eda_huggingface.ipynb`).
2. Generate the structured case summaries (`ingestion/ilpcsr_novita_summarize.ipynb`; `ingestion/ilpcsr_Colab_Distributed_Generation.ipynb` is a Colab variant).
3. Load cases into the database, chunk and embed them (`ingestion/ingest_ilpcsr_to_db.py`, `ingestion/ISAACUS_EMBEDDING_INGESTION.ipynb` for Kanon-2, `ingestion/embed_missing_chunks.py`).
4. Generate questions (`evaluation/ilpcsr_multiquery_prep_pipeline.py`), or use the released question file.
5. Run retrieval: `ilpcsr_multiquery_fill_eval.py` (CaseRAG), `dense_chunk_retrieval.py`, `bm25_retrieval.py`, `full_text_bm25_retrieval.py`; then `fuse_multiquery.py`, `evaluate.py` (retrieval metrics) and `hybrid_retrieval.py`.
6. Diagnostics: `gold_oracle_similarity.py`, `oracle_analysis.py`, `semantic_ceiling_plot.py`, `statutory_grounding_plot.py`, `field_similarity_matrix.py` and `weight_sensitivity.py`.
7. Audits: `ilpcsr_query_faithfulness_judge.py` and `ilpcsr_corpus_summary_faithfulness_judge.py`.

Each script's docstring gives its arguments and default paths. Most default to reading from `evaluation/data/` and `il-pcsr_dataset/`, which the steps above create.

**Environment variables** (names only; set your own values):

| Variable | Used for |
|---|---|
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_SSLMODE` | Postgres connection |
| `ISAACUS_API_KEY` | Kanon-2 embeddings |
| `NOVITA_API_KEY`, `NOVITA_BASE_URL` | Question generation, summaries, audits |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Gemini, an alternative provider in some scripts |
| `GROQ_API_KEY` | Groq, an alternative summary provider |
| `HF_ACCESS_TOKEN` | Downloading the gated IL-PCSR dataset |
| `CASERAG_DRIVE_DIR` | Your Google Drive folder, for the Colab summary notebook |

**Warning:** `notebooks/ilpcsr_dedup_analysis.ipynb` deletes duplicate chunk rows from the database and rebuilds an index. It records a clean-up that was already applied before the experiments; do not run it against a database you want to keep unchanged.

## License and data terms

The code is released under the MIT License (see `LICENSE`). IL-PCSR is licensed CC BY-NC-SA 4.0; the generated questions and the saved results derived from it are shared under the same terms.

## Citation

If you use this repository, please cite the paper and IL-PCSR:

```bibtex
@misc{joshi2026semantic,
  title  = {Evaluating the Limits of Semantic Similarity in Indian Legal Precedent Retrieval},
  author = {Joshi, Aroosh and Choudhari, Yaddhnyesh and Kanyawar, Aryan and Tiwari, Aditya and Mulla, Nikahat and Halbe, Aparna},
  year   = {2026},
  note   = {Manuscript under review}
}

@inproceedings{paul2025ilpcsr,
  title     = {{IL-PCSR}: Legal Corpus for Prior Case and Statute Retrieval},
  author    = {Paul, S. and Ghumare, D. and Goyal, P. and Ghosh, S. and Modi, A.},
  booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  note      = {arXiv:2511.00268}
}
```
