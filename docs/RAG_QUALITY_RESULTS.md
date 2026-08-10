# RAG quality optimization results

This report separates retrieval, answer, citation, and abstention diagnostics.
All numbers are local measurements on a three-paper, 127-chunk corpus and are not
claims about general-domain RAG quality.

## Retrieval A/B

BGE-M3 dense retrieval was compared with BGE-M3 top-12 candidate retrieval plus
`BAAI/bge-reranker-v2-m3`, returning top-3 in both cases. Embeddings ran on CPU;
reranking ran on an RTX 4090. The first query includes model loading, so steady
latency excludes it.

| Evaluation | Dense | Dense + reranker |
|---|---:|---:|
| Source Hit@3 (15 cases) | 1.000 | 1.000 |
| Source MRR | 1.000 | 1.000 |
| Evidence-chunk Hit@3 (8 cases) | 1.000 | 1.000 |
| Evidence-chunk MRR | 1.000 | 0.9375 |
| Evidence steady p50 retrieval | 105.9 ms | 166.0 ms |

The reranker did not improve the saturated source-level task, moved one relevant
evidence result from rank 1 to rank 2, and increased steady retrieval p50 by
56.8%. It remains an optional, measured trade-off rather than the default.

## Answer and citation diagnostic

Eight deterministic Qwen2.5-7B-Instruct Q4 cases were used: five answerable
questions and three out-of-corpus questions. Key-fact coverage is measured by
manually specified term groups; citation checks only verify that an answer uses
existing source IDs. These are useful regression signals, not semantic judges.

| Configuration | Fact-group recall | Citation presence/valid-ID | OOD abstention |
|---|---:|---:|---:|
| Dense, original prompt, 192 tokens | 0.6167 | 0.80 / 0.80 | 1.00 |
| Reranker, original prompt, 192 tokens | 0.7000 | 0.80 / 0.80 | 1.00 |
| Reranker, strict bullet citations, 192 tokens | 0.5000 | 1.00 / 1.00 | 1.00 |
| Reranker, balanced citations, 256 tokens | 0.6667 | 1.00 / 1.00 | 1.00 |

The final prompt uses the balanced version: it improves citation compliance over
the original dense baseline while retaining more factual coverage than the strict
format. The unconstrained reranker run had the best lexical fact coverage but
missed citations on one of five positive answers. This is a multi-objective
trade-off, not a universal quality win.

## Abstention boundary

The UI exposes the 0.5672 dense-score gate as **experimental and off by default**.
It was fitted on only 15 positive and 5 negative queries. Positive and negative
score ranges overlap; the calibration set retained 13/15 positives and rejected
5/5 negatives. Answer diagnostics used three of those negative topics and are not
an independent held-out validation.

When reranking is enabled, the gate uses the maximum dense score from the full
candidate pool, not only the reranked top-3. Reranker scores are deliberately not
thresholded because they have not been calibrated for abstention.

## Reproduce

```bash
RAG_EMBED_DEVICE=cpu python evaluate_retrieval.py \
  --output results/retrieval/dense_source_n15.json

RAG_EMBED_DEVICE=cpu RAG_RERANK_DEVICE=cuda:0 python evaluate_retrieval.py \
  --rerank --candidate-k 12 \
  --output results/retrieval/rerank_source_n15.json

RAG_EMBED_DEVICE=cpu python evaluate_evidence.py \
  --output results/retrieval/dense_evidence_n8.json

RAG_EMBED_DEVICE=cpu RAG_RERANK_DEVICE=cuda:0 python evaluate_evidence.py \
  --rerank --candidate-k 12 \
  --output results/retrieval/rerank_evidence_n8.json

RAG_EMBED_DEVICE=cpu RAG_RERANK_DEVICE=cuda:0 python evaluate_answers.py \
  --rerank --candidate-k 12 --max-tokens 256 \
  --output results/quality/ollama_q4_rerank_balanced_prompt_n8.json
```

## Remaining limitations

- The corpus has only three related systems papers and the test sets are small.
- Evidence labels were manually derived from the indexed PDFs but were not
  independently double-annotated.
- Term-group recall can miss valid paraphrases and does not measure entailment.
- Citation-ID validity does not prove that a cited chunk supports each claim.
- The abstention threshold needs a larger held-out positive/negative set.
