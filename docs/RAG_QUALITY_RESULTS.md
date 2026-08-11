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

## Sparse and hybrid retrieval

A dependency-free Okapi BM25 implementation (`k1=1.5`, `b=0.75`) was compared
with dense BGE-M3 and equal-weight reciprocal-rank fusion (`RRF k=60`). Hybrid
retrieval fused the top 12 candidates from each component and returned top-3.
The 12 lexical challenge cases were written before running the comparison and
target exact APIs, symbols, acronyms, and equations in the indexed papers.

| Evaluation | Retrieval | Hit@3 | MRR | Steady p50 |
|---|---|---:|---:|---:|
| Evidence labels (8) | Dense | 1.0000 | 1.0000 | 99.319 ms |
| Evidence labels (8) | BM25 | 1.0000 | 0.9375 | 0.233 ms |
| Evidence labels (8) | Equal RRF | 1.0000 | 0.9375 | 94.894 ms |
| Lexical challenge (12) | Dense | 0.9167 | 0.8333 | 102.655 ms |
| Lexical challenge (12) | BM25 | 1.0000 | 1.0000 | 0.211 ms |
| Lexical challenge (12) | Equal RRF | 0.9167 | 0.9167 | 96.639 ms |

The original 15 source-label cases were saturated: dense and RRF both scored
Hit@3/MRR 1.0000/1.0000, while BM25 scored 1.0000/0.9556. On the lexical set,
BM25 recovered all cases and improved both metrics. Equal RRF improved MRR over
dense but did not recover the missed top-3 case: the relevant SGLang cache-hit
chunk ranked 11th in dense, 1st in BM25, and 4th after fusion because nearby
overlapping chunks occupied the first three fused ranks.

No fusion weights were tuned after observing this result. Dense remains the
application default; sparse and hybrid modes are evaluation options. BM25's
sub-millisecond steady latency reflects a Python in-memory scan of only 127
chunks and must not be extrapolated to a distributed production index. The
lexical set is corpus-derived and single-annotator, not held-out external data.

Reproduce the lexical comparison:

```bash
RAG_EMBED_DEVICE=cpu python evaluate_evidence.py \
  --cases evaluation/lexical_cases.json --retrieval dense \
  --output results/retrieval/dense_lexical_n12.json
python evaluate_evidence.py --cases evaluation/lexical_cases.json \
  --retrieval sparse --output results/retrieval/sparse_lexical_n12.json
RAG_EMBED_DEVICE=cpu python evaluate_evidence.py \
  --cases evaluation/lexical_cases.json --retrieval hybrid \
  --candidate-k 12 --rrf-k 60 \
  --output results/retrieval/hybrid_rrf60_lexical_n12.json
```

## Parent-child retrieval

A separate Chroma collection was built with the same 127 labeled 512-token
parents as the dense baseline. Each parent was split into 192-token children
with 32-token overlap, producing 394 searchable child vectors. Retrieval queried
the top 12 children, deduplicated by `(source, parent_chunk_idx)`, and returned
the top three complete parents. This preserves the existing evidence labels and
changes only the search granularity.

| Evaluation | Retrieval | Hit@3 | MRR | Steady p50 |
|---|---|---:|---:|---:|
| Evidence labels (8) | Dense | 1.0000 | 1.0000 | 99.319 ms |
| Evidence labels (8) | Parent-child | 1.0000 | 1.0000 | 121.943 ms |
| Lexical challenge (12) | Dense | 0.9167 | 0.8333 | 102.655 ms |
| Lexical challenge (12) | Parent-child | 0.9167 | 0.8750 | 107.916 ms |
| Source labels (15) | Dense | 1.0000 | 1.0000 | 103.458 ms |
| Source labels (15) | Parent-child | 1.0000 | 1.0000 | 103.846 ms |

Parent-child retrieval preserved the saturated evidence and source metrics and
slightly improved lexical MRR, but it did not recover the missed SGLang cache-hit
case. Its returned parents for that query were chunks 43, 19, and 18; the labeled
relevant chunks were 9 and 12. It was also slower on the evidence and lexical
sets. Therefore this path remains an explicit evaluation mode rather than the
application default. Child size, overlap, and candidate count were not retuned
after observing the result.

Reproduce the separate index and evaluation:

```bash
python -m rag.ingest \
  --arxiv-ids 2309.06180 2211.17192 2312.07104 \
  --embed-device cuda:0 --parent-child

RAG_EMBED_DEVICE=cpu python evaluate_evidence.py \
  --retrieval parent_child --candidate-k 12 \
  --output results/retrieval/parent_child_evidence_n8.json

RAG_EMBED_DEVICE=cpu python evaluate_evidence.py \
  --cases evaluation/lexical_cases.json \
  --retrieval parent_child --candidate-k 12 \
  --output results/retrieval/parent_child_lexical_n12.json

RAG_EMBED_DEVICE=cpu python evaluate_retrieval.py \
  --retrieval parent_child --candidate-k 12 \
  --output results/retrieval/parent_child_source_n15.json
```

## External held-out retrieval: BEIR/SciFact

The retrieval methods were next evaluated without project-authored questions on
the full BEIR SciFact test split: 5,183 scientific abstracts, 300 test queries,
and official qrels. Documents were indexed once as title-plus-abstract units.
BGE-M3 used a separate cosine Chroma collection; BM25 retained `k1=1.5` and
`b=0.75`; equal-weight RRF retained `k=60`. All methods retrieved top-100, and
no parameters were changed after seeing the results.

| Retrieval | NDCG@10 | MAP@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| Dense BGE-M3 | 0.641458 | 0.591970 | 0.607956 | 0.775111 | 0.903667 |
| BM25 | 0.664451 | 0.620187 | 0.632503 | 0.784944 | 0.889222 |
| Equal RRF | **0.685165** | **0.637336** | **0.652460** | **0.813889** | **0.943667** |

Equal RRF improved every reported top-10 metric and Recall@100 over dense. The
absolute gains over dense were +0.043707 NDCG@10, +0.044504 MRR@10, and
+0.040000 Recall@100. BM25 outperformed dense at top-10 ranking but had lower
Recall@100, while fusion retained their complementary candidates. This is the
strongest positive retrieval result in the repository because it uses an
external test set and 300 queries rather than corpus-derived diagnostics.

Uncertainty was estimated by resampling the same 300 queries as paired units
20,000 times with seed 42. The table reports percentile 95% intervals for the
mean difference (`RRF - baseline`). Two-sided bootstrap tail probabilities use
an add-one correction; Holm-adjusted values control the family of eight metrics
within each baseline comparison.

| Comparison | Metric | Mean difference | 95% CI | Raw p | Holm p | Holm 0.05 |
|---|---|---:|---:|---:|---:|---|
| RRF - Dense | NDCG@10 | +0.043707 | [0.017389, 0.069269] | 0.001600 | 0.011200 | yes |
| RRF - Dense | MRR@10 | +0.044504 | [0.015975, 0.072334] | 0.002700 | 0.011200 | yes |
| RRF - Dense | Recall@10 | +0.038778 | [0.001833, 0.077224] | 0.040298 | 0.040298 | yes |
| RRF - Dense | Recall@100 | +0.040000 | [0.016667, 0.066667] | 0.002000 | 0.011200 | yes |
| RRF - BM25 | NDCG@10 | +0.020714 | [-0.003573, 0.044944] | 0.091195 | 0.455975 | no |
| RRF - BM25 | MRR@10 | +0.019958 | [-0.008311, 0.048015] | 0.162892 | 0.553172 | no |
| RRF - BM25 | Recall@10 | +0.028944 | [-0.000722, 0.059612] | 0.057197 | 0.343182 | no |
| RRF - BM25 | Recall@100 | +0.054444 | [0.026667, 0.083903] | 0.000100 | 0.000800 | yes |

The conservative conclusion is narrower than the point estimates: RRF is
supported over dense across the reported key metrics, while its top-10
advantages over BM25 are not statistically established here. The robust BM25
comparison is Recall@100, supporting the complementary-candidate explanation.

Dense retrieval uses a deterministic exact cosine matrix over the stored BGE-M3
vectors, with document ID as the tie-break. Two complete runs produced identical
aggregate metrics and all recorded top-10 rankings. Approximate HNSW rankings are
not used for the published quality numbers.

The benchmark is still one scientific fact-retrieval dataset, not proof of a
general production RAG improvement. It measures document retrieval rather than
answer generation or entailment. Bootstrap inference is conditional on this test
set and is not an independent cross-domain replication. The result file excludes
query text, abstracts, and qrels; it retains aggregate metrics, statistical
comparisons, and top document IDs/scores only.

Reproduce:

```bash
python evaluate_scifact.py --device cuda:0 --rebuild-index \
  --top-k 100 --record-k 10 --rrf-k 60 \
  --bootstrap-samples 20000 --bootstrap-seed 42 \
  --output results/retrieval/scifact_bge_m3_test_n300.json
```

Dataset provenance and license notes are recorded in
[`docs/DATASETS.md`](DATASETS.md).

## Cross-domain replication: BEIR/ArguAna

ArguAna was selected and its protocol committed before results because it changes
from scientific fact retrieval to best-counterargument retrieval. The official
run covers 8,674 documents and 1,406 queries and removes query-ID self matches.

| Retrieval | NDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Dense BGE-M3 | **0.543550** | **0.456290** | **0.821479** | 0.981508 |
| BM25 | 0.419138 | 0.335422 | 0.687767 | 0.915363 |
| Equal RRF | 0.508986 | 0.421387 | 0.788762 | **0.982930** |

The SciFact effect did not replicate. RRF minus dense NDCG@10 was -0.034564
with a 95% paired interval of [-0.047849, -0.021324] and Holm p=0.000800.
Recall@100 differed by only +0.001422, interval [-0.004267, 0.007112], Holm
p=0.722664. Equal RRF significantly improved every reported metric over BM25,
but mixing that weaker lexical ranker into dense retrieval damaged top-rank
quality.

A fixed-protocol repeat produced identical aggregate metrics, statistical
comparisons, and all recorded top-10 rankings. The published run fixes FP16
index/query batch size at 32. BM25 full-test search took 143.055 seconds versus
1.328 seconds for exact cosine search after embeddings, exposing the simple
Python sparse scan as the scaling bottleneck rather than an online latency win.

This negative cross-domain result remains part of the project. It narrows the
claim to: equal RRF can help when sparse retrieval contributes complementary
relevant candidates, but it must be validated per task and should not be enabled
universally. Full selection reasoning, statistics, execution notes, and the
reproduction command are in [`ARGUANA_RESULTS.md`](ARGUANA_RESULTS.md).

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

## Claim-level citation entailment

Citation syntax is now separated from semantic support. A local
`cross-encoder/nli-deberta-v3-base` verifier classifies each generated claim
against only its cited chunks as entailment, neutral, or contradiction. The
verifier is evaluated on a balanced, manually labeled 30-pair in-domain set.

| NLI input | Overall | Entailment | Neutral | Contradiction | Steady pair p50 |
|---|---:|---:|---:|---:|---:|
| Full 512-token chunk | 0.5667 | 0.1000 | 0.9000 | 0.7000 | 21.0 ms |
| Top-3 evidence sentences | 0.8000 | 0.7000 | 0.9000 | 0.8000 | 13.6 ms |

Evidence focusing ranks chunk sentences by lexical overlap with the claim, keeps
the top three in document order, and then runs NLI. It improved the main failure
mode without predicting entailment for any neutral or contradiction case in this
small set. Six of 30 pairs remain wrong, including mathematical or compound
claims, so the verifier is a diagnostic and not an automatic production block.

The same focused verifier was run over saved generated answers. Claim coverage is
sentence-level: a citation at the end of a paragraph does not silently support all
preceding sentences.

| Saved answer run | Claims | Citation coverage | NLI entailment | Contradiction |
|---|---:|---:|---:|---:|
| Dense historical baseline | 32 | 0.2812 | 0.2188 | 0.0000 |
| Strict citation experiment | 10 | 0.9000 | 0.5000 | 0.0000 |
| Balanced final, recorded protocol | 16 | 0.5625 | 0.3125 | 0.0000 |

The strict prompt is more auditable but its fact-group recall was only 0.5000,
versus 0.6667 for the balanced final run. A further sentence-citation prompt was
also rejected: it raised lexical fact recall to 0.7333 but claim citation coverage
was only 0.5625, NLI entailment was 0.1875, and one contradiction was flagged. Prompt-only
instructions therefore did not solve grounding reliably.

The final answer JSON records the full static grounding instructions in its
protocol. Older prompt-variant files did not, so they remain exploratory evidence
rather than fully reproducible prompt benchmarks. The NLI model threshold is not
calibrated on independently annotated generated claims.

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

Run the NLI diagnostic and the protocol-recorded final answer evaluation:

```bash
RAG_NLI_DEVICE=cuda:0 python evaluate_entailment.py --focus-sentences 3 \
  --output results/quality/nli_entailment_focused_gold_n30.json

RAG_EMBED_DEVICE=cpu RAG_RERANK_DEVICE=cuda:0 python evaluate_answers.py \
  --rerank --candidate-k 12 --max-tokens 256 \
  --output results/quality/ollama_q4_grounding_protocol_n8.json

RAG_NLI_DEVICE=cuda:0 python evaluate_answer_grounding.py \
  --answers results/quality/ollama_q4_grounding_protocol_n8.json \
  --output results/quality/nli_grounding_protocol_n5.json
```

## Remaining limitations

- The corpus has only three related systems papers and the test sets are small.
- Evidence labels were manually derived from the indexed PDFs but were not
  independently double-annotated.
- The lexical challenge set is corpus-derived, has one annotator, and was not
  evaluated on external documents.
- BM25 uses a simple alphanumeric tokenizer and an in-memory full-corpus scan;
  it has no stemming, Unicode-aware segmentation, or production index backend.
- Parent text is duplicated in child metadata for this small local experiment;
  a production index should store parents separately and reference them by ID.
- Parent-child sizes and candidate depth have one fixed configuration and no
  external held-out validation.
- SciFact and ArguAna cover two English retrieval tasks; they do not validate
  generation, multilingual retrieval, enterprise documents, or production scale.
- Paired bootstrap uncertainty is conditional on each fixed test set; the two
  datasets show opposite RRF effects rather than a universal fusion benefit.
- ArguAna has no separate development split in this protocol, so fusion weights
  were deliberately not fitted.
- Term-group recall can miss valid paraphrases and does not measure entailment.
- The 30-pair NLI set has one annotator and generated claims lack independent labels.
- Evidence focusing is lexical and can miss mathematical or low-overlap support.
- Claim-level thresholds are uncalibrated, so NLI is not an online blocking guardrail.
- Citation-ID validity does not prove that a cited chunk supports each claim.
- The abstention threshold needs a larger held-out positive/negative set.
