# External benchmark selection record

## Decision date and objective

Decision frozen on 2026-08-11, before running the second external benchmark.
The objective is to test whether the SciFact finding that equal-weight RRF
improves over dense BGE-M3 survives a materially different retrieval task. The
second dataset is not selected to maximize the expected score.

## Selection criteria

1. Different domain and relevance mechanism from scientific fact retrieval.
2. Official test queries and qrels, with enough queries for paired uncertainty.
3. Small enough for exact cosine retrieval and in-memory BM25 on one RTX 4090
   workstation with 32 GB system RAM.
4. Public provenance and a license boundary that can be documented without
   committing third-party text.
5. A task that can expose a failure of lexical, dense, or naive fusion methods.

## Candidates considered

| Dataset | BEIR test scale | Why it was considered | Decision |
|---|---:|---|---|
| ArguAna | 1,406 queries; about 8.67K documents | Social/web argument retrieval; the relevant item is an opposing counterargument rather than simply similar text; manageable exact evaluation; dataset card reports CC BY-SA 4.0 | Selected |
| FiQA-2018 | 648 queries; about 57K documents | Strong financial-domain shift and realistic QA passages | Rejected for this public portfolio run because the original challenge page limits train and test data to non-commercial use; also materially higher indexing cost |
| NFCorpus | 323 queries; about 3.6K documents | Cheap, densely judged biomedical retrieval | Rejected because it remains close to the scientific/biomedical domain and adds little query-count strength over SciFact's 300 queries |

ArguAna was chosen because it maximizes task and domain contrast while remaining
fully executable under the local hardware budget. Its special evaluation hazard
is recorded explicitly: query arguments also occur in the corpus, so a result
whose document ID equals the query ID must be excluded before scoring.

## Frozen protocol

The following settings are fixed before inspecting retrieval results:

- dataset: official BEIR-preprocessed ArguAna test split;
- archive MD5: `8ad3e3c2a5867cdced806d6503f29b99`;
- embedding model: `BAAI/bge-m3`, maximum input length 512;
- dense search: deterministic exact cosine with document-ID tie breaks;
- sparse search: the existing in-memory BM25, `k1=1.5`, `b=0.75`;
- fusion: equal-weight reciprocal rank fusion, `k=60`;
- candidate depth: top 100 after removing query-ID self matches;
- published audit depth: top 10 document IDs and scores, without raw text;
- primary metrics: NDCG@10 and Recall@100;
- secondary metrics: MAP@10/100, MRR@10/100, Recall@10, NDCG@100;
- uncertainty: 20,000 paired query bootstrap samples, seed 42, percentile
  95% intervals, two-sided add-one-corrected tail probabilities;
- multiplicity: Holm correction over all eight metrics separately for each
  baseline comparison;
- no fusion-weight, tokenizer, embedding, or candidate-depth tuning after the
  result is observed.

A cross-domain improvement claim will only be made for a metric when its paired
95% interval excludes zero and its Holm-adjusted value is at most 0.05. Point
estimate improvements that fail this rule will be reported as inconclusive.
Negative results will remain in the repository.

## Execution amendment

After the pre-result commit, the first batch-64 process lost its controlling WSL
connection while continuing in the background. No kernel OOM was recorded. Before
designating a published result, index batch size was made an explicit CLI and
protocol field and fixed at 32; query batch size was also recorded as 32. The
retrieval methods, primary metrics, cutoffs, BM25 parameters, RRF constant,
bootstrap settings, and decision thresholds above were unchanged. The final
batch-32 run was repeated independently under the same fixed protocol.

## Data and license boundary

The downloaded archive, corpus, queries, qrels, embeddings, and Chroma files stay
under ignored local paths. Published JSON contains aggregate metrics, statistical
comparisons, protocol metadata, and document IDs/scores only.

References:

- BEIR dataset list and checksums: <https://github.com/beir-cellar/beir#available-datasets>
- BEIR ArguAna dataset card: <https://huggingface.co/datasets/BeIR/arguana>
- Original ArguAna artifact: <https://doi.org/10.5281/zenodo.3973258>
- Original FiQA challenge terms: <https://sites.google.com/view/fiqa/home>
- BEIR license disclaimer: <https://github.com/beir-cellar/beir#disclaimer>
