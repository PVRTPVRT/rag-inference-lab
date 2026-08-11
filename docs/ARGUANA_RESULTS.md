# BEIR ArguAna cross-domain retrieval results

## Why this dataset

ArguAna was selected and committed before results were run. It changes both the
domain and relevance mechanism from SciFact: the query is an argument and the
relevant document is its best counterargument. The full comparison against
FiQA-2018 and NFCorpus is recorded in
[`DATASET_SELECTION.md`](DATASET_SELECTION.md).

The short version is:

- ArguAna provides 1,406 test queries and about 8.67K documents, enough for
  stronger paired analysis while remaining feasible for exact local retrieval;
- FiQA offered a useful financial shift but its original challenge page limits
  train and test data to non-commercial use;
- NFCorpus was cheaper but stayed close to the scientific/biomedical domain and
  had only 323 test queries.

## Frozen protocol

- official BEIR ArguAna test split, 8,674 documents and 1,406 queries;
- archive MD5 `8ad3e3c2a5867cdced806d6503f29b99`;
- BGE-M3 dense embeddings, FP16 on RTX 4090, maximum length 512;
- index and query batch size 32 for the published run;
- deterministic exact cosine ranking with document-ID tie breaks;
- in-memory BM25 with `k1=1.5`, `b=0.75`;
- equal-weight RRF with `k=60` and no fitted weights;
- top 100 after excluding any corpus document whose ID equals the query ID;
- top 10 document IDs and scores retained for audit, with no raw text;
- 20,000 paired query bootstrap samples, seed 42, percentile 95% intervals;
- Holm correction over eight metrics within each baseline comparison;
- NDCG@10 and Recall@100 declared primary before results.

## Aggregate results

| Retrieval | NDCG@10 | MAP@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|---:|
| Dense BGE-M3 | **0.543550** | **0.456290** | **0.456290** | **0.821479** | 0.981508 |
| BM25 | 0.419138 | 0.335422 | 0.335422 | 0.687767 | 0.915363 |
| Equal RRF | 0.508986 | 0.421387 | 0.421387 | 0.788762 | **0.982930** |

Unlike SciFact, equal RRF did not improve dense retrieval at the top of the
ranking. It reduced dense NDCG@10 by 0.034564, MRR@10 by 0.034903, and Recall@10
by 0.032717. Its Recall@100 point estimate was only 0.001422 above dense.

## Paired uncertainty

Differences are `RRF - baseline`.

| Comparison | Metric | Difference | 95% CI | Raw p | Holm p | Supported at 0.05 |
|---|---|---:|---:|---:|---:|---|
| RRF - Dense | NDCG@10 | -0.034564 | [-0.047849, -0.021324] | 0.000100 | 0.000800 | yes, negative |
| RRF - Dense | MRR@10 | -0.034903 | [-0.049471, -0.020417] | 0.000100 | 0.000800 | yes, negative |
| RRF - Dense | Recall@10 | -0.032717 | [-0.051920, -0.013514] | 0.001100 | 0.002200 | yes, negative |
| RRF - Dense | Recall@100 | +0.001422 | [-0.004267, 0.007112] | 0.722664 | 0.722664 | no |
| RRF - BM25 | NDCG@10 | +0.089849 | [0.079241, 0.100652] | 0.000100 | 0.000800 | yes, positive |
| RRF - BM25 | MRR@10 | +0.085964 | [0.074096, 0.098078] | 0.000100 | 0.000800 | yes, positive |
| RRF - BM25 | Recall@10 | +0.100996 | [0.083926, 0.118777] | 0.000100 | 0.000800 | yes, positive |
| RRF - BM25 | Recall@100 | +0.067568 | [0.054054, 0.081081] | 0.000100 | 0.000800 | yes, positive |

For NDCG@10, RRF beat dense on 342 queries, tied on 635, and lost on 429.
For Recall@100, it won on 10, tied on 1,388, and lost on 8. The primary
cross-domain replication therefore failed: the top-rank effect reversed, while
the small Recall@100 difference was inconclusive.

## Interpretation

The result rejects a universal "hybrid retrieval is better" claim. RRF greatly
improves over BM25, but equal fusion pulls a stronger dense ranking toward a
weaker lexical ranking. A plausible task-level explanation is that BM25 rewards
topic and word overlap, whereas ArguAna requires an opposing stance; this is a
hypothesis consistent with the task and measurements, not a directly annotated
error attribution.

The useful engineering rule is conditional:

- fuse when the sparse channel contributes complementary relevant candidates;
- do not assume an equal-weight lexical channel is harmless;
- evaluate fusion per domain and relevance mechanism before making it default;
- use learned or validated weighting only on a separate development split.

No weights are retuned here because ArguAna provides only the evaluated test
split in the BEIR configuration used by this project.

## Runtime and repeatability

| Stage | Published run | Independent repeat |
|---|---:|---:|
| Index build | 21.719 s | reused |
| Query embedding | 2.839 s | 3.075 s |
| Exact cosine search | 1.328 s | 1.273 s |
| BM25 build | 0.298 s | 0.295 s |
| BM25 search | 143.055 s | 142.591 s |
| RRF fusion | 0.251 s | 0.351 s |

The independent repeat produced identical aggregate metrics, statistical
comparisons, protocol metadata, and all 1,406 x 3 recorded top-10 rankings.
The repeat JSON remains in the ignored cache rather than being published twice.

An initial batch-64 execution lost its controlling WSL connection while the
Python process continued. Kernel logs did not show an OOM. Its observed aggregate
direction matched the final run, but FP16 embeddings differed at roughly the
fourth decimal place across batch groupings. The published protocol therefore
fixes both index and query batch size at 32 and does not claim cross-batch bitwise
reproducibility.

The in-memory Python BM25 scan is the clear runtime bottleneck at this scale. Its
143-second total covers all 1,406 queries, not a per-query online latency claim.

## Reproduce

```bash
python evaluate_arguana.py --device cuda:0 --rebuild-index \
  --index-batch-size 32 --top-k 100 --record-k 10 --rrf-k 60 \
  --bootstrap-samples 20000 --bootstrap-seed 42 \
  --output results/retrieval/arguana_bge_m3_test_n1406.json
```

Raw corpus text, query text, qrels, embeddings, archive files, and Chroma data
remain outside Git. See [`DATASETS.md`](DATASETS.md) for attribution and license
boundaries.
