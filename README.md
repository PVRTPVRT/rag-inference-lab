# RAG Inference Lab

A local retrieval-augmented generation prototype for comparing Ollama and vLLM
under an explicit, reproducible measurement protocol.

The project combines BGE-M3 embeddings, a persistent ChromaDB collection,
streaming generation, client-side latency instrumentation, and raw JSON benchmark
records. It is designed as an engineering lab: configuration labels are kept
separate from measured results, and no speedup is claimed without a saved run.

## Current status

- The RAG pipeline and dual inference backends are implemented.
- Token counting has been corrected to use Ollama server statistics or vLLM
  OpenAI-compatible usage data.
- The benchmark supports fail-fast backend checks, repeated trials, P50/P95
  summaries, failure accounting, and server metadata.
- Retrieval supports optional BGE cross-encoder reranking, stable source IDs,
  citation-constrained prompting, and an explicitly experimental abstention gate.
- Source, evidence-chunk, answer-fact, citation, and abstention diagnostics retain
  raw JSON outputs and negative results.
- A dependency-free BM25 path and equal-weight reciprocal-rank fusion (RRF) are
  available for controlled dense/sparse/hybrid retrieval diagnostics; dense
  retrieval remains the default.
- A separate parent-child index retrieves 192-token children and returns their
  512-token parents; its measured results are published without making it the
  application default.
- An external BEIR/SciFact evaluation covers 5,183 documents and all 300 test
  queries with official qrels; raw third-party text remains outside Git and the
  fixed equal-weight RRF result is reported without post-hoc tuning.
- Paired query bootstrap intervals and Holm correction separate supported gains
  over dense from inconclusive top-10 differences versus BM25.
- A predeclared BEIR/ArguAna run covers 8,674 documents and 1,406 queries; it
  reverses the SciFact result, with equal RRF significantly worse than dense at
  NDCG@10 and no supported Recall@100 gain.
- Claim-to-citation NLI has a balanced 30-pair diagnostic, evidence focusing,
  confusion matrices, and generated-answer checks; it is not an online blocker.
- Verified RTX 4090 trials and retrieval-quality diagnostics are summarized in
  [`docs/RTX4090_RESULTS.md`](docs/RTX4090_RESULTS.md).
- Quality A/B results: [`docs/RAG_QUALITY_RESULTS.md`](docs/RAG_QUALITY_RESULTS.md).

Earlier exploratory numbers were removed because streamed HTTP chunks and
whitespace-delimited words are not model tokens. The repository now prefers an
explicit `N/A` over a misleading throughput value.


- Download and parse arXiv PDFs with PyMuPDF.
- Split text using the BGE-M3 tokenizer with configurable token overlap.
- Generate BGE-M3 embeddings and persist them in ChromaDB.
- Retrieve top-k chunks and build source-labeled context.
- Compare dense retrieval, in-memory BM25, and equal-weight RRF while retaining
  component ranks and scores in raw evaluation output.
- Evaluate parent-child retrieval with child-level matching, parent deduplication,
  and full parent context returned to the generator.
- Optionally rerank dense candidates with BGE-reranker-v2-m3.
- Require stable `[S1]` citations and gate low-confidence queries before generation.
- Diagnose whether cited evidence entails each claim with a local three-way NLI
  model and a measured top-sentence evidence-focusing stage.
- Evaluate Dense, BM25, and RRF on external SciFact and ArguAna test splits with
  NDCG, MAP, Recall, MRR, paired intervals, and corrected comparisons.
- Serve a Streamlit UI across Ollama GGUF and vLLM FP16 backends.
- Measure client TTFT, end-to-end latency, decode throughput, and device-wide
  VRAM usage with clearly labeled measurement sources.
- Run fixed-question benchmarks and save every successful or failed trial.

## Project layout

```text
app.py                       Streamlit RAG interface
benchmark.py                 Repeated benchmark runner and JSON output
evaluate_retrieval.py        Labeled source-level Hit@k and MRR evaluation
evaluate_abstention.py       Out-of-corpus score-overlap diagnostic
evaluate_evidence.py         Manually labeled evidence-chunk evaluation
evaluate_answers.py          Answer facts, citations, and abstention diagnostic
evaluate_entailment.py       Balanced three-way cited-chunk NLI evaluation
evaluate_answer_grounding.py NLI diagnostic over saved generated answers
evaluate_scifact.py          External BEIR/SciFact retrieval evaluation
evaluate_arguana.py          External BEIR/ArguAna retrieval evaluation
backends/metrics.py          Dependency-free metric aggregation
backends/ollama_backend.py   Ollama streaming client and server metrics
backends/preflight.py        Server/model/version and VRAM-isolation checks
backends/vllm_backend.py     vLLM OpenAI-compatible streaming client
evaluation/                  Labeled positive and out-of-corpus query sets
rag/chunking.py              Tokenizer-aware chunking
rag/ingest.py                PDF ingestion and ChromaDB indexing
rag/retriever.py             BGE-M3 retrieval and context construction
rag/hybrid.py                Dependency-free BM25 and RRF fusion
rag/parent_child.py          Parent-child chunk construction and index constants
rag/reranker.py              Optional BGE cross-encoder reranking
rag/prompting.py             Citation contract and experimental gate
rag/entailment.py            Claim extraction, evidence focusing, and local NLI
docs/BENCHMARK_PROTOCOL.md   Claim and reproduction rules
docs/DATASETS.md             External dataset provenance and licenses
docs/DATASET_SELECTION.md    Pre-result external dataset decision record
docs/ARGUANA_RESULTS.md      Cross-domain results and execution notes
docs/RTX4090_RESULTS.md      Verified results and claim boundaries
results/                     Published raw serving and retrieval trials
tests/                       Unit tests for metrics, backends, and evaluation
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

No hosted LLM API key is required. Model downloads, PDFs, ChromaDB files, virtual
environments, and benchmark outputs are ignored by Git.

## Build the local index

```bash
python -m rag.ingest --query "LLM inference optimization" --n 3

python -m rag.ingest \
  --arxiv-ids 2309.06180 2211.17192 2312.07104 \
  --embed-device cuda:0 --parent-child
```

The default uses 512 **BGE-M3 tokenizer tokens** with 50-token overlap. Chunk
length is a retrieval configuration, not a claim about GPU kernel alignment.
The optional command creates a separate `papers_parent_child` collection with
192-token children, 32-token child overlap, and the same 512-token parents used
by the baseline labels; it does not overwrite the default `papers` collection.

## External held-out retrieval benchmark

```bash
python evaluate_scifact.py --device cuda:0 --rebuild-index \
  --top-k 100 --record-k 10 --rrf-k 60 \
  --output results/retrieval/scifact_bge_m3_test_n300.json
```

The command downloads and verifies the official BEIR-preprocessed SciFact archive,
builds a separate BGE-M3 index, and uses exact cosine search to evaluate Dense, BM25, and equal-weight
RRF on all 300 test queries. `.cache/beir/` and the local Chroma collection are
ignored by Git. See [`docs/DATASETS.md`](docs/DATASETS.md) for provenance and
license boundaries.

The second external run was selected and committed before results. Reproduce it
with:

```bash
python evaluate_arguana.py --device cuda:0 --rebuild-index \
  --index-batch-size 32 --top-k 100 --record-k 10 --rrf-k 60 \
  --bootstrap-samples 20000 --bootstrap-seed 42 \
  --output results/retrieval/arguana_bge_m3_test_n1406.json
```

ArguAna requires query-ID self-match removal. It produced a stable negative
cross-domain result for equal RRF versus dense. See
[`docs/DATASET_SELECTION.md`](docs/DATASET_SELECTION.md) for why it was chosen
and [`docs/ARGUANA_RESULTS.md`](docs/ARGUANA_RESULTS.md) for complete results.

## Run the UI

For Ollama:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M
streamlit run app.py
```

The sidebar can switch between dense retrieval and dense-plus-reranker. Reranking
defaults to CPU unless `RAG_RERANK_DEVICE=cuda:0` is set. The abstention gate is
off by default because its threshold is not production-calibrated.

For vLLM, start a server separately and enter its OpenAI-compatible URL in the
sidebar. Prefix caching and speculative decoding are server-startup settings;
the UI records a profile label but does not pretend to toggle them from a request.

## Reproducible benchmark

The three vLLM profiles below must be launched as distinct server configurations.
Run one profile at a time if the GPU cannot hold multiple model instances.

Baseline on port 8000:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --no-enable-prefix-caching \
  --port 8000
```

Prefix caching on port 8001:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-prefix-caching \
  --port 8001
```

N-gram speculation on port 8002:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}' \
  --port 8002
```

Example runs:

```bash
python benchmark.py --engines ollama_q4 --warmup 3 --repeats 3
python benchmark.py --engines vllm_baseline --warmup 3 --repeats 3 \
  --output outputs/vllm_baseline.json
python benchmark.py --engines vllm_prefix --warmup 3 --repeats 3 \
  --output outputs/vllm_prefix.json
python benchmark.py --engines vllm_ngram --warmup 3 --repeats 3 \
  --output outputs/vllm_ngram.json
```

The runner verifies that the selected API and model are available before loading
the embedding model. For Ollama it also refuses to run when a different model is
already resident, because NVML reports device-wide memory. Run Q4 and Q8
separately and use `ollama stop <model>` between VRAM profiles.

Each run records five fixed questions, all trial failures, environment metadata,
and summary statistics. For a resume-grade comparison, use the same GPU, model,
software versions, retrieval index, top-k, output limit, temperature, and seed.

## Metric definitions

| Metric | Definition | Source |
|---|---|---|
| TTFT | Client request start to first non-empty text chunk | `perf_counter` |
| E2E | Client request start to stream completion | `perf_counter` |
| Ollama decode TPS | `eval_count / eval_duration` | Ollama final response |
| vLLM decode TPS | completion usage tokens / client decode time | OpenAI-compatible stream usage |
| VRAM used | Total memory currently used on GPU 0 | NVML |

VRAM used is not isolated model weight size. It can include the embedding model,
inference server, UI, and other processes on the same device.

## Run tests

The core tests require only the Python standard library:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py benchmark.py evaluate_*.py backends rag evaluation tests
```

## Interpretation boundaries

This repository does not currently claim that:

- Q4 is universally a fixed percentage faster or smaller than Q8;
- prefix caching reduces TTFT for cold or unrelated prompts by the measured amount;
- n-gram speculation necessarily improves throughput;
- a particular chunk length triggers or avoids an SDPA fallback;
- five fixed questions measure retrieval or answer quality;
- cross-encoder reranking is universally better than dense retrieval;
- BM25 latency on 127 in-memory chunks represents production-scale sparse search;
- equal-weight RRF universally improves dense retrieval or has tuned fusion
  weights;
- parent-child retrieval improves every query, reduces latency, or has been tuned
  on an external held-out corpus;
- one SciFact run establishes general-domain or production-scale improvement;
  paired statistics support RRF over dense on this test set, but most top-10
  differences versus BM25 are not significant after Holm correction;
- the small one-annotator NLI diagnostic is production-calibrated groundedness;
- citation IDs establish claim-level entailment or the fitted gate generalizes.

Those are hypotheses to test under controlled conditions. See
[`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) before publishing
results.

## Connection to the BERT pruning project

Both projects distinguish a theoretical optimization from measured wall-clock
behavior. The BERT work studies logical versus physical head pruning; this lab
applies the same evidence discipline to RAG serving. It does not transfer a
BERT-specific hardware mechanism to vLLM without profiling evidence.
