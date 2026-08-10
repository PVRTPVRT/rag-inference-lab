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
- Verified RTX 4090 trials and retrieval-quality diagnostics are summarized in
  [`docs/RTX4090_RESULTS.md`](docs/RTX4090_RESULTS.md).

Earlier exploratory numbers were removed because streamed HTTP chunks and
whitespace-delimited words are not model tokens. The repository now prefers an
explicit `N/A` over a misleading throughput value.

## What is implemented

- Download and parse arXiv PDFs with PyMuPDF.
- Split text using the BGE-M3 tokenizer with configurable token overlap.
- Generate BGE-M3 embeddings and persist them in ChromaDB.
- Retrieve top-k chunks and build source-labeled context.
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
backends/metrics.py          Dependency-free metric aggregation
backends/ollama_backend.py   Ollama streaming client and server metrics
backends/preflight.py        Server/model/version and VRAM-isolation checks
backends/vllm_backend.py     vLLM OpenAI-compatible streaming client
evaluation/                  Labeled positive and out-of-corpus query sets
rag/chunking.py              Tokenizer-aware chunking
rag/ingest.py                PDF ingestion and ChromaDB indexing
rag/retriever.py             BGE-M3 retrieval and context construction
docs/BENCHMARK_PROTOCOL.md   Claim and reproduction rules
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
```

The default uses 512 **BGE-M3 tokenizer tokens** with 50-token overlap. Chunk
length is a retrieval configuration, not a claim about GPU kernel alignment.

## Run the UI

For Ollama:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M
streamlit run app.py
```

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
python -m compileall -q app.py benchmark.py backends rag tests
```

## Interpretation boundaries

This repository does not currently claim that:

- Q4 is universally a fixed percentage faster or smaller than Q8;
- prefix caching reduces TTFT for cold or unrelated prompts by the measured amount;
- n-gram speculation necessarily improves throughput;
- a particular chunk length triggers or avoids an SDPA fallback;
- five fixed questions measure retrieval or answer quality.

Those are hypotheses to test under controlled conditions. See
[`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) before publishing
results.

## Connection to the BERT pruning project

Both projects distinguish a theoretical optimization from measured wall-clock
behavior. The BERT work studies logical versus physical head pruning; this lab
applies the same evidence discipline to RAG serving. It does not transfer a
BERT-specific hardware mechanism to vLLM without profiling evidence.
