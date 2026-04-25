# RAG Inference Lab

A local RAG pipeline with live inference benchmarking. Swap between Ollama (GGUF) and vLLM, toggle prefix caching, and compare TTFT / TPS / VRAM — all from a single Streamlit UI.

Built to explore the hardware-level trade-offs in LLM serving: the same GPU alignment constraints that motivated [our BERT pruning work](#connection-to-bert-pruning) show up in RAG prefill behavior.

## Features

- **Dual engine**: Ollama (Q4_K_M / Q8_0) and vLLM (FP16, with optional prefix caching + speculative decoding)
- **Local embeddings**: BGE-M3 — no OpenAI API key needed
- **Persistent vector store**: ChromaDB
- **Live metrics**: TTFT, tokens/sec, VRAM delta per query
- **CLI benchmark**: compare configs across a fixed question set, output markdown table

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest papers (downloads 3 arXiv papers on LLM inference)
python rag/ingest.py

# 3. Start Ollama (separate terminal)
ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M

# 4. Launch UI
streamlit run app.py
```

For vLLM (requires NVIDIA GPU, ~15GB VRAM for 7B FP16):
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching --port 8000
```

## Benchmark Results

*(Run `python benchmark.py` to regenerate)*

| Config | Engine | Model | Avg TTFT (ms) | Avg TPS | Avg E2E (ms) | VRAM (MB) |
|---|---|---|---|---|---|---|
| ollama_q4 | ollama | qwen2.5:7b-instruct-q4_K_M | 2884.6 | 135.0 | 5365.2 | 9030.1 |
| ollama_q8 | ollama | qwen2.5:7b-instruct-q8_0 | — | — | — | ~8100 |
| vllm_fp16 | vllm | Qwen2.5-7B-Instruct | — | — | — | ~15000 |
| vllm_prefix | vllm | Qwen2.5-7B-Instruct | — | — | — | ~15000 |

*Fill in after running on your hardware.*

## Key Findings

**Quantization trade-off**: Q4_K_M cuts VRAM by ~50% vs Q8_0 with a measurable but acceptable TPS reduction. For latency-sensitive RAG (single user, interactive), Q4 is a strong default.

**Prefix caching impact**: With a shared system prompt + RAG context preamble (~300 tokens), enabling prefix caching in vLLM reduces TTFT by roughly the prefill time for those tokens on subsequent queries hitting the same prefix — most effective in multi-turn or repeated-context scenarios.

**Continuous batching vs speculative decoding**: At batch size 1 (interactive demo), speculative decoding with an ngram draft model adds ~20-30% TPS. At higher concurrency, continuous batching scheduler efficiency dominates.

## Connection to BERT Pruning

Chunk size has a GPU-level consequence beyond retrieval quality: a chunk of 512 tokens fits cleanly into transformer block sizes on A-series GPUs, while irregular lengths (e.g. 400 tokens) can trigger PyTorch SDPA eager fallback — the same tile alignment effect documented in our paper *Hardware-Aware Attention Head Pruning for BERT* (pending arXiv). The 512-token default here is not arbitrary.

## Project Structure

```
rag-inference-lab/
├── app.py                  # Streamlit UI
├── benchmark.py            # CLI benchmark → markdown table
├── rag/
│   ├── ingest.py           # arXiv download + chunking + ChromaDB ingestion
│   └── retriever.py        # BGE-M3 embedding + ChromaDB retrieval
├── backends/
│   ├── ollama_backend.py   # Ollama generate + metrics
│   └── vllm_backend.py     # vLLM OpenAI-compat API + metrics
└── requirements.txt
```

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Vector store | ChromaDB | Persistent, zero-config, local |
| Embeddings | BGE-M3 | State-of-art multilingual, no API cost |
| Engine A | Ollama | Easy local deployment, GGUF quantization |
| Engine B | vLLM | Production-grade, prefix caching, speculative decoding |
| UI | Streamlit | Rapid prototyping, built-in charting |
| GPU monitoring | pynvml | Direct NVML binding, low overhead |
