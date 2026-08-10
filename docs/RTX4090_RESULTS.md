# Verified RTX 4090 results

These are controlled local measurements, not universal serving claims. Raw trials
are retained under `results/rtx4090/` and retrieval evaluations under
`results/retrieval/`.

## Protocol

- RTX 4090 24 GB; WSL2; NVIDIA driver 610.74; CUDA UMD 13.3.
- Three fixed arXiv papers; 127 BGE-M3-tokenized ChromaDB chunks.
- BGE-M3 retrieval ran on CPU; generation ran on GPU.
- Five prompts per profile, 15 warmup and 50 measured requests, top-k 3,
  temperature 0, seed 42, and at most 128 output tokens.
- Ollama 0.32.6 with Qwen2.5-7B-Instruct Q4_K_M/Q8_0.
- vLLM 0.26.0 with Qwen2.5-7B-Instruct FP16, concurrency 1, FlashAttention 2,
  the V1 runner, and PyTorch sampler.

All five serving profiles completed 50/50 measured requests with zero failures.

## Ollama quantization

Q4 and Q8 were measured in separate isolated runs. This is necessary because
NVML reports total device memory and Ollama can keep both models resident.

| Profile | TTFT p50/p95 ms | E2E p50/p95 ms | Decode TPS mean | VRAM mean MB |
|---|---:|---:|---:|---:|
| Q4_K_M | 180.90 / 193.80 | 1143.20 / 1157.03 | 133.297 | 8371.46 |
| Q8_0 | 179.00 / 185.89 | 1272.55 / 1467.71 | 100.186 | 11357.84 |

In this setup Q4 delivered 33.0% higher server-reported decode throughput and
used 26.3% less device-wide VRAM. TTFT was effectively unchanged. Answer quality
equivalence was not evaluated, so this is not yet an end-to-end RAG-quality claim.

## vLLM prefix caching

| Profile | TTFT p50/p95 ms | E2E p50/p95 ms | Decode TPS mean | VRAM mean MB |
|---|---:|---:|---:|---:|
| Prefix off | 147.95 / 158.95 | 2195.05 / 2200.84 | 62.589 | 21786.67 |
| Prefix on | 28.05 / 31.46 | 2078.75 / 2081.77 | 62.535 | 21789.16 |

Prefix caching reduced TTFT p50 by 81.0% and p95 by 80.2%, while decode
throughput and VRAM were effectively unchanged. This is explicitly a warm,
repeated-prefix workload and must not be generalized to cold or unrelated prompts.

## vLLM n-gram speculation

| Profile | TTFT p50/p95 ms | E2E p50/p95 ms | Decode TPS mean | VRAM mean MB |
|---|---:|---:|---:|---:|
| Baseline | 147.95 / 158.95 | 2195.05 / 2200.84 | 62.589 | 21786.67 |
| N-gram | 147.15 / 158.47 | 2268.10 / 2311.20 | 60.316 | 21785.53 |

N-gram speculation did not accelerate this workload: decode throughput fell 3.6%
and E2E p50 increased 3.3%. Server logs showed roughly 26%-31% draft-token
acceptance and asynchronous scheduling was disabled for this profile.

## Retrieval quality and abstention

Fifteen manually labeled, source-specific questions (five per indexed paper)
achieved source-level Hit@3=1.00 and MRR=1.00; every correct paper ranked first.
This is a small, easy three-document corpus and does not establish chunk-level
relevance or generalization.

Five out-of-corpus questions covering MLA, LoRA, RLHF, diffusion, and BERT pruning
exposed the absence of a rejection policy. Positive top-1 scores ranged
0.5464-0.7360, while negative scores ranged 0.4875-0.5610. The distributions
overlap. A threshold fitted on this diagnostic set (0.5672) rejected all five
negatives but retained only 13/15 positives, so it is reported as calibration
evidence rather than installed as a production default.

## Safe interpretation

Supported:

- reproducible dual-backend measurement with raw trials and failure accounting;
- warm-prefix TTFT reduction in the recorded environment;
- Q4 throughput/VRAM trade-off without claiming quality equivalence;
- a measured negative speculative-decoding result;
- source-level retrieval evaluation and explicit discovery of an abstention gap;
- detection and correction of cross-model VRAM contamination.

Unsupported:

- arbitrary traffic receives an 81% prefix-cache improvement;
- Q4 and Q8 have equivalent answer quality;
- Ollama is faster than vLLM based on unlike TPS timing boundaries;
- fifteen source-specific questions establish complete RAG correctness;
- the fitted rejection threshold is production-ready.
