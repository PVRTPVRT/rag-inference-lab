# Benchmark protocol and claim policy

This document defines when a result from this repository is strong enough to
publish in a README, report, or resume.

## Controlled variables

For any comparison, keep the following fixed:

- physical GPU and competing GPU processes;
- model repository and revision;
- inference-engine version, CUDA version, and driver;
- precision or quantization format;
- question set, indexed documents, chunking, and top-k;
- sampling parameters, maximum output tokens, and random seed;
- warmup rounds and measured repetitions.

Change one experimental variable at a time. Run vLLM profiles sequentially on a
single GPU unless the device can hold isolated instances without contention.

## Default workload

`benchmark.py` uses five fixed questions. One warmup round executes all five
questions once. This makes the measured repeats a **warm repeated-prefix**
scenario, which is appropriate for testing automatic prefix caching but does not
represent cold-cache traffic.

The default measured workload is three repeats per question, or 15 requests per
configuration. Increase repetitions for a published result.

## Metric semantics

- Client TTFT starts immediately before the HTTP request and stops at the first
  non-empty generated text chunk.
- Client E2E stops when the stream closes.
- Ollama decode TPS uses the final response's `eval_count / eval_duration`.
- vLLM decode TPS uses API completion usage tokens divided by client-observed
  time after TTFT. It includes streaming and local networking overhead.
- NVML memory is total memory used on GPU 0; it is not model-only memory.
- Retrieval latency is measured separately from generation.

Do not compare Ollama server-side TPS with vLLM client-side TPS without noting
the different timing boundary. Prefer within-engine A/B comparisons.

## Prefix caching

Prefix caching is configured when the vLLM server starts. A client Boolean does
not enable it. Compare an explicitly disabled server with an explicitly enabled
server, using identical prompts and the same port mapping documented in the
README.

A prefix-cache result must be described as warm-cache unless cache state is
explicitly reset before every measured request.

## N-gram speculative decoding

Speculation is also a server-startup setting. Record the complete
`--speculative-config`, vLLM version, target model, concurrency, and output-token
limit. Report a gain only after comparing saved baseline and speculative JSON
runs under the same workload.

## Result acceptance checklist

A result is publication-ready only when:

1. Raw JSON exists for every compared profile.
2. No requests were silently dropped; failure counts are reported.
3. Token counts come from server/API usage, not stream chunks or words.
4. Hardware and software metadata are present.
5. Warmup and measured repetitions are stated.
6. Mean plus P50/P95 are available where relevant.
7. The claim is limited to the measured environment.
8. Retrieval or answer quality is evaluated separately before claiming an
   end-to-end RAG improvement.

## Current evidence status

No corrected GPU benchmark JSON is committed at this time. Consequently, the
repository makes no numeric claim for Q4 vs Q8, prefix caching, or speculative
decoding. That is an intentional evidence boundary, not a missing result to fill
with an estimate.
