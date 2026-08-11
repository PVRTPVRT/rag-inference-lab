"""Reproducible CLI benchmark for the RAG inference backends."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from backends.metrics import summarize
from backends.ollama_backend import generate as ollama_generate
from backends.preflight import inspect_backend
from backends.vllm_backend import generate as vllm_generate
from rag.prompting import build_rag_prompt
from rag.retriever import retrieve

QUESTIONS = [
    "What is speculative decoding and when does it improve throughput?",
    "How does prefix caching reduce time-to-first-token in vLLM?",
    "What is PagedAttention and how does it reduce KV cache waste?",
    "Explain the trade-off between continuous batching and speculative decoding.",
    "What is Multi-Head Latent Attention (MLA) and how does it compress KV cache?",
]

CONFIGS = {
    "ollama_q4": {
        "engine": "ollama",
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "server_profile": "ollama_q4",
    },
    "ollama_q8": {
        "engine": "ollama",
        "model": "qwen2.5:7b-instruct-q8_0",
        "server_profile": "ollama_q8",
    },
    "vllm_baseline": {
        "engine": "vllm",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "server_profile": "prefix_cache_off",
    },
    "vllm_prefix": {
        "engine": "vllm",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "base_url": "http://localhost:8001/v1",
        "server_profile": "prefix_cache_on",
    },
    "vllm_ngram": {
        "engine": "vllm",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "base_url": "http://localhost:8002/v1",
        "server_profile": "ngram_speculation_5",
    },
}

SUMMARY_FIELDS = ("retrieval_ms", "ttft_ms", "e2e_ms", "decode_tps", "vram_used_mb")


def environment_snapshot() -> dict:
    packages = {}
    for name in ("chromadb", "FlagEmbedding", "openai", "requests", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu, bytes):
            gpu = gpu.decode("utf-8")
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "packages": packages,
    }


def build_prompt(question: str, chunks: list[dict]) -> str:
    return build_rag_prompt(question, chunks)


def generate_for_config(prompt: str, cfg: dict, max_tokens: int, seed: int):
    common = {
        "model": cfg["model"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
    }
    if cfg["engine"] == "ollama":
        return ollama_generate(prompt, **common)
    return vllm_generate(
        prompt,
        base_url=cfg["base_url"],
        server_profile=cfg["server_profile"],
        **common,
    )


def execute_question(question: str, cfg: dict, top_k: int, max_tokens: int, seed: int) -> dict:
    retrieval_start = time.perf_counter()
    chunks = retrieve(question, top_k=top_k)
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 3)
    _, metrics = generate_for_config(build_prompt(question, chunks), cfg, max_tokens, seed)
    return {"retrieval_ms": retrieval_ms, **metrics}


def summarize_trials(trials: list[dict]) -> dict:
    successful = [trial for trial in trials if trial["status"] == "ok"]
    return {
        "successful_requests": len(successful),
        "failed_requests": len(trials) - len(successful),
        "metrics": {
            field: summarize([trial.get(field) for trial in successful])
            for field in SUMMARY_FIELDS
        },
    }


def run_config(
    name: str,
    cfg: dict,
    warmup: int,
    repeats: int,
    top_k: int,
    max_tokens: int,
    seed: int,
) -> dict:
    print(f"\nRunning {name}: warmup_rounds={warmup}, repeats={repeats}")
    server = inspect_backend(cfg)
    print(
        f"  backend preflight: engine={server['engine']} "
        f"version={server.get('version') or 'unknown'} model={cfg['model']}"
    )
    for warmup_round in range(warmup):
        for question_index, question in enumerate(QUESTIONS):
            try:
                execute_question(question, cfg, top_k, max_tokens, seed)
            except Exception as exc:
                print(
                    f"  warmup round={warmup_round + 1} question={question_index + 1} "
                    f"failed: {type(exc).__name__}: {exc}"
                )

    trials = []
    for repeat in range(repeats):
        for question_index, question in enumerate(QUESTIONS):
            trial = {"repeat": repeat, "question_id": question_index}
            try:
                trial.update(
                    execute_question(question, cfg, top_k, max_tokens, seed + repeat)
                )
                trial["status"] = "ok"
            except Exception as exc:
                trial.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
            trials.append(trial)
            print(f"  repeat={repeat + 1} question={question_index + 1}: {trial['status']}")
    return {
        "name": name,
        "config": cfg,
        "server": server,
        "summary": summarize_trials(trials),
        "trials": trials,
    }


def print_summary(results: list[dict]) -> None:
    print("\n| Config | Success | TTFT p50 ms | E2E p50 ms | Decode TPS mean |")
    print("|---|---:|---:|---:|---:|")
    for result in results:
        summary = result["summary"]
        metrics = summary["metrics"]
        print(
            f"| {result['name']} | {summary['successful_requests']} | "
            f"{metrics['ttft_ms']['p50']} | {metrics['e2e_ms']['p50']} | "
            f"{metrics['decode_tps']['mean']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", nargs="+", default=["ollama_q4"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    args = parser.parse_args()

    unknown = sorted(set(args.engines) - set(CONFIGS))
    if unknown:
        parser.error(f"unknown engines: {unknown}; available: {sorted(CONFIGS)}")
    if args.warmup < 0 or args.repeats < 1 or args.top_k < 1 or args.max_tokens < 1:
        parser.error("warmup must be >= 0 and repeats/top-k/max-tokens must be >= 1")

    results = [
        run_config(
            name,
            CONFIGS[name],
            args.warmup,
            args.repeats,
            args.top_k,
            args.max_tokens,
            args.seed,
        )
        for name in args.engines
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_snapshot(),
        "protocol": {
            "questions": len(QUESTIONS),
            "warmup_rounds": args.warmup,
            "warmup_requests": args.warmup * len(QUESTIONS),
            "repeats_per_question": args.repeats,
            "top_k": args.top_k,
            "max_output_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": args.seed,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nSaved raw trials and summaries to {args.output}")


if __name__ == "__main__":
    main()
