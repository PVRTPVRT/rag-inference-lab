"""vLLM backend via its OpenAI-compatible API."""

import os
import time

from openai import OpenAI

from backends.metrics import client_tokens_per_second

try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")


def _vram_mb() -> float:
    if not _nvml_ok:
        return -1.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return round(info.used / 1024 / 1024, 1)


def generate(
    prompt: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    base_url: str = VLLM_BASE_URL,
    server_profile: str = "unspecified",
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
):
    """Return generated text and client-observed metrics.

    Prefix caching and speculative decoding are server-startup settings. The
    client records a profile label but does not pretend to toggle either one.
    """
    client = OpenAI(base_url=base_url, api_key=VLLM_API_KEY)
    vram_before = _vram_mb()
    t_start = time.perf_counter()
    ttft_ms = None
    full_text = ""
    usage = None

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta and ttft_ms is None:
            ttft_ms = round((time.perf_counter() - t_start) * 1000, 1)
        full_text += delta

    e2e_ms = round((time.perf_counter() - t_start) * 1000, 1)
    vram_after = _vram_mb()
    output_tokens = getattr(usage, "completion_tokens", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)

    metrics = {
        "engine": "vllm",
        "model": model,
        "server_profile": server_profile,
        "base_url": base_url,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tps": client_tokens_per_second(output_tokens, e2e_ms, ttft_ms),
        "token_count_source": "openai_stream_usage.completion_tokens",
        "vram_used_mb": vram_after,
        "vram_delta_mb": round(vram_after - vram_before, 1) if vram_before > 0 else -1,
    }
    return full_text, metrics
