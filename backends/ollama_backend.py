"""Ollama streaming backend with authoritative server-side token metrics."""

import json
import time

import requests

from backends.metrics import nanoseconds_to_milliseconds, tokens_per_second

try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False

OLLAMA_URL = "http://localhost:11434/api/generate"


def _vram_mb() -> float:
    if not _nvml_ok:
        return -1.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return round(info.used / 1024 / 1024, 1)


def generate(
    prompt: str,
    model: str = "qwen2.5:7b-instruct-q4_K_M",
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 42,
    timeout_s: int = 180,
):
    """Return generated text and metrics derived from Ollama's final object."""
    vram_before = _vram_mb()
    t_start = time.perf_counter()
    ttft_ms = None
    full_text = ""
    final_data: dict = {}

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
        },
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(f"Ollama stream error: {data['error']}")
            chunk = data.get("response", "")
            if chunk and ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t_start) * 1000, 1)
            full_text += chunk
            if data.get("done"):
                final_data = data
                break

    e2e_ms = round((time.perf_counter() - t_start) * 1000, 1)
    vram_after = _vram_mb()
    output_tokens = final_data.get("eval_count")
    eval_duration_ns = final_data.get("eval_duration")

    metrics = {
        "engine": "ollama",
        "model": model,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "output_tokens": output_tokens,
        "prompt_tokens": final_data.get("prompt_eval_count"),
        "decode_tps": tokens_per_second(output_tokens, eval_duration_ns),
        "token_count_source": "ollama_final_response.eval_count",
        "server_eval_ms": nanoseconds_to_milliseconds(eval_duration_ns),
        "server_prompt_eval_ms": nanoseconds_to_milliseconds(
            final_data.get("prompt_eval_duration")
        ),
        "server_load_ms": nanoseconds_to_milliseconds(final_data.get("load_duration")),
        "vram_used_mb": vram_after,
        "vram_delta_mb": round(vram_after - vram_before, 1) if vram_before > 0 else -1,
    }
    return full_text, metrics
