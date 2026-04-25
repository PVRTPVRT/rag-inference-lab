"""
vLLM backend via OpenAI-compatible API.
Start vLLM first:
  vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching --port 8000

Optionally add: --speculative-model ngram --num-speculative-tokens 5
"""
import time
from openai import OpenAI

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"


def _vram_mb() -> float:
    if not _nvml_ok:
        return -1.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return round(info.used / 1024 / 1024, 1)


def generate(
    prompt: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    prefix_caching: bool = True,
    max_tokens: int = 512,
):
    """
    Returns: (full_text, metrics_dict)
    Note: prefix_caching is controlled at vLLM server startup (--enable-prefix-caching).
    This flag is recorded in metrics for display purposes.
    """
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    vram_before = _vram_mb()
    t_start = time.perf_counter()
    ttft_ms = None
    full_text = ""
    tokens = 0

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta and ttft_ms is None:
            ttft_ms = round((time.perf_counter() - t_start) * 1000, 1)
        full_text += delta
        if delta:
            tokens += len(delta.split())

    e2e_ms = round((time.perf_counter() - t_start) * 1000, 1)
    decode_s = (e2e_ms - (ttft_ms or 0)) / 1000
    tps = round(tokens / decode_s, 1) if decode_s > 0 else 0.0
    vram_after = _vram_mb()

    metrics = {
        "engine": "vllm",
        "model": model,
        "prefix_caching": prefix_caching,
        "ttft_ms": ttft_ms or 0,
        "tps": tps,
        "e2e_ms": e2e_ms,
        "vram_mb": vram_after,
        "vram_delta_mb": round(vram_after - vram_before, 1) if vram_before > 0 else -1,
        "tokens": tokens,
    }
    return full_text, metrics
