"""
Ollama backend: streaming generation with TTFT, TPS, and VRAM metrics.
Requires: ollama running locally (ollama serve)
"""
import time
import requests
import json

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


def generate(prompt: str, model: str = "qwen2.5:7b-instruct-q4_K_M", stream: bool = True):
    """
    Returns: (full_text, metrics_dict)
    metrics = {ttft_ms, tps, e2e_ms, vram_mb, model}
    """
    vram_before = _vram_mb()
    t_start = time.perf_counter()
    ttft_ms = None
    tokens = 0
    full_text = ""

    payload = {"model": model, "prompt": prompt, "stream": True}
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("response", "")
            if chunk and ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t_start) * 1000, 1)
            full_text += chunk
            tokens += 1
            if data.get("done"):
                break

    e2e_ms = round((time.perf_counter() - t_start) * 1000, 1)
    decode_s = (e2e_ms - (ttft_ms or 0)) / 1000
    tps = round(tokens / decode_s, 1) if decode_s > 0 else 0.0
    vram_after = _vram_mb()

    metrics = {
        "engine": "ollama",
        "model": model,
        "ttft_ms": ttft_ms or 0,
        "tps": tps,
        "e2e_ms": e2e_ms,
        "vram_mb": vram_after,
        "vram_delta_mb": round(vram_after - vram_before, 1) if vram_before > 0 else -1,
        "tokens": tokens,
    }
    return full_text, metrics
