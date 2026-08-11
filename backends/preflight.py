"""Fail-fast checks for local inference servers used by the benchmark."""

from __future__ import annotations

import requests

OLLAMA_ROOT = "http://localhost:11434"


def _get_json(url: str, timeout_s: float) -> dict:
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return payload


def _model_names(payload: dict) -> list[str]:
    return sorted(
        {
            model.get("name") or model.get("model")
            for model in payload.get("models", [])
            if model.get("name") or model.get("model")
        }
    )


def inspect_ollama(model: str, timeout_s: float = 5.0) -> dict:
    version = _get_json(f"{OLLAMA_ROOT}/api/version", timeout_s).get("version")
    available_models = _model_names(_get_json(f"{OLLAMA_ROOT}/api/tags", timeout_s))
    if model not in available_models:
        raise RuntimeError(
            f"Ollama model {model!r} is unavailable; installed models: "
            f"{available_models or 'none'}"
        )

    loaded_models = _model_names(_get_json(f"{OLLAMA_ROOT}/api/ps", timeout_s))
    other_loaded = [loaded for loaded in loaded_models if loaded != model]
    if other_loaded:
        commands = "; ".join(f"ollama stop {loaded}" for loaded in other_loaded)
        raise RuntimeError(
            "Ollama VRAM isolation failed because other models are loaded: "
            f"{other_loaded}. Unload them before benchmarking: {commands}"
        )

    return {
        "engine": "ollama",
        "version": version,
        "api_root": OLLAMA_ROOT,
        "model_available": True,
        "loaded_models_before": loaded_models,
    }


def inspect_vllm(model: str, base_url: str, timeout_s: float = 5.0) -> dict:
    api_root = base_url.rstrip("/")
    server_root = api_root[:-3] if api_root.endswith("/v1") else api_root
    version = _get_json(f"{server_root}/version", timeout_s).get("version")
    model_payload = _get_json(f"{api_root}/models", timeout_s)
    available_models = sorted(
        item.get("id") for item in model_payload.get("data", []) if item.get("id")
    )
    if model not in available_models:
        raise RuntimeError(
            f"vLLM model {model!r} is unavailable at {api_root}; served models: "
            f"{available_models or 'none'}"
        )
    return {
        "engine": "vllm",
        "version": version,
        "api_root": api_root,
        "model_available": True,
        "served_models": available_models,
    }


def inspect_backend(config: dict, timeout_s: float = 5.0) -> dict:
    engine = config["engine"]
    if engine == "ollama":
        return inspect_ollama(config["model"], timeout_s=timeout_s)
    if engine == "vllm":
        return inspect_vllm(config["model"], config["base_url"], timeout_s=timeout_s)
    raise ValueError(f"Unsupported backend engine: {engine}")
