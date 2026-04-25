"""
CLI benchmark: run a fixed question set across engine configs and output a markdown table.
Usage: python benchmark.py --engines ollama_q4 ollama_q8 vllm
"""
import argparse
import time
from rag.retriever import retrieve, build_context
from backends.ollama_backend import generate as ollama_gen
from backends.vllm_backend import generate as vllm_gen

QUESTIONS = [
    "What is speculative decoding and when does it improve throughput?",
    "How does prefix caching reduce time-to-first-token in vLLM?",
    "What is PagedAttention and how does it reduce KV cache waste?",
    "Explain the trade-off between continuous batching and speculative decoding.",
    "What is Multi-Head Latent Attention (MLA) and how does it compress KV cache?",
]

CONFIGS = {
    "ollama_q4": {"engine": "ollama", "model": "qwen2.5:7b-instruct-q4_K_M"},
    "ollama_q8": {"engine": "ollama", "model": "qwen2.5:7b-instruct-q8_0"},
    "vllm_fp16": {"engine": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct", "prefix_caching": False},
    "vllm_prefix": {"engine": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct", "prefix_caching": True},
}


def run_config(cfg_name: str, cfg: dict, top_k: int = 3) -> list[dict]:
    results = []
    for q in QUESTIONS:
        chunks = retrieve(q, top_k=top_k)
        prompt = f"Context:\n{build_context(chunks)}\n\nQuestion: {q}\nAnswer:"
        try:
            if cfg["engine"] == "ollama":
                _, m = ollama_gen(prompt, model=cfg["model"])
            else:
                _, m = vllm_gen(prompt, model=cfg["model"], prefix_caching=cfg.get("prefix_caching", True))
            results.append(m)
        except Exception as e:
            print(f"  ERROR [{cfg_name}] {q[:40]}: {e}")
    return results


def avg(vals: list[float]) -> str:
    return f"{sum(vals)/len(vals):.1f}" if vals else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", nargs="+", default=list(CONFIGS.keys()))
    args = parser.parse_args()

    rows = []
    for name in args.engines:
        if name not in CONFIGS:
            print(f"Unknown config: {name}. Available: {list(CONFIGS)}")
            continue
        cfg = CONFIGS[name]
        print(f"\nRunning: {name} ...")
        results = run_config(name, cfg)
        if not results:
            continue
        rows.append({
            "Config": name,
            "Engine": cfg["engine"],
            "Model": cfg["model"].split("/")[-1] if "/" in cfg["model"] else cfg["model"],
            "Avg TTFT (ms)": avg([r["ttft_ms"] for r in results]),
            "Avg TPS": avg([r["tps"] for r in results]),
            "Avg E2E (ms)": avg([r["e2e_ms"] for r in results]),
            "VRAM (MB)": avg([r["vram_mb"] for r in results if r["vram_mb"] > 0]),
        })

    # Print markdown table
    print("\n\n## Benchmark Results\n")
    if not rows:
        print("No results.")
        return

    headers = list(rows[0].keys())
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in headers) + " |")
    print()


if __name__ == "__main__":
    main()
