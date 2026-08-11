"""Evaluate whether retrieval returns manually labeled evidence chunks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from rag.hybrid import BM25_B, BM25_K1
from evaluation.retrieval_metrics import summarize_ranks
from rag.parent_child import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)
from rag.retriever import EMBED_MODEL, retrieve


def evidence_rank(chunks: list[dict], source: str, relevant_indices: set[int]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["source"] == source and chunk.get("chunk_idx") in relevant_indices:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evaluation/evidence_cases.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--retrieval",
        choices=("dense", "sparse", "hybrid", "parent_child"),
        default="dense",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("require top-k >= 1 and candidate-k >= top-k")

    if args.rerank and args.retrieval != "dense":
        parser.error("rerank requires --retrieval dense")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    trials = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        chunks = retrieve(
            case["question"], top_k=args.top_k, rerank=args.rerank,
            candidate_k=args.candidate_k,
            retrieval=args.retrieval, rrf_k=args.rrf_k,
        )
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        rank = evidence_rank(
            chunks, case["expected_source"], set(case["relevant_chunk_indices"])
        )
        trials.append({
            "case_id": case["id"],
            "question": case["question"],
            "expected_source": case["expected_source"],
            "relevant_chunk_indices": case["relevant_chunk_indices"],
            "returned": [
                {
                    "source": chunk["source"], "chunk_idx": chunk.get("chunk_idx"),
                    "score": chunk["score"],
                    "dense_score": chunk.get("dense_score"),
                    "sparse_score": chunk.get("sparse_score"),
                    "rrf_score": chunk.get("rrf_score"),
                    "dense_rank": chunk.get("dense_rank"),
                    "sparse_rank": chunk.get("sparse_rank"),
                    "child_chunk_idx": chunk.get("child_chunk_idx"),
                    "child_rank": chunk.get("child_rank"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in chunks
            ],
            "first_relevant_rank": rank,
            "retrieval_ms": elapsed,
        })
        print(f"[{index:02d}/{len(cases)}] {case['id']}: {'hit@' + str(rank) if rank else 'miss'}")

    ranks = [trial["first_relevant_rank"] for trial in trials]
    latencies = [trial["retrieval_ms"] for trial in trials]
    steady = latencies[1:]
    summary = summarize_ranks(ranks, args.top_k)
    summary.update({
        "cold_first_retrieval_ms": latencies[0] if latencies else None,
        "steady_retrieval_mean_ms": round(statistics.mean(steady), 3) if steady else None,
        "steady_retrieval_p50_ms": round(statistics.median(steady), 3) if steady else None,
    })
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "embedding_model": EMBED_MODEL if args.retrieval != "sparse" else None,
            "embedding_device": os.getenv("RAG_EMBED_DEVICE", "cpu") if args.retrieval != "sparse" else None,
            "reranker_model": "BAAI/bge-reranker-v2-m3" if args.rerank else None,
            "reranker_device": os.getenv("RAG_RERANK_DEVICE", "cpu") if args.rerank else None,
        },
        "protocol": {
            "task": "manually labeled evidence-chunk retrieval", "cases": len(cases),
            "retrieval": args.retrieval,
            "rrf_k": args.rrf_k if args.retrieval == "hybrid" else None,
            "bm25_k1": BM25_K1 if args.retrieval in {"sparse", "hybrid"} else None,
            "bm25_b": BM25_B if args.retrieval in {"sparse", "hybrid"} else None,
            "parent_chunk_size": PARENT_CHUNK_SIZE if args.retrieval == "parent_child" else None,
            "parent_chunk_overlap": PARENT_CHUNK_OVERLAP if args.retrieval == "parent_child" else None,
            "child_chunk_size": CHILD_CHUNK_SIZE if args.retrieval == "parent_child" else None,
            "child_chunk_overlap": CHILD_CHUNK_OVERLAP if args.retrieval == "parent_child" else None,
            "top_k": args.top_k, "rerank": args.rerank,
            "candidate_k": args.candidate_k if args.rerank or args.retrieval in {"hybrid", "parent_child"} else args.top_k,
        },
        "summary": summary,
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
