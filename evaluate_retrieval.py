"""Evaluate source-level retrieval quality on a labeled local query set."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.retrieval_metrics import rank_of_expected, summarize_ranks
from rag.hybrid import BM25_B, BM25_K1
from rag.retriever import EMBED_MODEL, retrieve
from rag.parent_child import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/retrieval_cases.json"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--retrieval",
        choices=("dense", "sparse", "hybrid", "parent_child"),
        default="dense",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/retrieval/source_retrieval_n15.json"),
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be >= 1")
    if args.candidate_k < args.top_k:
        parser.error("candidate-k must be >= top-k")

    if args.rerank and args.retrieval != "dense":
        parser.error("rerank requires --retrieval dense")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    trials = []
    for index, case in enumerate(cases):
        started = time.perf_counter()
        chunks = retrieve(
            case["question"],
            top_k=args.top_k,
            rerank=args.rerank,
            candidate_k=args.candidate_k,
            retrieval=args.retrieval,
            rrf_k=args.rrf_k,
        )
        retrieval_ms = round((time.perf_counter() - started) * 1000, 3)
        returned_sources = [chunk["source"] for chunk in chunks]
        rank = rank_of_expected(returned_sources, case["expected_source"])
        trial = {
            "case_id": case["id"],
            "question": case["question"],
            "expected_source": case["expected_source"],
            "returned_sources": returned_sources,
            "scores": [chunk["score"] for chunk in chunks],
            "dense_scores": [chunk.get("dense_score") for chunk in chunks],
            "sparse_scores": [chunk.get("sparse_score") for chunk in chunks],
            "rrf_scores": [chunk.get("rrf_score") for chunk in chunks],
            "dense_ranks": [chunk.get("dense_rank") for chunk in chunks],
            "sparse_ranks": [chunk.get("sparse_rank") for chunk in chunks],
            "child_chunk_indices": [chunk.get("child_chunk_idx") for chunk in chunks],
            "child_ranks": [chunk.get("child_rank") for chunk in chunks],
            "rerank_scores": [chunk.get("rerank_score") for chunk in chunks],
            "returned_chunk_indices": [chunk.get("chunk_idx") for chunk in chunks],
            "retrieval_ms": retrieval_ms,
            "first_relevant_rank": rank,
            "hit": rank is not None,
        }
        trials.append(trial)
        print(
            f"[{index + 1:02d}/{len(cases)}] {case['id']}: "
            f"{'hit@' + str(rank) if rank else 'miss'}"
        )

    ranks = [trial["first_relevant_rank"] for trial in trials]
    summary = summarize_ranks(ranks, top_k=args.top_k)
    per_source = {}
    for source in sorted({case["expected_source"] for case in cases}):
        source_ranks = [
            trial["first_relevant_rank"]
            for trial in trials
            if trial["expected_source"] == source
        ]
        per_source[source] = summarize_ranks(source_ranks, top_k=args.top_k)

    latencies = [trial["retrieval_ms"] for trial in trials]
    steady = latencies[1:]
    mean_retrieval_ms = round(sum(latencies) / len(latencies), 3) if latencies else None

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding_model": EMBED_MODEL if args.retrieval != "sparse" else None,
            "embedding_device": os.getenv("RAG_EMBED_DEVICE", "cpu") if args.retrieval != "sparse" else None,
            "reranker_model": "BAAI/bge-reranker-v2-m3" if args.rerank else None,
            "reranker_device": os.getenv("RAG_RERANK_DEVICE", "cpu") if args.rerank else None,
        },
        "protocol": {
            "task": "source-level retrieval",
            "cases": len(cases),
            "top_k": args.top_k,
            "retrieval": args.retrieval,
            "rrf_k": args.rrf_k if args.retrieval == "hybrid" else None,
            "bm25_k1": BM25_K1 if args.retrieval in {"sparse", "hybrid"} else None,
            "bm25_b": BM25_B if args.retrieval in {"sparse", "hybrid"} else None,
            "parent_chunk_size": PARENT_CHUNK_SIZE if args.retrieval == "parent_child" else None,
            "parent_chunk_overlap": PARENT_CHUNK_OVERLAP if args.retrieval == "parent_child" else None,
            "child_chunk_size": CHILD_CHUNK_SIZE if args.retrieval == "parent_child" else None,
            "child_chunk_overlap": CHILD_CHUNK_OVERLAP if args.retrieval == "parent_child" else None,
            "rerank": args.rerank,
            "candidate_k": args.candidate_k if args.rerank or args.retrieval in {"hybrid", "parent_child"} else args.top_k,
            "annotation": "one expected paper title per query",
        },
        "summary": {
            **summary,
            "mean_retrieval_ms_including_model_warmup": mean_retrieval_ms,
            "cold_first_retrieval_ms": latencies[0] if latencies else None,
            "steady_retrieval_mean_ms": round(statistics.mean(steady), 3) if steady else None,
            "steady_retrieval_p50_ms": round(statistics.median(steady), 3) if steady else None,
            "per_source": per_source,
        },
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved retrieval evaluation to {args.output}")


if __name__ == "__main__":
    main()
