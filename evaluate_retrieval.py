"""Evaluate source-level retrieval quality on a labeled local query set."""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from evaluation.retrieval_metrics import rank_of_expected, summarize_ranks
from rag.retriever import EMBED_MODEL, retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/retrieval_cases.json"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/retrieval/source_retrieval_n15.json"),
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be >= 1")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    trials = []
    for index, case in enumerate(cases):
        chunks = retrieve(case["question"], top_k=args.top_k)
        returned_sources = [chunk["source"] for chunk in chunks]
        rank = rank_of_expected(returned_sources, case["expected_source"])
        trial = {
            "case_id": case["id"],
            "question": case["question"],
            "expected_source": case["expected_source"],
            "returned_sources": returned_sources,
            "scores": [chunk["score"] for chunk in chunks],
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

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding_model": EMBED_MODEL,
            "embedding_device": os.getenv("RAG_EMBED_DEVICE", "cpu"),
        },
        "protocol": {
            "task": "source-level retrieval",
            "cases": len(cases),
            "top_k": args.top_k,
            "annotation": "one expected paper title per query",
        },
        "summary": {**summary, "per_source": per_source},
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved retrieval evaluation to {args.output}")


if __name__ == "__main__":
    main()
