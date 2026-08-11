"""Run claim-level NLI over saved RAG answers and their cited chunks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from rag.entailment import EntailmentVerifier, NLI_MODEL, verify_cited_answer
from rag.prompting import INSUFFICIENT_EVIDENCE
from rag.retriever import CHROMA_PATH, COLLECTION_NAME


def load_chunk_map() -> dict[tuple[str, int], str]:
    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION_NAME)
    payload = collection.get(include=["documents", "metadatas"])
    return {
        (metadata["source"], int(metadata["chunk_idx"])): document
        for document, metadata in zip(payload["documents"], payload["metadatas"])
    }


def mean_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=NLI_MODEL)
    parser.add_argument("--entailment-threshold", type=float, default=0.5)
    parser.add_argument("--contradiction-threshold", type=float, default=0.5)
    parser.add_argument("--focus-sentences", type=int, default=3)
    args = parser.parse_args()

    answer_payload = json.loads(args.answers.read_text(encoding="utf-8"))
    chunk_map = load_chunk_map()
    verifier = EntailmentVerifier(args.model, focus_sentences=args.focus_sentences)
    trials = []
    pair_latencies = []
    for trial in answer_payload["trials"]:
        if not trial["answerable"] or trial["answer"].strip() == INSUFFICIENT_EVIDENCE:
            continue
        chunks = []
        for source in trial["sources"]:
            key = (source["source"], int(source["chunk_idx"]))
            if key not in chunk_map:
                raise KeyError(f"missing indexed chunk for {key}")
            chunks.append({**source, "text": chunk_map[key]})
        verification = verify_cited_answer(
            trial["answer"],
            chunks,
            verifier=verifier,
            entailment_threshold=args.entailment_threshold,
            contradiction_threshold=args.contradiction_threshold,
        )
        for claim in verification["claims"]:
            pair_latencies.extend(row["latency_ms"] for row in claim["evidence"])
        trials.append({
            "case_id": trial["case_id"],
            "answer": trial["answer"],
            "verification": verification,
        })
        summary = verification["summary"]
        print(
            f"{trial['case_id']}: claims={summary['claim_count']} "
            f"supported={summary['supported_claim_count']} "
            f"contradicted={summary['contradicted_claim_count']}"
        )

    claims = sum(item["verification"]["summary"]["claim_count"] for item in trials)
    cited = sum(item["verification"]["summary"]["cited_claim_count"] for item in trials)
    supported = sum(
        item["verification"]["summary"]["supported_claim_count"] for item in trials
    )
    contradicted = sum(
        item["verification"]["summary"]["contradicted_claim_count"] for item in trials
    )
    invalid = sum(
        item["verification"]["summary"]["invalid_citation_count"] for item in trials
    )
    summary = {
        "answers": len(trials),
        "claims": claims,
        "cited_claims": cited,
        "citation_coverage": mean_ratio(cited, claims),
        "supported_claims": supported,
        "claim_entailment_rate": mean_ratio(supported, claims),
        "contradicted_claims": contradicted,
        "claim_contradiction_rate": mean_ratio(contradicted, claims),
        "invalid_citation_count": invalid,
        "nli_pairs": len(pair_latencies),
        "nli_pair_mean_ms": round(statistics.mean(pair_latencies), 3) if pair_latencies else None,
        "nli_pair_p50_ms": round(statistics.median(pair_latencies), 3) if pair_latencies else None,
        "warning": (
            "NLI labels are automated diagnostics. The thresholds require calibration "
            "against independently annotated generated-answer claims."
        ),
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model": args.model,
            "device": os.getenv("RAG_NLI_DEVICE", "cpu"),
        },
        "protocol": {
            "source_answers": str(args.answers),
            "entailment_threshold": args.entailment_threshold,
            "contradiction_threshold": args.contradiction_threshold,
            "focus_sentences": args.focus_sentences,
        },
        "summary": summary,
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
