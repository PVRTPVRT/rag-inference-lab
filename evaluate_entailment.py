"""Evaluate NLI citation verification on manually labeled paper-chunk pairs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from rag.entailment import EntailmentVerifier, NLI_MODEL
from rag.retriever import CHROMA_PATH, COLLECTION_NAME

LABELS = ("contradiction", "entailment", "neutral")


def load_chunk_map() -> dict[tuple[str, int], str]:
    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION_NAME)
    payload = collection.get(include=["documents", "metadatas"])
    return {
        (metadata["source"], int(metadata["chunk_idx"])): document
        for document, metadata in zip(payload["documents"], payload["metadatas"])
    }


def confusion_matrix(trials: list[dict]) -> dict:
    return {
        expected: {
            predicted: sum(
                trial["expected_label"] == expected
                and trial["predicted_label"] == predicted
                for trial in trials
            )
            for predicted in LABELS
        }
        for expected in LABELS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", type=Path, default=Path("evaluation/entailment_cases.json")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/quality/nli_entailment_gold_n30.json"),
    )
    parser.add_argument("--model", default=NLI_MODEL)
    parser.add_argument("--focus-sentences", type=int, default=0)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    chunk_map = load_chunk_map()
    verifier = EntailmentVerifier(args.model, focus_sentences=args.focus_sentences)
    trials = []
    for index, case in enumerate(cases, start=1):
        key = (case["source"], int(case["chunk_idx"]))
        if key not in chunk_map:
            raise KeyError(f"missing indexed chunk for {key}")
        result = verifier.predict(chunk_map[key], case["claim"])
        trial = {
            "case_id": case["id"],
            "source": case["source"],
            "chunk_idx": case["chunk_idx"],
            "claim": case["claim"],
            "expected_label": case["label"],
            "predicted_label": result.predicted_label,
            "probabilities": result.probabilities,
            "latency_ms": result.latency_ms,
            "correct": result.predicted_label == case["label"],
        }
        trials.append(trial)
        print(
            f"[{index:02d}/{len(cases)}] {case['id']}: "
            f"expected={case['label']} predicted={result.predicted_label}"
        )

    correct = sum(trial["correct"] for trial in trials)
    per_label = {}
    for label in LABELS:
        selected = [trial for trial in trials if trial["expected_label"] == label]
        per_label[label] = {
            "cases": len(selected),
            "accuracy": round(sum(trial["correct"] for trial in selected) / len(selected), 4),
        }
    latencies = [trial["latency_ms"] for trial in trials]
    steady = latencies[1:]
    summary = {
        "cases": len(trials),
        "label_counts": dict(Counter(case["label"] for case in cases)),
        "accuracy": round(correct / len(trials), 4),
        "per_label": per_label,
        "confusion_matrix": confusion_matrix(trials),
        "cold_first_pair_ms": latencies[0],
        "steady_pair_mean_ms": round(statistics.mean(steady), 3),
        "steady_pair_p50_ms": round(statistics.median(steady), 3),
        "warning": (
            "This is a small in-domain diagnostic set with one annotator. It is not "
            "a production groundedness accuracy claim or a calibrated decision threshold."
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
            "task": "three-way claim-to-cited-chunk NLI",
            "annotation": "single-annotator balanced entailment/neutral/contradiction set",
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
