"""Diagnostic for rejecting questions that are outside the indexed corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rag.retriever import retrieve


def score_summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "mean": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
    }


def best_calibration_threshold(positive: list[float], negative: list[float]) -> dict:
    candidates = sorted(set(positive + negative))
    candidates = [candidates[0] - 1e-6] + [
        (left + right) / 2 for left, right in zip(candidates, candidates[1:])
    ] + [candidates[-1] + 1e-6]
    best = None
    for threshold in candidates:
        true_positive = sum(score >= threshold for score in positive)
        true_negative = sum(score < threshold for score in negative)
        tpr = true_positive / len(positive)
        tnr = true_negative / len(negative)
        candidate = {
            "threshold": round(threshold, 4),
            "positive_recall": round(tpr, 4),
            "negative_rejection_rate": round(tnr, 4),
            "balanced_accuracy": round((tpr + tnr) / 2, 4),
        }
        if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
            best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/abstention_cases.json"),
    )
    parser.add_argument(
        "--positive-results",
        type=Path,
        default=Path("results/retrieval/source_retrieval_n15.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/retrieval/abstention_diagnostic_n5.json"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    negative_trials = []
    for index, case in enumerate(cases):
        chunks = retrieve(case["question"], top_k=args.top_k)
        trial = {
            "case_id": case["id"],
            "question": case["question"],
            "top_score": chunks[0]["score"],
            "returned_sources": [chunk["source"] for chunk in chunks],
            "scores": [chunk["score"] for chunk in chunks],
        }
        negative_trials.append(trial)
        print(f"[{index + 1}/{len(cases)}] {case['id']}: top_score={chunks[0]['score']}")

    positive_payload = json.loads(args.positive_results.read_text(encoding="utf-8"))
    positive_scores = [trial["scores"][0] for trial in positive_payload["trials"]]
    negative_scores = [trial["top_score"] for trial in negative_trials]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "task": "out-of-corpus abstention diagnostic",
            "positive_queries": len(positive_scores),
            "negative_queries": len(negative_scores),
            "top_k": args.top_k,
        },
        "summary": {
            "positive_top1_scores": score_summary(positive_scores),
            "negative_top1_scores": score_summary(negative_scores),
            "calibration_set_best_threshold": best_calibration_threshold(
                positive_scores, negative_scores
            ),
            "warning": (
                "The threshold is fitted on a small diagnostic set and requires "
                "validation on a larger held-out set before production use."
            ),
        },
        "negative_trials": negative_trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved abstention diagnostic to {args.output}")


if __name__ == "__main__":
    main()
