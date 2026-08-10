"""Small deterministic answer-quality diagnostic for the local RAG pipeline."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from backends.ollama_backend import generate
from rag.prompting import INSUFFICIENT_EVIDENCE, build_rag_prompt, retrieval_confidence, should_abstain
from rag.retriever import retrieve

CITATION_RE = re.compile(r"\[S(\d+)\]")


def term_group_recall(answer: str, groups: list[list[str]]) -> float | None:
    if not groups:
        return None
    lowered = answer.lower()
    hits = sum(any(term.lower() in lowered for term in group) for group in groups)
    return round(hits / len(groups), 4)


def citation_diagnostic(answer: str, source_count: int) -> dict:
    ids = [int(value) for value in CITATION_RE.findall(answer)]
    valid = [value for value in ids if 1 <= value <= source_count]
    return {
        "citation_ids": ids,
        "has_citation": bool(ids),
        "all_citation_ids_valid": bool(ids) and len(valid) == len(ids),
    }


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("evaluation/answer_cases.json"))
    parser.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.5672)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/quality/ollama_q4_answers_n8.json"))
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    trials = []
    for index, case in enumerate(cases, start=1):
        chunks = retrieve(
            case["question"], top_k=args.top_k, rerank=args.rerank,
            candidate_k=args.candidate_k,
        )
        confidence = retrieval_confidence(chunks)
        gated = should_abstain(chunks, args.threshold)
        metrics = None
        if gated:
            answer = INSUFFICIENT_EVIDENCE
        else:
            answer, metrics = generate(
                build_rag_prompt(case["question"], chunks),
                model=args.model, max_tokens=args.max_tokens, temperature=0.0, seed=42,
            )
        citations = citation_diagnostic(answer, len(chunks))
        recall = term_group_recall(answer, case["expected_term_groups"])
        correctly_abstained = (not case["answerable"]) and answer.strip() == INSUFFICIENT_EVIDENCE
        false_abstention = case["answerable"] and answer.strip() == INSUFFICIENT_EVIDENCE
        trial = {
            "case_id": case["id"], "question": case["question"],
            "answerable": case["answerable"], "retrieval_confidence": confidence,
            "gate_triggered": gated,
            "sources": [
                {"id": f"S{i}", "source": chunk["source"], "chunk_idx": chunk.get("chunk_idx"),
                 "dense_score": chunk["score"]}
                for i, chunk in enumerate(chunks, start=1)
            ],
            "answer": answer, "term_group_recall": recall,
            **citations,
            "correctly_abstained": correctly_abstained,
            "false_abstention": false_abstention,
            "generation_metrics": metrics,
        }
        trials.append(trial)
        print(
            f"[{index:02d}/{len(cases)}] {case['id']}: "
            f"gate={gated} recall={recall} citations={citations['citation_ids']}"
        )

    positives = [trial for trial in trials if trial["answerable"]]
    negatives = [trial for trial in trials if not trial["answerable"]]
    generated_positives = [trial for trial in positives if not trial["false_abstention"]]
    summary = {
        "cases": len(trials),
        "positive_cases": len(positives), "negative_cases": len(negatives),
        "positive_answer_rate": round(len(generated_positives) / len(positives), 4),
        "mean_positive_term_group_recall": mean([
            trial["term_group_recall"] for trial in generated_positives
            if trial["term_group_recall"] is not None
        ]),
        "generated_positive_citation_rate": mean([
            float(trial["has_citation"]) for trial in generated_positives
        ]),
        "generated_positive_valid_citation_id_rate": mean([
            float(trial["all_citation_ids_valid"]) for trial in generated_positives
        ]),
        "negative_abstention_rate": mean([
            float(trial["correctly_abstained"]) for trial in negatives
        ]),
        "warning": (
            "Term-group recall and citation-ID validity are deterministic diagnostics, "
            "not semantic correctness or citation-entailment judgments. The threshold "
            "was fitted on a small non-held-out calibration set."
        ),
    }
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model": args.model, "top_k": args.top_k, "max_tokens": args.max_tokens,
            "temperature": 0.0, "seed": 42, "dense_abstention_threshold": args.threshold,
            "rerank": args.rerank,
            "candidate_k": args.candidate_k if args.rerank else args.top_k,
        },
        "summary": summary,
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
