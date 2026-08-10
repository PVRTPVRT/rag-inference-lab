"""Grounded prompting and an explicitly experimental retrieval gate."""

from __future__ import annotations

from rag.retriever import build_context

INSUFFICIENT_EVIDENCE = "The retrieved context does not contain enough evidence."


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """Build a prompt with stable source IDs and a deterministic refusal string."""
    return (
        "Answer the question completely but concisely using only the retrieved context. "
        "Every factual claim must cite a source ID such as [S1]. End each paragraph or "
        "bullet with its supporting source IDs. Do not use outside knowledge. If the "
        "context is insufficient, reply exactly: "
        f"{INSUFFICIENT_EVIDENCE}\n\n"
        f"Context:\n{build_context(chunks)}\n\nQuestion: {question}\nAnswer:"
    )


def retrieval_confidence(chunks: list[dict]) -> float | None:
    """Return the best dense score; reranker scores are not threshold-calibrated yet."""
    if not chunks:
        return None
    candidate_scores = [
        chunk.get("candidate_max_dense_score") for chunk in chunks
        if chunk.get("candidate_max_dense_score") is not None
    ]
    if candidate_scores:
        return max(float(score) for score in candidate_scores)
    dense_scores = [chunk.get("score") for chunk in chunks]
    valid = [float(score) for score in dense_scores if score is not None]
    return max(valid) if valid else None


def should_abstain(chunks: list[dict], threshold: float) -> bool:
    """Apply a caller-supplied threshold to dense retrieval confidence."""
    confidence = retrieval_confidence(chunks)
    return confidence is None or confidence < threshold
