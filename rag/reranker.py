"""Optional BGE cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

import os
from collections.abc import Sequence

from FlagEmbedding import FlagReranker

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_model: FlagReranker | None = None


def _get_model() -> FlagReranker:
    global _model
    if _model is None:
        device = os.environ.get("RAG_RERANK_DEVICE", "cpu").strip() or "cpu"
        _model = FlagReranker(
            RERANK_MODEL,
            devices=device,
            use_fp16=device.startswith("cuda"),
            normalize=True,
        )
    return _model


def _as_score_list(scores: object) -> list[float]:
    """Normalize scalar, list, tuple, or NumPy-like reranker output."""
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        return [float(scores)]
    if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
        return [float(score) for score in scores]
    raise TypeError(f"unsupported reranker score type: {type(scores).__name__}")


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Rerank dense candidates and retain both dense and reranker scores."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not chunks:
        return []
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = _as_score_list(_get_model().compute_score(pairs))
    if len(scores) != len(chunks):
        raise RuntimeError("reranker returned a different number of scores than candidates")
    ranked = []
    for chunk, score in zip(chunks, scores):
        item = dict(chunk)
        item["rerank_score"] = round(score, 6)
        ranked.append(item)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_k]
