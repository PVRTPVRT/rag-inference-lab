"""Dependency-free source retrieval metrics."""

from __future__ import annotations


def rank_of_expected(returned_sources: list[str], expected_source: str) -> int | None:
    for rank, source in enumerate(returned_sources, start=1):
        if source == expected_source:
            return rank
    return None


def summarize_ranks(ranks: list[int | None], top_k: int) -> dict:
    if not ranks:
        return {
            "queries": 0,
            f"hit_rate_at_{top_k}": None,
            "mean_reciprocal_rank": None,
        }
    hits = sum(rank is not None and rank <= top_k for rank in ranks)
    reciprocal_ranks = [
        0.0 if rank is None or rank > top_k else 1.0 / rank for rank in ranks
    ]
    return {
        "queries": len(ranks),
        "hits": hits,
        f"hit_rate_at_{top_k}": round(hits / len(ranks), 4),
        "mean_reciprocal_rank": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks), 4
        ),
    }
