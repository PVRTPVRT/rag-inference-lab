"""Standard ranked-retrieval metrics for external BEIR-style qrels."""

from __future__ import annotations

import math


def _relevant(qrels: dict[str, int]) -> dict[str, int]:
    return {doc_id: score for doc_id, score in qrels.items() if score > 0}


def query_metrics(
    ranking: list[str], qrels: dict[str, int], cutoff: int = 10
) -> dict[str, float]:
    if cutoff < 1:
        raise ValueError("cutoff must be >= 1")
    relevant = _relevant(qrels)
    retrieved = ranking[:cutoff]
    gains = [relevant.get(doc_id, 0) for doc_id in retrieved]

    reciprocal_rank = 0.0
    precisions = []
    hits = 0
    dcg = 0.0
    for rank, gain in enumerate(gains, start=1):
        if gain > 0:
            hits += 1
            if not reciprocal_rank:
                reciprocal_rank = 1 / rank
            precisions.append(hits / rank)
        dcg += (2**gain - 1) / math.log2(rank + 1)

    ideal = sorted(relevant.values(), reverse=True)[:cutoff]
    idcg = sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(ideal, start=1)
    )
    denominator = min(len(relevant), cutoff)
    return {
        f"ndcg_at_{cutoff}": dcg / idcg if idcg else 0.0,
        f"map_at_{cutoff}": sum(precisions) / denominator if denominator else 0.0,
        f"recall_at_{cutoff}": hits / len(relevant) if relevant else 0.0,
        f"mrr_at_{cutoff}": reciprocal_rank,
    }


def aggregate_metrics(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    cutoffs: tuple[int, ...] = (10, 100),
) -> dict[str, float]:
    if not rankings:
        return {}
    rows = []
    for query_id, ranking in rankings.items():
        row = {}
        for cutoff in cutoffs:
            row.update(query_metrics(ranking, qrels.get(query_id, {}), cutoff))
        rows.append(row)
    return {
        key: round(sum(row[key] for row in rows) / len(rows), 6)
        for key in rows[0]
    }


def per_query_metrics(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    cutoffs: tuple[int, ...] = (10, 100),
) -> dict[str, dict[str, float]]:
    """Return metric rows keyed by query ID for paired analysis."""
    rows = {}
    for query_id, ranking in rankings.items():
        row = {}
        for cutoff in cutoffs:
            row.update(query_metrics(ranking, qrels.get(query_id, {}), cutoff))
        rows[query_id] = row
    return rows
