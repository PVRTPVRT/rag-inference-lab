"""Dependency-free BM25 scoring and reciprocal-rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

TOKEN_RE = re.compile(r"[a-z0-9]+")
BM25_K1 = 1.5
BM25_B = 0.75


def lexical_tokens(text: str) -> list[str]:
    """Tokenize technical English while preserving digits in identifiers."""
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Small in-memory Okapi BM25 index for deterministic local evaluation."""

    def __init__(
        self,
        documents: list[str],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(lexical_tokens(doc)) for doc in documents]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.document_count = len(documents)
        self.average_length = (
            sum(self.lengths) / self.document_count if self.document_count else 0.0
        )
        self.document_frequencies = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequencies.update(frequencies.keys())

    def score(self, query: str) -> list[float]:
        if not self.document_count:
            return []
        query_terms = set(lexical_tokens(query))
        scores = []
        for frequencies, length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequencies[term]
                inverse_frequency = math.log(
                    1
                    + (self.document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                normalization = 1 - self.b
                if self.average_length:
                    normalization += self.b * length / self.average_length
                denominator = frequency + self.k1 * normalization
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        ranked = sorted(
            enumerate(self.score(query)),
            key=lambda item: (-item[1], item[0]),
        )
        return [(index, score) for index, score in ranked[:top_k] if score > 0]


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict]],
    *,
    key: Callable[[dict], tuple] = lambda row: (row["source"], row["chunk_idx"]),
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse named rankings and retain their component ranks and scores."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    fused = {}
    for name, rows in rankings.items():
        for rank, row in enumerate(rows, start=1):
            identifier = key(row)
            item = fused.setdefault(identifier, {**row, "rrf_score": 0.0})
            for field, value in row.items():
                if field not in item or item[field] is None:
                    item[field] = value
            item[f"{name}_rank"] = rank
            item["rrf_score"] += 1 / (rrf_k + rank)
    for item in fused.values():
        item["rrf_score"] = round(item["rrf_score"], 8)
    return [
        item
        for identifier, item in sorted(
            fused.items(),
            key=lambda pair: (
                -pair[1]["rrf_score"],
                tuple(str(value) for value in pair[0]),
            ),
        )
    ]
