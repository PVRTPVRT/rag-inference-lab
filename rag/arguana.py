"""ArguAna preparation and self-match-safe retrieval helpers."""

from __future__ import annotations

from pathlib import Path

from rag.scifact import (
    build_scifact_index,
    dense_rankings,
    get_scifact_index,
    hybrid_rankings,
    load_beir_dataset,
    prepare_beir_dataset,
    sparse_rankings,
)

ARGUANA_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/arguana.zip"
)
ARGUANA_MD5 = "8ad3e3c2a5867cdced806d6503f29b99"
ARGUANA_COLLECTION = "beir_arguana"


def prepare_arguana(cache_dir: Path = Path(".cache/beir")) -> Path:
    return prepare_beir_dataset(
        "arguana", ARGUANA_URL, ARGUANA_MD5, cache_dir
    )


def load_arguana(dataset: Path) -> tuple[dict, dict, dict]:
    return load_beir_dataset(dataset)


def build_arguana_index(corpus: dict, model, **kwargs):
    return build_scifact_index(
        corpus, model, collection_name=ARGUANA_COLLECTION, **kwargs
    )


def get_arguana_index(chroma_path: str = "./chroma_db"):
    return get_scifact_index(
        chroma_path=chroma_path, collection_name=ARGUANA_COLLECTION
    )


def remove_query_self_matches(
    rankings: dict[str, list[dict]], top_k: int
) -> dict[str, list[dict]]:
    """Exclude an ArguAna query's identical corpus item from its ranking."""
    return {
        query_id: [
            row for row in rows if row["doc_id"] != query_id
        ][:top_k]
        for query_id, rows in rankings.items()
    }


def arguana_dense_rankings(queries, model, collection, top_k):
    candidate_k = min(top_k + 1, collection.count())
    rankings, timing = dense_rankings(
        queries, model, collection, candidate_k
    )
    return remove_query_self_matches(rankings, top_k), timing


def arguana_sparse_rankings(corpus, queries, top_k):
    candidate_k = min(top_k + 1, len(corpus))
    rankings, timing = sparse_rankings(corpus, queries, candidate_k)
    return remove_query_self_matches(rankings, top_k), timing


def arguana_hybrid_rankings(dense, sparse, *, rrf_k=60, top_k=100):
    rankings, timing = hybrid_rankings(
        dense, sparse, rrf_k=rrf_k, top_k=top_k
    )
    return remove_query_self_matches(rankings, top_k), timing
