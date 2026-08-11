"""Evaluate dense, BM25, and RRF retrieval on the external SciFact test set."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from evaluation.beir_metrics import aggregate_metrics
from rag.hybrid import BM25_B, BM25_K1
from rag.scifact import (
    EMBED_MODEL,
    SCIFACT_MD5,
    SCIFACT_URL,
    build_scifact_index,
    dense_rankings,
    embedding_model,
    get_scifact_index,
    hybrid_rankings,
    load_scifact,
    prepare_scifact,
    sparse_rankings,
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def doc_ids(rankings: dict[str, list[dict]]) -> dict[str, list[str]]:
    return {
        query_id: [row["doc_id"] for row in rows]
        for query_id, rows in rankings.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--record-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/retrieval/scifact_bge_m3_test_n300.json"),
    )
    args = parser.parse_args()
    if args.top_k < 100:
        parser.error("top-k must be >= 100 for Recall@100")
    if not 1 <= args.record_k <= args.top_k:
        parser.error("record-k must satisfy 1 <= record-k <= top-k")

    dataset = prepare_scifact()
    corpus, queries, qrels = load_scifact(dataset)
    print(f"SciFact: {len(corpus)} documents, {len(queries)} test queries")
    model = embedding_model(args.device)
    index_seconds = None
    if args.rebuild_index:
        collection, index_seconds = build_scifact_index(corpus, model)
    else:
        try:
            collection = get_scifact_index()
            if collection.count() != len(corpus):
                raise ValueError("collection size does not match corpus")
        except Exception:
            collection, index_seconds = build_scifact_index(corpus, model)
    print(f"Dense index documents: {collection.count()}")

    dense, dense_timing = dense_rankings(
        queries, model, collection, args.top_k
    )
    sparse, sparse_timing = sparse_rankings(corpus, queries, args.top_k)
    hybrid, fusion_seconds = hybrid_rankings(
        dense, sparse, rrf_k=args.rrf_k, top_k=args.top_k
    )
    modes = {"dense": dense, "bm25": sparse, "rrf": hybrid}
    results = {
        name: aggregate_metrics(doc_ids(ranking), qrels, cutoffs=(10, 100))
        for name, ranking in modes.items()
    }
    for name, metrics in results.items():
        print(name, json.dumps(metrics, sort_keys=True))

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "BEIR/SciFact",
            "source_url": SCIFACT_URL,
            "archive_md5": SCIFACT_MD5,
            "documents": len(corpus),
            "test_queries": len(queries),
            "licenses": {
                "claims_and_evidence": "CC BY 4.0",
                "abstract_corpus": "ODC-By 1.0",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding_model": EMBED_MODEL,
            "embedding_device": args.device,
            "vector_search": "deterministic exact cosine matrix",
            "packages": {
                "chromadb": package_version("chromadb"),
                "FlagEmbedding": package_version("FlagEmbedding"),
                "torch": package_version("torch"),
            },
        },
        "protocol": {
            "split": "test",
            "modes": list(modes),
            "top_k": args.top_k,
            "record_k": args.record_k,
            "rrf_k": args.rrf_k,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "query_order": "sorted qrel query IDs",
            "post_hoc_tuning": False,
        },
        "timing": {
            "index_build_seconds": index_seconds,
            **dense_timing,
            **sparse_timing,
            "rrf_fusion_seconds": fusion_seconds,
        },
        "results": results,
        "queries": {
            query_id: {
                "rankings": {
                    name: rows[query_id][:args.record_k]
                    for name, rows in modes.items()
                },
            }
            for query_id in queries
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved external retrieval benchmark to {args.output}")


if __name__ == "__main__":
    main()
