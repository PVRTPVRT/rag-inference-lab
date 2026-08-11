"""Evaluate dense, BM25, and RRF retrieval on BEIR ArguAna."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from evaluation.beir_metrics import aggregate_metrics, per_query_metrics
from evaluation.bootstrap import holm_adjust, paired_bootstrap
from rag.arguana import (
    ARGUANA_MD5,
    ARGUANA_URL,
    arguana_dense_rankings,
    arguana_hybrid_rankings,
    arguana_sparse_rankings,
    build_arguana_index,
    get_arguana_index,
    load_arguana,
    prepare_arguana,
)
from rag.hybrid import BM25_B, BM25_K1
from rag.scifact import EMBED_MODEL, embedding_model


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
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--index-batch-size", type=int, default=32)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/retrieval/arguana_bge_m3_test_n1406.json"),
    )
    args = parser.parse_args()
    if args.top_k < 100:
        parser.error("top-k must be >= 100 for Recall@100")
    if not 1 <= args.record_k <= args.top_k:
        parser.error("record-k must satisfy 1 <= record-k <= top-k")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be >= 1")

    dataset = prepare_arguana()
    corpus, queries, qrels = load_arguana(dataset)
    print(f"ArguAna: {len(corpus)} documents, {len(queries)} test queries")
    model = embedding_model(args.device)
    index_seconds = None
    if args.rebuild_index:
        collection, index_seconds = build_arguana_index(
            corpus, model, batch_size=args.index_batch_size
        )
    else:
        try:
            collection = get_arguana_index()
            if collection.count() != len(corpus):
                raise ValueError("collection size does not match corpus")
        except Exception:
            collection, index_seconds = build_arguana_index(
                corpus, model, batch_size=args.index_batch_size
            )
    print(f"Dense index documents: {collection.count()}")

    dense, dense_timing = arguana_dense_rankings(
        queries, model, collection, args.top_k
    )
    sparse, sparse_timing = arguana_sparse_rankings(
        corpus, queries, args.top_k
    )
    hybrid, fusion_seconds = arguana_hybrid_rankings(
        dense, sparse, rrf_k=args.rrf_k, top_k=args.top_k
    )
    modes = {"dense": dense, "bm25": sparse, "rrf": hybrid}
    ranking_ids = {name: doc_ids(ranking) for name, ranking in modes.items()}
    results = {
        name: aggregate_metrics(ids, qrels, cutoffs=(10, 100))
        for name, ids in ranking_ids.items()
    }
    query_rows = {
        name: per_query_metrics(ids, qrels, cutoffs=(10, 100))
        for name, ids in ranking_ids.items()
    }
    metric_names = list(next(iter(query_rows["rrf"].values())))
    query_ids = sorted(queries)
    comparisons = {}
    for baseline in ("dense", "bm25"):
        name = f"rrf_vs_{baseline}"
        comparisons[name] = {
            metric: paired_bootstrap(
                [query_rows[baseline][query_id][metric] for query_id in query_ids],
                [query_rows["rrf"][query_id][metric] for query_id in query_ids],
                samples=args.bootstrap_samples,
                confidence=0.95,
                seed=args.bootstrap_seed,
            )
            for metric in metric_names
        }
        adjusted = holm_adjust(
            {
                metric: row["two_sided_bootstrap_p"]
                for metric, row in comparisons[name].items()
            }
        )
        for metric, adjusted_p in adjusted.items():
            comparisons[name][metric]["holm_adjusted_p"] = adjusted_p
            comparisons[name][metric]["significant_at_0_05_holm"] = (
                adjusted_p <= 0.05
            )
    for name, metrics in results.items():
        print(name, json.dumps(metrics, sort_keys=True))

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "BEIR/ArguAna",
            "source_url": ARGUANA_URL,
            "archive_md5": ARGUANA_MD5,
            "documents": len(corpus),
            "test_queries": len(queries),
            "license": "CC BY-SA 4.0 (per BEIR Hugging Face dataset card)",
            "task": "retrieve the best counterargument",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding_model": EMBED_MODEL,
            "embedding_device": args.device,
            "vector_search": "deterministic exact cosine matrix",
            "packages": {
                "numpy": package_version("numpy"),
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
            "exclude_query_id_from_corpus_results": True,
            "index_batch_size": args.index_batch_size,
            "query_batch_size": 32,
            "post_hoc_tuning": False,
            "dataset_selected_before_results": True,
            "selection_record": "docs/DATASET_SELECTION.md",
            "paired_bootstrap": {
                "samples": args.bootstrap_samples,
                "confidence": 0.95,
                "seed": args.bootstrap_seed,
                "resampling_unit": "test query",
                "interval": "percentile",
                "tail_probability": "two-sided, add-one corrected, exploratory",
                "multiple_comparison_correction": {
                    "method": "Holm step-down",
                    "family": "eight metrics within each baseline comparison",
                },
            },
        },
        "timing": {
            "index_build_seconds": index_seconds,
            **dense_timing,
            **sparse_timing,
            "rrf_fusion_seconds": fusion_seconds,
        },
        "results": results,
        "statistical_comparisons": comparisons,
        "queries": {
            query_id: {
                "rankings": {
                    name: rows[query_id][:args.record_k]
                    for name, rows in modes.items()
                }
            }
            for query_id in queries
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved external retrieval benchmark to {args.output}")


if __name__ == "__main__":
    main()
