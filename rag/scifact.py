"""SciFact/BEIR data preparation and retrieval helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import chromadb
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from rag.hybrid import BM25Index, reciprocal_rank_fusion

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
SCIFACT_COLLECTION = "beir_scifact"
EMBED_MODEL = "BAAI/bge-m3"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe zip member: {member.filename}")
        zipped.extractall(destination)


def prepare_beir_dataset(
    name: str,
    url: str,
    expected_md5: str,
    cache_dir: Path = Path(".cache/beir"),
) -> Path:
    """Download, verify, and extract a BEIR archive outside version control."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"{name}.zip"
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    actual = _md5(archive)
    if actual != expected_md5:
        raise ValueError(
            f"{name} MD5 mismatch: expected {expected_md5}, got {actual}"
        )
    dataset = cache_dir / name
    if not (dataset / "corpus.jsonl").exists():
        _safe_extract(archive, cache_dir)
    return dataset


def prepare_scifact(cache_dir: Path = Path(".cache/beir")) -> Path:
    """Download, verify, and extract SciFact without committing third-party data."""
    return prepare_beir_dataset(
        "scifact", SCIFACT_URL, SCIFACT_MD5, cache_dir
    )


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_beir_dataset(dataset: Path) -> tuple[dict, dict, dict]:
    corpus = {
        str(row["_id"]): {
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }
        for row in _jsonl(dataset / "corpus.jsonl")
    }
    queries = {
        str(row["_id"]): row["text"]
        for row in _jsonl(dataset / "queries.jsonl")
    }
    qrels: dict[str, dict[str, int]] = {}
    with (dataset / "qrels" / "test.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            query_id = str(row["query-id"])
            qrels.setdefault(query_id, {})[str(row["corpus-id"])] = int(row["score"])
    test_queries = {query_id: queries[query_id] for query_id in sorted(qrels)}
    return corpus, test_queries, qrels


def load_scifact(dataset: Path) -> tuple[dict, dict, dict]:
    return load_beir_dataset(dataset)


def document_text(document: dict) -> str:
    return "\n".join(
        part.strip() for part in (document.get("title", ""), document["text"])
        if part.strip()
    )


def embedding_model(device: str) -> BGEM3FlagModel:
    return BGEM3FlagModel(
        EMBED_MODEL,
        devices=device,
        use_fp16=device.startswith("cuda"),
    )


def build_scifact_index(
    corpus: dict,
    model: BGEM3FlagModel,
    *,
    chroma_path: str = "./chroma_db",
    batch_size: int = 64,
    collection_name: str = SCIFACT_COLLECTION,
) -> tuple[object, float]:
    ordered = sorted(corpus)
    texts = [document_text(corpus[doc_id]) for doc_id in ordered]
    started = time.perf_counter()
    embeddings = model.encode(
        texts, batch_size=batch_size, max_length=512
    )["dense_vecs"].tolist()
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )
    for start in range(0, len(ordered), 500):
        end = start + 500
        ids = ordered[start:end]
        collection.add(
            ids=ids,
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=[{"title": corpus[doc_id]["title"]} for doc_id in ids],
        )
    return collection, round(time.perf_counter() - started, 3)


def get_scifact_index(
    chroma_path: str = "./chroma_db",
    collection_name: str = SCIFACT_COLLECTION,
):
    return chromadb.PersistentClient(path=chroma_path).get_collection(
        collection_name
    )


def exact_cosine_top_k(
    query_embeddings,
    document_embeddings,
    document_ids: list[str],
    top_k: int,
) -> list[list[dict]]:
    """Return deterministic exact cosine rankings with document-ID tie breaks."""
    if top_k < 1 or top_k > len(document_ids):
        raise ValueError("top_k must satisfy 1 <= top_k <= document count")
    queries = np.asarray(query_embeddings, dtype=np.float32)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    if queries.ndim != 2 or documents.ndim != 2:
        raise ValueError("query and document embeddings must be 2D")
    if queries.shape[1] != documents.shape[1]:
        raise ValueError("query and document dimensions must match")
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    document_norms = np.linalg.norm(documents, axis=1, keepdims=True)
    queries = queries / np.maximum(query_norms, 1e-12)
    documents = documents / np.maximum(document_norms, 1e-12)
    similarities = queries @ documents.T
    ids = np.asarray(document_ids, dtype=str)
    rankings = []
    for scores in similarities:
        order = np.lexsort((ids, -scores))[:top_k]
        rankings.append([
            {
                "doc_id": document_ids[index],
                "dense_score": round(float(scores[index]), 6),
            }
            for index in order
        ])
    return rankings


def dense_rankings(
    queries: dict[str, str],
    model: BGEM3FlagModel,
    collection,
    top_k: int,
    batch_size: int = 32,
) -> tuple[dict[str, list[dict]], dict]:
    query_ids = list(queries)
    started = time.perf_counter()
    embeddings = model.encode(
        [queries[query_id] for query_id in query_ids],
        batch_size=batch_size,
        max_length=512,
    )["dense_vecs"]
    encoded_seconds = time.perf_counter() - started
    searched = time.perf_counter()
    corpus = collection.get(include=["embeddings"])
    ranked_rows = exact_cosine_top_k(
        embeddings,
        corpus["embeddings"],
        corpus["ids"],
        top_k,
    )
    search_seconds = time.perf_counter() - searched
    rankings = {
        query_id: ranked_rows[index]
        for index, query_id in enumerate(query_ids)
    }
    return rankings, {
        "query_embedding_seconds": round(encoded_seconds, 3),
        "exact_cosine_search_seconds": round(search_seconds, 3),
    }


def sparse_rankings(
    corpus: dict, queries: dict[str, str], top_k: int
) -> tuple[dict[str, list[dict]], dict]:
    doc_ids = sorted(corpus)
    started = time.perf_counter()
    index = BM25Index([document_text(corpus[doc_id]) for doc_id in doc_ids])
    build_seconds = time.perf_counter() - started
    searched = time.perf_counter()
    rankings = {}
    for query_id, query in queries.items():
        rankings[query_id] = [
            {"doc_id": doc_ids[position], "sparse_score": round(score, 6)}
            for position, score in index.rank(query, top_k)
        ]
    return rankings, {
        "bm25_build_seconds": round(build_seconds, 3),
        "bm25_search_seconds": round(time.perf_counter() - searched, 3),
    }


def hybrid_rankings(
    dense: dict[str, list[dict]],
    sparse: dict[str, list[dict]],
    *,
    rrf_k: int = 60,
    top_k: int = 100,
) -> tuple[dict[str, list[dict]], float]:
    started = time.perf_counter()
    rankings = {
        query_id: reciprocal_rank_fusion(
            {"dense": dense[query_id], "sparse": sparse[query_id]},
            key=lambda row: (row["doc_id"],),
            rrf_k=rrf_k,
        )[:top_k]
        for query_id in dense
    }
    return rankings, round(time.perf_counter() - started, 3)
