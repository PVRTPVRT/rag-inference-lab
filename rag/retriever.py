"""ChromaDB retrieval with local BGE-M3 embeddings."""
import os

import chromadb
from FlagEmbedding import BGEM3FlagModel
from rag.hybrid import BM25Index, reciprocal_rank_fusion

from rag.parent_child import PARENT_CHILD_COLLECTION_NAME
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "papers"
EMBED_MODEL = "BAAI/bge-m3"
MAX_LENGTH = 512

_model: BGEM3FlagModel | None = None
_sparse_chunks = None
_sparse_index = None
_parent_child_col = None
_col = None


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        device = os.environ.get("RAG_EMBED_DEVICE", "cpu").strip() or "cpu"
        _model = BGEM3FlagModel(
            EMBED_MODEL,
            devices=device,
            use_fp16=device.startswith("cuda"),
        )
    return _model


def _get_collection():
    global _col
    if _col is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _col = client.get_collection(COLLECTION_NAME)
    return _col

def _get_parent_child_collection():
    global _parent_child_col
    if _parent_child_col is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _parent_child_col = client.get_collection(PARENT_CHILD_COLLECTION_NAME)
    return _parent_child_col


def _get_sparse_corpus() -> tuple[list[dict], BM25Index]:
    global _sparse_chunks, _sparse_index
    if _sparse_chunks is None or _sparse_index is None:
        payload = _get_collection().get(include=["documents", "metadatas"])
        rows = [
            {
                "source": metadata["source"],
                "chunk_idx": metadata.get("chunk_idx"),
                "text": document,
            }
            for document, metadata in zip(payload["documents"], payload["metadatas"])
        ]
        _sparse_chunks = sorted(
            rows,
            key=lambda row: (str(row["source"]), int(row["chunk_idx"])),
        )
        _sparse_index = BM25Index([row["text"] for row in _sparse_chunks])
    return _sparse_chunks, _sparse_index


def _sparse_retrieve(query: str, top_k: int) -> list[dict]:
    chunks, index = _get_sparse_corpus()
    return [
        {
            **chunks[position],
            "score": round(score, 6),
            "sparse_score": round(score, 6),
        }
        for position, score in index.rank(query, top_k)
    ]

def _parent_child_retrieve(query: str, top_k: int, candidate_k: int) -> list[dict]:
    model = _get_model()
    query_embedding = model.encode(
        [query], batch_size=1, max_length=MAX_LENGTH
    )["dense_vecs"][0].tolist()
    results = _get_parent_child_collection().query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    parents = []
    seen = set()
    candidate_max_dense_score = None
    for child_rank, (child_text, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), start=1):
        dense_score = round(1 / (1 + distance), 4)
        if candidate_max_dense_score is None:
            candidate_max_dense_score = dense_score
        key = (metadata["source"], metadata["parent_chunk_idx"])
        if key in seen:
            continue
        seen.add(key)
        parents.append({
            "source": metadata["source"],
            "chunk_idx": metadata["parent_chunk_idx"],
            "child_chunk_idx": metadata["child_chunk_idx"],
            "child_rank": child_rank,
            "matched_child_text": child_text,
            "text": metadata["parent_text"],
            "score": dense_score,
            "dense_score": dense_score,
            "candidate_max_dense_score": candidate_max_dense_score,
        })
        if len(parents) == top_k:
            break
    return parents


def retrieve(
    query: str,
    top_k: int = 3,
    *,
    rerank: bool = False,
    candidate_k: int = 12,
    retrieval: str = "dense",
    rrf_k: int = 60,
) -> list[dict]:
    """Retrieve with dense, sparse, or RRF hybrid ranking."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be >= top_k")
    if retrieval not in {"dense", "sparse", "hybrid", "parent_child"}:
        raise ValueError("retrieval must be dense, sparse, hybrid, or parent_child")
    if rerank and retrieval != "dense":
        raise ValueError("reranking currently requires dense retrieval")
    if retrieval == "sparse":
        return _sparse_retrieve(query, top_k)
    if retrieval == "parent_child":
        return _parent_child_retrieve(query, top_k, candidate_k)
    model = _get_model()
    q_emb = model.encode(
        [query], batch_size=1, max_length=MAX_LENGTH
    )["dense_vecs"][0].tolist()
    col = _get_collection()
    n_results = candidate_k if rerank or retrieval == "hybrid" else top_k
    results = col.query(
        query_embeddings=[q_emb], n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # ChromaDB returns L2 distance; convert to similarity score
        chunks.append(
            {
                "source": meta["source"],
                "chunk_idx": meta.get("chunk_idx"),
                "text": doc,
                "score": round(1 / (1 + dist), 4),
                "dense_score": round(1 / (1 + dist), 4),
            }
        )
    if retrieval == "hybrid":
        sparse_chunks = _sparse_retrieve(query, candidate_k)
        fused = reciprocal_rank_fusion(
            {"dense": chunks, "sparse": sparse_chunks}, rrf_k=rrf_k
        )
        candidate_max_dense_score = max(chunk["dense_score"] for chunk in chunks)
        for chunk in fused:
            chunk["score"] = chunk["rrf_score"]
            chunk["candidate_max_dense_score"] = candidate_max_dense_score
        return fused[:top_k]
    if rerank:
        from rag.reranker import rerank_chunks

        candidate_max_dense_score = max(chunk["score"] for chunk in chunks)
        ranked = rerank_chunks(query, chunks, top_k)
        for chunk in ranked:
            chunk["candidate_max_dense_score"] = candidate_max_dense_score
        return ranked
    return chunks


def build_context(chunks: list[dict]) -> str:
    entries = []
    for index, chunk in enumerate(chunks, start=1):
        scores = []
        if chunk.get("dense_score") is not None:
            scores.append(f"Dense score: {chunk['dense_score']}")
        if chunk.get("sparse_score") is not None:
            scores.append(f"BM25 score: {chunk['sparse_score']}")
        if chunk.get("rrf_score") is not None:
            scores.append(f"RRF score: {chunk['rrf_score']}")
        if not scores:
            scores.append(f"Dense score: {chunk['score']}")
        label = " | ".join(scores)
        entries.append(
            f"[S{index} | Source: {chunk['source']} | Chunk: {chunk.get('chunk_idx')} | "
            f"{label}]\n{chunk['text']}"
        )
    return "\n\n".join(entries)
