"""ChromaDB retrieval with local BGE-M3 embeddings."""
import os

import chromadb
from FlagEmbedding import BGEM3FlagModel

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "papers"
EMBED_MODEL = "BAAI/bge-m3"
MAX_LENGTH = 512

_model: BGEM3FlagModel | None = None
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


def retrieve(
    query: str,
    top_k: int = 3,
    *,
    rerank: bool = False,
    candidate_k: int = 12,
) -> list[dict]:
    """Retrieve dense top-k chunks, optionally reranking a larger candidate set."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be >= top_k")
    model = _get_model()
    q_emb = model.encode(
        [query], batch_size=1, max_length=MAX_LENGTH
    )["dense_vecs"][0].tolist()
    col = _get_collection()
    n_results = candidate_k if rerank else top_k
    results = col.query(query_embeddings=[q_emb], n_results=n_results, include=["documents", "metadatas", "distances"])
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
            }
        )
    if rerank:
        from rag.reranker import rerank_chunks

        candidate_max_dense_score = max(chunk["score"] for chunk in chunks)
        ranked = rerank_chunks(query, chunks, top_k)
        for chunk in ranked:
            chunk["candidate_max_dense_score"] = candidate_max_dense_score
        return ranked
    return chunks


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[S{index} | Source: {chunk['source']} | Chunk: {chunk.get('chunk_idx')} | "
        f"Dense score: {chunk['score']}]\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )
