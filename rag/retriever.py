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


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Return top_k chunks as [{"source": str, "text": str, "score": float}]."""
    model = _get_model()
    q_emb = model.encode(
        [query], batch_size=1, max_length=MAX_LENGTH
    )["dense_vecs"][0].tolist()
    col = _get_collection()
    results = col.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # ChromaDB returns L2 distance; convert to similarity score
        chunks.append({"source": meta["source"], "text": doc, "score": round(1 / (1 + dist), 4)})
    return chunks


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c['source']} | Score: {c['score']}]\n{c['text']}" for c in chunks
    )
