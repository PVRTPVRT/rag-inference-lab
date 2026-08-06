"""Download arXiv papers, chunk them by tokenizer tokens, and ingest ChromaDB."""

import argparse
import os

import arxiv
import chromadb
import fitz
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

from rag.chunking import chunk_text

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "papers"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_MODEL = "BAAI/bge-m3"


def download_papers(query: str, n: int, save_dir: str = "./papers") -> list[str]:
    os.makedirs(save_dir, exist_ok=True)
    search = arxiv.Search(query=query, max_results=n, sort_by=arxiv.SortCriterion.Relevance)
    paths = []
    for result in search.results():
        safe = "".join(c for c in result.title if c.isalnum() or c in " -_").strip()
        path = os.path.join(save_dir, f"{safe}.pdf")
        if not os.path.exists(path):
            result.download_pdf(dirpath=save_dir, filename=f"{safe}.pdf")
            print(f"  Downloaded: {safe}")
        else:
            print(f"  Already exists: {safe}")
        paths.append(path)
    return paths


def extract_text(pdf_path: str) -> str:
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def ingest(
    query: str = "LLM inference optimization",
    n: int = 3,
    embed_device: str = "cuda:0",
):
    print(f"[1/4] Downloading {n} papers for: '{query}'")
    paths = download_papers(query, n)

    print("[2/4] Extracting and chunking text by BGE-M3 tokenizer tokens")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    all_chunks, all_ids, all_metas = [], [], []
    for path in paths:
        title = os.path.splitext(os.path.basename(path))[0]
        text = extract_text(path)
        chunks = chunk_text(text, tokenizer, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{title}__chunk_{i}")
            all_metas.append({"source": title, "chunk_idx": i})
    print(f"   Total chunks: {len(all_chunks)}")

    print(f"[3/4] Generating embeddings with BGE-M3 on {embed_device}")
    model = BGEM3FlagModel(
        EMBED_MODEL,
        devices=embed_device,
        use_fp16=embed_device.startswith("cuda"),
    )
    embeddings = model.encode(
        all_chunks, batch_size=32, max_length=CHUNK_SIZE
    )["dense_vecs"].tolist()

    print("[4/4] Storing in ChromaDB")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME)
    batch = 500
    for start in range(0, len(all_chunks), batch):
        col.add(
            documents=all_chunks[start : start + batch],
            embeddings=embeddings[start : start + batch],
            ids=all_ids[start : start + batch],
            metadatas=all_metas[start : start + batch],
        )
    print(f"   Stored {col.count()} chunks. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="LLM inference optimization vLLM speculative decoding")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument(
        "--embed-device",
        default="cuda:0",
        help="Embedding device for ingestion, for example cuda:0 or cpu",
    )
    args = parser.parse_args()
    ingest(args.query, args.n, args.embed_device)
