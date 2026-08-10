"""Download fixed or searched arXiv papers and build a local ChromaDB index."""

import argparse
import os
from urllib.request import urlretrieve

import arxiv
import chromadb
import fitz
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

from rag.chunking import chunk_text
from rag.parent_child import (
    CHILD_CHUNK_SIZE,
    PARENT_CHILD_COLLECTION_NAME,
    PARENT_CHUNK_SIZE,
    build_parent_child_records,
)

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "papers"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_MODEL = "BAAI/bge-m3"


def download_papers(
    query: str,
    n: int,
    save_dir: str = "./papers",
    arxiv_ids: list[str] | None = None,
) -> list[str]:
    os.makedirs(save_dir, exist_ok=True)
    if arxiv_ids:
        search = arxiv.Search(id_list=arxiv_ids)
    else:
        search = arxiv.Search(
            query=query,
            max_results=n,
            sort_by=arxiv.SortCriterion.Relevance,
        )
    paths = []
    for result in arxiv.Client().results(search):
        safe = "".join(c for c in result.title if c.isalnum() or c in " -_").strip()
        path = os.path.join(save_dir, f"{safe}.pdf")
        if not os.path.exists(path):
            if not result.pdf_url:
                raise ValueError(f"arXiv result has no PDF URL: {result.entry_id}")
            urlretrieve(result.pdf_url, path)
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
    arxiv_ids: list[str] | None = None,
    parent_child: bool = False,
):
    source = f"fixed arXiv IDs: {', '.join(arxiv_ids)}" if arxiv_ids else f"query: '{query}'"
    print(f"[1/4] Downloading papers from {source}")
    paths = download_papers(query, n, arxiv_ids=arxiv_ids)

    print("[2/4] Extracting and chunking text by BGE-M3 tokenizer tokens")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    collection_name = PARENT_CHILD_COLLECTION_NAME if parent_child else COLLECTION_NAME
    all_chunks, all_ids, all_metas = [], [], []
    for path in paths:
        title = os.path.splitext(os.path.basename(path))[0]
        text = extract_text(path)
        if parent_child:
            records = build_parent_child_records(text, tokenizer)
            for record in records:
                parent_index = record["parent_chunk_idx"]
                child_index = record["child_chunk_idx"]
                all_chunks.append(record["child_text"])
                all_ids.append(f"{title}__parent_{parent_index}__child_{child_index}")
                all_metas.append({
                    "source": title,
                    "parent_chunk_idx": parent_index,
                    "child_chunk_idx": child_index,
                    "parent_text": record["parent_text"],
                })
        else:
            chunks = chunk_text(text, tokenizer, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{title}__chunk_{i}")
                all_metas.append({"source": title, "chunk_idx": i})
    unit = "child chunks" if parent_child else "chunks"
    print(f"   Total {unit}: {len(all_chunks)}")

    print(f"[3/4] Generating embeddings with BGE-M3 on {embed_device}")
    model = BGEM3FlagModel(
        EMBED_MODEL,
        devices=embed_device,
        use_fp16=embed_device.startswith("cuda"),
    )
    embedding_length = CHILD_CHUNK_SIZE if parent_child else CHUNK_SIZE
    embeddings = model.encode(
        all_chunks, batch_size=32, max_length=embedding_length
    )["dense_vecs"].tolist()

    print(f"[4/4] Storing in ChromaDB collection '{collection_name}'")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
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
    parser.add_argument(
        "--arxiv-ids",
        nargs="+",
        help="Fixed arXiv IDs for a reproducible corpus; overrides query-based discovery",
    )
    parser.add_argument(
        "--parent-child",
        action="store_true",
        help="Build a separate child-vector index that returns 512-token parents",
    )
    args = parser.parse_args()
    ingest(args.query, args.n, args.embed_device, args.arxiv_ids, args.parent_child)
