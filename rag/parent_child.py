"""Parent-child chunk construction shared by ingestion and evaluation."""

from __future__ import annotations

from rag.chunking import chunk_text

PARENT_CHILD_COLLECTION_NAME = "papers_parent_child"
PARENT_CHUNK_SIZE = 512
PARENT_CHUNK_OVERLAP = 50
CHILD_CHUNK_SIZE = 192
CHILD_CHUNK_OVERLAP = 32


def build_parent_child_records(
    text: str,
    tokenizer,
    *,
    parent_size: int = PARENT_CHUNK_SIZE,
    parent_overlap: int = PARENT_CHUNK_OVERLAP,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> list[dict]:
    """Create searchable child chunks that retain their complete parent text."""
    if child_size >= parent_size:
        raise ValueError("child_size must be smaller than parent_size")
    parents = chunk_text(
        text, tokenizer, size=parent_size, overlap=parent_overlap
    )
    records = []
    for parent_index, parent_text in enumerate(parents):
        children = chunk_text(
            parent_text,
            tokenizer,
            size=child_size,
            overlap=child_overlap,
        )
        for child_index, child_text in enumerate(children):
            records.append({
                "parent_chunk_idx": parent_index,
                "child_chunk_idx": child_index,
                "parent_text": parent_text,
                "child_text": child_text,
            })
    return records
