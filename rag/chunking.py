"""Tokenizer-aware text chunking with overlap."""

from __future__ import annotations


def chunk_token_ids(token_ids: list[int], size: int = 512, overlap: int = 50) -> list[list[int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    step = size - overlap
    chunks = []
    for start in range(0, len(token_ids), step):
        chunks.append(token_ids[start : start + size])
        if start + size >= len(token_ids):
            break
    return chunks


def chunk_text(text: str, tokenizer, size: int = 512, overlap: int = 50) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(chunk, skip_special_tokens=True).strip()
        for chunk in chunk_token_ids(token_ids, size=size, overlap=overlap)
        if chunk
    ]
