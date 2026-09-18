"""Text chunkers."""

from __future__ import annotations

from typing import List


def sliding_window_chunk(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    step = max(1, chunk_chars - overlap)
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i + chunk_chars])
        if i + chunk_chars >= len(text):
            break
        i += step
    return out
