"""Build a document from the original draft plus the model edits the user chose to keep."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class MergeError(ValueError):
    """The chosen edits cannot be applied to this draft. The message is safe to show."""


def merge(original: str, blocks: Iterable[dict[str, Any]], keep: set[int]) -> str:
    """Apply only the ``edited`` blocks whose index is in ``keep``, leaving the rest untouched.

    Every block's ``original`` text is located in the draft in order, kept or not, so a
    paragraph that appears twice is replaced at the right place and blank lines, code and
    links stay exactly as they were.
    """
    crlf = "\r\n" in original
    work = original.replace("\r\n", "\n")
    pieces: list[str] = []
    cursor = 0
    for block in sorted(blocks, key=lambda b: int(b.get("index", 0))):
        before = str(block.get("original", ""))
        wanted = block.get("status") == "edited" and int(block.get("index", -1)) in keep
        position = work.find(before, cursor) if before else -1
        if position < 0:
            if wanted:
                raise MergeError(f"block {block.get('index')} no longer matches the draft")
            continue
        if wanted:
            pieces.append(work[cursor:position])
            pieces.append(str(block.get("result", "")))
            cursor = position + len(before)
        else:
            pieces.append(work[cursor : position + len(before)])
            cursor = position + len(before)
    pieces.append(work[cursor:])
    merged = "".join(pieces)
    return merged.replace("\n", "\r\n") if crlf else merged
