"""Show what a model changed."""

from __future__ import annotations

import difflib
import re

_TOKEN = re.compile(r"\s+|\w+|[^\w\s]")


def unified_diff(original: str, edited: str, name: str = "draft.md") -> str:
    """A classic line-based unified diff. Empty when nothing changed."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        edited.splitlines(keepends=True),
        fromfile=f"a/{name}",
        tofile=f"b/{name}",
    )
    return "".join(diff)


def word_diff(original: str, edited: str) -> list[tuple[str, str]]:
    """Word-level changes as ``(op, text)`` pairs, ``op`` being equal, delete or insert."""
    a, b = _TOKEN.findall(original), _TOKEN.findall(edited)
    pairs: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            pairs.append(("equal", "".join(a[i1:i2])))
            continue
        if i2 > i1:
            pairs.append(("delete", "".join(a[i1:i2])))
        if j2 > j1:
            pairs.append(("insert", "".join(b[j1:j2])))
    return pairs
