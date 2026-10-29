"""Split a masked document into blocks without losing a single character."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .masking import PLACEHOLDER_RE

EDITABLE_KINDS = frozenset({"paragraph", "list", "quote"})

_SEPARATOR = re.compile(r"(\n(?:[ \t]*\n)+)")
_OUTER_WS = re.compile(r"\A(?P<lead>\s*)(?P<core>.*?)(?P<trail>\s*)\Z", re.DOTALL)

_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(?:=+|-+)\s*$")
_HRULE = re.compile(r"^ {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_QUOTE = re.compile(r"^\s*>")
_FOOTNOTE_DEF = re.compile(r"^\s*\[\^[^\]]+\]:")


@dataclass(frozen=True)
class Chunk:
    index: int
    kind: str
    lead: str
    core: str
    trail: str
    sep: str

    @property
    def editable(self) -> bool:
        return self.kind in EDITABLE_KINDS

    def render(self, core: str | None = None) -> str:
        return self.lead + (self.core if core is None else core) + self.trail + self.sep


def classify(core: str) -> str:
    """Name the kind of Markdown block ``core`` is."""
    if not core.strip():
        return "blank"
    if not PLACEHOLDER_RE.sub("", core).strip():
        return "protected"
    lines = core.split("\n")
    first = lines[0]
    if _ATX_HEADING.match(first):
        return "heading"
    if len(lines) > 1 and _SETEXT_UNDERLINE.match(lines[1]) and not _LIST_ITEM.match(first):
        return "heading"
    if _HRULE.match(first):
        return "protected"
    if _FOOTNOTE_DEF.match(first):
        return "protected"
    if "|" in first and len(lines) > 1 and _TABLE_DIVIDER.match(lines[1]):
        return "protected"
    if _QUOTE.match(first):
        return "quote"
    if _LIST_ITEM.match(first):
        return "list"
    return "paragraph"


def split(text: str) -> list[Chunk]:
    """Cut ``text`` at blank lines. ``"".join(c.render() for c in split(t)) == t``."""
    parts = _SEPARATOR.split(text)
    chunks: list[Chunk] = []
    for n in range(0, len(parts), 2):
        block = parts[n]
        sep = parts[n + 1] if n + 1 < len(parts) else ""
        m = _OUTER_WS.match(block)
        assert m is not None  # the pattern matches every string
        core = m["core"]
        chunks.append(
            Chunk(
                index=len(chunks),
                kind=classify(core),
                lead=m["lead"],
                core=core,
                trail=m["trail"],
                sep=sep,
            )
        )
    return chunks
