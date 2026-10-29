"""Hide the parts of a Markdown draft that must never change.

Code, links and markup are swapped for placeholders such as ``[[CODE_1]]`` before
the text reaches a model, and swapped back afterwards. A model cannot corrupt
what it never sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PLACEHOLDER_RE = re.compile(r"\[\[(?:BLOCK|CODE|HTML|URL|LINK)_\d+\]\]")


class MaskingError(ValueError):
    """The document cannot be masked safely."""


@dataclass
class Masked:
    text: str
    originals: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


_FRONTMATTER = re.compile(
    r"\A(?P<fence>---|\+\+\+)[ \t]*\n.*?\n(?P=fence)[ \t]*(?:\n|\Z)", re.DOTALL
)
_FENCE_OPEN = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_INLINE_CODE = re.compile(
    r"(?<!`)(?P<ticks>`+)(?!`)(?P<body>(?:(?!\n[ \t]*\n).)+?)(?<!`)(?P=ticks)(?!`)",
    re.DOTALL,
)
_HTML_TAG = re.compile(
    r"<(?:/?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?|(?:https?|ftp|mailto):[^<>\s]+)>"
)
_BARE_URL = re.compile(r"(?:https?|ftp)://[^\s<>\[\]]+")
_LINK_DEST = re.compile(
    r"\]\((?P<dest>[^)\s]+(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?)\)",
)
_REF_DEF = re.compile(r"^ {0,3}\[[^\]\n]+\]:[ \t]+\S.*$", re.MULTILINE)

_URL_TRAILING = ".,;:!?\"'*_~"


class _Store:
    def __init__(self) -> None:
        self.originals: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def put(self, kind: str, original: str) -> str:
        # Expand placeholders nested inside the span so that unmask() is a single pass.
        expanded = PLACEHOLDER_RE.sub(
            lambda m: self.originals.get(m.group(0), m.group(0)), original
        )
        self._counts[kind] = self._counts.get(kind, 0) + 1
        key = f"[[{kind}_{self._counts[kind]}]]"
        self.originals[key] = expanded
        return key


def _mask_frontmatter(text: str, store: _Store) -> str:
    m = _FRONTMATTER.match(text)
    if not m:
        return text
    block = m.group(0)
    newline = "\n" if block.endswith("\n") else ""
    return store.put("BLOCK", block[: len(block) - len(newline)]) + newline + text[m.end() :]


def _mask_fences(text: str, store: _Store, warnings: list[str]) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN.match(lines[i])
        if m is None or (m["fence"][0] == "`" and "`" in m["info"]):
            out.append(lines[i])
            i += 1
            continue
        char, size = m["fence"][0], len(m["fence"])
        closer = re.compile(rf"^[ \t]*{re.escape(char)}{{{size},}}[ \t]*$")
        end = next((j for j in range(i + 1, len(lines)) if closer.match(lines[j])), None)
        if end is None:
            warnings.append(
                f"unclosed code fence starting at line {i + 1}; masked to end of document"
            )
            end = len(lines) - 1
        out.append(store.put("BLOCK", "\n".join(lines[i : end + 1])))
        i = end + 1
    return "\n".join(out)


def _trim_url(url: str) -> str:
    while url:
        last = url[-1]
        if last in _URL_TRAILING or (last == ")" and url.count(")") > url.count("(")):
            url = url[:-1]
        else:
            break
    return url


def _mask_bare_urls(text: str, store: _Store) -> str:
    def repl(m: re.Match[str]) -> str:
        url = _trim_url(m.group(0))
        if not url:
            return m.group(0)
        return store.put("URL", url) + m.group(0)[len(url) :]

    return _BARE_URL.sub(repl, text)


def _mask_link_destinations(text: str, store: _Store) -> str:
    def repl(m: re.Match[str]) -> str:
        dest = m.group("dest")
        if PLACEHOLDER_RE.fullmatch(dest):
            return m.group(0)
        return "](" + store.put("LINK", dest) + ")"

    return _LINK_DEST.sub(repl, text)


def mask(text: str) -> Masked:
    """Replace code, links and raw HTML in ``text`` with placeholders."""
    if PLACEHOLDER_RE.search(text):
        raise MaskingError("document already contains placeholder-like text such as [[CODE_1]]")

    store = _Store()
    warnings: list[str] = []

    text = _mask_frontmatter(text, store)
    text = _mask_fences(text, store, warnings)
    text = _HTML_COMMENT.sub(lambda m: store.put("HTML", m.group(0)), text)
    text = _INLINE_CODE.sub(lambda m: store.put("CODE", m.group(0)), text)
    text = _HTML_TAG.sub(lambda m: store.put("HTML", m.group(0)), text)
    text = _mask_bare_urls(text, store)
    text = _mask_link_destinations(text, store)
    text = _REF_DEF.sub(lambda m: store.put("LINK", m.group(0)), text)

    return Masked(text=text, originals=store.originals, warnings=warnings)


def unmask(text: str, originals: dict[str, str]) -> str:
    """Put the original spans back. Unknown placeholders are left untouched."""
    return PLACEHOLDER_RE.sub(lambda m: originals.get(m.group(0), m.group(0)), text)
