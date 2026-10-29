"""Reject model output that changed something it should not have."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .masking import PLACEHOLDER_RE

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_LINE_PREFIX = re.compile(r"^\s*(?:>\s?)*(?:(?:[-*+]|\d+[.)])\s+)?(?:\[[ xX]\]\s+)?")


@dataclass(frozen=True)
class CheckConfig:
    min_ratio: float = 0.5
    max_ratio: float = 1.8
    ratio_min_length: int = 30
    strict_order: bool = True
    check_numbers: bool = True


def _numbers(text: str) -> Counter[str]:
    return Counter(_NUMBER.findall(PLACEHOLDER_RE.sub(" ", text)))


def _prefix(line: str) -> str:
    m = _LINE_PREFIX.match(line)
    return m.group(0) if m else ""


def check_block(
    kind: str, original: str, edited: str, config: CheckConfig | None = None
) -> list[str]:
    """Return the reasons ``edited`` is unsafe to use. An empty list means it passed."""
    cfg = config or CheckConfig()
    if not edited.strip():
        return ["model returned an empty result"]

    problems: list[str] = []

    before = PLACEHOLDER_RE.findall(original)
    after = PLACEHOLDER_RE.findall(edited)
    if Counter(before) != Counter(after):
        lost = sorted((Counter(before) - Counter(after)).elements())
        extra = sorted((Counter(after) - Counter(before)).elements())
        if lost:
            problems.append(f"placeholders lost: {', '.join(lost)}")
        if extra:
            problems.append(f"placeholders added or duplicated: {', '.join(extra)}")
    elif cfg.strict_order and before != after:
        problems.append("placeholders were reordered")

    if edited.count("`") > original.count("`"):
        problems.append("model introduced backticks")
    if edited.count("://") > original.count("://"):
        problems.append("model introduced a URL")

    if cfg.check_numbers and _numbers(original) != _numbers(edited):
        problems.append("numbers changed")

    if len(original) >= cfg.ratio_min_length:
        ratio = len(edited) / len(original)
        if not cfg.min_ratio <= ratio <= cfg.max_ratio:
            problems.append(f"length changed too much (x{ratio:.2f})")

    if kind in {"list", "quote"}:
        old_lines, new_lines = original.split("\n"), edited.split("\n")
        if len(old_lines) != len(new_lines):
            problems.append(f"line count changed ({len(old_lines)} -> {len(new_lines)})")
        elif any(_prefix(a) != _prefix(b) for a, b in zip(old_lines, new_lines, strict=True)):
            problems.append("list or quote markers changed")
    elif kind == "paragraph":
        if _BLANK_LINE.search(edited):
            problems.append("paragraph was split by a blank line")
        elif "\n" not in original and "\n" in edited:
            problems.append("single-line paragraph gained line breaks")

    return problems
