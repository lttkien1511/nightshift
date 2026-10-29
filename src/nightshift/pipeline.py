"""Run a draft through an editor, keeping only the edits that pass every check."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .checks import CheckConfig, check_block
from .chunking import EDITABLE_KINDS, split
from .masking import PLACEHOLDER_RE, mask, unmask

Editor = Callable[[str], str]
Status = Literal["edited", "unchanged", "rejected", "skipped", "error"]


@dataclass(frozen=True)
class PipelineConfig:
    edit_headings: bool = False
    checks: CheckConfig = field(default_factory=CheckConfig)


@dataclass(frozen=True)
class BlockResult:
    index: int
    kind: str
    status: Status
    original: str
    result: str
    proposal: str | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentResult:
    original: str
    edited: str
    blocks: tuple[BlockResult, ...]
    warnings: tuple[str, ...] = ()

    @property
    def stats(self) -> dict[str, int]:
        return dict(Counter(block.status for block in self.blocks))

    @property
    def changed(self) -> bool:
        return self.original != self.edited


def process_document(
    text: str, editor: Editor, config: PipelineConfig | None = None
) -> DocumentResult:
    """Edit ``text`` block by block.

    ``editor`` only ever sees masked prose. Output that fails a check is dropped and the
    original block is kept, so the result can never be worse than the input.
    """
    cfg = config or PipelineConfig()
    crlf = "\r\n" in text
    work = text.replace("\r\n", "\n") if crlf else text

    masked = mask(work)
    editable = EDITABLE_KINDS | ({"heading"} if cfg.edit_headings else frozenset())

    rendered: list[str] = []
    results: list[BlockResult] = []
    for chunk in split(masked.text):
        applied = chunk.core
        status: Status = "skipped"
        proposal: str | None = None
        reasons: tuple[str, ...] = ()

        if chunk.kind in editable:
            try:
                raw = editor(chunk.core)
            except Exception as exc:
                status, reasons = "error", (f"editor failed: {exc}",)
            else:
                candidate = raw.strip()
                problems = check_block(chunk.kind, chunk.core, candidate, cfg.checks)
                if problems:
                    status, reasons = "rejected", tuple(problems)
                    proposal = unmask(raw, masked.originals)
                elif candidate == chunk.core:
                    status = "unchanged"
                else:
                    status, applied = "edited", candidate

        results.append(
            BlockResult(
                index=chunk.index,
                kind=chunk.kind,
                status=status,
                original=unmask(chunk.core, masked.originals),
                result=unmask(applied, masked.originals),
                proposal=proposal,
                reasons=reasons,
            )
        )
        rendered.append(chunk.render(applied))

    edited = unmask("".join(rendered), masked.originals)
    if PLACEHOLDER_RE.search(edited):
        raise RuntimeError("a placeholder leaked into the output")
    if crlf:
        edited = edited.replace("\n", "\r\n")

    return DocumentResult(
        original=text,
        edited=edited,
        blocks=tuple(results),
        warnings=tuple(masked.warnings),
    )
