"""The JSON a finished run leaves in the outbox. Shared by the nightly worker and Run now."""

from __future__ import annotations

import time

from .pipeline import DocumentResult
from .storage import digest


def result_payload(model: str, text: str, result: DocumentResult) -> dict[str, object]:
    return {
        "model": model,
        "draft_sha256": digest(text),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stats": result.stats,
        "warnings": list(result.warnings),
        "edited": result.edited,
        "blocks": [
            {
                "index": b.index,
                "kind": b.kind,
                "status": b.status,
                "original": b.original,
                "result": b.result,
                "proposal": b.proposal,
                "reasons": list(b.reasons),
            }
            for b in result.blocks
        ],
    }


def never_answered(result: DocumentResult) -> str:
    """Why a run produced nothing worth saving, or an empty string if it did.

    When the model never answered a single block (it is down, or the request timed out) the
    result would only say "error" everywhere, so callers keep nothing and try again later.
    """
    stats = result.stats
    if stats.get("error") and not (
        stats.get("edited") or stats.get("unchanged") or stats.get("rejected")
    ):
        first = next((b.reasons[0] for b in result.blocks if b.reasons), "")
        return first or "the model did not answer"
    return ""
