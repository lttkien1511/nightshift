"""The nightly worker: edit every queued draft with each configured model, one model at a time."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .ollama import OllamaConfig, OllamaEditor, unload
from .pipeline import DocumentResult, Editor, process_document
from .storage import Store
from .web import store_from_env

log = logging.getLogger("nightshift.worker")

EditorFactory = Callable[[str], Editor]


@dataclass
class Summary:
    done: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    timed_out: bool = False


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def run(
    store: Store,
    models: list[str],
    editor_for: EditorFactory,
    release: Callable[[str], None] = lambda model: None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Summary:
    """Process the queue. Models run one after another so only one is in memory at a time."""
    summary = Summary()
    store.ensure()
    for model in models:
        editor = editor_for(model)
        try:
            for draft in store.drafts():
                if deadline is not None and clock() >= deadline:
                    summary.timed_out = True
                    return summary
                text = store.read(draft.slug)
                if text is None:
                    continue
                key = (draft.slug, model)
                existing = store.result(draft.slug, model)
                if existing and existing.get("draft_sha256") == digest(text):
                    summary.skipped.append(key)
                    continue
                result = process_document(text, editor)
                outcomes = result.stats
                if outcomes.get("error") and not (
                    outcomes.get("edited") or outcomes.get("unchanged") or outcomes.get("rejected")
                ):
                    # The model never answered: leave no result so tomorrow's run retries.
                    log.error("%s / %s: every block failed", draft.slug, model)
                    summary.failed.append(key)
                    continue
                store.write_result(draft.slug, model, result_payload(model, text, result))
                log.info("%s / %s: %s", draft.slug, model, outcomes)
                summary.done.append(key)
        finally:
            release(model)
    return summary


def _models_from_env() -> list[str]:
    raw = os.environ.get("NIGHTSHIFT_MODELS", "qwen2.5:7b")
    return [m.strip() for m in raw.split(",") if m.strip()]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = store_from_env()
    store.ensure()
    voice = store.voice_path.read_text(encoding="utf-8") if store.voice_path.is_file() else ""
    base = OllamaConfig(
        url=os.environ.get("NIGHTSHIFT_OLLAMA_URL", OllamaConfig.url),
        timeout=float(os.environ.get("NIGHTSHIFT_OLLAMA_TIMEOUT", OllamaConfig.timeout)),
        num_ctx=int(os.environ.get("NIGHTSHIFT_OLLAMA_NUM_CTX", OllamaConfig.num_ctx)),
    )

    def cfg_for(model: str) -> OllamaConfig:
        return OllamaConfig(url=base.url, model=model, timeout=base.timeout, num_ctx=base.num_ctx)

    budget = float(os.environ.get("NIGHTSHIFT_BUDGET_SECONDS", "19800"))
    summary = run(
        store,
        _models_from_env(),
        lambda model: OllamaEditor(cfg_for(model), voice),
        release=lambda model: unload(cfg_for(model)),
        deadline=time.monotonic() + budget,
    )
    log.info(
        "done=%d skipped=%d failed=%d timed_out=%s",
        len(summary.done),
        len(summary.skipped),
        len(summary.failed),
        summary.timed_out,
    )
    return 1 if summary.failed and not summary.done else 0


if __name__ == "__main__":
    raise SystemExit(main())
