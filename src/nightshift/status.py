"""Machine-readable state for the separate monitor app. It carries no draft text."""

from __future__ import annotations

from .storage import Store, digest


def queue_counts(store: Store) -> tuple[int, int, int]:
    """How many drafts are waiting, have results, and have results older than the draft."""
    waiting = ready = stale = 0
    for draft in store.drafts():
        text = store.read(draft.slug) or ""
        results = store.results(draft.slug)
        if not results:
            waiting += 1
            continue
        ready += 1
        if any(r.get("draft_sha256") != digest(text) for r in results):
            stale += 1
    return waiting, ready, stale


def snapshot(store: Store, version: str) -> dict[str, object]:
    waiting, ready, stale = queue_counts(store)
    return {
        "version": version,
        "queue": {"waiting": waiting, "ready": ready, "stale": stale},
        "worker": store.worker_status() or {},
    }
